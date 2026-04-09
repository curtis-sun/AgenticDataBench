"""
Parallel Eval Case Synthesizer

Architecture:
  Main process (Coordinator):
    - Owns PathSampler (~1.1 GB embeddings, single instance)
    - Samples skill paths and submits tasks to a ProcessPoolExecutor
    - Uses as_completed to process results as soon as any worker finishes
    - Immediately commits coverage and flushes to disk, then backfills a new task

  Worker processes:
    - Each initializes its own QwenClient + DatasetLoader + EvalCaseSynthesizer
    - Receives a serialized task dict, reconstructs SampledPath, calls synthesize()
    - Returns (case_dict, used_skill_ids) or None
"""

import os
import re
import json
import time
import random
import argparse
import traceback
from collections import Counter
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_classes import Skill, EvalCase
from dataset_loader import DatasetLoader
from sample_graph import PathSampler
from synthesizer import EvalCaseSynthesizer
import sys
sys.path.append('..')
from utils.config import DATASETS_DIR, OUTPUT_DIR, QWEN_MODEL, get_min_files_for_domain
from utils.llm_client import QwenClient

# Extra skills to sample beyond the target step count, providing a buffer
# so that a few auxiliary-skill failures don't drop the case below the tier minimum.
SKILL_BUFFER = 3


# ---------------------------------------------------------------------------
# Uncovered-skills loader
# ---------------------------------------------------------------------------

def _load_uncovered_skills(filepath: str, path_sampler: PathSampler) -> List[str]:
    """Load uncovered skills from a JSONL file and resolve names to graph node IDs.

    Each line is ``{"skill": "<name>", "type": "<category>"}``.
    Returns a list of node IDs that exist in the PathSampler graph.
    """
    name_to_id: Dict[str, str] = {}
    for node_id in path_sampler.top_node_indices:
        node = path_sampler.nodes_id_dict[node_id]
        name_to_id[node.name] = node_id

    resolved: List[str] = []
    not_found: List[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            skill_name = entry["skill"]
            if skill_name in name_to_id:
                resolved.append(name_to_id[skill_name])
            else:
                not_found.append(skill_name)

    print(f"[Uncovered] Loaded {len(resolved)} uncovered skills from {filepath}")
    if not_found:
        print(f"  [Warning] {len(not_found)} skills not found in graph "
              f"(showing first 5):")
        for name in not_found[:5]:
            print(f"    - {name}")
        if len(not_found) > 5:
            print(f"    ... and {len(not_found) - 5} more")

    return resolved


# ---------------------------------------------------------------------------
# Round directory management
# ---------------------------------------------------------------------------

def _next_round_dir(base_dir):
    """Create and return the next ``roundN`` directory under *base_dir*."""
    os.makedirs(base_dir, exist_ok=True)
    existing_nums = []
    for name in os.listdir(base_dir):
        m = re.fullmatch(r"round(\d+)", name)
        if m and os.path.isdir(os.path.join(base_dir, name)):
            existing_nums.append(int(m.group(1)))
    next_num = max(existing_nums, default=0) + 1
    round_dir = os.path.join(base_dir, f"round{next_num}")
    os.makedirs(round_dir, exist_ok=True)
    return round_dir


# ---------------------------------------------------------------------------
# Helper data classes (picklable, used across process boundaries)
# ---------------------------------------------------------------------------

class SkillExample:
    def __init__(self, id, covered_skills, steps, title="", answer=""):
        self.id = id
        self.covered_skills = covered_skills
        self.steps = steps
        self.title = title
        self.answer = answer

    def get_text(self):
        parts = []
        if self.title:
            parts.append(f"**Problem**: {self.title}")
        if self.answer:
            parts.append(f"**Solution**:\n{self.answer}")
        parts.append("**Steps**:\n" + "\n".join([f"- {s['text']}" for s in self.steps]))
        return "\n\n".join(parts)


class SampledPath:
    def __init__(self, path_id, skills, examples):
        self.path_id = path_id
        self.skills = skills
        self.examples = examples

    def get_examples_for_skill(self, skill_name, max_examples=3):
        relevant = [ex for ex in self.examples if skill_name in ex.covered_skills]
        result = []
        for ex in relevant[:max_examples]:
            result.append(f"### Example {ex.id}\n{ex.get_text()}")
        return "\n\n".join(result)



# ---------------------------------------------------------------------------
# Worker-side globals (initialised once per process via pool initializer)
# ---------------------------------------------------------------------------

_worker_llm: Optional[QwenClient] = None
_worker_loader: Optional[DatasetLoader] = None
_worker_synthesizer: Optional[EvalCaseSynthesizer] = None


def _init_worker(datasets_dir: str, min_files: int):
    global _worker_llm, _worker_loader, _worker_synthesizer
    _worker_llm = QwenClient(model=QWEN_MODEL)
    _worker_loader = DatasetLoader(datasets_dir, llm_client=_worker_llm)
    _worker_synthesizer = EvalCaseSynthesizer(_worker_llm, min_files=min_files)


def _worker_synthesize(task: dict) -> Optional[dict]:
    """
    Worker entry point.  Receives a serialised task dict, runs synthesis,
    and returns a result dict (or None on failure).
    """
    try:
        skills = [Skill(**s) for s in task["skills"]]
        examples = [SkillExample(**e) for e in task["examples"]]
        sampled_path = SampledPath(
            path_id=task["task_id"],
            skills=skills,
            examples=examples,
        )

        domain = task["domain"]
        domain_dataset = _worker_loader.load_domain(domain)
        if not domain_dataset or not domain_dataset.files:
            return {"task_id": task["task_id"], "domain": domain,
                    "case": None, "used_skill_ids": [],
                    "synthesis_log": {"outcome": "failed",
                                      "failure_reason": "dataset_load_failed"}}

        _worker_llm.reset_token_count()
        t_start = time.time()

        case, used_skill_ids, syn_log = _worker_synthesizer.synthesize(
            sampled_path,
            domain_dataset,
            min_skills=task.get("min_skills", 2),
            target_steps=task.get("target_steps"),
            min_files=task.get("min_files"),
            rare_skill_names=task.get("rare_skill_names"),
            required_files=task.get("required_files"),
        )

        elapsed = time.time() - t_start
        token_usage = _worker_llm.get_token_count()

        syn_log["task_id"] = task["task_id"]
        syn_log["elapsed_seconds"] = round(elapsed, 2)
        syn_log["token_usage"] = token_usage

        if case is None:
            print(f"[Worker] task {task.get('task_id')} failed | "
                  f"Time: {elapsed:.1f}s | Tokens: {token_usage['total_tokens']}")
            return {
                "task_id": task["task_id"],
                "domain": domain,
                "case": None,
                "used_skill_ids": [],
                "synthesis_log": syn_log,
            }

        case.synthesis_time_seconds = round(elapsed, 2)
        case.token_usage = token_usage

        print(f"[Worker] task {task.get('task_id')} done | "
              f"Time: {elapsed:.1f}s | Tokens: {token_usage['total_tokens']} "
              f"(prompt: {token_usage['prompt_tokens']}, completion: {token_usage['completion_tokens']})")

        return {
            "task_id": task["task_id"],
            "domain": domain,
            "case": case.to_dict(),
            "used_skill_ids": used_skill_ids,
            "synthesis_log": syn_log,
        }
    except Exception as e:
        print(f"[Worker] task {task.get('task_id')} error: {e}")
        traceback.print_exc()
        return {"task_id": task.get("task_id"), "domain": task.get("domain"),
                "case": None, "used_skill_ids": [],
                "synthesis_log": {"outcome": "error",
                                  "failure_reason": str(e)}}


# ---------------------------------------------------------------------------
# Coordinator (main process)
# ---------------------------------------------------------------------------

class ParallelCoordinator:
    def __init__(
        self,
        path_sampler: PathSampler,
        datasets_dir: str,
        output_dir: str,
        n_workers: int = 4,
        min_files: int = 5,
        min_skills: int = 2,
        rare_skill_ratio: float = 0.1,
        min_new_skills_per_path: int = 3,
        uncovered_skill_ids: List[str] = None,
        uncovered_per_case: int = 2,
        required_files: List[str] = None,
    ):
        self.path_sampler = path_sampler
        self.datasets_dir = datasets_dir
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.min_files = min_files
        self.min_skills = min_skills
        self.rare_skill_ratio = rare_skill_ratio
        self.min_new_skills_per_path = min_new_skills_per_path
        self._in_flight_skill_ids: set = set()
        self._required_files: List[str] = required_files or []

        # Uncovered-skills mode
        self._uncovered_pool: List[str] = list(uncovered_skill_ids or [])
        self._uncovered_per_case: int = uncovered_per_case
        self._uncovered_usage: Counter = Counter()

        os.makedirs(output_dir, exist_ok=True)

    # ----- uncovered-skills helpers -----

    def _pick_uncovered_skills(self) -> List[str]:
        """Select uncovered skills for one task, prioritising least-used ones.

        Returns up to ``self._uncovered_per_case`` skill IDs from the pool,
        avoiding skills already in-flight when possible.
        """
        if not self._uncovered_pool or self._uncovered_per_case <= 0:
            return []

        candidates = [
            sid for sid in self._uncovered_pool
            if sid not in self._in_flight_skill_ids
        ]
        if not candidates:
            candidates = list(self._uncovered_pool)

        # Least-used first, random tie-break
        candidates.sort(key=lambda sid: (self._uncovered_usage[sid], random.random()))
        selected = candidates[: self._uncovered_per_case]

        for sid in selected:
            self._uncovered_usage[sid] += 1

        return selected

    def _inject_skill_into_path(self, path_nodes: list, skill_id: str) -> list:
        """Insert a skill node at the position with the strongest edge affinity."""
        if skill_id in {n.id for n in path_nodes}:
            return path_nodes
        if skill_id not in self.path_sampler.nodes_id_dict:
            return path_nodes

        skill_node = self.path_sampler.nodes_id_dict[skill_id]
        G = self.path_sampler.G

        best_pos = len(path_nodes)
        best_weight = -1.0
        for i, node in enumerate(path_nodes):
            w = 0.0
            if G.has_edge(node.id, skill_id):
                w = max(w, G[node.id][skill_id].get("weight", 0))
            if G.has_edge(skill_id, node.id):
                w = max(w, G[skill_id][node.id].get("weight", 0))
            if w > best_weight:
                best_weight = w
                best_pos = i + 1

        path_nodes = list(path_nodes)
        path_nodes.insert(best_pos, skill_node)
        return path_nodes

    # ----- batch sampling with temporary coverage bumps -----

    def batch_sample_paths(
        self,
        batch_size: int,
        max_steps: int,
    ) -> List[Tuple[list, list, list]]:
        """
        Sample *batch_size* diverse skill paths.

        Returns a list of (path_nodes, path_ids, evidence) tuples.
        Skill diversity is ensured by temporarily bumping coverage after
        each sample so the next sample naturally avoids the same skills.

        We sample ``max_steps - 1 + SKILL_BUFFER`` skills so that a few
        auxiliary-skill failures during synthesis don't drop the case
        below the minimum step count.
        """
        paths = []
        reserved: List[list] = []
        batch_skill_ids: set = set()
        skill_budget = max_steps - 1 + SKILL_BUFFER

        for _ in range(batch_size):
            path_nodes = self.path_sampler.sample_with_novelty(
                max_steps=skill_budget,
                existing_skill_ids=batch_skill_ids,
                min_new_skills=self.min_new_skills_per_path,
            )
            path_ids = [n.id for n in path_nodes]
            evidence = self.path_sampler.sample_evidence(path_ids, max_examples=10)

            paths.append((path_nodes, path_ids, evidence))
            batch_skill_ids.update(path_ids)

            self.path_sampler.reserve_skills(path_ids)
            reserved.append(path_ids)

        for skill_ids in reserved:
            self.path_sampler.release_skills(skill_ids)

        return paths

    # ----- serialise a sampled path into a worker task dict -----

    def _serialize_task(
        self,
        task_id: int,
        path_nodes: list,
        path_ids: list,
        evidence: list,
        domain: str,
        min_skills: int,
        target_steps: int = None,
        min_files: int = None,
        rare_skill_names: List[str] = None,
        required_files: List[str] = None,
    ) -> dict:
        skills_data = []
        seen = set()
        for node in path_nodes:
            if node.id in seen:
                continue
            seen.add(node.id)
            skills_data.append({
                "id": node.id,
                "name": node.name,
                "description": self.path_sampler.get_skill_description(node.id),
            })

        examples_data = []
        for ex_id, steps in evidence:
            ex = self.path_sampler.example_dict[ex_id]
            covered = [
                s.name
                for s in self.path_sampler.get_example_covered_skills(ex, path_ids)
            ]
            examples_data.append({
                "id": ex_id,
                "covered_skills": covered,
                "steps": [{"step_id": step.id, "text": step.text} for step in steps],
                "title": ex.title,
                "answer": ex.answer,
            })

        return {
            "task_id": task_id,
            "skills": skills_data,
            "examples": examples_data,
            "domain": domain,
            "min_skills": min_skills,
            "path_ids": path_ids,
            "target_steps": target_steps,
            "min_files": min_files,
            "rare_skill_names": rare_skill_names,
            "required_files": required_files,
        }

    # ----- apply real weight updates for successful cases -----

    def _commit_results(self, results: List[Optional[dict]], evidence_map: dict):
        """Call add_path for every successful result."""
        for r in results:
            if r is None or r.get("case") is None:
                continue
            used_ids = r["used_skill_ids"]
            if not used_ids:
                continue
            original_evidence = evidence_map.get(r["task_id"], [])
            used_evidence = []
            for ex_id, steps in original_evidence:
                relevant = [
                    s for s in steps
                    if any(sid in s.skills for sid in used_ids)
                ]
                if relevant:
                    used_evidence.append((ex_id, relevant))
            self.path_sampler.add_path(used_ids, used_evidence)

    # ----- mixed-batch helpers -----

    def _sample_one_task_for_slot(
        self,
        slot: Dict,
        global_task_counter: List[int],
    ) -> Tuple[dict, list]:
        """Sample a single skill path and build one worker task for *slot*.

        When uncovered-skills mode is active, selected uncovered skills are
        injected into the sampled path (the first as a rare-injection seed,
        the rest inserted at positions with the strongest edge affinity).

        Returns:
            (task_dict, evidence) — the serialised task and the raw evidence
            list (needed later for ``_commit_results``).
        """
        domain = slot["domain"]
        steps_target = random.randint(slot["min_steps"], slot["max_steps"])
        skill_budget = steps_target - 1 + SKILL_BUFFER

        uncovered_for_task = self._pick_uncovered_skills()

        if uncovered_for_task:
            # Use the first uncovered skill as seed for rare injection
            seed_skill = uncovered_for_task[0]
            path_nodes = self.path_sampler.sample_with_rare_and_novelty(
                seed_skill, skill_budget,
                existing_skill_ids=self._in_flight_skill_ids,
                min_new_skills=self.min_new_skills_per_path,
            )
            # Inject remaining uncovered skills at their best positions
            for uc_id in uncovered_for_task[1:]:
                path_nodes = self._inject_skill_into_path(path_nodes, uc_id)
        else:
            # Normal path: occasional rare-skill injection
            rare_skills = self.path_sampler.get_rare_skills()
            if rare_skills and random.random() < self.rare_skill_ratio:
                rare_skill_id = random.choice(rare_skills)
                path_nodes = self.path_sampler.sample_with_rare_and_novelty(
                    rare_skill_id, skill_budget,
                    existing_skill_ids=self._in_flight_skill_ids,
                    min_new_skills=self.min_new_skills_per_path,
                )
            else:
                path_nodes = self.path_sampler.sample_with_novelty(
                    max_steps=skill_budget,
                    existing_skill_ids=self._in_flight_skill_ids,
                    min_new_skills=self.min_new_skills_per_path,
                )

        path_ids = [n.id for n in path_nodes]
        self._in_flight_skill_ids.update(path_ids)
        evidence = self.path_sampler.sample_evidence(path_ids, max_examples=10)

        tid = global_task_counter[0]
        global_task_counter[0] += 1
        slot["attempts"] += 1

        # Treat both graph-rare and uncovered skills as rare_skill_names
        # so the synthesiser gives them extra attention
        rare_skill_names = [
            n.name for n in path_nodes if self.path_sampler.is_rare_skill(n.id)
        ]
        if uncovered_for_task:
            uncovered_name_set = {
                self.path_sampler.nodes_id_dict[sid].name
                for sid in uncovered_for_task
                if sid in self.path_sampler.nodes_id_dict
            }
            for name in uncovered_name_set:
                if name not in rare_skill_names:
                    rare_skill_names.append(name)

        min_files_this_case = get_min_files_for_domain(domain)
        task = self._serialize_task(
            tid, path_nodes, path_ids, evidence, domain, self.min_skills,
            target_steps=steps_target,
            min_files=min_files_this_case,
            rare_skill_names=rare_skill_names or None,
            required_files=self._required_files or None,
        )
        return task, evidence

    # ----- top-level run across all domains (mixed-batch round-robin) -----

    @staticmethod
    def _count_existing_cases(output_dir: str, domains: List[str]) -> Dict[str, int]:
        """Count existing cases per domain from output files.

        Scans *output_dir* and all ``roundN`` subdirectories.
        Returns {domain: count}.
        """
        counts: Dict[str, int] = {d: 0 for d in domains}
        scan_dirs = [output_dir]
        if os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                sub = os.path.join(output_dir, name)
                if os.path.isdir(sub) and re.fullmatch(r"round\d+", name):
                    scan_dirs.append(sub)

        for scan_dir in scan_dirs:
            for domain in domains:
                out_path = os.path.join(scan_dir, f"eval_cases_({domain}).jsonl")
                if not os.path.exists(out_path):
                    continue
                with open(out_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            counts[domain] += 1
        return counts

    def run(
        self,
        domains: List[str],
        cases_per_domain: int = 20,
        max_attempts_multiplier: int = 3,
        append: bool = False,
        min_steps: int = 3,
        max_steps: int = 8,
    ):
        """Generate cases for every domain.

        Uses **async dispatch** scheduling: tasks are submitted to a
        ``ProcessPoolExecutor`` and processed via ``as_completed``.
        Each completed result is immediately committed (coverage update
        + disk flush) and a replacement task is submitted from the next
        active slot, keeping all workers saturated without synchronous
        batch boundaries.

        When *append* is True, all existing round directories are scanned.
        """
        # -- 0. Count existing cases when continuing / appending -------------
        existing_counts: Dict[str, int] = {}
        if append:
            existing_counts = self._count_existing_cases(self.output_dir, domains)
            total_existing = sum(existing_counts.values())
            print(f"\n[Append] Found {total_existing} existing cases across "
                  f"{len(domains)} domains")
            for domain in domains:
                if existing_counts.get(domain):
                    print(f"  {domain}: {existing_counts[domain]}")

        # -- 1. Build the slot queue: one entry per domain ----------
        slots: List[Dict] = []
        for domain in domains:
            already = existing_counts.get(domain, 0)
            remaining = max(0, cases_per_domain - already)
            slots.append({
                "domain": domain,
                "min_steps": min_steps,
                "max_steps": max_steps,
                "target": remaining,
                "max_attempts": remaining * max_attempts_multiplier,
                "attempts": 0,
                "in_flight": 0,
                "cases": [],
                "traces": [],
            })

        active_slots = [s for s in slots if s["target"] > 0]
        print(f"\n[Schedule] {len(active_slots)} active domain slots")
        for s in slots:
            if s["target"] > 0:
                print(f"  {s['domain']}: target {s['target']} cases "
                      f"(steps {s['min_steps']}~{s['max_steps']})")

        if not active_slots:
            print("[Done] All targets already met, nothing to generate")
            return

        # -- 2. Prepare output directory --------------------------------------
        round_dir = _next_round_dir(self.output_dir)
        print(f"\n[Output] Writing results to {round_dir}")

        domain_out_paths: Dict[str, str] = {}
        domain_trace_paths: Dict[str, str] = {}
        for domain in domains:
            domain_out_paths[domain] = os.path.join(
                round_dir, f"eval_cases_({domain}).jsonl")
            domain_trace_paths[domain] = os.path.join(
                round_dir, f"synthesis_trace_({domain}).jsonl")

        # -- 3. Async dispatch with ProcessPoolExecutor ---------------------
        global_task_counter = [0]

        def _next_active_slot():
            """Randomly pick a slot that still has room for new tasks.

            A slot has room when ``completed + in_flight < target`` and
            attempts budget is not exhausted.  This prevents over-subscription:
            in-flight tasks count as "reserved" capacity, so new tasks are only
            dispatched when there is genuine demand.
            """
            active = [
                s for s in slots
                if len(s["cases"]) + s["in_flight"] < s["target"]
                and s["attempts"] < s["max_attempts"]
            ]
            return random.choice(active) if active else None

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker,
            initargs=(self.datasets_dir, self.min_files),
        ) as executor:
            in_flight: Dict = {}  # future -> {task, slot, evidence}

            # Pre-submit up to n_workers tasks
            for _ in range(self.n_workers):
                slot = _next_active_slot()
                if slot is None:
                    break
                task, evidence = self._sample_one_task_for_slot(
                    slot, global_task_counter,
                )
                fut = executor.submit(_worker_synthesize, task)
                slot["in_flight"] += 1
                in_flight[fut] = {"task": task, "slot": slot, "evidence": evidence}

            print(f"\n  [Async] pre-submitted {len(in_flight)} tasks")

            while in_flight:
                for fut in as_completed(list(in_flight)):
                    meta = in_flight.pop(fut)
                    slot = meta["slot"]
                    tid = meta["task"]["task_id"]
                    slot["in_flight"] -= 1

                    completed_path_ids = set(meta["task"].get("path_ids", []))
                    self._in_flight_skill_ids -= completed_path_ids

                    try:
                        r = fut.result()
                    except Exception as e:
                        print(f"  [Error] task {tid} raised: {e}")
                        traceback.print_exc()
                        r = None

                    # Immediate coverage commit + disk flush
                    if r is not None:
                        self._commit_results([r], {tid: meta["evidence"]})

                        domain = slot["domain"]
                        if r.get("synthesis_log"):
                            slot["traces"].append(r["synthesis_log"])
                            with open(domain_trace_paths[domain], "a", encoding="utf-8") as f:
                                f.write(json.dumps(r["synthesis_log"], ensure_ascii=False, default=str) + "\n")
                        if r.get("case") is not None:
                            slot["cases"].append(r["case"])
                            with open(domain_out_paths[domain], "a", encoding="utf-8") as f:
                                f.write(json.dumps(r["case"], ensure_ascii=False) + "\n")
                            print(f"  [{domain}] task {tid} OK "
                                  f"({len(slot['cases'])}/{slot['target']})")
                        else:
                            print(f"  [{domain}] task {tid} failed "
                                  f"({len(slot['cases'])}/{slot['target']})")

                    # Submit a replacement task if any slot is still active
                    next_slot = _next_active_slot()
                    if next_slot is not None:
                        task, evidence = self._sample_one_task_for_slot(
                            next_slot, global_task_counter,
                        )
                        new_fut = executor.submit(_worker_synthesize, task)
                        next_slot["in_flight"] += 1
                        in_flight[new_fut] = {
                            "task": task, "slot": next_slot, "evidence": evidence,
                        }

                    break  # re-enter while-loop so as_completed sees new futures

        # -- 4. Summary ------------------------------------------------------
        for domain in domains:
            n_cases = sum(
                len(s["cases"]) for s in slots if s["domain"] == domain
            )
            n_traces = sum(
                len(s["traces"]) for s in slots if s["domain"] == domain
            )
            print(f"\n  [{domain}] {n_cases} cases -> {domain_out_paths[domain]}")
            print(f"  [{domain}] {n_traces} trace entries -> {domain_trace_paths[domain]}")

        # -- 5. Coverage report ---------------------------------------------
        coverage = self.path_sampler.get_coverage_ratio()
        print(f"\n[Coverage] Current skill coverage: {coverage:.2%}")

        self.path_sampler.print_node_statistics(topk=10)
        self.path_sampler.print_example_statistics(topk=10)

        # -- 6. Uncovered-skills report (when enabled) ----------------------
        if self._uncovered_pool:
            used = sum(1 for sid in self._uncovered_pool
                       if self._uncovered_usage[sid] > 0)
            total = len(self._uncovered_pool)
            total_injections = sum(self._uncovered_usage.values())
            print(f"\n[Uncovered Skills] {used}/{total} skills used "
                  f"({total_injections} total injections)")
            if self._uncovered_usage:
                top5 = self._uncovered_usage.most_common(5)
                print("  Most injected:")
                for sid, cnt in top5:
                    name = self.path_sampler.nodes_id_dict[sid].name
                    print(f"    - {name}: {cnt} times")
                unused = [
                    self.path_sampler.nodes_id_dict[sid].name
                    for sid in self._uncovered_pool
                    if self._uncovered_usage[sid] == 0
                ]
                if unused:
                    print(f"  Never injected ({len(unused)}):")
                    for name in unused[:5]:
                        print(f"    - {name}")
                    if len(unused) > 5:
                        print(f"    ... and {len(unused) - 5} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel eval case generation (multi-process)"
    )
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--datasets_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--cases_per_domain", type=int, default=20)
    parser.add_argument("--path_record", type=str,
                        default="../skill_cluster/data/sample_paths.jsonl")
    parser.add_argument("--min_skills", type=int, default=2)
    parser.add_argument("--min_files", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--domains", type=str, nargs="*", default=None,
                        help="Domains to process; omit to process all")
    parser.add_argument("--rare_skill_ratio", type=float, default=0.3,
                        help="Fraction of tasks that inject a rare skill (default: 0.1)")
    parser.add_argument("--min_new_skills_per_path", type=int, default=5,
                        help="Min number of novel skills each new path must have "
                             "compared to currently in-flight paths (default: 3)")
    parser.add_argument("--min_steps", type=int, default=3,
                        help="Minimum number of solution steps per case (default: 3)")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Maximum number of solution steps per case (default: 8)")
    parser.add_argument("--append", action="store_true",
                        help="Continue from existing output (don't clear files). "
                             "Use after greedy_select.py to fill remaining quota.")
    parser.add_argument("--uncovered_skills", action="store_true",
                        help="Inject uncovered skills from uncovered_skills.jsonl "
                             "into every generated case")
    parser.add_argument("--uncovered_skills_file", type=str,
                        default="uncovered_skills.jsonl",
                        help="Path to uncovered skills JSONL file "
                             "(default: uncovered_skills.jsonl)")
    parser.add_argument("--uncovered_per_case", type=int, default=2,
                        help="Number of uncovered skills to inject per case "
                             "(default: 2)")
    parser.add_argument("--required_file", type=str, default=None, action="append",
                        help="File (relative path within domain) that MUST be included "
                             "in every generated eval case. Can be specified multiple times.")
    args = parser.parse_args()

    required_files = args.required_file  # None or list of strings

    print("=" * 60)
    print("Parallel Eval Case Synthesizer")
    print(f"  Workers:            {args.workers}")
    print(f"  Cases per domain:   {args.cases_per_domain}")
    print(f"  Steps per case:     {args.min_steps}~{args.max_steps}")
    print(f"  Rare skill ratio:   {args.rare_skill_ratio}")
    print(f"  Min new skills/path:{args.min_new_skills_per_path}")
    if required_files:
        print(f"  Required files:     {required_files}")
    if args.uncovered_skills:
        print(f"  Uncovered skills:   ON ({args.uncovered_per_case}/case)")
        print(f"  Uncovered file:     {args.uncovered_skills_file}")
    print("=" * 60)

    path_sampler = PathSampler(
        path_record_file=args.path_record,
        alpha=1.0,
        beta=1.0,
        random_prob=0.1,
    )
    print(f"[Init] PathSampler: {len(path_sampler.top_node_indices)} skills loaded")

    # Load uncovered skills when enabled
    uncovered_skill_ids: List[str] = []
    if args.uncovered_skills:
        uncovered_skill_ids = _load_uncovered_skills(
            args.uncovered_skills_file, path_sampler,
        )
        if not uncovered_skill_ids:
            print("[Warning] No uncovered skills resolved — "
                  "falling back to normal mode")

    llm = QwenClient(model=QWEN_MODEL)
    loader = DatasetLoader(args.datasets_dir, llm_client=llm)
    domains = args.domains if args.domains else loader.get_all_domains()
    if not domains:
        print("No domains found, please check --datasets_dir")
        return

    print(f"[Init] Domains ({len(domains)}): {domains}")

    coordinator = ParallelCoordinator(
        path_sampler=path_sampler,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        n_workers=args.workers,
        min_files=args.min_files,
        min_skills=args.min_skills,
        rare_skill_ratio=args.rare_skill_ratio,
        min_new_skills_per_path=args.min_new_skills_per_path,
        uncovered_skill_ids=uncovered_skill_ids or None,
        uncovered_per_case=args.uncovered_per_case,
        required_files=required_files,
    )

    coordinator.run(
        domains=domains,
        cases_per_domain=args.cases_per_domain,
        append=args.append,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )

    print("\n" + "=" * 60)
    print("All Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
