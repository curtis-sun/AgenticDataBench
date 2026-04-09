"""
Greedy Subset Selection for Eval Cases

Uses a greedy maximum-coverage algorithm to retain a minimal subset of
already-generated eval cases that maximises skill coverage.  After
selection the output files are rewritten and sample_paths.jsonl is
rebuilt so that subsequent generation runs continue from the retained
coverage state.

Usage:
    python greedy_select.py                       # default: read from ./output, write back
    python greedy_select.py --output_dir ./output --coverage_target 1.0
    python greedy_select.py --dry_run              # preview without writing
"""

import os
import json
import glob
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import sys
sys.path.append('..')
from utils.config import OUTPUT_DIR


def load_skill_name_to_id(
    clusters_file: str = "../skill_cluster/data/step-clusters.jsonl",
) -> Dict[str, str]:
    """Build a case-insensitive skill-name -> node-id mapping from step-clusters."""
    name_to_id: Dict[str, str] = {}
    with open(clusters_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            name_to_id[obj["name"].lower()] = obj["graph_id"]
    return name_to_id


def load_all_cases(output_dir: str) -> Dict[str, List[dict]]:
    """Load eval cases grouped by domain.

    Returns {domain: [case_dict, ...]}
    """
    pattern = os.path.join(output_dir, "eval_cases_(*).jsonl")
    files = sorted(glob.glob(pattern))

    domain_cases: Dict[str, List[dict]] = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        domain = fname.replace("eval_cases_(", "").replace(").jsonl", "")
        cases = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cases.append(json.loads(line))
        domain_cases[domain] = cases
    return domain_cases


def case_skill_ids(case: dict, name_to_id: Dict[str, str]) -> Set[str]:
    """Map a case's skill names to node IDs."""
    ids = set()
    for name in case.get("skills", []):
        nid = name_to_id.get(name.lower())
        if nid:
            ids.add(nid)
    return ids


def greedy_max_coverage(
    cases: List[Tuple[str, int, dict]],
    name_to_id: Dict[str, str],
    all_skill_ids: Set[str],
    coverage_target: float = 1.0,
    pre_covered: Set[str] = None,
) -> List[Tuple[str, int, dict]]:
    """Greedy weighted maximum-coverage selection.

    Each iteration picks the case covering the most uncovered skills.
    Stops when ``coverage_target`` fraction of all skills is covered or
    no case adds new coverage.

    Args:
        cases: list of (domain, original_index, case_dict)
        name_to_id: skill name -> node id mapping
        all_skill_ids: universe of skill node IDs we want to cover
        coverage_target: stop when covered/total >= this value
        pre_covered: skills already covered by frozen/protected cases

    Returns:
        Selected subset in selection order.
    """
    target_count = int(len(all_skill_ids) * coverage_target)
    covered: Set[str] = set(pre_covered) if pre_covered else set()
    selected: List[Tuple[str, int, dict]] = []
    remaining = list(range(len(cases)))

    case_skills = [case_skill_ids(c[2], name_to_id) for c in cases]

    while covered.__len__() < target_count and remaining:
        best_idx = -1
        best_gain = 0
        best_pos = -1

        for pos, idx in enumerate(remaining):
            gain = len(case_skills[idx] - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_pos = pos

        if best_gain == 0:
            break

        covered |= case_skills[best_idx]
        selected.append(cases[best_idx])
        remaining.pop(best_pos)

    return selected


def filter_trace_files(
    output_dir: str,
    domain_cases: Dict[str, List[dict]],
    retained_case_indices: Dict[str, Set[int]],
):
    """Remove trace entries only for deleted cases, keep everything else.

    Matching strategy: traces and cases are written in the same order per
    domain file.  Successful trace entries (outcome=="success") map 1:1 to
    eval case lines.  We walk through the trace file, count successes, and
    remove only those successful entries whose case was deleted by greedy
    selection.  Failed/error trace entries are always kept.

    Args:
        retained_case_indices: {domain: set of original case indices that
            were kept by greedy selection}
    """

    total_kept = 0
    total_removed = 0
    for domain in domain_cases:
        trace_path = os.path.join(output_dir, f"synthesis_trace_({domain}).jsonl")
        if not os.path.exists(trace_path):
            continue

        entries = []
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            continue

        kept_entries = []
        success_counter = 0
        domain_retained = retained_case_indices.get(domain, set())

        for entry in entries:
            if entry.get("outcome") == "success":
                if success_counter in domain_retained:
                    kept_entries.append(entry)
                    total_kept += 1
                else:
                    total_removed += 1
                success_counter += 1
            else:
                kept_entries.append(entry)
                total_kept += 1

        with open(trace_path, "w", encoding="utf-8") as f:
            for entry in kept_entries:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"  {domain}: {len(entries)} -> {len(kept_entries)} trace entries")

    return total_kept, total_removed


def prune_sample_paths(
    removed_cases: List[Tuple[str, int, dict]],
    name_to_id: Dict[str, str],
    path_record_file: str,
):
    """Remove sample_paths entries that correspond to deleted cases.

    Uses a reverse-deletion strategy: builds fingerprints for the
    *removed* cases and walks through the original file, consuming
    (deleting) one matching entry per removed case.  All unmatched
    entries are kept, preserving coverage history from other rounds.
    """
    # 1. Read original entries
    original_entries: List[dict] = []
    if os.path.exists(path_record_file):
        with open(path_record_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    original_entries.append(json.loads(line))

    # 2. Build fingerprint counts for removed cases
    remove_fp_counts: Dict[frozenset, int] = defaultdict(int)
    for _, _, case in removed_cases:
        skill_ids = []
        for name in case.get("skills", []):
            nid = name_to_id.get(name.lower())
            if nid:
                skill_ids.append(nid)
        if skill_ids:
            remove_fp_counts[frozenset(skill_ids)] += 1

    # 3. Walk original entries; consume matching fingerprints
    kept: List[dict] = []
    pruned = 0
    for entry in original_entries:
        fp = frozenset(entry.get("path", []))
        if remove_fp_counts.get(fp, 0) > 0:
            remove_fp_counts[fp] -= 1
            pruned += 1
        else:
            kept.append(entry)

    # 4. Write back
    os.makedirs(os.path.dirname(path_record_file) or ".", exist_ok=True)
    with open(path_record_file, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Pruned {pruned} path entries, kept {len(kept)}")
    return len(kept)


def run_global_select(
    round_dirs: List[str],
    name_to_id: Dict[str, str],
    all_skill_ids: Set[str],
    coverage_target: float,
    path_record: str,
    dry_run: bool,
):
    """Jointly select across multiple round directories.

    All rounds are mutable -- cases can be removed from any of them.
    After greedy selection, each round directory is rewritten in-place
    with only its retained cases.
    """
    # 1. Load cases from every round, tracking source directory
    per_dir_domain_cases: Dict[str, Dict[str, List[dict]]] = {}
    # flat list for greedy; parallel list tracks source dir
    flat_cases: List[Tuple[str, int, dict]] = []
    flat_source_dirs: List[str] = []

    total_all = 0
    for rdir in round_dirs:
        domain_cases = load_all_cases(rdir)
        per_dir_domain_cases[rdir] = domain_cases
        n = sum(len(v) for v in domain_cases.values())
        total_all += n
        for domain, cases in domain_cases.items():
            for idx, case in enumerate(cases):
                flat_cases.append((domain, idx, case))
                flat_source_dirs.append(rdir)
        covered = set()
        for cases_list in domain_cases.values():
            for case in cases_list:
                covered |= case_skill_ids(case, name_to_id)
        print(f"[Load] {rdir}: {n} cases, "
              f"{len(covered)} skills covered")

    if total_all == 0:
        print("[Warning] No eval cases found across specified rounds")
        return

    covered_before: Set[str] = set()
    for _, _, case in flat_cases:
        covered_before |= case_skill_ids(case, name_to_id)
    print(f"\n[Before] {total_all} cases across {len(round_dirs)} rounds, "
          f"covering {len(covered_before)}/{len(all_skill_ids)} skills "
          f"({len(covered_before)/len(all_skill_ids):.1%})")

    # 2. Global greedy selection
    selected = greedy_max_coverage(
        flat_cases, name_to_id, all_skill_ids,
        coverage_target=coverage_target,
    )

    covered_after: Set[str] = set()
    for _, _, case in selected:
        covered_after |= case_skill_ids(case, name_to_id)

    # 3. Build selected / removed indices per source directory
    selected_ids = {id(c) for _, _, c in selected}
    selected_global_indices = {
        i for i, (_, _, case) in enumerate(flat_cases) if id(case) in selected_ids
    }

    removed_cases_by_dir: Dict[str, List[Tuple[str, int, dict]]] = defaultdict(list)
    selected_by_dir: Dict[str, Dict[str, List[dict]]] = {
        rdir: defaultdict(list) for rdir in round_dirs
    }
    retained_indices_by_dir: Dict[str, Dict[str, Set[int]]] = {
        rdir: defaultdict(set) for rdir in round_dirs
    }

    for i, (domain, orig_idx, case) in enumerate(flat_cases):
        rdir = flat_source_dirs[i]
        if i in selected_global_indices:
            selected_by_dir[rdir][domain].append(case)
            retained_indices_by_dir[rdir][domain].add(orig_idx)
        else:
            removed_cases_by_dir[rdir].append((domain, orig_idx, case))

    # 4. Print statistics
    print(f"\n[After]  {len(selected)} cases retained, "
          f"covering {len(covered_after)}/{len(all_skill_ids)} skills "
          f"({len(covered_after)/len(all_skill_ids):.1%})")
    print(f"[Reduction] {total_all} -> {len(selected)} "
          f"(removed {total_all - len(selected)} redundant cases)")

    print(f"\nPer-round breakdown:")
    for rdir in round_dirs:
        before = sum(len(v) for v in per_dir_domain_cases[rdir].values())
        after = sum(len(v) for v in selected_by_dir[rdir].values())
        removed = before - after
        print(f"  {rdir}: {before} -> {after} (removed {removed})")

    uncovered = all_skill_ids - covered_after
    if uncovered:
        print(f"\n[Gap] {len(uncovered)} skills still uncovered after selection")

    if dry_run:
        print("\n[Dry run] No files written")
        return

    # 5. Rewrite each round directory
    all_removed: List[Tuple[str, int, dict]] = []
    for rdir in round_dirs:
        domain_cases = per_dir_domain_cases[rdir]
        sel_domains = selected_by_dir[rdir]
        ret_indices = retained_indices_by_dir[rdir]
        removed_list = removed_cases_by_dir.get(rdir, [])
        all_removed.extend(removed_list)

        print(f"\n[Write] {rdir}...")
        for domain in domain_cases:
            out_path = os.path.join(rdir, f"eval_cases_({domain}).jsonl")
            new_cases = sel_domains.get(domain, [])
            with open(out_path, "w", encoding="utf-8") as f:
                for case in new_cases:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")
            print(f"  {domain}: {len(domain_cases[domain])} -> {len(new_cases)}")

        print(f"  [Trace] Filtering...")
        kept, removed = filter_trace_files(rdir, domain_cases, ret_indices)
        print(f"  Kept {kept}, removed {removed} trace entries")

    # 6. Prune sample_paths once with all removed cases
    print(f"\n[Paths] Pruning {path_record}...")
    n_kept = prune_sample_paths(all_removed, name_to_id, path_record)
    print(f"  {n_kept} path records remaining")

    print(f"\n{'='*60}")
    print("Done (global mode)!")
    print(f"  Coverage: {len(covered_after)}/{len(all_skill_ids)} "
          f"({len(covered_after)/len(all_skill_ids):.1%})")
    print(f"  Retained: {len(selected)} cases across {len(round_dirs)} rounds")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Greedy subset selection for eval cases")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--path_record", type=str, default="../skill_cluster/data/sample_paths.jsonl")
    parser.add_argument("--clusters_file", type=str, default="../skill_cluster/data/step-clusters.jsonl",
                        help="Top-level skill clusters file (default: ../skill_cluster/data/step-clusters.jsonl)")
    parser.add_argument("--coverage_target", type=float, default=1.0,
                        help="Target skill coverage ratio (0~1, default 1.0 = all skills)")
    parser.add_argument("--frozen_rounds", type=str, nargs="*", default=None,
                        help="Directories with cases that count toward coverage "
                             "but cannot be removed (e.g. output/round1)")
    parser.add_argument("--global_rounds", type=str, nargs="+", default=None,
                        help="Multiple round directories to jointly select from. "
                             "All rounds are mutable (cases can be removed from any). "
                             "Overrides --output_dir and --frozen_rounds.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview selection without writing files")
    args = parser.parse_args()

    print("=" * 60)
    print("Greedy Subset Selection")
    print("=" * 60)

    # 1. Load skill graph
    name_to_id = load_skill_name_to_id(args.clusters_file)
    all_skill_ids = set(name_to_id.values())
    print(f"[Graph] {len(all_skill_ids)} skills loaded")

    # -- Global mode: joint selection across multiple mutable rounds --
    if args.global_rounds:
        if args.frozen_rounds:
            print("[Note] --frozen_rounds ignored when --global_rounds is used")
        run_global_select(
            round_dirs=args.global_rounds,
            name_to_id=name_to_id,
            all_skill_ids=all_skill_ids,
            coverage_target=args.coverage_target,
            path_record=args.path_record,
            dry_run=args.dry_run,
        )
        return

    # 2. Load all existing cases
    domain_cases = load_all_cases(args.output_dir)
    total_before = sum(len(v) for v in domain_cases.values())
    if total_before == 0:
        print("[Warning] No eval cases found, nothing to select")
        return

    flat_cases: List[Tuple[str, int, dict]] = []
    for domain, cases in domain_cases.items():
        for idx, case in enumerate(cases):
            flat_cases.append((domain, idx, case))

    # Pre-selection coverage (target round only)
    covered_before: Set[str] = set()
    for _, _, case in flat_cases:
        covered_before |= case_skill_ids(case, name_to_id)
    print(f"[Target] {total_before} cases across {len(domain_cases)} domains, "
          f"covering {len(covered_before)}/{len(all_skill_ids)} skills "
          f"({len(covered_before)/len(all_skill_ids):.1%})")

    # 2b. Load frozen rounds (protected cases that contribute coverage
    #     but are never removed)
    frozen_covered: Set[str] = set()
    frozen_total = 0
    if args.frozen_rounds:
        for frozen_dir in args.frozen_rounds:
            frozen_cases = load_all_cases(frozen_dir)
            n = sum(len(v) for v in frozen_cases.values())
            frozen_total += n
            for cases_list in frozen_cases.values():
                for case in cases_list:
                    frozen_covered |= case_skill_ids(case, name_to_id)
            print(f"[Frozen] {frozen_dir}: {n} cases, "
                  f"{len(frozen_covered)} skills covered so far")
        combined = frozen_covered | covered_before
        print(f"[Combined] {frozen_total} frozen + {total_before} target = "
              f"{frozen_total + total_before} total cases, "
              f"covering {len(combined)}/{len(all_skill_ids)} skills "
              f"({len(combined)/len(all_skill_ids):.1%})")

    # 3. Greedy selection (with frozen coverage as baseline)
    selected = greedy_max_coverage(
        flat_cases, name_to_id, all_skill_ids,
        coverage_target=args.coverage_target,
        pre_covered=frozen_covered if frozen_covered else None,
    )

    covered_after: Set[str] = set(frozen_covered)
    for _, _, case in selected:
        covered_after |= case_skill_ids(case, name_to_id)

    # Group by domain (preserving original indices for trace matching)
    selected_by_domain: Dict[str, List[dict]] = defaultdict(list)
    retained_case_indices: Dict[str, Set[int]] = defaultdict(set)
    selected_set: Set[Tuple[str, int]] = set()
    for domain, orig_idx, case in selected:
        selected_by_domain[domain].append(case)
        retained_case_indices[domain].add(orig_idx)
        selected_set.add((domain, orig_idx))

    removed_cases = [
        (domain, idx, case)
        for domain, idx, case in flat_cases
        if (domain, idx) not in selected_set
    ]

    if frozen_covered:
        selected_only_covered = set()
        for _, _, case in selected:
            selected_only_covered |= case_skill_ids(case, name_to_id)
        print(f"\n[After]  {len(selected)} target cases retained "
              f"(+ {frozen_total} frozen), "
              f"combined coverage {len(covered_after)}/{len(all_skill_ids)} "
              f"({len(covered_after)/len(all_skill_ids):.1%})")
        print(f"  Frozen contribution: {len(frozen_covered)} skills")
        print(f"  Target contribution: {len(selected_only_covered - frozen_covered)} "
              f"unique skills (+ {len(selected_only_covered & frozen_covered)} overlap)")
    else:
        print(f"\n[After]  {len(selected)} cases retained, "
              f"covering {len(covered_after)}/{len(all_skill_ids)} skills "
              f"({len(covered_after)/len(all_skill_ids):.1%})")
    print(f"[Reduction] {total_before} -> {len(selected)} "
          f"(removed {total_before - len(selected)} redundant cases)")

    print(f"\nPer-domain breakdown:")
    for domain in sorted(set(list(domain_cases.keys()) + list(selected_by_domain.keys()))):
        before = len(domain_cases.get(domain, []))
        after = len(selected_by_domain.get(domain, []))
        print(f"  {domain}: {before} -> {after}")

    # Show uncovered skills if any
    uncovered = all_skill_ids - covered_after
    if uncovered:
        print(f"\n[Gap] {len(uncovered)} skills still uncovered after selection")

    if args.dry_run:
        print("\n[Dry run] No files written")
        return

    # 4. Rewrite output files
    print(f"\n[Write] Rewriting output files...")
    for domain, cases_before in domain_cases.items():
        out_path = os.path.join(args.output_dir, f"eval_cases_({domain}).jsonl")
        new_cases = selected_by_domain.get(domain, [])
        with open(out_path, "w", encoding="utf-8") as f:
            for case in new_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"  {domain}: {len(cases_before)} -> {len(new_cases)} ({out_path})")

    # Filter trace files: keep only entries for retained cases
    print(f"\n[Trace] Filtering synthesis traces...")
    kept, removed = filter_trace_files(
        args.output_dir, domain_cases, retained_case_indices,
    )
    print(f"  Kept {kept} trace entries, removed {removed}")

    # 5. Prune sample_paths.jsonl (remove only deleted cases' entries)
    n_kept = prune_sample_paths(removed_cases, name_to_id, args.path_record)
    print(f"\n[Paths] Pruned {args.path_record}, {n_kept} path records remaining")

    print(f"\n{'='*60}")
    print("Done! You can now re-run generation to continue from this state.")
    print(f"  Combined coverage: {len(covered_after)}/{len(all_skill_ids)} "
          f"({len(covered_after)/len(all_skill_ids):.1%})")
    if frozen_covered:
        print(f"  Frozen cases: {frozen_total}")
    print(f"  Retained target cases: {len(selected)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
