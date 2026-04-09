"""
Skill Graph Path Sampling Module

This module provides functionality for sampling skill paths from a knowledge graph.
It supports weighted sampling based on node/edge weights, coverage tracking for
diversity, rare skill injection, and novelty constraints.

Key Components:
- Step: Represents a single step with associated skills
- Example: Represents a QA example with multiple steps
- PathSampler: Main class for graph construction and path sampling

Sampling Strategy:
1. Node weights are based on cluster sizes and real-world frequency data
2. Edge weights are based on co-occurrence in examples
3. Coverage tracking ensures diversity across sampled paths
4. Rare skill injection promotes underrepresented skills
"""

import networkx as nx
from collections import defaultdict, Counter
import random
import json
import os
import sys
sys.path.append('..')
from utils.graph_utils import GraphNode
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Temperature for weight scaling (higher = more uniform distribution)
temperature = 2
DEFAULT_EXCLUDED_SKILL_KEYWORDS = []


class Step:
    """
    Represents a single step in an example.

    Attributes:
        id: Unique identifier (format: "{example_id}-{step_idx}")
        skills: Set of skill IDs associated with this step
        text: Raw text content of the step
        embedding: Pre-computed embedding vector for similarity scoring
    """

    def __init__(self, step_id, skills, text, embedding=None):
        self.id: str = step_id
        self.skills: set[str] = set(skills)
        self.text: str = text
        self.embedding = embedding


class Example:
    """
    Represents a QA example with multiple steps.

    Attributes:
        id: Unique identifier (format: "{question_id}-{answer_id}")
        title: Question title
        question: Question body text
        answer: Answer body text
        steps: List of Step objects in this example
        coverage: Number of times this example has been used as evidence
    """

    def __init__(self, example_id, title, question, answer, steps):
        self.id: str = example_id
        self.title: str = title
        self.question: str = question
        self.answer: str = answer
        self.steps: list[Step] = steps
        self.coverage: int = 0


class PathSampler:
    """
    Main class for skill graph construction and path sampling.

    This class builds a directed graph where nodes represent skills and edges
    represent sequential relationships between skills in examples. It supports
    various sampling strategies to generate diverse and meaningful skill paths.

    Attributes:
        G: NetworkX DiGraph representing the skill graph
        top_node_indices: List of top-level skill node IDs
        nodes_id_dict: Mapping from node ID to GraphNode object
        cluster_groups: Mapping for step-to-skill association
        example_dict: Mapping from example ID to Example object
        skill_to_examples: Mapping from skill ID to set of example IDs
        node_counter: Counter for node appearances in sampled paths
        example_counter: Counter for example usage as evidence
    """

    def __init__(
        self,
        path_record_file=None,
        alpha=1.0,
        beta=1.0,
        random_prob=0.1,
        excluded_skill_keywords=None,
    ):
        """
        Initialize the PathSampler.

        Args:
            path_record_file: File to record sampled paths (for persistence)
            alpha: Weight multiplier for node scores in sampling
            beta: Weight multiplier for edge scores in sampling
            random_prob: Probability of random sampling (vs weighted)
            excluded_skill_keywords: Keywords to exclude from sampling
        """
        self.path_record_file = path_record_file
        self.alpha = alpha
        self.beta = beta
        self.random_prob = random_prob
        self.excluded_skill_keywords = excluded_skill_keywords if excluded_skill_keywords is not None else DEFAULT_EXCLUDED_SKILL_KEYWORDS

        # Graph structure
        self.G = nx.DiGraph()
        self.top_node_indices = []
        self.nodes_id_dict = {}
        self.cluster_groups = defaultdict(dict)

        # Example and skill data
        self.step_nums = {}
        self.example_dict = {}
        self.skill_to_examples = defaultdict(set)
        self.example_skill_steps = defaultdict(dict)

        # Skill descriptions (optional)
        self.skill_descriptions = {}

        # Tracking counters
        self.node_counter = Counter()
        self.example_counter = Counter()

        # Initialize data structures
        self._load_data()
        self._build_graph()
        self._score_example_skills()
        self._load_history_paths()
        self._build_candidate_roots()
        self._build_rare_skill_set()

    # ==================== Data Loading ====================

    def _load_data(self):
        """
        Load nodes, examples, and pre-computed embeddings from files.

        Builds:
        - top_node_indices: List of skill node IDs
        - nodes_id_dict: Mapping from ID to GraphNode
        - cluster_groups: Step-to-skill mappings
        - example_dict: Example objects with steps
        - skill_to_examples: Reverse mapping from skills to examples
        """
        # Load skill clusters (top-level nodes)
        with open(f'../skill_cluster/data/step-clusters.jsonl') as fin:
            for line in fin:
                obj = json.loads(line)
                node_id = obj['graph_id']
                self.top_node_indices.append(node_id)
                node = GraphNode(id=node_id, name=obj['name'], cluster=obj['cluster'])
                self.nodes_id_dict[node_id] = node
                # Build cluster_groups: maps {example_id -> {step_idx -> set of skill IDs}}
                for c in node.cluster:
                    prefix = "-".join(c.split('-')[:2])  # example_id
                    suffix = int(c.split('-')[2])        # step_idx
                    self.cluster_groups[prefix].setdefault(suffix, set()).add(node_id)

        raw_examples = {}
        with open('../skill_cluster/data/stackoverflow-data-science.jsonl') as fin:
            for line in fin:
                obj = json.loads(line)
                example_id = f'{obj["id"]}-{obj["answer_id"]}'
                raw_examples[example_id] = {
                    'question_title': obj['question_title'],
                    'question_body': obj['question_body'],
                    'answer_body': obj['answer_body']
                }
        
        # Load pre-computed step embeddings
        pre_embeddings = np.load(f'../skill_cluster/data/steps-embed.npy')

        # Load examples with steps
        i = 0
        with open(f'../skill_cluster/data/steps.jsonl') as fin:
            for line in fin:
                obj = json.loads(line)
                example_id = f'{obj["id"]}-{obj["answer_id"]}'
                steps = []

                for step_idx, step_text in enumerate(obj['steps']):
                    # Get skill IDs associated with this step
                    step_node_ids = self.cluster_groups.get(example_id, {}).get(step_idx, [])
                    steps.append(Step(f'{example_id}-{step_idx}', step_node_ids, step_text, pre_embeddings[i]))
                    i += 1

                raw_example = raw_examples[example_id]
                self.example_dict[example_id] = Example(
                    example_id, obj.get('question_title', ''), obj.get('question_body', ''), obj.get('answer_body', ''), steps
                )
                self.step_nums[example_id] = len(obj['steps'])

        assert i == len(pre_embeddings), f"Embedding count mismatch: {i} vs {len(pre_embeddings)}"

        # Build reverse mapping: skill -> examples
        for example in self.example_dict.values():
            for step in example.steps:
                for skill in step.skills:
                    self.skill_to_examples[skill].add(example.id)

        # Load optional skill descriptions
        desc_path = '../skill_cluster/data/skill-descriptions.jsonl'
        if os.path.exists(desc_path):
            with open(desc_path) as fin:
                for line in fin:
                    obj = json.loads(line)
                    skill_id = obj.get('id') or obj.get('skill')
                    desc = obj.get('response') or obj.get('description')
                    if skill_id and desc:
                        self.skill_descriptions[skill_id] = desc
            print(f"  Loaded skill descriptions for {len(self.skill_descriptions)} skills")

    # ==================== Graph Construction ====================

    def _build_graph(self):
        """
        Build the initial graph structure with weighted nodes and edges.

        Node weights are initialized based on cluster sizes and adjusted using
        real-world frequency data from skills_pairs_frequency.json.

        Edge weights are based on sequential co-occurrence in examples.
        """
        name_node_id_dict = {}

        # Add nodes with initial weights based on cluster size
        for node_id in self.top_node_indices:
            node = self.nodes_id_dict[node_id].to_dict()
            name_node_id_dict[node['name']] = node_id
            base_weight = len(self.nodes_id_dict[node_id].cluster)

            node_attr = {k: v for k, v in node.items() if k not in ('weight',)}
            node_attr.update(base_weight=base_weight, weight=base_weight, coverage=0)
            self.G.add_node(node_id, **node_attr)

        # Add edges based on sequential co-occurrence in examples
        for prefix in self.cluster_groups:
            suffices = sorted(self.cluster_groups[prefix].keys())
            for i in range(len(suffices) - 1):
                # Connect step i to step i+1 in the same example
                for u in self.cluster_groups[prefix][suffices[i]]:
                    for v in self.cluster_groups[prefix][suffices[i + 1]]:
                        if u == v:
                            continue
                        if self.G.has_edge(u, v):
                            self.G[u][v]['base_weight'] += 1
                            self.G[u][v]['weight'] += 1
                        else:
                            self.G.add_edge(u, v, base_weight=1, weight=1, coverage=0)

        # Load real-world frequency data for weight adjustment
        with open('../skill_cluster/data/skills_pairs_frequency.json', 'r') as fin:
            obj = json.load(fin)
            real_skills_frequency = obj['skills_frequency']
            real_skill_pairs_frequency = obj['skill_pairs_frequency']
            real_skills_summary = obj['summary']

        # Calculate relative weights for combining local and global frequencies
        sum_node_weights = sum([self.G.nodes[n]['base_weight'] for n in self.G.nodes])
        node_relative_weight = sum_node_weights / real_skills_summary['total_skill_occurrences'] * 0.1
        sum_edge_weights = sum([self.G[u][v]['base_weight'] for u, v in self.G.edges])
        edge_relative_weight = sum_edge_weights / real_skills_summary['total_skill_pair_occurrences'] * 0.05
        print(f"Node relative weight: {node_relative_weight}, Edge relative weight: {edge_relative_weight}")

        # Adjust node weights with real-world frequency
        for skill in real_skills_frequency:
            node_id = name_node_id_dict[skill]
            assert node_id in self.G.nodes, f"Skill {skill} in real data but not in graph nodes."
            # Blend local cluster weight with real-world frequency
            self.G.nodes[node_id]['base_weight'] = (
                self.G.nodes[node_id]['base_weight'] * 0.9 + real_skills_frequency[skill] * node_relative_weight
            )
            self.G.nodes[node_id]['weight'] = self.G.nodes[node_id]['base_weight'] ** (1 / temperature)

        # Adjust edge weights with real-world frequency and add missing edges
        for skill_pair in real_skill_pairs_frequency:
            skill1, skill2 = skill_pair.split(' -> ')
            node_id1 = name_node_id_dict[skill1]
            node_id2 = name_node_id_dict[skill2]
            if self.G.has_edge(node_id1, node_id2):
                self.G[node_id1][node_id2]['base_weight'] = (
                    self.G[node_id1][node_id2]['base_weight'] * 0.95 + real_skill_pairs_frequency[skill_pair] * edge_relative_weight
                )
                self.G[node_id1][node_id2]['weight'] = self.G[node_id1][node_id2]['base_weight'] ** (1 / temperature)
            else:
                # Add edge that doesn't exist in local data but exists in real-world
                e_base_weight = real_skill_pairs_frequency[skill_pair] * edge_relative_weight
                e_weight = e_base_weight ** (1 / temperature)
                self.G.add_edge(node_id1, node_id2, base_weight=e_base_weight, weight=e_weight, coverage=0)

        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize node and edge weights to [0, 1] range for consistent scoring."""
        node_weights = [self.G.nodes[n]['weight'] for n in self.G.nodes]
        max_w = max(node_weights)

        for n in self.G.nodes:
            self.G.nodes[n]['weight_norm'] = self.G.nodes[n]['weight'] / max_w

        edge_weights = [self.G[u][v]['weight'] for u, v in self.G.edges]
        max_e = max(edge_weights)

        for u, v in self.G.edges:
            self.G[u][v]['weight_norm'] = self.G[u][v]['weight'] / max_e

    # ==================== Skill Scoring ====================

    def _score_example_skills(self):
        """
        Score example steps for each skill based on embedding similarity.

        For each skill, steps are scored by average cosine similarity to other
        steps in the same skill cluster. Higher scores indicate more representative steps.
        """
        skill_step_scores = {}

        for node_id in self.top_node_indices:
            node = self.nodes_id_dict[node_id]
            cur_steps = []
            for source in node.cluster:
                example_id = '-'.join(source.split('-')[:-1])
                step = self.example_dict[example_id].steps[int(source.split('-')[-1])]
                cur_steps.append({'embed': step.embedding, 'source': source})

            # Compute pairwise cosine distances and convert to similarity scores
            cur_embeds = np.array([step['embed'] for step in cur_steps])
            dist_matrix = cosine_distances(cur_embeds)
            total_dist = dist_matrix.sum(axis=1)
            avg_scores = 1. - total_dist / (len(cur_embeds) - 1)  # Higher = more similar

            skill_step_scores[node_id] = {}
            for i, step in enumerate(cur_steps):
                skill_step_scores[node_id][step['source']] = avg_scores[i]

        # Store max step score per skill for each example
        for ex in self.example_dict.values():
            cur_skill_scores = defaultdict(list)
            for step in ex.steps:
                for skill in step.skills:
                    cur_skill_scores[skill].append(skill_step_scores[skill][step.id])
            for skill in cur_skill_scores:
                self.example_skill_steps[ex.id][skill] = max(cur_skill_scores[skill])

    # ==================== Utility Methods ====================

    def _is_skill_excluded(self, node) -> bool:
        """Return True if a node's name matches any excluded keyword (case-insensitive)."""
        name_lower = node.name.lower()
        return any(kw.lower() in name_lower for kw in self.excluded_skill_keywords)

    def get_skill_description(self, skill_id: str) -> str:
        """Return the description for a skill, or empty string if not available."""
        return self.skill_descriptions.get(skill_id, "")

    # ==================== Coverage Management ====================

    def reserve_skills(self, skill_ids):
        """
        Temporarily increase coverage to discourage re-sampling the same skills.

        Useful for batch sampling to ensure diversity within a batch.
        Call release_skills() to undo.

        Args:
            skill_ids: List of skill IDs to reserve
        """
        for sid in skill_ids:
            if sid in self.G:
                self.G.nodes[sid]['coverage'] += 1
                c = self.G.nodes[sid]['coverage']
                self.G.nodes[sid]['weight'] = self.G.nodes[sid]['base_weight'] / (1 + c)
        for u, v in zip(skill_ids[:-1], skill_ids[1:]):
            if self.G.has_edge(u, v):
                self.G[u][v]['coverage'] += 1
                c = self.G[u][v]['coverage']
                self.G[u][v]['weight'] = self.G[u][v]['base_weight'] / (1 + c)
        self._normalize_weights()

    def release_skills(self, skill_ids):
        """Undo a previous reserve_skills() call."""
        for sid in skill_ids:
            if sid in self.G:
                self.G.nodes[sid]['coverage'] = max(0, self.G.nodes[sid]['coverage'] - 1)
                c = self.G.nodes[sid]['coverage']
                self.G.nodes[sid]['weight'] = self.G.nodes[sid]['base_weight'] / (1 + c)
        for u, v in zip(skill_ids[:-1], skill_ids[1:]):
            if self.G.has_edge(u, v):
                self.G[u][v]['coverage'] = max(0, self.G[u][v]['coverage'] - 1)
                c = self.G[u][v]['coverage']
                self.G[u][v]['weight'] = self.G[u][v]['base_weight'] / (1 + c)
        self._normalize_weights()

    # ==================== Path Recording ====================

    def add_path(self, path, evidence):
        """
        Record a sampled path and update coverage statistics.

        Args:
            path: List of skill node IDs in the path
            evidence: List of (example_id, steps) tuples used as evidence
        """
        self.node_counter.update(path)

        # Write to record file if configured
        if self.path_record_file is not None:
            record = {"path": path}
            record["evidence"] = [
                {
                    "example_id": ex_id,
                    "step_ids": [step.id for step in steps],
                    "skill_coverage": self.example_skill_coverage(
                        self.example_dict[ex_id], path
                    )
                }
                for ex_id, steps in evidence
            ]
            with open(self.path_record_file, "a") as fout:
                fout.write(json.dumps(record) + "\n")

        # Update node coverage
        for n in path:
            if n not in self.G:
                continue
            self.G.nodes[n]["coverage"] += 1
            c = self.G.nodes[n]["coverage"]
            self.G.nodes[n]["weight"] = (
                self.G.nodes[n]["base_weight"] ** (1 / temperature) / (1 + c)
            )

        # Update edge coverage
        for u, v in zip(path[:-1], path[1:]):
            if not self.G.has_edge(u, v):
                continue
            self.G[u][v]["coverage"] += 1
            c = self.G[u][v]["coverage"]
            self.G[u][v]["weight"] = (
                self.G[u][v]["base_weight"] ** (1 / temperature) / (1 + c)
            )

        # Update example coverage
        if evidence is not None:
            for ex_id, _ in evidence:
                if ex_id in self.example_dict:
                    self.example_dict[ex_id].coverage += 1
                    self.example_counter[ex_id] += 1

        self._normalize_weights()

    def _load_history_paths(self):
        """Load historical paths from record file and restore coverage state."""
        if not self.path_record_file or not os.path.exists(self.path_record_file):
            return

        with open(self.path_record_file) as fin:
            for line in fin:
                record = json.loads(line)
                path = record["path"]

                # Restore node coverage
                for n in path:
                    if n in self.G:
                        self.G.nodes[n]["coverage"] += 1

                # Restore edge coverage
                for u, v in zip(path[:-1], path[1:]):
                    if self.G.has_edge(u, v):
                        self.G[u][v]["coverage"] += 1

                # Restore example coverage
                if "evidence" in record:
                    for ev in record["evidence"]:
                        ex_id = ev["example_id"]
                        if ex_id in self.example_dict:
                            self.example_dict[ex_id].coverage += 1

        # Update weights based on restored coverage
        for n in self.G.nodes:
            c = self.G.nodes[n]["coverage"]
            self.G.nodes[n]["weight"] = (
                self.G.nodes[n]["base_weight"] ** (1 / temperature) / (1 + c)
            )

        for u, v in self.G.edges:
            c = self.G[u][v]["coverage"]
            self.G[u][v]["weight"] = (
                self.G[u][v]["base_weight"] ** (1 / temperature) / (1 + c)
            )

        self._normalize_weights()

    # ==================== Candidate Selection ====================

    def _build_candidate_roots(self, top_ratio=0.1):
        """
        Build candidate root nodes from the top steps of each example.

        Args:
            top_ratio: Fraction of steps from start to consider as roots
        """
        candidate_roots = set()
        for prefix in self.cluster_groups:
            top_count = int(self.step_nums[prefix] * top_ratio)
            for suffix in range(top_count):
                if suffix in self.cluster_groups[prefix]:
                    candidate_roots.update(self.cluster_groups[prefix][suffix])
        self.candidate_roots = list(candidate_roots)

    def _build_rare_skill_set(self, percentile=30):
        """
        Cache the set of rare skill IDs (base_weight in the bottom percentile).

        Args:
            percentile: Bottom percentile threshold for rare skills
        """
        base_weights = [self.G.nodes[n]['base_weight'] for n in self.top_node_indices]
        threshold = np.percentile(base_weights, percentile)
        self._rare_skill_ids = set()
        for n in self.top_node_indices:
            node = self.nodes_id_dict[n]
            if self.G.nodes[n]['base_weight'] <= threshold and not self._is_skill_excluded(node):
                self._rare_skill_ids.add(n)
        print(f"  Identified {len(self._rare_skill_ids)} rare skills "
              f"(bottom {percentile}th percentile, threshold={threshold:.3f})")

    # ==================== Sampling Methods ====================

    def weighted_root_choice(self):
        """
        Sample a root node for path starting.

        Returns:
            Node ID selected with probability proportional to weight
        """
        if random.random() < self.random_prob:
            return random.choice(self.candidate_roots)

        total = sum(self.G.nodes[n]['weight'] for n in self.candidate_roots)
        r = random.uniform(0, total)
        acc = 0
        for n in self.candidate_roots:
            acc += self.G.nodes[n]['weight']
            if acc >= r:
                return n

    def sample_path(self, start_node, max_steps):
        """
        Sample a path starting from the given node.

        Args:
            start_node: Starting node ID
            max_steps: Maximum number of steps in the path

        Returns:
            List of GraphNode objects representing the path
        """
        path = [self.nodes_id_dict[start_node]]
        visited = {start_node}
        current = start_node

        for _ in range(max_steps):
            neighbors = [n for n in self.G.successors(current) if n not in visited]
            if not neighbors:
                break

            next_node = self._weighted_choice(current, neighbors)
            path.append(self.nodes_id_dict[next_node])
            visited.add(next_node)
            current = next_node

        return path

    def _path_coherence(self, path, excluded_node_ids=None):
        """
        Calculate the average edge weight of a path (coherence score).

        Args:
            path: List of GraphNode objects
            excluded_node_ids: Nodes whose edges should be excluded from scoring

        Returns:
            Average edge weight_norm (0-1, higher is more coherent)
        """
        if len(path) < 2:
            return 1.0
        edge_weights = []
        for i in range(len(path) - 1):
            u, v = path[i].id, path[i + 1].id
            if excluded_node_ids and (u in excluded_node_ids or v in excluded_node_ids):
                continue
            if self.G.has_edge(u, v):
                edge_weights.append(self.G[u][v].get('weight_norm', 0.0))
            else:
                edge_weights.append(0.0)
        return sum(edge_weights) / len(edge_weights) if edge_weights else 1.0

    def sample(self, max_steps, max_retries=5):
        """
        Sample a coherent skill path.

        Args:
            max_steps: Maximum number of steps in the path
            max_retries: Maximum number of retry attempts

        Returns:
            List of GraphNode objects representing the path
        """
        path = []
        for _ in range(max_retries):
            root = self.weighted_root_choice()
            path = self.sample_path(root, max_steps)
            path = [n for n in path if not self._is_skill_excluded(n)]
            if len(path) < 2:
                continue
            rare_in_path = {n.id for n in path if self.is_rare_skill(n.id)}
            coherence = self._path_coherence(path, excluded_node_ids=rare_in_path or None)
            if coherence >= 0.1:
                return path
        return path

    def sample_with_rare_injection(self, rare_skill_id, max_steps, max_retries=5):
        """
        Sample a backbone path and inject a rare skill at the best position.

        Args:
            rare_skill_id: Skill ID to inject
            max_steps: Maximum path length
            max_retries: Maximum retry attempts

        Returns:
            Path with rare skill injected at optimal position
        """
        if rare_skill_id not in self.G or rare_skill_id not in self.nodes_id_dict:
            return self.sample(max_steps, max_retries=max_retries)

        rare_node = self.nodes_id_dict[rare_skill_id]

        for _ in range(max_retries):
            backbone = self.sample(max_steps=max_steps - 1, max_retries=max_retries)
            if len(backbone) < 2:
                continue

            if rare_skill_id in {n.id for n in backbone}:
                return backbone

            # Find best insertion position based on edge weights
            best_pos = len(backbone)
            best_weight = -1.0
            for i, node in enumerate(backbone):
                w = 0.0
                if self.G.has_edge(node.id, rare_skill_id):
                    w = max(w, self.G[node.id][rare_skill_id].get('weight', 0))
                if self.G.has_edge(rare_skill_id, node.id):
                    w = max(w, self.G[rare_skill_id][node.id].get('weight', 0))
                if w > best_weight:
                    best_weight = w
                    best_pos = i + 1

            backbone.insert(best_pos, rare_node)
            return backbone

        return self.sample(max_steps, max_retries=max_retries)

    def sample_with_novelty(
        self,
        max_steps,
        existing_skill_ids,
        min_new_skills=5,
        max_retries=20,
    ):
        """
        Sample a path where at least min_new_skills are new (not in existing_skill_ids).

        Args:
            max_steps: Maximum path length
            existing_skill_ids: Set of skill IDs already used
            min_new_skills: Minimum number of new skills required
            max_retries: Maximum retry attempts

        Returns:
            Path meeting novelty constraint, or best path found
        """
        best_path = None
        best_new_count = -1

        for _ in range(max_retries):
            path = self.sample(max_steps)
            path_skill_ids = {n.id for n in path}
            new_count = len(path_skill_ids - existing_skill_ids)

            if new_count >= min_new_skills:
                return path
            if new_count > best_new_count:
                best_new_count = new_count
                best_path = path

        print(f"  [Novelty] Could not find path with {min_new_skills} new skills "
              f"after {max_retries} retries (best: {best_new_count})")
        return best_path if best_path is not None else path

    def sample_with_rare_and_novelty(
        self,
        rare_skill_id,
        max_steps,
        existing_skill_ids,
        min_new_skills=5,
        max_retries=20,
    ):
        """Rare-skill injection + novelty constraint combined."""
        best_path = None
        best_new_count = -1

        for _ in range(max_retries):
            path = self.sample_with_rare_injection(rare_skill_id, max_steps)
            path_skill_ids = {n.id for n in path}
            new_count = len(path_skill_ids - existing_skill_ids)

            if new_count >= min_new_skills:
                return path
            if new_count > best_new_count:
                best_new_count = new_count
                best_path = path

        print(f"  [Novelty+Rare] Could not find path with {min_new_skills} new skills "
              f"after {max_retries} retries (best: {best_new_count})")
        return best_path if best_path is not None else path

    def _weighted_choice(self, current, neighbors, min_edge_weight_norm=0.05):
        """
        Choose next node with probability proportional to combined node+edge score.

        Args:
            current: Current node ID
            neighbors: Candidate neighbor node IDs
            min_edge_weight_norm: Minimum edge weight for viability

        Returns:
            Selected neighbor node ID
        """
        # Filter neighbors with insufficient edge weight
        viable = [
            n for n in neighbors
            if self.G.has_edge(current, n) and self.G[current][n].get('weight_norm', 0) >= min_edge_weight_norm
        ]
        if not viable:
            viable = neighbors  # Fallback when no viable neighbors

        if random.random() < self.random_prob:
            return random.choice(viable)

        # Compute combined scores
        scores = []
        for n in viable:
            node_score = self.alpha * self.G.nodes[n]['weight_norm']
            edge_score = self.beta * self.G[current][n]['weight_norm']
            scores.append(max(node_score + edge_score, 1e-6))

        r = random.uniform(0, sum(scores))
        acc = 0
        for n, s in zip(viable, scores):
            acc += s
            if acc >= r:
                return n
        return viable[-1]

    # ==================== Coverage Queries ====================

    def get_uncovered_skills(self):
        """Return a list of skill node IDs with coverage == 0."""
        return [n for n in self.top_node_indices if self.G.nodes[n].get('coverage', 0) == 0]

    def get_coverage_ratio(self):
        """Return the ratio of covered skills (0~1)."""
        total = len(self.top_node_indices)
        if total == 0:
            return 1.0
        covered = sum(1 for n in self.top_node_indices if self.G.nodes[n].get('coverage', 0) > 0)
        return covered / total

    def get_rare_skills(self):
        """Return uncovered rare skill IDs."""
        return [n for n in self._rare_skill_ids if self.G.nodes[n].get('coverage', 0) == 0]

    def is_rare_skill(self, node_id):
        """Return whether a skill is rare (based on base_weight)."""
        return node_id in self._rare_skill_ids

    def force_sample_with_main(self, skill_id, max_aux, max_retries=5):
        """
        Sample a path starting from a specific skill with coherence constraints.

        Args:
            skill_id: Starting skill ID (forced)
            max_aux: Maximum number of auxiliary skills
            max_retries: Maximum retry attempts

        Returns:
            Path starting with the forced skill
        """
        if skill_id not in self.G or skill_id not in self.nodes_id_dict:
            return []
        path = []
        for _ in range(max_retries):
            path = [self.nodes_id_dict[skill_id]]
            visited = {skill_id}
            current = skill_id
            for _ in range(max_aux):
                neighbors = [n for n in self.G.successors(current) if n not in visited]
                if not neighbors:
                    break
                next_node = self._weighted_choice(current, neighbors)
                path.append(self.nodes_id_dict[next_node])
                visited.add(next_node)
                current = next_node
            # Keep forced skill, filter excluded from auxiliaries
            path = [path[0]] + [n for n in path[1:] if not self._is_skill_excluded(n)]
            if len(path) >= 2 and self._path_coherence(path) >= 0.1:
                return path
            if len(path) == 1:
                return path
        return path

    # ==================== Evidence Sampling ====================

    def sample_evidence(self, skill_path, max_examples=3):
        """
        Sample example evidence for a skill path.

        Args:
            skill_path: List of skill IDs
            max_examples: Maximum number of evidence examples

        Returns:
            List of (example_id, steps) tuples
        """
        uncovered = set(skill_path)
        evidences = []
        used_examples = set()

        while uncovered and len(evidences) < max_examples:
            example = self._sample_example_for_skills(uncovered, used_examples)
            if example is None:
                break

            steps = self._select_relevant_steps(example, uncovered)
            if not steps:
                used_examples.add(example.id)
                continue

            evidences.append((example.id, steps))
            used_examples.add(example.id)

            for step in steps:
                uncovered -= step.skills

        return evidences

    def score_example_for_path(self, example_id, path):
        """
        Score an example by summing its skill scores for the given path.

        Args:
            example_id: Example ID to score
            path: List of skill IDs

        Returns:
            Total score (sum of skill step scores)
        """
        score = 0.0
        for skill in path:
            if skill not in self.example_skill_steps[example_id]:
                continue
            score += self.example_skill_steps[example_id][skill]
        return score

    def _sample_example_for_skills(self, skills, used_examples):
        """
        Sample an example that covers the given skills.

        Args:
            skills: Set of skill IDs to cover
            used_examples: Set of already-used example IDs

        Returns:
            Example object or None if no candidates
        """
        candidate_ids = set()
        for s in skills:
            candidate_ids |= self.skill_to_examples.get(s, set())

        candidate_ids -= used_examples
        if not candidate_ids:
            return None

        scores = []
        examples = []

        for ex_id in candidate_ids:
            ex = self.example_dict[ex_id]
            score = self.score_example_for_path(ex_id, skills)
            if score <= 0:
                continue
            # Penalize already-covered examples
            score = score / (1 + ex.coverage)
            scores.append(score)
            examples.append(ex)

        if not examples:
            return None

        return random.choices(examples, weights=scores, k=1)[0]

    def _select_relevant_steps(self, example: Example, skills):
        """
        Select steps from an example that cover the given skills.

        Args:
            example: Example object
            skills: Set of skill IDs to cover

        Returns:
            List of relevant Step objects, or None if no matches
        """
        target = set(skills)
        selected = []

        for step in example.steps:
            if step.skills & target:
                selected.append(step)

        if not selected:
            return None

        return selected

    def few_shot_for_skill(self, skill, sample_num):
        """
        Sample few-shot examples for a given skill.

        Args:
            skill: Skill ID
            sample_num: Number of examples to sample

        Returns:
            List of example dictionaries with step information
        """
        candidate_ids = self.skill_to_examples.get(skill, set())
        if not candidate_ids:
            return None

        scores = []
        examples = []

        for ex_id in candidate_ids:
            ex = self.example_dict[ex_id]
            score = self.score_example_for_path(ex_id, [skill])
            if score <= 0:
                continue
            scores.append(score)
            examples.append(ex)

        if not examples:
            return None

        # Sample without replacement based on scores
        samples = np.random.choice(
            examples,
            size=sample_num,
            replace=False,
            p=np.array(scores) / sum(scores)
        ).tolist() if len(examples) > sample_num else examples

        steps = []
        for example in samples:
            selected_steps = self._select_relevant_steps(example, [skill])
            if selected_steps:
                steps.append({
                    "example_id": example.id,
                    "steps": [{"step_id": step.id, "text": step.text} for step in selected_steps]
                })
        return steps

    # ==================== Query Methods ====================

    def get_skills(self) -> list[GraphNode]:
        """Return all top-level skill nodes."""
        return [self.nodes_id_dict[n] for n in self.top_node_indices]

    def example_skill_coverage(self, example, skills):
        """Get the number of skills covered by an example."""
        ex_skills = set()
        for step in example.steps:
            ex_skills |= step.skills
        return len(set(skills) & ex_skills)

    def get_example_covered_skills(self, example, skills):
        """Get the skill nodes covered by an example."""
        covered = set()
        for step in example.steps:
            covered |= (step.skills & set(skills))
        return [self.nodes_id_dict[s] for s in covered]

    # ==================== Statistics ====================

    def print_node_statistics(self, topk=20):
        """Print statistics about node appearances in sampled paths."""
        print("\n=== Node Appearance Counts (Top {}) ===".format(topk))
        for n, c in self.node_counter.most_common(topk):
            node = self.G.nodes[n]
            print(f"Node ID: {n}, Name: {node.get('name','')}, Count: {c}")

        print(f"\nTotal number of different nodes covered: {len(self.node_counter)}")

    def print_example_statistics(self, topk=20):
        """Print statistics about example usage as evidence."""
        print("\n=== Example Appearance Counts (Top {}) ===".format(topk))
        for ex_id, c in self.example_counter.most_common(topk):
            ex = self.example_dict[ex_id]
            print(f"Example ID: {ex_id}, Coverage: {ex.coverage}, Count: {c}")

        print(f"\nTotal number of different examples used: {len(self.example_counter)}")


if __name__ == "__main__":
    # Demo usage
    sampler = PathSampler(
        path_record_file='../skill_cluster/data/sample_paths.jsonl',
        alpha=1.0,
        beta=1.0,
        random_prob=0.1
    )

    for _ in range(1):
        path = sampler.sample(max_steps=5)
        for n in path:
            print(f'{n.id}: {n.name}')
        path_ids = [n.id for n in path]
        evidence = sampler.sample_evidence(path_ids, max_examples=5)
        for ex_id, steps in evidence:
            print(f'  Example {ex_id}:')
            example_covered_skills = sampler.get_example_covered_skills(
                sampler.example_dict[ex_id],
                [n.id for n in path]
            )
            print(f'  Covered Skills: {", ".join([s.name for s in example_covered_skills])}')
            for step in steps:
                print(f'    Step {step.id}: {step.text}')
        sampler.add_path(path_ids, evidence)