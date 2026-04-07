"""
Remove Name Covers Script

Functionality:
1. Load graph nodes and identify top-level nodes
2. Remove nodes whose names are covered by other nodes' names
   (e.g., "Load Data" covers "Load Data from CSV" - shorter name is more general)
3. Process top nodes first, then handle deeper nodes via BFS
4. Maintain graph integrity after node removal (update children/parents)
5. Export cleaned cluster data and graph structure

This script improves cluster quality by removing redundant nodes
where one node's name semantically covers another's.
"""

import argparse
import json
from collections import defaultdict, deque
import logging
import os

from check_name_covers import check_name_covers
from log_utils import set_logger
import sys
sys.path.append('..')
from utils.graph_utils import GraphNode, remove

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='Layer to process')
args = parser.parse_args()

# Global data structures
nodes_id_dict = {}              # Maps node ID -> GraphNode object
name_node_ids_dict = defaultdict(list)  # Maps lowercase name -> list of node IDs

# Load graph nodes from file
with open(f'data/layer{args.layer}/graph-nodes.jsonl', 'r') as fin:
    for line in fin:
        obj = json.loads(line)
        nodes_id_dict[obj['id']] = GraphNode(**obj)
        name_lower = obj['name'].lower()
        name_node_ids_dict[name_lower].append(obj['id'])

# Identify top-level nodes (nodes without parents)
top_node_indices = set()
with open(f'data/layer{args.layer}/step-clusters.jsonl', 'r') as fin:
    for line in fin:
        obj = json.loads(line)
        top_node_indices.add(obj['graph_id'])

# Configure logging
set_logger(f'data/remove_name_covers.log')

def get_node_depths() -> dict[str, int]:
    """
    Compute depth of each node in the graph using BFS.

    Depth is measured as the minimum distance from any top node.
    BFS guarantees each node is visited at its minimum depth.

    Returns:
        Dictionary mapping node ID -> depth (distance from top)
    """
    node_depths: dict[str, int] = {}
    visited: set[str] = set()
    q = deque()

    # Initialize BFS with all top nodes at depth 0
    for top_node_id in top_node_indices:
        node = nodes_id_dict[top_node_id]
        node_depths[node.id] = 0
        visited.add(node.id)
        q.append(node)

    # BFS traversal: each node visited once at minimum depth
    while q:
        node = q.popleft()
        depth = node_depths[node.id]

        for child_id in node.children:
            if child_id in visited:
                continue

            child_node = nodes_id_dict[child_id]
            visited.add(child_id)
            node_depths[child_id] = depth + 1
            q.append(child_node)

    return node_depths

def get_node_count(node_id: str) -> int:
    """
    Count the number of unique sources (posters/answers) in a node's cluster.

    Source format: {id}-{answer_id}-{step_index}
    We count unique {id}-{answer_id} prefixes.

    Args:
        node_id: Node ID to count sources for

    Returns:
        Number of unique source prefixes
    """
    node = nodes_id_dict[node_id]
    prefices = set()
    for c in node.cluster:
        # Extract {id}-{answer_id} prefix
        prefix = "-".join(c.split('-')[:2])
        prefices.add(prefix)
    return len(prefices)

def update_top_node_indices(top_node_indices: set[str]) -> None:
    """
    Clean up top nodes by removing those with insufficient sources.

    A top node is removed if:
    1. Its ID starts with "2-" (higher layer node, skip processing)
    2. It has only 1 or fewer unique sources (too small)

    When a node is removed, its children may become orphaned and also
    need to be removed if they have no parents.

    Args:
        top_node_indices: Set of top node IDs (modified in place)

    Returns:
        List of removed node names (lowercase)
    """
    removed_names_lower = set()
    queue = deque()

    # Identify nodes to remove
    for node_id in top_node_indices:
        if node_id.startswith('2-'):
            # Skip higher layer nodes
            logging.info(f'Skipped top node {node_id} name "{nodes_id_dict[node_id].name}" starting with "2-"')
            queue.append(node_id)
            continue
        node_count = get_node_count(node_id)
        if node_count <= 1:
            # Remove nodes with insufficient sources
            logging.info(f'Removed top node {node_id} name "{nodes_id_dict[node_id].name}" with origin size {node_count}')
            queue.append(node_id)

    # Process removal queue (handles cascade of orphaned children)
    while queue:
        node_id = queue.popleft()
        children = nodes_id_dict[node_id].children.copy()
        removed_names_lower.add(nodes_id_dict[node_id].name.lower())
        remove(nodes_id_dict[node_id], top_node_indices, nodes_id_dict)

        # Check for orphaned children
        for child_id in children:
            if len(nodes_id_dict[child_id].parents) == 0 and child_id not in top_node_indices:
                # Child is now orphaned, remove it too
                assert get_node_count(child_id) <= 1
                logging.info(f'Removed separated node {child_id} name "{nodes_id_dict[child_id].name}" with origin size {get_node_count(child_id)}')
                queue.append(child_id)

    # Update name->ID mapping after removals
    for name in removed_names_lower:
        node_ids = [node_id for node_id in name_node_ids_dict[name] if node_id in nodes_id_dict]
        if len(node_ids) == 0:
            name_node_ids_dict.pop(name)
        else:
            name_node_ids_dict[name] = node_ids

# ==================== Phase 1: Process Top Nodes ====================

# Clean up top nodes first
update_top_node_indices(top_node_indices)

# Iteratively remove covered nodes among top nodes
while True:
    top_nodes_hash = set(top_node_indices)
    top_node_names_lower = set()
    for node_id in top_nodes_hash:
        top_node_names_lower.add(nodes_id_dict[node_id].name.lower())

    # Check for name coverage relationships
    sorted_clusters, wordset_cache = check_name_covers(top_node_names_lower, threshold=1.0)
    update_flag = False

    for label, names in sorted_clusters:
        # label is the covering name, names are all names in this cluster
        label_length = len(wordset_cache[label])
        # Filtered names: names covered by label (longer/more specific)
        filtered_names = [name for name in names if len(wordset_cache[name]) > label_length]
        # Label names: names with same length as label
        label_names = [name for name in names if len(wordset_cache[name]) == label_length]

        if not filtered_names:
            continue

        # Get node IDs for label nodes and covered nodes
        label_node_ids = set()
        for label in label_names:
            label_node_ids.update(name_node_ids_dict[label])
        top_label_node_ids = [label_node_id for label_node_id in label_node_ids if label_node_id in top_nodes_hash]

        covered_node_ids = set()
        for name in filtered_names:
            covered_node_ids.update(name_node_ids_dict[name])
        top_covered_node_ids = [covered_node_id for covered_node_id in covered_node_ids if covered_node_id in top_nodes_hash]

        # If there are covered nodes among top nodes, remove the label nodes
        if len(top_covered_node_ids) > 0:
            update_flag = True
            for label_node_id in top_label_node_ids:
                children = nodes_id_dict[label_node_id].children.copy()
                label_node_name = nodes_id_dict[label_node_id].name
                logging.info(f'Removed top node {label_node_id} name "{label_node_name}" covered by top node {top_covered_node_ids[0]} name "{nodes_id_dict[top_covered_node_ids[0]].name}"')
                remove(nodes_id_dict[label_node_id], top_node_indices, nodes_id_dict)

                # Promote orphaned children to top nodes
                for child_id in children:
                    if len(nodes_id_dict[child_id].parents) == 0:
                        top_node_indices.add(child_id)
                        logging.info(f'Add child node {child_id} name "{nodes_id_dict[child_id].name}" to top nodes after removing its parent node {label_node_id} name "{label_node_name}"')

            # Clean up name->ID mapping if all label nodes removed
            if len(top_label_node_ids) == len(label_node_ids):
                for label in label_names:
                    name_node_ids_dict.pop(label)

    if not update_flag:
        break
    update_top_node_indices(top_node_indices)

# ==================== Phase 2: Process Bottom Nodes ====================

# Iteratively remove covered nodes at all depths
while True:
    node_depths = get_node_depths()
    sorted_clusters, wordset_cache = check_name_covers(set(name_node_ids_dict.keys()), threshold=1.0)
    update_flag = False
    add_top_node_flag = False

    for label, names in sorted_clusters:
        label_length = len(wordset_cache[label])
        filtered_names = [name for name in names if len(wordset_cache[name]) > label_length]
        label_names = [name for name in names if len(wordset_cache[name]) == label_length]

        if not filtered_names:
            continue

        # Get node IDs for all nodes with these names
        label_node_ids = set()
        for label in label_names:
            label_node_ids.update(name_node_ids_dict[label])

        covered_node_ids = set()
        for name in filtered_names:
            covered_node_ids.update(name_node_ids_dict[name])

        # Check each label node for coverage by shallower nodes
        for label_node_id in label_node_ids:
            # Find covered nodes that are at same depth or shallower
            shallow_covered_node_ids = [
                covered_node_id for covered_node_id in covered_node_ids
                if node_depths[covered_node_id] <= node_depths[label_node_id]
            ]

            if len(shallow_covered_node_ids) > 0:
                children = nodes_id_dict[label_node_id].children.copy()
                logging.info(f'Removed node {label_node_id} depth {node_depths[label_node_id]} name "{nodes_id_dict[label_node_id].name}" covered by node {shallow_covered_node_ids[0]} depth {node_depths[shallow_covered_node_ids[0]]} name "{nodes_id_dict[shallow_covered_node_ids[0]].name}"')
                remove(nodes_id_dict[label_node_id], top_node_indices, nodes_id_dict)
                update_flag = True

                # If removed node was at depth 0 (top), promote children
                if node_depths[label_node_id] == 0:
                    for child_id in children:
                        if len(nodes_id_dict[child_id].parents) == 0:
                            top_node_indices.add(child_id)
                            logging.info(f'Add child node {child_id} name "{nodes_id_dict[child_id].name}" to top nodes after removing its parent node {label_node_id} name "{nodes_id_dict[label_node_id].name}"')
                            add_top_node_flag = True

        # Clean up name->ID mapping
        for label in label_names:
            if all([label_node_id not in nodes_id_dict for label_node_id in name_node_ids_dict[label]]):
                name_node_ids_dict.pop(label)

    if not update_flag:
        break
    if add_top_node_flag:
        update_top_node_indices(top_node_indices)

# ==================== Export Results ====================

# Export cluster data for top nodes
with open(f'data/step-clusters.jsonl', 'w') as fout:
    sorted_top_node_indices = sorted(top_node_indices, key=lambda x: tuple(int(i) for i in x.split('-')))
    for top_node_id in sorted_top_node_indices:
        cur_node = nodes_id_dict[top_node_id]
        fout.write(json.dumps({
            'graph_id': cur_node.id,
            'name': cur_node.name,
            'cluster': list(sorted(cur_node.cluster))
        }) + '\n')

# Export all graph nodes
with open(f'data/graph-nodes.jsonl', 'w') as fout:
    for node in nodes_id_dict.values():
        fout.write(json.dumps(node.to_dict()) + '\n')