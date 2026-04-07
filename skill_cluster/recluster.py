"""
RAPTOR Cluster Reclustering Script

Functionality:
1. Recover cluster tree from RAPTOR clustering results
2. Post-process outliers by recovering their sub-clusters
3. Recluster leaf nodes to meet minimum/maximum size constraints
4. Merge small clusters with siblings to achieve target cluster sizes
5. Export reclustered results for downstream LLM processing

This script ensures clusters have enough unique sources (posters/answers)
for meaningful LLM-based skill clustering analysis.
"""

import argparse
import json
import random
import os
from collections import defaultdict

from cluster_utils import RaptorClusteringRecovery, ClusterNode

# Configuration constants
random.seed(42)  # For reproducibility in random cluster selection
max_cluster_nodes = 30  # Maximum number of unique sources per cluster
min_cluster_nodes = 20  # Minimum number of unique sources per cluster

def postprocess_raptor(layer: int, cluster_node: ClusterNode):
    """
    Recursively post-process RAPTOR cluster tree to handle outliers.

    Outliers are nodes that were not assigned to any child cluster during
    RAPTOR clustering. If outlier sub-clustering was performed, this function
    recovers those results and attaches them to the cluster tree.

    Args:
        layer: Layer number for file paths
        cluster_node: Current cluster node to process
    """
    grand_children = []        # Collect grand children for recursive processing
    clustered_indices = set()  # Track indices assigned to child clusters

    # Iterate through children to identify clustered nodes
    for child_node in cluster_node.children:
        if child_node.child_num == 0:
            # Leaf node: nodes directly in cluster
            clustered_indices.update(child_node.cluster)
        else:
            # Non-leaf node: collect from grand children
            for grand_child_node in child_node.children:
                clustered_indices.update(grand_child_node.cluster)
                if grand_child_node.child_num > 0:
                    grand_children.append(grand_child_node)

    # Identify outliers (indices not in any child cluster)
    outlier_indices = [index for index in cluster_node.cluster if index not in clustered_indices]
    if len(outlier_indices) > 0:
        outlier_name = f'{cluster_node.index}-outlier'
        outlier_child_node = ClusterNode('local', outlier_indices, outlier_name)

        # If outlier clustering was performed, recover and attach results
        if os.path.exists(f'data/layer{layer}/raptor/{outlier_name}.log'):
            outlier_child_node.children = RaptorClusteringRecovery(layer, outlier_name).recover()
            outlier_child_node.child_num = len(outlier_child_node.children)
            for n in outlier_child_node.children:
                n.parent = outlier_child_node
            grand_children.append(outlier_child_node)

        # Attach outlier node to cluster tree
        assert cluster_node.child_num > 0
        cluster_node.children.append(outlier_child_node)
        cluster_node.child_num += 1
        outlier_child_node.parent = cluster_node

    # Recursively process grand children
    for grand_child_node in grand_children:
        postprocess_raptor(layer, grand_child_node)

def get_leaf_nodes(cluster_node: ClusterNode) -> list:
    """
    Extract all leaf nodes from cluster tree.

    Leaf nodes are clusters with no children (child_num == 0).
    Returns leaf nodes sorted by priority: regular leaves first, then outlier leaves.

    Args:
        cluster_node: Root of cluster tree

    Returns:
        List of leaf ClusterNode objects
    """
    def _get_leaf_nodes(cluster_node: ClusterNode, leaf_nodes: list, outlier_leaf_nodes: list):
        if cluster_node.child_num == 0:
            # Separate outlier leaves from regular leaves
            if 'outlier' in cluster_node.index:
                outlier_leaf_nodes.append(cluster_node)
            else:
                leaf_nodes.append(cluster_node)
            return

        # Recursively collect leaves from children
        for child_node in cluster_node.children:
            _get_leaf_nodes(child_node, leaf_nodes, outlier_leaf_nodes)

    leaf_nodes = []
    outlier_leaf_nodes = []
    _get_leaf_nodes(cluster_node, leaf_nodes, outlier_leaf_nodes)
    # Return with outlier leaves at end (lower priority for merging)
    leaf_nodes.extend(outlier_leaf_nodes)
    return leaf_nodes

def get_nodes_index_dict(cluster_node: ClusterNode, nodes_index_dict: dict):
    """
    Build dictionary mapping node index to ClusterNode object.

    Recursively traverses tree to collect all nodes.

    Args:
        cluster_node: Current node to process
        nodes_index_dict: Dictionary to populate (index -> ClusterNode)
    """
    assert cluster_node.index not in nodes_index_dict
    nodes_index_dict[cluster_node.index] = cluster_node

    for child_node in cluster_node.children:
        get_nodes_index_dict(child_node, nodes_index_dict)

def run_recluster(layer: int):
    """
    Execute the reclustering process for a given layer.

    This function performs:
    1. Load step sources and cluster representatives
    2. Recover cluster tree from RAPTOR results
    3. Post-process outliers
    4. Merge small clusters to meet minimum size constraints
    5. Export reclustered results

    Args:
        layer: Layer number to process
    """
    # Load step source mapping for layer 1
    steps_source_dict = {}
    with open(f'data/steps.jsonl', 'r') as fin:
        for line in fin:
            obj = json.loads(line)
            for j, step in enumerate(obj['steps']):
                # source format: {id}-{answer_id}-{step_index}
                source = f"{obj['id']}-{obj['answer_id']}-{j}"
                steps_source_dict[source] = step

    # Load cluster representatives for higher layers
    reps = []
    if layer > 1:
        with open(f'data/layer{layer}/step-clusters-representatives.jsonl', 'r') as fin:
            for line in fin:
                obj = json.loads(line)
                reps.append(obj)

    # Recover cluster tree from RAPTOR results
    root_node = ClusterNode(
        'local',
        [str(i) for i in range(len(reps))] if layer > 1 else list(steps_source_dict.keys()),
        'main'
    )
    root_node.children = RaptorClusteringRecovery(layer, 'main').recover()
    root_node.child_num = len(root_node.children)

    # Post-process to recover outlier sub-clusters
    postprocess_raptor(layer, root_node)

    # Extract leaf nodes and build index dictionary
    leaf_nodes = get_leaf_nodes(root_node)
    nodes_index_dict = {}
    get_nodes_index_dict(root_node, nodes_index_dict)

    # Clear parent reference for top-level children
    for child in root_node.children:
        child.parent = None

    # Process each leaf node to ensure minimum cluster size
    finished_clusters = set()  # Track clusters that have been processed
    for leaf_node in leaf_nodes:
        if leaf_node.index in finished_clusters:
            continue

        # Collect unique poster IDs (source documents/answers) in this cluster
        poster_ids = set()
        if layer > 1:
            poster_ids.update(leaf_node.cluster)
        else:
            for source in leaf_node.cluster:
                # Extract poster ID from source (remove step index)
                poster_id = source[:source.rindex('-')]
                poster_ids.add(poster_id)

        current = leaf_node
        added_clusters = set([leaf_node.index])  # Clusters merged into current group

        # Merge with siblings if cluster is too small
        while len(poster_ids) < min_cluster_nodes:
            # Stop if reached root (no more parent to merge from)
            if current.parent is None:
                break

            # Move up to parent level
            current = current.parent

            # Find sibling leaf nodes that haven't been processed
            pending_clusters = []
            cur_leaf_nodes = get_leaf_nodes(current)
            for child in cur_leaf_nodes:
                if child.index not in finished_clusters and child.index not in added_clusters:
                    pending_clusters.append(child)

            # No available siblings, continue to higher parent
            if not pending_clusters:
                continue

            # Randomly try siblings to find suitable merge candidates
            random.shuffle(pending_clusters)
            for sibling in pending_clusters:
                poster_ids_copy = poster_ids.copy()

                # Add sibling's poster IDs
                if layer > 1:
                    poster_ids_copy.update(sibling.cluster)
                else:
                    for source in sibling.cluster:
                        poster_id = source[:source.rindex('-')]
                        poster_ids_copy.add(poster_id)

                # Check if merge exceeds maximum
                if len(poster_ids_copy) > max_cluster_nodes:
                    continue

                # Accept merge
                poster_ids = poster_ids_copy
                added_clusters.add(sibling.index)

                # Stop if minimum size achieved
                if len(poster_ids) >= min_cluster_nodes:
                    break

        # Write merged cluster to output file
        with open(f'data/layer{layer}/step-clusters-raptor-recluster.jsonl', 'a') as fout:
            if layer > 1:
                # Collect representative IDs from all merged clusters
                cur_rep_ids = []
                for cluster_id in added_clusters:
                    cur_rep_ids.extend(nodes_index_dict[cluster_id].cluster)

                # Build step dictionary to avoid duplicates and maintain order
                # Uses set to deduplicate repeated representative IDs
                cur_steps_source_dict = defaultdict(dict)
                for rep_id in cur_rep_ids:
                    source = reps[int(rep_id)]['source']
                    # Extract step numbers from sources
                    nums = set([int(s.split('-')[-1]) for s in source])
                    poster_id = source[0][:source[0].rindex('-')]
                    num = min(nums)  # Use earliest step number for ordering
                    if num not in cur_steps_source_dict[poster_id]:
                        cur_steps_source_dict[poster_id][num] = set()
                    cur_steps_source_dict[poster_id][num].add(rep_id)

                # Build output list sorted by poster and step number
                cur_reps = []
                for poster_id in sorted(cur_steps_source_dict.keys()):
                    for num in sorted(cur_steps_source_dict[poster_id].keys()):
                        for rep_id in cur_steps_source_dict[poster_id][num]:
                            cur_reps.append({
                                'text': reps[int(rep_id)]['text'],
                                'children': [rep_id]
                            })
                fout.write(json.dumps(cur_reps) + '\n')
            else:
                # Layer 1: collect raw step sources
                cur_sources = []
                for cluster_id in added_clusters:
                    cur_sources.extend(nodes_index_dict[cluster_id].cluster)

                # Build step dictionary maintaining order within each poster
                cur_steps_source_dict = defaultdict(dict)
                for source in cur_sources:
                    num = source.split('-')[-1]
                    poster_id = source[:source.rindex('-')]
                    cur_steps_source_dict[poster_id][int(num)] = steps_source_dict[source]

                # Merge steps from same poster into single text
                merged_steps = []
                for poster_id in sorted(cur_steps_source_dict.keys()):
                    merged_steps.append({
                        'text': '\n'.join(cur_steps_source_dict[poster_id][num]
                                          for num in sorted(cur_steps_source_dict[poster_id].keys())),
                        'source': [f"{poster_id}-{num}"
                                   for num in sorted(cur_steps_source_dict[poster_id].keys())]
                    })
                fout.write(json.dumps(merged_steps) + '\n')

        # Mark all merged clusters as finished
        finished_clusters.update(added_clusters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True, help='Layer to process')
    args = parser.parse_args()

    run_recluster(args.layer)