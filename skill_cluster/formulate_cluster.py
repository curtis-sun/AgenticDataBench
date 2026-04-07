"""
Cluster Formulation Script

Functionality:
1. Merge and rename clusters based on LLM-generated output and DBSCAN results
2. Build hierarchical graph structure for cluster relationships
3. Handle node merging and parent-child connections
4. Export cluster data and graph nodes for next layer processing
"""

import argparse
import json
from collections import defaultdict
import logging
import os

from cluster_utils import length
from log_utils import set_logger
import sys
sys.path.append('..')
from utils.graph_utils import GraphNode, replace, remove

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='Layer to process')
args = parser.parse_args()

# Global data structures for cluster management
cluster_names_dict = {}        # Maps lowercase name -> original name (preserves case)
merged_clusters = defaultdict(set)   # Maps lowercase name -> set of cluster members/sources
merged_sources = defaultdict(set)    # Maps lowercase name -> set of source IDs (for layer > 1)
graph_ids_source_list = []     # List of graph IDs from previous layer
clusters_source_list = []      # List of cluster source lists from previous layer

if args.layer > 1:
    # Load existing cluster data for higher layers
    with open(f'data/layer{args.layer}/step-clusters.jsonl', 'r') as fin:
        for id, line in enumerate(fin.readlines()):
            obj = json.loads(line)
            clusters_source_list.append(obj['cluster'])  # Cluster member IDs
            graph_ids_source_list.append(obj['graph_id'])  # Graph node ID

# Load LLM-generated cluster output and merge clusters with same name
with open(f'data/layer{args.layer}/step-clusters-output.jsonl', 'r') as fin:
    for line in fin:
        obj = json.loads(line)
        for name, c in obj['clusters'].items():
            name_lower = name.lower()
            # Initialize name mapping for first occurrence
            if name_lower not in merged_clusters:
                cluster_names_dict[name_lower] = name
            # Merge cluster members
            merged_clusters[name_lower].update(c)
            # For higher layers, merge underlying sources from referenced clusters
            if args.layer > 1:
                for rep_id in c:
                    merged_sources[name_lower].update(clusters_source_list[int(rep_id)])

# Load DBSCAN results for cluster renaming/merging
rename_dict = {}  # Maps lowercase name -> new name (from DBSCAN labels)
with open(f'data/layer{args.layer}/step-clusters-dbscan.jsonl', 'r') as fin:
    for line in fin:
        obj = json.loads(line)
        label_lower = obj['label'].lower()
        for c in obj['cluster']:
            name_lower = c['name'].lower()
            # Skip if name already matches the label
            if name_lower == label_lower:
                continue
            # Record rename mapping
            rename_dict[name_lower] = obj['label']
            if name_lower not in merged_clusters:
                continue
            # Perform merge: add members to label cluster and remove old cluster
            if label_lower not in merged_clusters:
                cluster_names_dict[label_lower] = obj['label']
            merged_clusters[label_lower].update(merged_clusters[name_lower])
            merged_clusters.pop(name_lower, None)
            if args.layer > 1:
                merged_sources[label_lower].update(merged_sources[name_lower])
                merged_sources.pop(name_lower, None)

if not os.path.exists(f'data/layer{args.layer+1}'):
    os.makedirs(f'data/layer{args.layer+1}')

if args.layer == 1:
    # Layer 1: Simple output, no graph structure needed
    with open(f'data/layer{args.layer+1}/step-clusters.jsonl', 'w') as fout1:
        with open(f'data/layer{args.layer+1}/graph-nodes.jsonl', 'w') as fout2:
            for i, name_lower in enumerate(merged_clusters):
                # Create graph node for each merged cluster
                cur_node = GraphNode(
                    id=f'{args.layer+1}-{i}',
                    name=cluster_names_dict[name_lower],
                    cluster=merged_clusters[name_lower]
                )
                fout1.write(json.dumps({
                    'graph_id': cur_node.id,
                    'name': cur_node.name,
                    'cluster': list(sorted(cur_node.cluster))
                }) + '\n')
                fout2.write(json.dumps(cur_node.to_dict()) + '\n')
else:
    # Higher layers: Build hierarchical graph structure with parent-child relationships
    set_logger(f'data/layer{args.layer}/formulate_cluster.log')

    # Load existing graph nodes
    nodes_id_dict = {}  # Maps node ID -> GraphNode object
    name_node_ids_dict = defaultdict(list)  # Maps lowercase name -> list of node IDs
    with open(f'data/layer{args.layer}/graph-nodes.jsonl', 'r') as fin:
        for line in fin:
            obj = json.loads(line)
            nodes_id_dict[obj['id']] = GraphNode(**obj)

    # Create new top-level nodes from merged clusters
    top_node_indices = set()  # IDs of nodes without parents (top of hierarchy)
    for i, name_lower in enumerate(merged_clusters):
        cur_node = GraphNode(
            id=f'{args.layer+1}-{i}',
            name=cluster_names_dict[name_lower],
            cluster=merged_sources[name_lower],
            children=[graph_ids_source_list[int(rep_id)] for rep_id in merged_clusters[name_lower]]
        )
        # Update parent references for children
        for child_id in cur_node.children:
            nodes_id_dict[child_id].parents.add(cur_node.id)
        nodes_id_dict[cur_node.id] = cur_node
        top_node_indices.add(cur_node.id)

    # Apply rename mappings to all nodes
    for node_id in nodes_id_dict:
        name_lower = nodes_id_dict[node_id].name.lower()
        if name_lower in rename_dict:
            nodes_id_dict[node_id].name = rename_dict[name_lower]
            name_lower = rename_dict[name_lower].lower()
        name_node_ids_dict[name_lower].append(node_id)

    # Step 1: Merge nodes with single child (parent-child collapse)
    for node_id in list(top_node_indices):
        cur_node = nodes_id_dict[node_id]
        assert len(cur_node.parents) == 0
        # If node has exactly one child, consider merging
        if len(cur_node.children) == 1:
            child_node = nodes_id_dict[list(cur_node.children)[0]]
            # Verify they represent same cluster
            assert sorted(cur_node.cluster) == sorted(child_node.cluster)
            # Choose name based on length metric (longer/more descriptive wins)
            if length(child_node.name) >= length(cur_node.name):
                logging.info(f'Replace node {cur_node.id} name "{cur_node.name}" with its child node {child_node.id} name "{child_node.name}"')
                replace(cur_node, child_node, top_node_indices, nodes_id_dict)
            else:
                logging.info(f'Replace child node {child_node.id} name "{child_node.name}" with its parent node {cur_node.id} name "{cur_node.name}"')
                replace(child_node, cur_node, top_node_indices, nodes_id_dict)

    # Step 2: Merge nodes with same name
    for v in name_node_ids_dict.values():
        # Filter to existing nodes (some may have been removed)
        node_ids = [node_id for node_id in v if node_id in nodes_id_dict]
        if len(node_ids) <= 1:
            continue

        # Separate top nodes (no parents) from bottom nodes (have parents)
        cur_top_nodes = []
        cur_bottom_nodes = []
        for node_id in node_ids:
            cur_node = nodes_id_dict[node_id]
            if node_id in top_node_indices:
                cur_top_nodes.append(cur_node)
            else:
                cur_bottom_nodes.append(cur_node)

        # Merge multiple top nodes with same name
        if len(cur_top_nodes) > 1:
            base_node = cur_top_nodes[0]
            for i in range(1, len(cur_top_nodes)):
                merge_node = cur_top_nodes[i]
                base_node.cluster.update(merge_node.cluster)
                logging.info(f'Merge node {merge_node.id} name "{merge_node.name}" into node {base_node.id} name "{base_node.name}"')
                replace(merge_node, base_node, top_node_indices, nodes_id_dict)

        # Merge multiple bottom nodes with same name
        if len(cur_bottom_nodes) > 1:
            base_node = cur_bottom_nodes[0]
            for i in range(1, len(cur_bottom_nodes)):
                merge_node = cur_bottom_nodes[i]
                base_node.cluster.update(merge_node.cluster)
                logging.info(f'Merge node {merge_node.id} name "{merge_node.name}" into node {base_node.id} name "{base_node.name}"')
                replace(merge_node, base_node, top_node_indices, nodes_id_dict)

        # Connect top node to bottom node if both exist
        # Bottom nodes may be overly general categories that should be children of top nodes
        if len(cur_top_nodes) >= 1 and len(cur_bottom_nodes) >= 1:
            top_node = cur_top_nodes[0]
            bottom_node = cur_bottom_nodes[0]
            # Establish parent-child relationship
            bottom_node.parents.add(top_node.id)
            top_node.children.add(bottom_node.id)
            logging.info(f'Connect top node {top_node.id} name "{top_node.name}" with bottom node {bottom_node.id} name "{bottom_node.name}"')
            # Remove bottom node from top indices (it now has a parent)
            remove(bottom_node, top_node_indices, nodes_id_dict)

    # Verify all nodes have unique names
    check_names = set()
    for node in nodes_id_dict.values():
        name_lower = node.name.lower()
        assert name_lower not in check_names, f"Duplicate name: {name_lower}"
        check_names.add(name_lower)

    # Export cluster data for next layer
    with open(f'data/layer{args.layer+1}/step-clusters.jsonl', 'w') as fout:
        # Sort by ID for consistent ordering
        sorted_top_node_indices = sorted(top_node_indices, key=lambda x: tuple(int(i) for i in x.split('-')))
        for top_node_id in sorted_top_node_indices:
            cur_node = nodes_id_dict[top_node_id]
            fout.write(json.dumps({
                'graph_id': cur_node.id,
                'name': cur_node.name,
                'cluster': list(sorted(cur_node.cluster))
            }) + '\n')

    # Export all graph nodes
    with open(f'data/layer{args.layer+1}/graph-nodes.jsonl', 'w') as fout:
        for node in nodes_id_dict.values():
            fout.write(json.dumps(node.to_dict()) + '\n')