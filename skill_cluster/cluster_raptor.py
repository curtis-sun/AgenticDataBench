"""
RAPTOR Clustering Script for Skill Steps

Functionality:
1. Load step embeddings from layer 1 or cluster representatives from higher layers
2. Perform hierarchical clustering using RAPTOR algorithm
3. Post-process clusters to handle outliers recursively
"""

import argparse
import numpy as np
import json
import jsonlines
import os

from cluster_utils import Node, RAPTOR_Clustering, cnt_cluster_sources, RaptorClusteringRecovery, ClusterNode
from log_utils import set_logger

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='Layer to process (1 for base layer, higher for clustered layers)')
args = parser.parse_args()

# Clustering hyperparameters
clustering_algorithm = RAPTOR_Clustering  # Clustering algorithm to use
cluster_embedding_model = 'sbert'          # Embedding model name for clustering
reduction_dimension = 10                   # Dimension for UMAP reduction before clustering
threshold = 0.5                            # Distance threshold for cluster merging
max_nodes_in_cluster = 30                  # Maximum nodes allowed in a single cluster

# Global data structures
nodes = []                    # List of Node objects for clustering
nodes_source_dict = {}       # Mapping from source ID to Node object (layer 1 only)
document_embeddings = None  # Numpy array of document embeddings

# Load data based on layer number
if args.layer > 1:
    # Higher layers: load pre-computed cluster representative embeddings
    document_embeddings = np.load(f'data/layer{args.layer}/step-clusters-embed.npy')
    with open(f'data/layer{args.layer}/step-clusters-representatives.jsonl', 'r') as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line)
            # Create Node with index as both index and parent (for cluster representatives)
            nodes.append(Node(str(i), str(i), {'sbert': document_embeddings[i]}))
else:
    # Layer 1: load individual step embeddings
    document_embeddings = np.load(f'data/steps-embed.npy')
    i = 0
    with open(f'data/steps.jsonl', 'r') as fin:
        for line in fin:
            obj = json.loads(line)
            for j, step in enumerate(obj['steps']):
                # Source format: {id}-{answer_id}-{step_index}
                source = f"{obj['id']}-{obj['answer_id']}-{j}"
                node = Node(source, obj['answer_id'], {'sbert': document_embeddings[i]})
                nodes.append(node)
                nodes_source_dict[source] = node
                i += 1
    assert len(document_embeddings) == i  # Verify embedding count matches step count

def cluster_raptor(name: str, nodes: list):
    """
    Perform RAPTOR clustering on a set of nodes.

    Args:
        name: Identifier for this clustering run (used for log files)
        nodes: List of Node objects to cluster

    This function skips if the log file already exists (resume capability).
    """
    # Check if clustering already performed (resume capability)
    if os.path.exists(f'data/layer{args.layer}/raptor/{name}.log'):
        return

    # Set up logging for this clustering run
    set_logger(f'data/layer{args.layer}/raptor/{name}.log')

    # Perform RAPTOR clustering
    clusters = clustering_algorithm.perform_clustering(
                    nodes,
                    cluster_embedding_model,
                    log_filename=f'data/layer{args.layer}/raptor/{name}-inner.jsonl',
                    max_nodes_in_cluster=max_nodes_in_cluster,
                    reduction_dimension=reduction_dimension,
                    threshold=threshold,
                    verbose=True
                )

    # Save leaf cluster assignments to file
    with open(f'data/layer{args.layer}/raptor/{name}-leaf.jsonl', 'w') as f:
        writer = jsonlines.Writer(f)
        for cluster in clusters:
            obj = [node.index for node in cluster]
            writer.write(obj)
        writer.close()

def get_node_from_index(index: str) -> Node:
    """
    Retrieve a Node object by its index string.

    Args:
        index: Node index string

    Returns:
        Node object corresponding to the index
    """
    if args.layer > 1:
        # Higher layers: index is numeric, access by position
        return nodes[int(index)]
    else:
        # Layer 1: index is source string, use dictionary lookup
        return nodes_source_dict[index]

def postprocess_raptor(cluster_node: ClusterNode):
    """
    Recursively post-process cluster tree to handle outliers.

    Outliers are nodes that were not assigned to any child cluster.
    If outliers exceed max_nodes_in_cluster, they are re-clustered recursively.

    Args:
        cluster_node: Current cluster node to process
    """
    grand_children = []         # Collect all grand children for recursive processing
    clustered_indices = set()   # Track which indices have been assigned to clusters

    # Iterate through children to identify clustered nodes and grand children
    for child_node in cluster_node.children:
        if child_node.child_num == 0:
            # Leaf node: nodes are directly in cluster
            clustered_indices.update(child_node.cluster)
        else:
            # Non-leaf node: collect indices from grand children
            for grand_child_node in child_node.children:
                clustered_indices.update(grand_child_node.cluster)
                if grand_child_node.child_num > 0:
                    # This grand child has further children, needs recursive processing
                    grand_children.append(grand_child_node)

    # Identify outliers (nodes not in any child cluster)
    outlier_indices = [index for index in cluster_node.cluster if index not in clustered_indices]
    outlier_nodes = [get_node_from_index(index) for index in outlier_indices]
    outlier_name = f'{cluster_node.index}-outlier'

    # If outliers exceed threshold, perform recursive clustering
    if cnt_cluster_sources(outlier_nodes) > max_nodes_in_cluster:
        cluster_raptor(outlier_name, outlier_nodes)
        # Create cluster node for outliers
        outlier_child_node = ClusterNode('local', outlier_indices, outlier_name)
        outlier_child_node.children = RaptorClusteringRecovery(args.layer, outlier_name).recover()
        outlier_child_node.child_num = len(outlier_child_node.children)
        grand_children.append(outlier_child_node)

    # Recursively process grand children
    for grand_child_node in grand_children:
        postprocess_raptor(grand_child_node)

# Create output directory
if not os.path.exists(f'data/layer{args.layer}/raptor'):
    os.mkdir(f'data/layer{args.layer}/raptor')

# Perform main clustering on all nodes
cluster_raptor('main', nodes)

# Build cluster tree from saved clustering results
root_node = ClusterNode(
    'local',
    [str(i) for i in range(len(nodes))] if args.layer > 1 else list(nodes_source_dict.keys()),
    'main'
)
root_node.children = RaptorClusteringRecovery(args.layer, 'main').recover()
root_node.child_num = len(root_node.children)

# Post-process to handle outliers recursively
postprocess_raptor(root_node)

# ==================== Run Reclustering ====================

# Import and run reclustering after clustering is complete
from recluster import run_recluster

run_recluster(args.layer)