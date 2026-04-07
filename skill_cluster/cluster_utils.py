"""
Clustering Utilities for RAPTOR Algorithm

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is a hierarchical
clustering approach that recursively clusters documents using Gaussian Mixture Models (GMM)
with UMAP dimensionality reduction.

Reference:
    RAPTOR algorithm adapted from: https://github.com/parthsarthi03/raptor
    Paper: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
    Authors: Parth Sarthi, Abdullah Salman, et al.

This module provides:
1. Node class for representing nodes in hierarchical tree structure
2. UMAP-based dimensionality reduction for embeddings
3. GMM-based clustering with automatic cluster number selection
4. RAPTOR_Clustering class for recursive hierarchical clustering
5. ClusterNode class for tree structure representation
6. RaptorClusteringRecovery class for reconstructing cluster tree from logs
"""

import logging
import os
import random
import re
import json
import jsonlines
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import umap
from sklearn.mixture import GaussianMixture

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


class Node:
    """
    Represents a node in the hierarchical tree structure.

    Attributes:
        index: Unique identifier for this node (e.g., "{id}-{answer_id}-{step_index}")
        post_index: Parent identifier (e.g., answer_id for steps)
        embeddings: Dictionary mapping model names to embedding vectors
    """

    def __init__(self, index: str, post_index: str, embeddings) -> None:
        self.index = index
        self.post_index = post_index
        self.embeddings = embeddings


def log_inner_cluster(cluster: List[Node], filename: str) -> None:
    """
    Log inner cluster nodes to file for tracking recursive clustering.

    Args:
        cluster: List of Node objects in the cluster
        filename: Log file path
    """
    obj = [node.index for node in cluster]
    with open(filename, 'a') as f:
        f.write(json.dumps(obj) + '\n')


def cnt_cluster_sources(cluster: List[Node]) -> int:
    """
    Count the number of unique sources (post_indices) in a cluster.

    This is used to enforce the max_nodes_in_cluster constraint at the source level,
    ensuring clusters don't exceed a certain number of unique documents.

    Args:
        cluster: List of Node objects

    Returns:
        Number of unique post_indices in the cluster
    """
    post_indices = set()
    for node in cluster:
        post_indices.add(node.post_index)
    return len(post_indices)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP for global clustering.

    Uses adaptive n_neighbors based on dataset size for better global structure preservation.

    Args:
        embeddings: Input embedding vectors (n_samples, n_features)
        dim: Target dimensionality after reduction
        n_neighbors: Number of neighbors for UMAP (auto-computed if None)
        metric: Distance metric for UMAP

    Returns:
        Reduced embeddings (n_samples, dim)
    """
    if n_neighbors is None:
        # Adaptive n_neighbors: sqrt(n-1) balances local/global structure
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP for local clustering.

    Uses fixed small n_neighbors for better local structure preservation.

    Args:
        embeddings: Input embedding vectors (n_samples, n_features)
        dim: Target dimensionality after reduction
        num_neighbors: Number of neighbors for UMAP (default 10)
        metric: Distance metric for UMAP

    Returns:
        Reduced embeddings (n_samples, dim)
    """
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine optimal number of clusters using BIC (Bayesian Information Criterion).

    Fits GMM models with different cluster counts and selects the one with lowest BIC.

    Args:
        embeddings: Input embedding vectors
        max_clusters: Maximum number of clusters to test
        random_state: Random seed for reproducibility

    Returns:
        Optimal number of clusters
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    valid_n_clusters = []

    # Try each cluster count and record BIC score
    for n in n_clusters:
        try:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            valid_n_clusters.append(n)
        except ValueError as e:
            # Skip invalid configurations
            continue

    if not valid_n_clusters:
        raise RuntimeError("No valid GMM fits found. Try reducing max_clusters.")

    # Select cluster count with minimum BIC
    optimal_clusters = valid_n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Perform soft clustering using Gaussian Mixture Model.

    Uses probabilistic assignment: nodes can belong to multiple clusters
    if their probability exceeds the threshold.

    Args:
        embeddings: Input embedding vectors
        threshold: Probability threshold for cluster assignment
        random_state: Random seed for reproducibility

    Returns:
        labels: List of cluster assignments for each node (soft clustering)
        n_clusters: Number of clusters used
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    # Soft clustering: each node can belong to multiple clusters
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    nodes: List[Node], embeddings: np.ndarray, log_filename: str, max_nodes_in_cluster: int, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    """
    Perform two-level RAPTOR clustering: global then local.

    Process:
    1. Global clustering: UMAP reduction + GMM on all embeddings
    2. Local clustering: For each global cluster, apply UMAP + GMM again
       if cluster size exceeds max_nodes_in_cluster

    Args:
        nodes: List of Node objects to cluster
        embeddings: Embedding vectors for nodes
        log_filename: File to log inner clusters for recursive processing
        max_nodes_in_cluster: Maximum unique sources allowed per cluster
        dim: Dimensionality reduction target
        threshold: Probability threshold for soft clustering
        verbose: Whether to print progress logs

    Returns:
        all_local_clusters: Cluster assignments for each node
    """
    # Step 1: Global clustering with UMAP reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) - 2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    # Initialize cluster assignments
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Step 2: Local clustering for each global cluster
    for i in range(n_global_clusters):
        # Extract embeddings belonging to global cluster i
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue

        # Get indices and nodes for this global cluster
        global_cluster_indices = [idx for idx, gc in enumerate(global_clusters) if i in gc]
        global_cluster_nodes_ = [nodes[idx] for idx in global_cluster_indices]

        # Check if local clustering needed (based on unique sources)
        if cnt_cluster_sources(global_cluster_nodes_) <= max_nodes_in_cluster:
            # Cluster is small enough, no further subdivision needed
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Cluster is too large, perform local clustering
            log_inner_cluster(global_cluster_nodes_, log_filename)
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        # Assign local cluster labels to nodes
        for j in range(n_local_clusters):
            local_cluster_indices = [
                global_cluster_indices[idx] for idx, lc in enumerate(local_clusters) if j in lc
            ]
            for idx in local_cluster_indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms.
    """
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    """
    RAPTOR clustering algorithm for hierarchical document clustering.

    Implements recursive clustering:
    1. Initial clustering of all nodes
    2. For each resulting cluster, check if it exceeds max_nodes_in_cluster
    3. If exceeded, recursively cluster that subset
    """

    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        log_filename: str,
        max_nodes_in_cluster: int = 30,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        """
        Perform RAPTOR hierarchical clustering.

        Args:
            nodes: List of Node objects to cluster
            embedding_model_name: Key to access embeddings from node.embeddings
            log_filename: File to log clusters for recovery
            max_nodes_in_cluster: Maximum unique sources per cluster
            reduction_dimension: UMAP reduction dimension
            threshold: GMM probability threshold
            verbose: Print progress logs

        Returns:
            List of clusters, each cluster is a list of Node objects
        """
        # Extract embeddings from nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # Perform initial clustering
        clusters = perform_clustering(
            nodes, embeddings, log_filename=log_filename, max_nodes_in_cluster=max_nodes_in_cluster, dim=reduction_dimension, threshold=threshold, verbose=verbose
        )

        # Convert cluster labels to node groups
        node_clusters = []
        pre_label = -1
        for label in np.unique(np.concatenate(clusters)):
            if verbose:
                # Log empty clusters for completeness
                for l in range(pre_label + 1, int(label)):
                    logging.info(f"Nodes in Local Cluster {l}: 0")
            pre_label = int(label)

            # Get nodes belonging to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: single node cluster, no recursion needed
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Recursive case: cluster too large, re-cluster
            if cnt_cluster_sources(cluster_nodes) > max_nodes_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster {int(label)} with {len(cluster_nodes)} nodes"
                    )
                log_inner_cluster(cluster_nodes, log_filename)
                # Recursive clustering
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes,
                        embedding_model_name,
                        log_filename=log_filename,
                        max_nodes_in_cluster=max_nodes_in_cluster,
                        reduction_dimension=reduction_dimension,
                        threshold=threshold,
                        verbose=verbose
                    )
                )
            else:
                # Cluster is acceptable size
                node_clusters.append(cluster_nodes)

        if verbose:
            logging.info(f"Finalize Clustering")
        return node_clusters


class ClusterNode:
    """
    Represents a node in the cluster tree structure.

    Used for storing and navigating the hierarchical cluster results.

    Attributes:
        type: 'global' or 'local' cluster type
        index: Cluster identifier
        num: Number of nodes (when initialized with int)
        cluster: List of node indices (when initialized with list)
        parent: Parent ClusterNode reference
        child_num: Number of children
        children: List of child ClusterNode objects
    """

    def __init__(self, type: str, arg, index=None):
        self.type = type
        self.index = index

        # Initialize from count or from list
        if isinstance(arg, int):
            self.num = arg
            self.cluster = None
        else:
            assert isinstance(arg, list)
            self.cluster = arg
            self.num = len(arg)

        self.parent = None
        self.child_num = 0
        self.children = []

    def remove_empty_children(self):
        """Remove children with zero nodes and update child count."""
        self.children = [child for child in self.children if child.num > 0]
        self.child_num = len(self.children)


class RaptorClusteringRecovery:
    """
    Recover cluster tree structure from RAPTOR log files.

    Parses log files and cluster files to reconstruct the hierarchical
    cluster tree, enabling inspection and manipulation of results.

    Attributes:
        layer: Layer number for file paths
        name: Cluster run name (e.g., 'main', 'main-outlier')
        clusters: Leaf cluster assignments from -leaf.jsonl
        inner_clusters: Inner cluster assignments from -inner.jsonl
        lines: Log file lines for parsing
    """

    def __init__(self, layer, name):
        self.layer = layer
        self.name = name

        # Log parsing prefixes
        self.prefix0 = '- root - INFO: - '
        self.prefix1 = 'Global Clusters: '
        self.prefix2 = 'Nodes in Global Cluster'
        self.prefix3 = 'Local Clusters in Global Cluster'
        self.prefix4 = 'Total Clusters: '
        self.prefix5 = r'reclustering cluster (\d+) with (\d+) nodes'
        self.prefix6 = 'Nodes in Local Cluster'
        self.prefix7 = 'Finalize Clustering'

        # Load leaf cluster assignments
        self.clusters_front = 0
        self.clusters = []
        with open(f'data/layer{layer}/raptor/{name}-leaf.jsonl', 'r') as fin:
            reader = jsonlines.Reader(fin)
            for obj in reader:
                self.clusters.append(obj)  # Sources or representative IDs

        # Load inner cluster assignments (for recursive clusters)
        self.inner_clusters_front = 0
        self.inner_clusters = []
        if os.path.exists(f'data/layer{layer}/raptor/{name}-inner.jsonl'):
            with open(f'data/layer{layer}/raptor/{name}-inner.jsonl', 'r') as fin:
                reader = jsonlines.Reader(fin)
                for obj in reader:
                    self.inner_clusters.append(obj)  # Sources or representative IDs

        # Load log file for parsing
        self.front = 0
        with open(f'data/layer{layer}/raptor/{name}.log', 'r') as fin:
            self.lines = list(fin.readlines())

    def _next_cluster(self):
        """Get next leaf cluster and advance pointer."""
        self.clusters_front += 1
        return self.clusters[self.clusters_front - 1], f'{self.name}-leaf-{self.clusters_front - 1}'

    def _next_inner_cluster(self):
        """Get next inner cluster and advance pointer."""
        self.inner_clusters_front += 1
        return self.inner_clusters[self.inner_clusters_front - 1], f'{self.name}-inner-{self.inner_clusters_front - 1}'

    def _next(self):
        """Get next log line and advance pointer."""
        self.front += 1
        line = self.lines[self.front - 1]
        return line[line.index(self.prefix0) + len(self.prefix0):]

    def _check(self, node):
        """
        Verify cluster node integrity.

        Checks that child counts match actual children and indices are set.
        """
        if len(node.children) != node.child_num or len(node.cluster) != node.num or node.index is None:
            return False
        for child_node in node.children:
            if not self._check(child_node):
                return False
        return True

    def _recover(self):
        """
        Recursively recover cluster tree from logs.

        Parses log structure to reconstruct global -> local cluster hierarchy,
        handling recursive reclustering cases.

        Returns:
            List of ClusterNode objects representing the recovered tree
        """
        # Parse global cluster count
        content = self._next()
        assert content.startswith(self.prefix1)
        global_num = int(content[content.index(self.prefix1) + len(self.prefix1):])
        cur_nodes = []

        # Parse each global cluster
        for _ in range(global_num):
            content = self._next()
            assert content.startswith(self.prefix2)
            _, num = [int(x) for x in content[content.index(self.prefix2) + len(self.prefix2):].split(':')]
            if num == 0:
                continue
            cur_node = ClusterNode('global', num)
            cur_nodes.append(cur_node)

            # Parse local cluster count for this global cluster
            content = self._next()
            assert content.startswith(self.prefix3)
            _, local_num = [int(x) for x in content[content.index(self.prefix3) + len(self.prefix3):].split(':')]
            if local_num > 1:
                # Multiple local clusters: load inner cluster data
                cur_node.cluster, cur_node.index = self._next_inner_cluster()
                assert len(cur_node.cluster) == cur_node.num
                cur_node.child_num = local_num

        # Parse total cluster count
        content = self._next()
        assert content.startswith(self.prefix4)
        total_num = int(content[content.index(self.prefix4) + len(self.prefix4):])
        assert sum([cur_node.child_num if cur_node.cluster is not None else 1 for cur_node in cur_nodes]) == total_num

        # Parse cluster details and handle reclustering
        pre_child_id = 0
        global_cluster_id = 0
        while True:
            content = self._next()
            if content.startswith(self.prefix7):
                # End of clustering
                cur_child_id = total_num
                num = None
            elif content.startswith(self.prefix6):
                # Regular local cluster
                cur_child_id, num = [int(x) for x in content[content.index(self.prefix6) + len(self.prefix6):].split(':')]
            else:
                # Reclustering case
                match = re.match(self.prefix5, content)
                assert match
                cur_child_id = int(match.group(1))
                num = int(match.group(2))

            # Fill in missing cluster nodes
            while pre_child_id < cur_child_id:
                cur_node = cur_nodes[global_cluster_id]
                if cur_node.child_num == 0:
                    # Leaf cluster
                    cur_node.cluster, cur_node.index = self._next_cluster()
                    cur_node.type = 'local'
                    global_cluster_id += 1
                else:
                    # Non-leaf cluster: add child
                    cur_local_node = ClusterNode('local', *self._next_cluster())
                    cur_node.children.append(cur_local_node)
                    cur_local_node.parent = cur_node
                    if len(cur_node.children) == cur_node.child_num:
                        global_cluster_id += 1
                pre_child_id += 1

            if num is None:
                break

            # Handle reclustering case
            if num == 0:
                cur_local_node = ClusterNode('local', [])
            else:
                cur_local_node = ClusterNode('local', *self._next_inner_cluster())
                assert cur_local_node.num == num
                # Recursively recover nested clusters
                cur_local_node.children = self._recover()
                cur_local_node.child_num = len(cur_local_node.children)
                for n in cur_local_node.children:
                    n.parent = cur_local_node

            assert cur_nodes[global_cluster_id].child_num > 0
            cur_nodes[global_cluster_id].children.append(cur_local_node)
            cur_local_node.parent = cur_nodes[global_cluster_id]
            if len(cur_nodes[global_cluster_id].children) == cur_nodes[global_cluster_id].child_num:
                global_cluster_id += 1
            pre_child_id += 1

        # Clean up empty children
        for cur_node in cur_nodes:
            cur_node.remove_empty_children()
        return cur_nodes

    def recover(self):
        """
        Recover complete cluster tree and verify integrity.

        Returns:
            List of ClusterNode objects representing the cluster tree
        """
        nodes = self._recover()
        # Verify all nodes are properly constructed
        for node in nodes:
            assert self._check(node)
        # Verify all data consumed
        assert self.clusters_front == len(self.clusters) and self.front == len(self.lines) and self.inner_clusters_front == len(self.inner_clusters)
        return nodes


def length(s: str) -> tuple[int, int]:
    """
    Compute string length metrics.

    Args:
        s: Input string

    Returns:
        Tuple of (word segment count, character count)
    """
    return (len([seg for seg in re.split(r'[\s\-_/&]+', s) if seg]), len(s))