"""
DBSCAN Clustering for Cluster Names

Functionality:
1. Load cluster names from multiple sources (graph nodes and LLM output)
2. Generate embeddings for all unique cluster names
3. Use DBSCAN to cluster similar names together
4. Select representative name for each cluster (shortest/most concise)
5. Output merged clusters for downstream processing

This script helps merge clusters with similar or duplicate names,
improving cluster quality by consolidating redundant labels.
"""

import argparse
import numpy as np
import json
import jsonlines
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from cluster_utils import Node, length

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='Layer to process')
parser.add_argument('--model', type=str, default='Qwen3-Embedding-4B', help='HuggingFace embedding model name')
args = parser.parse_args()

# Initialize embedding model for name similarity computation
Settings.embed_model = HuggingFaceEmbedding(
    model_name=args.model,
    max_length=32768,
    device='cuda:0'
)

# Collect all cluster names from multiple sources
cluster_names_dict = {}  # Maps lowercase name -> original name (preserves case)

if args.layer > 1:
    # Load cluster names from graph nodes (from previous layer processing)
    with open(f'data/layer{args.layer}/graph-nodes.jsonl', 'r') as fin:
        for line in fin:
            obj = json.loads(line)
            name = obj['name']
            name_lower = name.lower()
            if name_lower not in cluster_names_dict:
                cluster_names_dict[name_lower] = name

# Load cluster names from LLM clustering output
with open(f'data/layer{args.layer}/step-clusters-output.jsonl', 'r') as fin:
    for line in fin:
        obj = json.loads(line)
        for name in obj['clusters']:
            name_lower = name.lower()
            if name_lower not in cluster_names_dict:
                cluster_names_dict[name_lower] = name

# Build document list and generate embeddings
documents = [cluster_names_dict[name_lower] for name_lower in cluster_names_dict]
# Embedding generation: approximately 100 iterations per second
document_embeddings = Settings.embed_model.get_text_embedding_batch(documents, show_progress=True)

# Create Node objects for clustering
nodes = []
reps = []  # Representative info for each name
for i, name in enumerate(documents):
    nodes.append(Node(str(i), str(i), {'sbert': document_embeddings[i]}))
    reps.append({'name': name})

def cluster_dbscan(name: str, nodes: list[Node]):
    """
    Perform DBSCAN clustering on name embeddings.

    Uses cosine distance to identify similar cluster names.
    Names in the same DBSCAN cluster should be merged as they
    represent the same or very similar concepts.

    Args:
        name: Output file name prefix
        nodes: List of Node objects with name embeddings
    """
    # Extract embeddings and compute cosine distance matrix
    embeddings = np.array([node.embeddings['sbert'] for node in nodes])
    dist_matrix = cosine_distances(embeddings)

    # DBSCAN parameters:
    # - eps=0.05: Very small distance threshold (names must be very similar)
    # - min_samples=2: Minimum cluster size (loners are noise, label=-1)
    # - metric="precomputed": Use pre-computed distance matrix
    db = DBSCAN(eps=0.05, min_samples=2, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    # Group nodes by cluster label (exclude noise with label=-1)
    clusters = defaultdict(list)
    for node, label in zip(nodes, labels):
        if label != -1:
            clusters[label].append(node)

    # Write merged clusters to output file
    with open(f'data/layer{args.layer}/{name}-dbscan.jsonl', 'w') as f:
        writer = jsonlines.Writer(f)
        for label in clusters:
            # Select representative name: shortest/most concise name in cluster
            # This ensures merged clusters have clean, simple labels
            best_label = None

            for node in clusters[label]:
                name = reps[int(node.index)]['name']
                custom_len, str_len = length(name)

                # Tuple comparison: (word_count, char_count, name)
                # Smaller tuple = shorter/more concise name
                candidate = (custom_len, str_len, name)

                if best_label is None or candidate < best_label:
                    best_label = candidate

            # best_label is now (custom_len, str_len, name) with minimal values
            representative_name = best_label[2]

            # Build output object with representative label and cluster members
            obj = {
                'label': representative_name,
                'cluster': [
                    {'index': node.index, 'name': reps[int(node.index)]['name']}
                    for node in clusters[label]
                ]
            }
            writer.write(obj)
        writer.close()

# Perform DBSCAN clustering on all cluster names
cluster_dbscan('step-clusters', nodes)