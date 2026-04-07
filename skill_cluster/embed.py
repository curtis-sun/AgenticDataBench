"""
Skill Clustering Embedding Generation Script

Functionality:
1. Load skill step data
2. Generate text embeddings using HuggingFace model
3. Select representative steps for clusters (using medoid algorithm)
"""

import json
import numpy as np
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import os
import re
import argparse

from file_utils import STEP_PATTERN

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='Layer to process (1 or higher)')
parser.add_argument('--embed', action='store_true', help='Whether to generate embeddings')
parser.add_argument('--model', type=str, default='Qwen3-Embedding-4B', help='HuggingFace embedding model name')
args = parser.parse_args()

# Initialize embedding model if --embed flag is set
if args.embed:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.model,  # Use Qwen3 embedding model
        max_length=32768,                  # Maximum sequence length
        device='cuda:0'                   # Use GPU 0
    )

documents = []

if args.layer == 1:
    # Layer 1: Load data directly from steps file
    with open(f'data/steps.jsonl', 'r') as fin:
        for line in fin.readlines():
            obj = json.loads(line)
            documents.extend(obj['steps'])  # Collect all step texts

    # Generate and save embeddings for layer 1 steps
    if args.embed:
        document_embeddings = Settings.embed_model.get_text_embedding_batch(documents, show_progress=True)
        np.save(f'data/steps-embed.npy', document_embeddings)
else:
    # Higher layers: Process cluster representatives
    if os.path.exists(f'data/layer{args.layer}/step-clusters-representatives.jsonl'):
        # If representatives file exists, load directly
        with open(f'data/layer{args.layer}/step-clusters-representatives.jsonl', 'r') as fin:
            for line in fin.readlines():
                obj = json.loads(line)
                documents.append(obj['text'])
    else:
        # Otherwise, build cluster representatives from layer 1
        # Load pre-computed layer 1 embeddings
        pre_embeddings = np.load(f'data/steps-embed.npy')

        # Build steps dictionary: associate step text with embeddings
        i = 0
        steps = {}
        with open(f'data/steps.jsonl', 'r') as fin:
            for line in fin.readlines():
                obj = json.loads(line)
                for j, step in enumerate(obj['steps']):
                    # source format: {id}-{answer_id}-{step_index}
                    source = f"{obj['id']}-{obj['answer_id']}-{j}"
                    steps[source] = {'text': step, 'embed': pre_embeddings[i]}
                    i += 1
        assert len(pre_embeddings) == i  # Ensure embedding count matches step count

        # Select representative step for each cluster
        representatives = []
        with open(f'data/layer{args.layer}/step-clusters.jsonl', 'r') as fin:
            for id, line in tqdm(enumerate(fin.readlines())):
                obj = json.loads(line)
                name = obj['name']      # Cluster name
                cluster = obj['cluster']  # List of step sources in this cluster

                cur_steps = []
                cur_steps_dict = {}

                # Collect all step information in the cluster
                for source in cluster:
                    step = steps[source]
                    cur_steps.append({'embed': step['embed'], 'source': source})
                    step_text = step['text']

                    # Parse source to get id and step number
                    num = source.split('-')[-1]
                    id = source[:source.rindex('-')]

                    if id not in cur_steps_dict:
                        cur_steps_dict[id] = {}

                    # Use regex to extract step body content
                    match = re.search(STEP_PATTERN, step_text, re.DOTALL)
                    assert match, f"Pattern not matched in step text: {step_text}"
                    step_body = match.group(2).strip()
                    cur_steps_dict[id][int(num)] = step_body

                # Use medoid algorithm to select representative step
                # medoid is the point with minimum total distance to all other points in cluster
                cur_embeds = np.array([step['embed'] for step in cur_steps])
                dist_matrix = cosine_distances(cur_embeds)  # Compute cosine distance matrix
                total_dist = dist_matrix.sum(axis=1)        # Compute total distance from each point to all others
                medoid_idx = total_dist.argmin()            # Find medoid index

                # Build representative text
                rep_source = cur_steps[medoid_idx]['source']
                rep_id = rep_source[:rep_source.rindex('-')]
                # Representative text = cluster name + all steps from that answer
                rep_text = f'**{name}**: ' + '\n'.join(cur_steps_dict[rep_id][i] for i in sorted(cur_steps_dict[rep_id].keys()))
                rep_sources = [f"{rep_id}-{i}" for i in sorted(cur_steps_dict[id].keys())]
                representatives.append({'text': rep_text, 'source': rep_sources})
                documents.append(rep_text)

        # Save cluster representatives to file
        with open(f'data/layer{args.layer}/step-clusters-representatives.jsonl', 'w') as fout:
            for rep in representatives:
                fout.write(json.dumps(rep) + '\n')

    # Generate and save embeddings for current layer
    if args.embed:
        document_embeddings = Settings.embed_model.get_text_embedding_batch(documents, show_progress=True)
        np.save(f'data/layer{args.layer}/step-clusters-embed.npy', document_embeddings)