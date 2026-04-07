"""
Cluster Generation Script using LLM

Functionality:
1. Extract skills from step texts using LLM (layer 1)
2. Cluster steps into skills based on extracted skill list (layer 1)
3. Perform unsupervised clustering for higher layers
4. Support multi-processing for parallel execution

Log Levels:
- WARNING: Result obtained but with quality concerns (partial coverage, duplicate skills, etc.)
- ERROR: No valid result obtained after all retries
"""

import jsonlines
from collections import defaultdict
from math import floor
import argparse
import multiprocessing as mp
import logging
import re
import sys

from prompts import *
from log_utils import set_logger
from file_utils import prepare_clusters
sys.path.append('..')
from utils.llm_client import QwenClient
from utils.llm_utils import extract_code_from_text

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='Layer to process')
parser.add_argument('--process_num', type=int, default=1, required=False, help='Number of parallel processes')
parser.add_argument('--debug', type=int, default=0, required=False, help='Debug mode: 1 for single document')
parser.add_argument('--model', type=str, default='qwen3-32b', help='LLM model to use')
parser.add_argument('--retry', type=int, default=4, help='Number of retry attempts for clustering')
args = parser.parse_args()

# Initialize LLM client
openai_client = QwenClient()

def extract_cluster(doc_id, steps):
    """
    Extract skill names from step texts using LLM.

    The LLM identifies distinct skills/techniques mentioned across the steps,
    returning a list of skill names for subsequent clustering.

    Args:
        doc_id: Document identifier for logging
        steps: List of step dictionaries containing 'text' field

    Returns:
        skills: List of extracted skill names, or None if extraction fails
        response: Raw LLM response text
        usage: Token usage information
    """
    # Format steps for LLM prompt
    steps_str = '\n\n'.join([f'Step {i + 1}:\n"""{s["text"]}"""' for i, s in enumerate(steps)])
    messages = [
        {'role': 'system', 'content': EXTRACT_SYS_PROMPT},
        {'role': 'user', 'content': EXTRACT_PROMPT.format(steps=steps_str)}
    ]

    skills_log = []
    # Retry up to args.retry times for extraction
    for attempt in range(max(args.retry // 2, 1)):
        try:
            _, response, usage = openai_client.generate(messages, model=args.model)
            # Extract Python code block containing skill list
            list_str = extract_code_from_text(response, ('```python', '```'))
            skills = eval(list_str)

            # Validate: skills should be unique and fewer than total steps
            if len(set(skills)) == len(skills) and len(skills) < len(steps):
                return skills, response, usage

            # Validation failed: log the issue
            if len(set(skills)) != len(skills):
                logging.warning(f"[EXTRACT] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                               f"Extracted {len(skills)} skills but {len(skills) - len(set(skills))} are duplicates. "
                               f"Skills: {skills[:5]}{'...' if len(skills) > 5 else ''}")
            elif len(skills) >= len(steps):
                logging.warning(f"[EXTRACT] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                               f"Extracted {len(skills)} skills >= {len(steps)} total steps (too many). "
                               f"Skills: {skills[:5]}{'...' if len(skills) > 5 else ''}")

            # Save for potential fallback
            skills_log.append({'skills': skills, 'response': response, 'usage': usage})

        except Exception as e:
            logging.error(f"[EXTRACT] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                         f"Failed to parse LLM response - {type(e).__name__}: {e}")

    # Fallback: return best attempt based on skill count
    if len(skills_log) == 0:
        logging.error(f"[EXTRACT] doc={doc_id}: FAILED - No valid skill extraction after {args.retry} attempts")
        return None, None, None

    log = sorted(skills_log, key=lambda x: len(x['skills']))[0]
    logging.warning(f"[EXTRACT] doc={doc_id}: Using fallback - extracted {len(log['skills'])} skills "
                   f"(validation failed but partial result available)")
    return log['skills'], log['response'], log['usage']

def recluster(doc_id, steps, skills):
    """
    Cluster steps into the provided skill categories using LLM.

    Given a list of skills extracted from steps, the LLM assigns each step
    to one or more skills, creating a mapping from skill name to step sources.

    Args:
        doc_id: Document identifier for logging
        steps: List of step dictionaries with 'text' and 'source' fields
        skills: List of skill names to cluster steps into

    Returns:
        clusters: Dictionary mapping skill name to list of source IDs
        response: LLM response with reasoning and clustering
        usage: Token usage information
    """
    # Format steps and skills for LLM prompt
    steps_str = '\n\n'.join([f'Step {i + 1}:\n"""{s["text"]}"""' for i, s in enumerate(steps)])
    skills_str = '\n\n'.join([f'Skill {i + 1}:\n"""{s}"""' for i, s in enumerate(skills)])
    messages = [
        {'role': 'system', 'content': CLUSTER_SYS_PROMPT},
        {'role': 'user', 'content': CLUSTER_PROMPT.format(steps=steps_str, skills=skills_str)}
    ]

    clusters_log = []
    # Retry up to args.retry times for clustering
    for attempt in range(args.retry):
        try:
            # Enable thinking mode for better reasoning
            reasoning, response, usage = openai_client.generate(messages, model=args.model, enable_thinking=True)
            # Extract Python code block containing cluster dictionary
            dict_str = extract_code_from_text(response, ('```python', '```'))
            raw_clusters = eval(dict_str)

            # Track which steps have been assigned
            clustered_skill_indices = set()
            for indices in raw_clusters.values():
                clustered_skill_indices.update(indices)

            # Validate: step indices must be in valid range [1, len(steps)]
            if min(clustered_skill_indices) >= 1 and max(clustered_skill_indices) <= len(steps):
                clusters = defaultdict(list)
                failed_skill_flag = False

                # Process each skill cluster
                for raw_skill, indices in raw_clusters.items():
                    # Parse skill reference (e.g., "Skill 1" -> index 1)
                    skill_match = re.match(r'Skill\s*(\d+)', raw_skill, re.IGNORECASE)
                    if skill_match:
                        skill_idx = int(skill_match.group(1))
                        if skill_idx <= 0 or skill_idx > len(skills):
                            failed_skill_flag = True
                            logging.warning(f"[CLUSTER] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                                          f"Invalid skill reference '{raw_skill}' (valid range: 1-{len(skills)})")
                            break
                        skill = skills[skill_idx - 1]
                    else:
                        # Use raw skill name if not a reference
                        skill = raw_skill

                    # Collect source IDs for steps in this skill cluster
                    for i in indices:
                        clusters[skill].extend(steps[i - 1]['source'])

                if not failed_skill_flag:
                    coverage = len(clustered_skill_indices)
                    total = len(steps)

                    # Perfect coverage: all steps assigned
                    if coverage == total:
                        return clusters, reasoning + '\n</think>\n\n' + response, usage

                    # Partial coverage: save for potential fallback
                    clusters_log.append({
                        'clusters': clusters,
                        'reasoning': reasoning,
                        'response': response,
                        'usage': usage,
                        'coverage': coverage
                    })
                    logging.warning(f"[CLUSTER] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                                   f"Partial coverage {coverage}/{total} steps ({coverage*100//total}%). "
                                   f"Missing steps: {set(range(1, total+1)) - clustered_skill_indices}")
            else:
                min_idx = min(clustered_skill_indices) if clustered_skill_indices else 'N/A'
                max_idx = max(clustered_skill_indices) if clustered_skill_indices else 'N/A'
                logging.warning(f"[CLUSTER] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                               f"Invalid step indices: range [{min_idx}, {max_idx}], valid range [1, {len(steps)}]")

        except Exception as e:
            logging.error(f"[CLUSTER] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                         f"Failed to parse LLM response - {type(e).__name__}: {e}")

    # Fallback: return best attempt based on coverage
    if len(clusters_log) > 0:
        log = sorted(clusters_log, key=lambda x: x['coverage'], reverse=True)[0]
        coverage = log['coverage']
        total = len(steps)
        if coverage != total:
            logging.warning(f"[CLUSTER] doc={doc_id}: Using fallback - {coverage}/{total} steps covered "
                           f"({coverage*100//total}% coverage)")
        return log['clusters'], log['reasoning'] + '\n</think>\n\n' + log['response'], log['usage']

    logging.error(f"[CLUSTER] doc={doc_id}: FAILED - No valid clustering after {args.retry} attempts "
                 f"for {len(steps)} steps into {len(skills)} skills")
    return None, None, None

def recluster_unsupervised(doc_id, steps):
    """
    Cluster steps into skills without pre-defined skill list (higher layers).

    Used for layer > 1 where steps are already grouped clusters.
    The LLM directly identifies skill patterns and assigns step clusters.

    Args:
        doc_id: Document identifier for logging
        steps: List of step cluster dictionaries with 'text' and 'children' fields

    Returns:
        clusters: Dictionary mapping skill name to list of source IDs
        response: Raw LLM response
        usage: Token usage information
    """
    # Format steps as skills for unsupervised prompt
    steps_str = '\n\n'.join([f'Skill {i + 1}:\n{s["text"]}' for i, s in enumerate(steps)])
    messages = [
        {'role': 'system', 'content': CLUSTER_UNSUPERVISED_SYS_PROMPT},
        {'role': 'user', 'content': CLUSTER_UNSUPERVISED_PROMPT.format(steps=steps_str)}
    ]

    for attempt in range(args.retry):
        try:
            _, response, usage = openai_client.generate(messages, model=args.model)
            dict_str = extract_code_from_text(response, ('```python', '```'))
            raw_clusters = eval(dict_str)

            # Track assigned step indices
            clustered_skill_indices = set()
            for indices in raw_clusters.values():
                clustered_skill_indices.update(indices)

            # Validate: all steps assigned, unique skill names, valid indices
            skill_names = list(raw_clusters.keys())
            unique_skills = len(set(skill_names)) == len(skill_names)
            all_covered = len(clustered_skill_indices) == len(steps)
            valid_range = min(clustered_skill_indices) == 1 and max(clustered_skill_indices) == len(steps)

            if unique_skills and all_covered and valid_range:
                clusters = {}
                for skill, indices in raw_clusters.items():
                    if len(indices) > 0:
                        clusters[skill] = []
                        # Collect child source IDs from each step cluster
                        for i in indices:
                            clusters[skill].extend(steps[i - 1]['children'])
                return clusters, response, usage

            # Log specific validation failures
            issues = []
            if not unique_skills:
                duplicates = [name for name in skill_names if skill_names.count(name) > 1]
                issues.append(f"duplicate skill names: {set(duplicates)}")
            if not all_covered:
                missing = set(range(1, len(steps)+1)) - clustered_skill_indices
                issues.append(f"missing steps: {missing if len(missing) <= 5 else str(list(missing)[:5]) + '...'}")
            if not valid_range:
                issues.append(f"invalid indices: [{min(clustered_skill_indices)}, {max(clustered_skill_indices)}] "
                            f"(expected [1, {len(steps)}])")

            logging.warning(f"[UNSUPERVISED] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                           f"Validation failed - {'; '.join(issues)}")

        except Exception as e:
            logging.error(f"[UNSUPERVISED] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                         f"Failed to parse LLM response - {type(e).__name__}: {e}")

    logging.error(f"[UNSUPERVISED] doc={doc_id}: FAILED - No valid clustering after {args.retry} attempts "
                 f"for {len(steps)} step clusters")
    return None, None, None

def main(process_id):
    """
    Main processing function for each worker process.

    Args:
        process_id: ID of the current process (used to determine document slice)
    """
    # Determine the slice of documents for this process
    if process_id + 1 == process_num:
        cur_docs = docs[process_id*split_size:]
    else:
        cur_docs = docs[process_id*split_size: (process_id+1)*split_size]

    for doc in cur_docs:
        steps = doc['cluster']

        if args.layer == 1:
            # Layer 1: Two-stage process (extract skills then cluster)
            skills, extract_response, extract_usage = extract_cluster(doc['id'], steps)
            if skills is None:
                continue
            clusters, cluster_response, cluster_usage = recluster(doc['id'], steps, skills)
            if clusters is None:
                continue
            out_dict = {
                'id': doc['id'],
                'skills': skills,
                'extract_response': extract_response,
                'extract_usage': extract_usage.model_dump(),
                'clusters': clusters,
                'cluster_response': cluster_response,
                'cluster_usage': cluster_usage.model_dump()
            }
            out_file.write(out_dict)
        else:
            # Higher layers: Direct unsupervised clustering
            clusters, cluster_response, usage = recluster_unsupervised(doc['id'], steps)
            if clusters is None:
                continue
            out_dict = {
                'id': doc['id'],
                'clusters': clusters,
                'response': cluster_response,
                'usage': usage.model_dump()
            }
            out_file.write(out_dict)

        # Debug mode: only process first document
        if debug == 1:
            break

# Global configuration
process_num = args.process_num
input_file = f'data/layer{args.layer}/step-clusters-raptor-recluster.jsonl'
output_file = f'data/layer{args.layer}/step-clusters-output.jsonl'
debug = args.debug

# Load documents and filter those already processed
docs = prepare_clusters(input_file, output_file)
split_size = int(floor(len(docs) / process_num))

# Open output file in append mode with auto-flush
out_file = jsonlines.open(output_file, "a")
out_file._flush = True

# Configure logging
set_logger(f'data/layer{args.layer}/cluster.log')

if __name__ == '__main__':
    if debug == 1:
        # Debug mode: run single process
        main(0)
    else:
        # Production mode: use multiprocessing pool
        with mp.Pool(processes=process_num) as pool:
            pool.map(main, range(process_num))