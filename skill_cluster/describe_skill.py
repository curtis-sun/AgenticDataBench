"""
Skill Description Generation Script

Functionality:
1. Load skills from the graph structure
2. Sample example paths for each skill
3. Use LLM to generate detailed skill descriptions based on examples
4. Support multi-processing for parallel execution

Log Levels:
- WARNING: Result obtained but with quality concerns (not currently used)
- ERROR: No valid result obtained after all retries
"""

import argparse
import multiprocessing as mp
import logging
import time
import jsonlines
from math import floor

from sample_graph import PathSampler
from prompts import *
from log_utils import set_logger
from file_utils import read_idxes
import sys
sys.path.append('..')
from utils.llm_client import QwenClient

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, required=True, help='Output file path')
parser.add_argument('--log_file', type=str, default='describe_skill.log', required=False, help='Log file path')
parser.add_argument('--process_num', type=int, default=1, required=False, help='Number of parallel processes')
parser.add_argument('--debug', type=int, default=0, required=False, help='Debug mode: 1 for single document')
parser.add_argument('--model', type=str, default='qwen3-32b', help='LLM model to use')
parser.add_argument('--retry', type=int, default=4, help='Number of retry attempts for API calls')
args = parser.parse_args()

# Initialize LLM client
openai_client = QwenClient()

def main(process_id):
    """
    Main processing function for each worker process.

    Processes a subset of skills and generates descriptions using LLM.

    Args:
        process_id: ID of the current process (used to determine document slice)
    """
    # Determine the slice of documents for this process
    if process_id + 1 == process_num:
        cur_docs = docs[process_id*split_size:]
    else:
        cur_docs = docs[process_id*split_size: (process_id+1)*split_size]

    for doc in cur_docs:
        skill_id = doc['id']
        skill_name = doc['skill']

        # Format examples for LLM prompt
        examples = []
        for ex in doc['examples']:
            # Join all step texts in each example into a single string
            examples.append('\n'.join([step['text'] for step in ex['steps']]))
        examples_str = '\n\n'.join([f"### Example {i+1}\n{ex}" for i, ex in enumerate(examples)])

        # Build LLM prompt
        sys_prompt = DESCRIBE_SKILL_SYS_PROMPT
        user_prompt = DESCRIBE_SKILL_USER_PROMPT.format(skill=skill_name, examples=examples_str)
        messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_prompt}]

        # Retry loop for API calls
        for attempt in range(args.retry):
            start = time.perf_counter()
            try:
                _, response, usage = openai_client.generate(messages, model=args.model)
                end = time.perf_counter()
                elapsed = end - start

                # Write successful result to output file
                out_dict = {
                    'id': skill_id,
                    'skill': skill_name,
                    'examples': doc['examples'],
                    'description': response,
                    'usage': usage.model_dump(),
                    'time (s)': elapsed
                }
                out_file.write(out_dict)
                break

            except Exception as e:
                if attempt < args.retry - 1:
                    logging.warning(f"[DESCRIBE] skill_id={skill_id} skill=\"{skill_name}\" attempt={attempt+1}/{args.retry}: "
                                   f"API call failed - {type(e).__name__}: {e}, will retry")
                else:
                    logging.error(f"[DESCRIBE] skill_id={skill_id} skill=\"{skill_name}\": FAILED - "
                                 f"No valid description after {args.retry} attempts. Last error: {type(e).__name__}: {e}")

        # In debug mode, only process first document
        if args.debug == 1:
            break

# Global configuration
process_num = args.process_num
output_file = args.output_file
debug = args.debug

# Initialize path sampler for skill extraction
# Parameters:
# - alpha=1.0: Weight for path quality
# - beta=1.0: Weight for path diversity
# - random_prob=0.1: Probability of random sampling
sampler = PathSampler(
    alpha=1.0,
    beta=1.0,
    random_prob=0.1
)

# Load all skills from the graph structure
docs = [{'id': skill.id, 'skill': skill.name} for skill in sampler.get_skills()]

# Filter out already processed skills (resume capability)
finished_idxes = read_idxes(output_file)
docs = [obj for obj in docs if obj['id'] not in finished_idxes]

# Sample example paths for each skill
# sample_num=10: Number of example paths to sample per skill
for doc in docs:
    doc['examples'] = sampler.few_shot_for_skill(doc['id'], sample_num=10)

# Calculate split size for multi-processing
split_size = int(floor(len(docs) / process_num))

# Open output file in append mode with auto-flush
out_file = jsonlines.open(output_file, "a")
out_file._flush = True

# Configure logging
set_logger(args.log_file)

if __name__ == '__main__':
    if debug == 1:
        # Debug mode: run single process
        main(0)
    else:
        # Production mode: use multiprocessing pool
        with mp.Pool(processes=process_num) as pool:
            pool.map(main, range(process_num))