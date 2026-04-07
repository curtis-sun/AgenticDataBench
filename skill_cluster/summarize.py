"""
QA Summarization Script

Functionality:
1. Load QA documents from input file
2. Use LLM to summarize QA pairs into step-by-step solutions
3. Support multi-processing for parallel summarization
4. Track usage and timing metrics

Log Levels:
- WARNING: Result obtained but with quality concerns (unexpected format in steps)
- ERROR: No valid result obtained after all retries
"""

import jsonlines
from math import floor
import argparse
import multiprocessing as mp
import logging
import re
import time
import os

from prompts import *
from file_utils import prepare_qas
from log_utils import set_logger
import sys
sys.path.append('..')
from utils.llm_client import QwenClient

# Command line argument configuration
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help='Input file path')
parser.add_argument('--output_file', type=str, required=True, help='Output file path')
parser.add_argument('--log_file', type=str, default='summarize.log', required=False, help='Log file path')
parser.add_argument('--process_num', type=int, default=1, required=False, help='Number of parallel processes')
parser.add_argument('--debug', type=int, default=0, required=False, help='Debug mode: 1 for single document processing')
parser.add_argument('--model', type=str, default='qwen3-32b', help='LLM model to use')
parser.add_argument('--retry', type=int, default=4, help='Number of retry attempts for summarization')
args = parser.parse_args()

# Initialize LLM client
openai_client = QwenClient()

def summarize_qa(doc_id, title, question_body, answer_body):
    """
    Summarize a QA pair into step-by-step solution using LLM.

    Args:
        doc_id: Document identifier for logging (format: "{id}-{answer_id}")
        title: Question title
        question_body: Question content
        answer_body: Answer content

    Returns:
        steps: List of formatted step strings, or None if parsing fails
        response: Raw LLM response
        usage: Token usage information
    """
    sys_prompt = SUMMARIZE_QA_SYS_PROMPT
    user_prompt = STACKOVERFLOW_QA_PROMPT.format(title=title, question_body=question_body, answer_body=answer_body)

    messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_prompt}]
    _, response, usage = openai_client.generate(messages, model=args.model)

    # Parse response to extract steps
    prefix = '### Step-by-Step Solution:'
    prefix_idx = response.find(prefix)
    if prefix_idx == -1:
        logging.error(f"[SUMMARIZE] doc={doc_id}: No step-by-step solution prefix found in response. "
                     f"Response preview: {response[:200]}{'...' if len(response) > 200 else ''}")
        return None, response, usage

    # Extract steps after the prefix using regex
    # Pattern captures: bold title content, optional parenthetical suffix (e.g., "(optional)"), and description
    response_after_prefix = response[prefix_idx + len(prefix):].strip()
    step_pattern = r'\n?\d+\.\s+\*\*(.*?)\*\*(\s*\([^)]*\))?\s*:\s+(.*?)(?=\n\d+\.|\n?---|\n###|$)'
    matches = re.findall(step_pattern, response_after_prefix, re.DOTALL)

    # Format each step as "**Title**: Description"
    # Combine bold title with optional parenthetical suffix (e.g., "Rename Axes" + "(optional)")
    steps = [f"**{title.strip() + (parenthetical or '')}**: {desc.strip()}" for title, parenthetical, desc in matches]
    if not steps:
        logging.error(f"[SUMMARIZE] doc={doc_id}: No valid steps extracted from response. "
                     f"Response after prefix: {response_after_prefix[:200]}{'...' if len(response_after_prefix) > 200 else ''}")
        return None, response, usage

    # Check for malformed extraction: if any step contains lines starting with numbered patterns
    # (e.g., "9.", "10.", "11."), it indicates steps were merged incorrectly during extraction
    number_pattern = re.compile(r'^\d+\.')
    for i, step in enumerate(steps):
        for line in step.split('\n'):
            if number_pattern.match(line):
                logging.error(f"[SUMMARIZE] doc={doc_id}: Malformed step extraction - step {i+1} contains "
                             f"numbered line '{line[:50]}...', indicating steps were incorrectly merged. "
                             f"Total steps: {len(steps)}")
                return None, response, usage

    # Check for unexpected format in last step (double newline indicates potential parsing issue)
    if '\n\n' in steps[-1]:
        logging.warning(f"[SUMMARIZE] doc={doc_id}: Last step contains double newline (potential formatting issue). "
                       f"Last step preview: {steps[-1][:100]}{'...' if len(steps[-1]) > 100 else ''}")

    return steps, response, usage

def main(process_id):
    """
    Main processing function for each worker process.

    Args:
        process_id: ID of the current process (used to determine document slice)
    """
    # Determine the slice of documents for this process
    if process_id + 1 == process_num:
        # Last process handles remaining documents
        cur_docs = docs[process_id*split_size:]
    else:
        cur_docs = docs[process_id*split_size: (process_id+1)*split_size]

    # Process each document with retry logic
    for doc in cur_docs:
        doc_id = f"{doc['id']}-{doc['answer_id']}"

        for attempt in range(args.retry):
            start = time.perf_counter()
            try:
                steps, response, usage = summarize_qa(doc_id, doc['question_title'], doc['question_body'], doc['answer_body'])
                end = time.perf_counter()
                elapsed = end - start

                if steps is not None:
                    # Write successful result to output file
                    out_dict = {
                        'id': doc['id'],
                        'answer_id': doc['answer_id'],
                        'response': response,
                        'steps': steps,
                        'usage': usage.model_dump(),
                        'time (s)': elapsed
                    }
                    out_file.write(out_dict)
                    break

                # Log failed attempt
                if attempt < args.retry - 1:
                    logging.warning(f"[SUMMARIZE] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                                   f"Step extraction failed, will retry")

            except Exception as e:
                logging.error(f"[SUMMARIZE] doc={doc_id} attempt={attempt+1}/{args.retry}: "
                             f"Exception during summarization - {type(e).__name__}: {e}")

        # Log if all retries exhausted without success
        else:
            logging.error(f"[SUMMARIZE] doc={doc_id}: FAILED - No valid summarization after {args.retry} attempts")

        # In debug mode, only process first document
        if args.debug == 1:
            break

# Global configuration
process_num = args.process_num
debug = args.debug

# Load documents and filter those already processed in output file
output_file = args.output_file
log_file = args.log_file

# Create output directory if it doesn't exist
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load QA documents and prepare for processing
docs = prepare_qas(args.input_file, output_file)
split_size = int(floor(len(docs) / process_num))

# Open output file in append mode with auto-flush
out_file = jsonlines.open(output_file, "a")
out_file._flush = True

# Configure logging
set_logger(log_file)

if __name__ == '__main__':
    if debug == 1:
        # Debug mode: run single process
        main(0)
    else:
        # Production mode: use multiprocessing pool
        with mp.Pool(processes=process_num) as pool:
            pool.map(main, range(process_num))