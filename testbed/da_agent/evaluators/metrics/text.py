"""
Text Metrics Module - Text comparison metric for evaluation.

This module compares text output:
- Case-insensitive, whitespace-normalized comparison
- Provides detailed error messages for mismatches
- Returns score (0.0 or 1.0) and comparison details
"""

from typing import Dict, List, Union
import os
import logging

def compare_text(
    output_file_name: str,
    gold_file_name: str
) -> Dict[str, Union[float, List[str], str]]:
    """
    Evaluate LLM output by comparing text content.

    Args:
        output_file_name: Path to the output/predicted text file
        gold_file_name: Path to the gold/expected text file
        options: Configuration dictionary (currently unused, reserved for future features)
            - tolerance: Tolerance for numeric comparisons (if supported in future)
            - case_sensitive: Whether to perform case-sensitive comparison (default: False)

    Returns:
        dict: {'score': float, 'errors': list[str], 'meaning': str}
            - score: normalized score in range [0, 1]
            - errors: detailed error messages explaining mismatches
            - meaning: description of what the comparison did
    """
    # Load text files
    try:
        with open(output_file_name, 'r', encoding='utf-8') as f:
            output_text = f.read().strip()
    except (FileNotFoundError, UnicodeDecodeError) as e:
        logging.warning(f"Failed to read output file {output_file_name}: {e}")
        errors = [f"Failed to read output file '{os.path.basename(output_file_name)}': {e}"]
        meaning = f"Failed to compare text files: cannot read output file '{os.path.basename(output_file_name)}'"
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }

    try:
        with open(gold_file_name, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip()
    except (FileNotFoundError, UnicodeDecodeError) as e:
        logging.warning(f"Failed to read gold file {gold_file_name}: {e}")
        errors = [f"Failed to read gold file '{os.path.basename(gold_file_name)}': {e}"]
        meaning = f"Failed to compare text files: cannot read gold file '{os.path.basename(gold_file_name)}'"
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }

    # Case-insensitive, whitespace-normalized comparison
    output_normalized = ' '.join(output_text.lower().split())
    reference_normalized = ' '.join(reference_text.lower().split())

    errors = []

    # Save output_data and gold_data with truncation for long texts
    if len(output_text) < 1000:
        output_data = output_text
    else:
        output_data = output_text[:500] + f"\n... ({len(output_text)} characters total)"

    if len(reference_text) < 1000:
        gold_data = reference_text
    else:
        gold_data = reference_text[:500] + f"\n... ({len(reference_text)} characters total)"

    if output_normalized != reference_normalized:
        # Provide detailed error information
        if len(output_text) == 0:
            errors.append("Output is empty but expected text is not")
        elif len(reference_text) == 0:
            errors.append("Output contains text but expected empty")
        else:
            # Show first difference
            min_len = min(len(output_normalized), len(reference_normalized))
            diff_pos = None
            for i in range(min_len):
                if output_normalized[i] != reference_normalized[i]:
                    diff_pos = i
                    break

            if diff_pos is not None:
                # Show context around the difference
                start = max(0, diff_pos - 20)
                end = min(min_len, diff_pos + 20)
                context_output = output_normalized[start:end]
                context_ref = reference_normalized[start:end]
                errors.append(f"Text mismatch at position {diff_pos}: output has '{context_output}' but expected '{context_ref}'")
            else:
                # One is longer than the other
                if len(output_normalized) > len(reference_normalized):
                    extra = output_normalized[min_len:min_len+50]
                    errors.append(f"Output has extra text: '{extra}...' (length {len(output_text)} vs {len(reference_text)})")
                else:
                    missing = reference_normalized[min_len:min_len+50]
                    errors.append(f"Output is missing text: expected '{missing}...' (length {len(output_text)} vs {len(reference_text)})")
    else:
        # Perfect match
        meaning = f"Text comparison successful: output matches gold exactly after normalization (case-insensitive, whitespace-normalized). Both texts contain {len(output_text)} characters."
        return {'score': 1.0, 'errors': [], 'meaning': meaning, 'output_data': output_data, 'gold_data': gold_data}

    # Build meaning description
    meaning = f"Text comparison: case-insensitive, whitespace-normalized comparison between output ({len(output_text)} chars) and gold ({len(reference_text)} chars). Score: 0.0"

    return {'score': 0.0, 'errors': errors, 'meaning': meaning, 'output_data': output_data, 'gold_data': gold_data}
