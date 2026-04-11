"""
ML Metrics Module - Machine learning evaluation metrics.

This module provides ML-specific evaluation:
- Model performance metrics (accuracy, F1, etc.)
- Prediction comparison with tolerance
- Handles sklearn-style outputs

Reference: https://github.com/yiyihum/da-code/tree/main/da_agent/evaluators/metrics/ml.py
"""

from typing import Optional, Union, Callable, Dict, Any
import logging
import numpy as np
import pandas as pd
from .script.ml_script import CalculateML

TYPES = ['binary classification', 'multi classification', 'cluster', 'regression']
LOWER_METRICS = ["logloss_class", "logloss_total", "rmsle", "mae", "mse", "smape", "medae", "crps", "ks"]

def compare_model(
    output_file_name: str,
    gold_file_name: str,
    matched_columns: Optional[Union[list[str], list[int]]] = None,
    calculate_columns: Optional[Union[list[str], list[int]]] = None,
    metric: Union[str, Callable] = 'accuracy',
    lower_bound: float = 0.0,
    upper_bound: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate model performance with flexible table matching.

    Args:
        output_file_name: Path to the output/predicted CSV file
        gold_file_name: Path to the gold/expected CSV file
        matched_columns: Column names (list[str]) or indices (list[int], 0-based) for row alignment.
            If None, aligns by row order (requires equal length).
        calculate_columns: Column names (list[str]) or indices (list[int], 0-based) for metric calculation.
            If None, uses all columns. Multiple columns will be flattened and concatenated.
        metric: Metric name (str) or custom function (Callable[[pd.DataFrame, pd.DataFrame], float])
        lower_bound: Lower bound for score normalization
        upper_bound: Upper bound for score normalization

    Returns:
        dict: {'score': float, 'errors': list[str]}
            - score: normalized score in range [0, 1]
            - errors: detailed error messages

    Note:
        - For "lower is better" metrics (e.g., mse), set lower_bound > upper_bound
        - Score normalization: score = clamp((value - lower) / (upper - lower), 0, 1)
    """
    errors = []
    meaning = f"Evaluating model predictions using metric '{metric}' with lower_bound={lower_bound}, upper_bound={upper_bound}"
    if matched_columns is not None:
        meaning += f" and row alignment based on matched_columns={matched_columns}"
    if calculate_columns is not None:
        meaning += f", calculating on columns={calculate_columns}"

    output_data = None
    gold_data = None

    # Load CSV files (keep original error handling)
    try:
        result_df = pd.read_csv(output_file_name)
    except FileNotFoundError as e:
        logging.warning(f"Failed to read result CSV {output_file_name}: {e}")
        errors.append(f"Failed to read result CSV: {output_file_name} not found. Please ensure the file exists and is accessible.")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }
    except pd.errors.EmptyDataError as e:
        logging.warning(f"Failed to read result CSV {output_file_name}: {e}")
        errors.append(f"Failed to read result CSV: {output_file_name} is empty or has no data. Please provide a valid CSV file with data.")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }
    except pd.errors.ParserError as e:
        logging.warning(f"Failed to read result CSV {output_file_name}: {e}")
        errors.append(f"Failed to read result CSV: {output_file_name} has invalid CSV format. Check for parsing errors like mismatched quotes or delimiters.")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }
    except Exception as e:
        logging.warning(f"Failed to read result CSV {output_file_name}: {e}")
        errors.append(f"Failed to read result CSV: {output_file_name} - {type(e).__name__}: {str(e)}")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }

    try:
        gold_df = pd.read_csv(gold_file_name)
    except FileNotFoundError as e:
        logging.warning(f"Failed to read expected CSV {gold_file_name}: {e}")
        errors.append(f"Failed to read gold CSV: {gold_file_name} not found. Please ensure the gold standard file exists.")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }
    except pd.errors.EmptyDataError as e:
        logging.warning(f"Failed to read expected CSV {gold_file_name}: {e}")
        errors.append(f"Failed to read gold CSV: {gold_file_name} is empty or has no data. Please provide a valid gold standard CSV file.")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }
    except pd.errors.ParserError as e:
        logging.warning(f"Failed to read expected CSV {gold_file_name}: {e}")
        errors.append(f"Failed to read gold CSV: {gold_file_name} has invalid CSV format. Check for parsing errors like mismatched quotes or delimiters.")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }
    except Exception as e:
        logging.warning(f"Failed to read expected CSV {gold_file_name}: {e}")
        errors.append(f"Failed to read gold CSV: {gold_file_name} - {type(e).__name__}: {str(e)}")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }

    # Row alignment based on matched_columns (keep original logic, add merge option)
    match_ratio = 1.0  # Default: no penalty for unmatched rows
    if matched_columns is None:
        # Original behavior: align by row order, require equal length
        if not len(gold_df) == len(result_df):
            logging.warning(f'The length of the result content is not equal to the length of the true value.')
            errors.append(f"Row count mismatch: result file has {len(result_df)} rows but gold file has {len(gold_df)} rows. Both files must have the same number of rows when matched_columns is None.")
            return {
                'score': 0.0,
                'errors': errors,
                'meaning': meaning,
                'output_data': None,
                'gold_data': None
            }
        gold_aligned = gold_df
        result_aligned = result_df
    else:
        # Handle matched_columns: support both column names (list[str]) and indices (list[int])
        if all(isinstance(col, int) for col in matched_columns):
            # Using indices: directly compare by position, allow different column names
            matched_columns_int = [col for col in matched_columns if isinstance(col, int)]
            max_index = max(matched_columns_int)
            if max_index >= len(result_df.columns) or max_index >= len(gold_df.columns):
                logging.warning(f"Column index out of range. Result columns: {len(result_df.columns)}, Gold columns: {len(gold_df.columns)}")
                errors.append(f"Column index out of range: the highest matched column index is {max_index}, but result file has {len(result_df.columns)} columns and gold file has {len(gold_df.columns)} columns. Please use valid column indices.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }

            # Check if lengths match for direct alignment
            if len(result_df) != len(gold_df):
                logging.warning(f"Row lengths don't match: result has {len(result_df)} rows, gold has {len(gold_df)} rows")
                errors.append(f"Row count mismatch when using column indices for matching: result file has {len(result_df)} rows but gold file has {len(gold_df)} rows. Files must have the same length when matching by column indices.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }

            # Extract matched columns by index for comparison
            result_matched = result_df.iloc[:, matched_columns_int]
            gold_matched = gold_df.iloc[:, matched_columns_int]

            # Find matching rows by comparing values
            comparison = (result_matched.values == gold_matched.values)
            matched_mask = np.all(comparison, axis=1)

            if not matched_mask.any():
                logging.warning("No rows matched based on matched_columns")
                errors.append(f"No rows matched: after comparing the matched columns {matched_columns}, no rows in result file match the corresponding rows in gold file. Check if the matched columns contain the correct values for alignment.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }

            # Filter rows that match
            result_aligned = result_df[matched_mask].reset_index(drop=True)
            gold_aligned = gold_df[matched_mask].reset_index(drop=True)
        else:
            # Using column names: align by matched columns using merge
            missing_cols = [col for col in matched_columns if col not in result_df.columns or col not in gold_df.columns]
            if missing_cols:
                logging.warning(f"Missing matched columns: {missing_cols}")
                errors.append(f"Missing matched columns in input files: columns {missing_cols} are not found in either result file or gold file. Available columns in result: {list(result_df.columns)}. Available columns in gold: {list(gold_df.columns)}.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }

            # Inner join to keep only matched rows
            merged = pd.merge(
                result_df,
                gold_df,
                on=matched_columns,
                how='inner',
                suffixes=('_result', '_gold')
            )

            # Check if gold has rows that don't match output
            gold_matched_count = len(gold_df)
            gold_unmatched_count = gold_matched_count - len(merged)
            if gold_unmatched_count > 0:
                match_ratio = len(merged) / gold_matched_count
                logging.warning(f"{gold_unmatched_count} row(s) in gold file have no matching rows in output file based on matched_columns={matched_columns}")
                errors.append(f"Gold file has {gold_unmatched_count} row(s) that could not be matched to output file based on matched_columns={matched_columns}. These rows are excluded from evaluation, resulting in a match ratio of {match_ratio:.2%} applied as penalty.")

            if len(merged) == 0:
                logging.warning("No rows matched based on matched_columns")
                errors.append(f"No rows matched: after merging on columns {matched_columns}, no common rows were found between result and gold files. Please verify that the matched columns contain matching values in both files.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }

            # Separate result and gold data
            result_cols = [col for col in merged.columns if col.endswith('_result') or col in matched_columns]
            gold_cols = [col for col in merged.columns if col.endswith('_gold') or col in matched_columns]

            result_aligned = merged[result_cols].rename(columns={col: col.replace('_result', '') for col in result_cols if col.endswith('_result')})
            gold_aligned = merged[gold_cols].rename(columns={col: col.replace('_gold', '') for col in gold_cols if col.endswith('_gold')})

    # Extract calculate_columns (flatten multiple columns)
    if calculate_columns:
        # Handle calculate_columns: support both column names (list[str]) and indices (list[int])
        if all(isinstance(col, int) for col in calculate_columns):
            calculate_columns_int = [col for col in calculate_columns if isinstance(col, int)]
            max_calc_index = max(calculate_columns_int) if calculate_columns_int else -1
            if max_calc_index >= len(result_aligned.columns) or max_calc_index >= len(gold_aligned.columns):
                logging.warning(f"Column index out of range. Result columns: {len(result_aligned.columns)}, Gold columns: {len(gold_aligned.columns)}")
                errors.append(f"Calculate column index out of range: the highest calculate column index is {max_calc_index}, but result file has {len(result_aligned.columns)} columns and gold file has {len(gold_aligned.columns)} columns after alignment. Please use valid column indices.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }

            # Extract by index (allows different column names in result and gold)
            result_data = result_aligned.iloc[:, calculate_columns_int]
            gold_data = gold_aligned.iloc[:, calculate_columns_int]
        else:
            missing_calc_cols = [col for col in calculate_columns if col not in result_aligned.columns]
            if missing_calc_cols:
                logging.warning(f"Missing calculate columns in result: {missing_calc_cols}")
                errors.append(f"Missing calculate columns in result file: columns {missing_calc_cols} are not found. Available columns in aligned result: {list(result_aligned.columns)}. Please specify valid column names that exist in the result file.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }

            result_data = result_aligned[calculate_columns]
            gold_calc_cols = [col for col in calculate_columns if col in gold_aligned.columns]
            if not gold_calc_cols:
                gold_calc_cols = calculate_columns
            gold_data = gold_aligned[gold_calc_cols]
    else:
        # Use all columns if calculate_columns is None
        result_data = result_aligned
        gold_data = gold_aligned

    # Calculate metric (keep CalculateML logic, add custom function support)
    if isinstance(metric, str):
        # Use CalculateML methods (keep original logic)
        metric_lower = metric.lower().strip().replace(' ', "_")
        metric_func = getattr(CalculateML, f'calculate_{metric_lower}', None)

        if metric_func:
            try:
                # Flatten multi-column data into single series for pred and label
                if len(result_data.columns) > 1:
                    # Flatten all columns into single array
                    pred = result_data.values.flatten()
                    label = gold_data.values.flatten()
                else:
                    # Single column
                    pred = result_data.iloc[:, 0]
                    label = gold_data.iloc[:, 0]

                score, output = metric_func(pred, label)
                if len(output['errors']) > 0:
                    errors.extend(output['errors'])
            except ValueError as e:
                logging.warning(f"Failed to calculate metric {metric}: {e}")
                errors.append(f"Value error when calculating metric '{metric}': {str(e)}. This may be due to invalid data types, NaN values, or incompatible values between result and gold data.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }
            except TypeError as e:
                logging.warning(f"Failed to calculate metric {metric}: {e}")
                errors.append(f"Type error when calculating metric '{metric}': {str(e)}. Please ensure the data types in the result and gold files are compatible with the metric requirements.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }
            except Exception as e:
                logging.warning(f"Failed to calculate metric {metric}: {e}")
                errors.append(f"Failed to calculate metric '{metric}': {type(e).__name__}: {str(e)}. Please check the data format and metric compatibility.")
                return {
                    'score': 0.0,
                    'errors': errors,
                    'meaning': meaning,
                    'output_data': None,
                    'gold_data': None
                }
        else:
            logging.warning(f"Evaluation Scripts don't have metric: {metric}")
            errors.append(f"Evaluation Scripts don't have metric: '{metric}'. Available metrics are defined in CalculateML class. Please check the supported metrics or use a valid metric name.")
            return {
                'score': 0.0,
                'errors': errors,
                'meaning': meaning,
                'output_data': None,
                'gold_data': None
            }
    elif callable(metric):
        # Custom metric function (new feature)
        try:
            score = metric(result_data, gold_data)
        except Exception as e:
            logging.warning(f"Custom metric function failed: {e}")
            errors.append(f"Custom metric function failed with error: {type(e).__name__}: {str(e)}. Please verify the custom metric function implementation and input data compatibility.")
            return {
                'score': 0.0,
                'errors': errors,
                'meaning': meaning,
                'output_data': None,
                'gold_data': None
            }
    else:
        logging.warning(f"Unsupported metric type: {type(metric)}")
        errors.append(f"Unsupported metric type: {type(metric).__name__}. Metric must be either a string (metric name) or a callable function that takes two DataFrames and returns a float score.")
        return {
            'score': 0.0,
            'errors': errors,
            'meaning': meaning,
            'output_data': None,
            'gold_data': None
        }

    if metric in LOWER_METRICS and lower_bound < upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    # Normalize score (keep original normalization logic)
    if upper_bound != lower_bound:
        normalized_score = (score - lower_bound) / (upper_bound - lower_bound)
        normalized_score = max(0.0, min(1.0, normalized_score))
    else:
        normalized_score = 1.0 if score == lower_bound else 0.0

    # Apply penalty for unmatched gold rows
    normalized_score = normalized_score * match_ratio

    # Save result_data and gold_data content (as CSV string) with truncation for long texts
    result_data_str = result_data.to_csv(index=False)
    gold_data_str = gold_data.to_csv(index=False)

    if len(result_data_str) < 1000:
        result_data_content = result_data_str
    else:
        result_data_content = result_data_str[:1000] + f"\n... ({len(result_data_str)} characters total)"

    if len(gold_data_str) < 1000:
        gold_data_content = gold_data_str
    else:
        gold_data_content = gold_data_str[:1000] + f"\n... ({len(gold_data_str)} characters total)"

    output_dict = {'score': normalized_score, 'errors': errors, 'meaning': meaning}
    if result_data_content is not None:
        output_dict['result_data'] = result_data_content
    if gold_data_content is not None:
        output_dict['gold_data'] = gold_data_content

    return output_dict
