"""
Image Metrics Module - Image comparison metric for visualization evaluation.

This module compares plot outputs:
- Compares .json metadata (colors, labels, plot type)
- Compares .npy data arrays with tolerance
- Supports multiple gold standard images

Reference: https://github.com/yiyihum/da-code/tree/main/da_agent/evaluators/metrics/image.py
"""

import numpy as np
from PIL import Image
import os, logging, json
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from fuzzywuzzy import fuzz
import cv2 

@dataclass
class ImageTest:

    @classmethod
    def compare_key(cls, key: str, result: Dict, gold: Dict, subplot_name: str = "") -> tuple[float, Dict[str, bool], str]:
        def hex_to_rgb(hex_color):
            import matplotlib.colors as mcolors
            try:
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                rgb_float = mcolors.to_rgb(hex_color)
                return tuple(int(c * 255) for c in rgb_float)
        def color_distance(c1, c2):
            return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))
        def is_color_similar(c1, gold, threshold=15):
            for c2 in gold:
                distance = color_distance(c1, c2)
                if distance <= threshold:
                    return True
            return False

        key = key.lower()
        prefix = f"{subplot_name}: " if subplot_name else ""
        if key == 'figsize':
            result_fig, gold_fig = result.get(key, []), gold.get(key, [])
            result_fig = result_fig if isinstance(result_fig, list)\
                else list(result_fig)
            gold_fig = gold_fig if isinstance(gold_fig, list)\
                else list(gold_fig)
            if result_fig == gold_fig:
                return (1.0, {key: True}, "")
            else:
                error_detail = f"Result figsize is {result_fig}, but expected {gold_fig}. Figure dimensions do not match."
                logging.warning(f"{prefix}figsize mismatch: {error_detail}")
                return (0.0, {key: False}, f"{prefix}figsize mismatch: {error_detail}")
        elif key == 'color':
            result_fig, gold_fig = result.get(key, []), gold.get(key, [])
            result_fig = result_fig if isinstance(result_fig, list)\
                else list(result_fig)
            gold_fig = gold_fig if isinstance(gold_fig, list)\
                else list(gold_fig)
            result_fig_rgb = list(map(hex_to_rgb, result_fig))
            gold_fig_rgb = list(map(hex_to_rgb, gold_fig))
            if all(is_color_similar(c1, gold_fig_rgb) for c1 in result_fig_rgb):
                return (1.0, {key: True}, "")
            else:
                error_detail = f"Result colors {result_fig} (RGB: {result_fig_rgb}) do not match expected colors {gold_fig} (RGB: {gold_fig_rgb})."
                logging.warning(f"{prefix}color mismatch: {error_detail}")
                return (0.0, {key: False}, f"{prefix}color mismatch: {error_detail}")
        elif key == 'type':
            result_fig, gold_fig = result.get(key, ''), gold.get(key, '')
            if result_fig.lower() == gold_fig.lower():
                return (1.0, {key: True}, "")
            else:
                error_detail = f"Result plot type is '{result_fig}', but expected '{gold_fig}'. Plot type does not match."
                logging.warning(f"{prefix}type mismatch: {error_detail}")
                return (0.0, {key: False}, f"{prefix}type mismatch: {error_detail}")
        elif key == 'graph_title' or key == 'x_label' or key == 'y_label' or key == 'legend_title':
            result_fig, gold_fig = result.get(key, ''), gold.get(key, '')
            if not result_fig and gold_fig:
                error_detail = f"{key.replace('_', ' ')} is missing. Expected '{gold_fig}' but got empty string."
                return (0.0, {key: False}, f"{prefix}{error_detail}")
            similarity_score = fuzz.ratio(result_fig.lower(), gold_fig.lower())
            if similarity_score >= 60:
                return (1.0, {key: True}, "")
            else:
                error_detail = f"Result {key.replace('_', ' ')} is '{result_fig}', but expected '{gold_fig}'. Text similarity is {similarity_score}% (threshold: 60%)."
                logging.warning(f"{prefix}{key} mismatch: {error_detail}")
                return (0.0, {key: False}, f"{prefix}{error_detail}")
        elif key == 'labels' or key == 'xtick_labels' or key == 'ytick_labels':
            result_fig, gold_fig = result.get(key, []), gold.get(key, [])
            if len(result_fig) != len(gold_fig):
                error_detail = f"{key.replace('_', ' ')} count mismatch: result has {len(result_fig)} items, but expected {len(gold_fig)} items."
                logging.warning(f"{prefix}{error_detail}")
                return (0.0, {key: False}, f"{prefix}{error_detail}")
            result_fig_lower = list(map(lambda x: x.lower().replace('\u2212', '-'), result_fig))
            gold_fig_lower = list(map(lambda x: x.lower().replace('\u2212', '-'), gold_fig))

            result = all(any(fuzz.ratio(x, y) > 95 for y in gold_fig_lower) for x in result_fig_lower)
            if result:
                return (1.0, {key: True}, "")
            else:
                error_detail = f"Result {key.replace('_', ' ')} {result_fig} do not match expected {gold_fig}. Some labels have low similarity (<95%)."
                logging.warning(f"{prefix}{key} mismatch: {error_detail}")
                return (0.0, {key: False}, f"{prefix}{error_detail}")
        else:
            raise \
            ValueError(f'please check your key: {key}, it must in [figsize, type, labels, x_label, y_label, graph_title, legend_title, color, xtick_labels, ytick_lables]')
            
    @classmethod
    def scale_to_percentage(cls, arr):
        total_sum = np.sum(arr)
        return arr / total_sum

    @classmethod
    def compare_numpy(cls, hyp_np: np.ndarray,
        ref_np: np.ndarray, tol=1e-2, is_sacle: bool=False, subplot_name: str = ""):
        errors = []

        # Check shape mismatch
        if hyp_np.shape != ref_np.shape:
            error_detail = f"Shape mismatch detected. Result array shape is {hyp_np.shape}, but expected shape is {ref_np.shape}."
            errors.append(f"{subplot_name}{error_detail}")
            logging.warning(f"{subplot_name}Shape mismatch: {error_detail}")
            return False, errors

        if is_sacle:
            hyp_np_scaled = cls.scale_to_percentage(hyp_np)
            ref_np_scaled = cls.scale_to_percentage(ref_np)
        else:
            hyp_np_scaled = hyp_np.copy()
            ref_np_scaled = ref_np.copy()

        hyp_np_sorted = np.sort(hyp_np_scaled, axis=0).reshape(hyp_np_scaled.shape)
        ref_np_sorted = np.sort(ref_np_scaled, axis=0).reshape(ref_np_scaled.shape)

        # Check if arrays are close
        if np.allclose(hyp_np_sorted, ref_np_sorted, atol=tol, equal_nan=True):
            return True, errors

        # If not close, generate detailed error information about distribution differences
        scale_text = " (normalized to percentage)" if is_sacle else ""
        errors.append(f"{subplot_name}Data distribution mismatch detected{scale_text} with tolerance {tol}:")

        # Calculate statistics for both arrays
        hyp_flat = hyp_np_sorted.flatten()
        ref_flat = ref_np_sorted.flatten()

        hyp_finite = hyp_flat[np.isfinite(hyp_flat)]
        ref_finite = ref_flat[np.isfinite(ref_flat)]

        if len(hyp_finite) == 0 or len(ref_finite) == 0:
            errors.append(f"\nArrays contain only NaN/Inf values (result finite: {len(hyp_finite)}/{len(hyp_flat)}, expected finite: {len(ref_finite)}/{len(ref_flat)})")
            return False, errors

        errors.append(f"\nResult data statistics:")
        errors.append(f"  Total elements: {len(hyp_flat)} (finite: {len(hyp_finite)})")
        errors.append(f"  Min value: {np.min(hyp_finite):.6f}")
        errors.append(f"  Max value: {np.max(hyp_finite):.6f}")
        errors.append(f"  Mean value: {np.mean(hyp_finite):.6f}")
        errors.append(f"  Median value: {np.median(hyp_finite):.6f}")
        errors.append(f"  Standard deviation: {np.std(hyp_finite):.6f}")

        errors.append(f"\nExpected data statistics:")
        errors.append(f"  Total elements: {len(ref_flat)} (finite: {len(ref_finite)})")
        errors.append(f"  Min value: {np.min(ref_finite):.6f}")
        errors.append(f"  Max value: {np.max(ref_finite):.6f}")
        errors.append(f"  Mean value: {np.mean(ref_finite):.6f}")
        errors.append(f"  Median value: {np.median(ref_finite):.6f}")
        errors.append(f"  Standard deviation: {np.std(ref_finite):.6f}")

        # Calculate histogram differences (for large arrays)
        total_elements = hyp_finite.size
        if total_elements > 100 and ref_finite.size > 100:
            errors.append(f"\nHistogram distribution comparison (10 bins, showing bins with >5% difference):")
            hyp_hist, _ = np.histogram(hyp_finite, bins=10, density=True)
            ref_hist, _ = np.histogram(ref_finite, bins=10, density=True)

            significant_diff_found = False
            for i in range(10):
                hyp_pct = hyp_hist[i] * 100
                ref_pct = ref_hist[i] * 100
                diff = abs(hyp_pct - ref_pct)
                if diff > 5:  # Only show bins with significant difference
                    errors.append(f"  Bin {i+1} (range: {i*10}%-{(i+1)*10}% of data range):")
                    errors.append(f"    Result: {hyp_pct:.2f}% of data")
                    errors.append(f"    Expected: {ref_pct:.2f}% of data")
                    errors.append(f"    Absolute difference: {diff:.2f}%")
                    significant_diff_found = True

            if not significant_diff_found:
                errors.append(f"  Note: No single bin has >5% difference, but overall distribution still differs.")
        else:
            # For small arrays, show individual element differences
            errors.append(f"\nElement-by-element comparison (showing first 20 mismatched elements):")
            diff = np.abs(hyp_flat - ref_flat)
            mismatched_indices = np.where(diff > tol)[0]
            mismatched_count = len(mismatched_indices)

            if mismatched_count > 0:
                errors.append(f"  Total mismatched elements: {mismatched_count} out of {len(hyp_flat)}")
                for i in range(min(20, mismatched_count)):
                    idx = mismatched_indices[i]
                    result_val = hyp_flat[idx]
                    expected_val = ref_flat[idx]
                    actual_diff = diff[idx]
                    relative_diff = abs(actual_diff / expected_val) * 100 if expected_val != 0 else float('inf')
                    errors.append(f"  Element [{idx}]:")
                    errors.append(f"    Result value: {result_val:.6f}")
                    errors.append(f"    Expected value: {expected_val:.6f}")
                    errors.append(f"    Absolute difference: {actual_diff:.6f}")
                    if relative_diff != float('inf'):
                        errors.append(f"    Relative difference: {relative_diff:.2f}%")
                if mismatched_count > 20:
                    errors.append(f"\n  ... and {mismatched_count - 20} more mismatched elements (total {mismatched_count})")
            else:
                errors.append(f"  Warning: No individual elements exceed tolerance {tol}, but arrays are still not close enough.")
                errors.append(f"  This may indicate a systematic small bias across many elements.")

        return False, errors

    @staticmethod
    def test_image(results: str | List[str], golds: str | List[str], iscolor:bool=False, issize:bool=False):
        results = results if isinstance(results, list) else [results]
        golds = golds if isinstance(golds, list) else [golds]
        score = 0.0
        for gold in golds:
            assert os.path.exists(gold), f'gold path {gold} do not exist'
            image_name = os.path.basename(gold)
            result = next((file for file in results if image_name in file), '')
            if not result or not os.path.exists(result):
                # logging.error(f"could not find {image_name} in agent's results")
                return (0.0, {'img': False})
            result_img = np.array(Image.open(result).convert('L')) if not iscolor \
                else np.array(Image.open(result))
            gold_img = np.array(Image.open(gold).convert('L')) if not iscolor \
                else np.array(Image.open(gold))
            if gold_img.ndim == 3:
                if result_img.ndim != 3:
                    return (0.0, {'img': False})
                else:
                    if gold_img.shape[-1] != result_img.shape[-1]:
                        return (0.0, {'img': False})
            if issize:
                image_stat = (
                    result_img.shape == gold_img.shape
                    and np.allclose(result_img, gold_img,  atol=1e-2)
                )
            else:
                result_img = cv2.resize(result_img, (gold_img.shape[1], gold_img.shape[0]))
                image_stat = np.allclose(result_img, gold_img, atol=1e-2)
            score += float(image_stat)
        return (1.0, {'img': True}) if score == float(len(golds)) else (0.0, {'img': False})

    @classmethod
    def test_numpy(cls, result_np: str, gold_np: str):
        if not os.path.exists(result_np):
            return {'score': 0.0, 'subplot_matches': {'single': False}, 'errors': ['Result numpy file not found']}
        assert os.path.exists(gold_np), f'the gold file {gold_np} does not exist'
        results = np.load(result_np, allow_pickle=True).item()
        golds = np.load(gold_np, allow_pickle=True).item()

        all_errors = []

        # Check if it's a multi-subplot result
        if isinstance(results, dict) and isinstance(golds, dict):
            # Multi-subplot case - check keys match subplot pattern
            has_subplot_keys = any(k.startswith('subplot_') for k in results.keys())

            if has_subplot_keys:
                if len(results) != len(golds):
                    return {'score': 0.0, 'subplot_matches': {},
                            'errors': [f"Subplot count mismatch: result has {len(results)} subplots, expected {len(golds)}"]}

                subplot_matches = {}  # Each subplot: True if data matches (or scale matches), False otherwise

                for subplot_key in results.keys():
                    if subplot_key not in golds:
                        all_errors.append(f"Subplot '{subplot_key}' not found in gold")
                        subplot_matches[subplot_key] = False
                        continue

                    result_arr = results[subplot_key]
                    gold_arr = golds[subplot_key]

                    # Reshape to 2D if necessary
                    result_arr = result_arr.reshape(-1, 1) if result_arr.ndim == 1 else result_arr
                    gold_arr = gold_arr.reshape(-1, 1) if gold_arr.ndim == 1 else gold_arr

                    # Record shape mismatch but continue evaluating other subplots
                    if result_arr.shape != gold_arr.shape:
                        error_detail = f"Subplot '{subplot_key}': shape mismatch - result {result_arr.shape}, expected {gold_arr.shape}"
                        all_errors.append(error_detail)
                        subplot_matches[subplot_key] = False
                        continue

                    finds, data_errors = cls.compare_numpy(hyp_np=result_arr, ref_np=gold_arr, is_sacle=False, subplot_name=f"Subplot '{subplot_key}': ")
                    scale_finds, scale_errors = cls.compare_numpy(hyp_np=result_arr, ref_np=gold_arr, is_sacle=True, subplot_name=f"Subplot '{subplot_key}': ")

                    # Subplot matches if either data or scale_data matches
                    subplot_matches[subplot_key] = finds or scale_finds

                    if not finds:
                        all_errors.extend(data_errors)
                    if not scale_finds and not finds:
                        all_errors.extend(scale_errors)

                # Calculate average score across subplots (each subplot is 1 or 0)
                avg_score = sum(subplot_matches.values()) / len(subplot_matches) if subplot_matches else 0.0
                return {'score': avg_score,
                        'subplot_matches': subplot_matches,
                        'errors': all_errors}
            else:
                # Dict but not subplot format - treat as single
                pass

        # Single subplot case
        if hasattr(results, 'ndim'):
            results = results.reshape(-1, 1) if results.ndim == 1 else results
        if hasattr(golds, 'ndim'):
            golds = golds.reshape(-1, 1) if golds.ndim == 1 else golds

        # Check shapes
        if hasattr(results, 'shape') and hasattr(golds, 'shape') and results.shape != golds.shape:
            return {'score': 0.0, 'subplot_matches': {'single': False},
                    'errors': [f"Shape mismatch: result {results.shape}, expected {golds.shape}"]}

        finds, data_errors = cls.compare_numpy(hyp_np=results, ref_np=golds, is_sacle=False)
        scale_finds, scale_errors = cls.compare_numpy(hyp_np=results, ref_np=golds, is_sacle=True)

        all_errors = []
        if not finds:
            all_errors.extend(data_errors)
        if not scale_finds and not finds:
            all_errors.extend(scale_errors)

        match = finds or scale_finds
        return {'score': 1.0 if match else 0.0,
                'subplot_matches': {'single': match},
                'errors': all_errors}
        
    @classmethod
    def test_info(cls, result_js: str, gold_js: str, fig_keys: Optional[List[str]]=None):
        output_dict = {}
        errors = []
        if not os.path.exists(result_js):
            logging.warning(f"Result JSON file {result_js} not found, scoring 0 for figure info")
            return (0.0, {}, {'subplot_scores': {}, 'errors': [f"Result JSON file {result_js} not found"]})
        assert os.path.exists(gold_js), f'the gold file {gold_js} does not exist'
        with open(result_js, 'r') as js:
            result = json.load(js)
        with open(gold_js, 'r') as js:
            gold = json.load(js)

        # Check if it's a multi-subplot result
        if 'total_subplots' in result or 'total_subplots' in gold:
            # Multi-subplot case
            result_subplots = {k: v for k, v in result.items() if k.startswith('subplot_')}
            gold_subplots = {k: v for k, v in gold.items() if k.startswith('subplot_')}

            # Check number of subplots
            if len(result_subplots) != len(gold_subplots):
                logging.warning(f"Subplot count mismatch: result has {len(result_subplots)} subplots, expected {len(gold_subplots)}")
                return (0.0, {}, {'subplot_scores': {}, 'errors': [f"Subplot count mismatch: result has {len(result_subplots)} subplots, expected {len(gold_subplots)}"]})

            subplot_scores = {}  # Store score for each subplot (1.0 if all keys match, 0.0 otherwise)

            # Compare each subplot
            for subplot_key in sorted(result_subplots.keys()):
                if subplot_key not in gold_subplots:
                    logging.warning(f"Subplot '{subplot_key}' not found in gold")
                    errors.append(f"Subplot '{subplot_key}' not found in gold")
                    subplot_scores[subplot_key] = 0.0
                    continue

                result_subplot = result_subplots[subplot_key]
                gold_subplot = gold_subplots[subplot_key]

                # Determine keys to compare
                keys_compare = fig_keys if fig_keys is not None else gold_subplot.keys()
                subplot_total_keys = len(keys_compare)
                subplot_score = 0.0

                for key in keys_compare:
                    key_score, key_dict, error_msg = cls.compare_key(key, result=result_subplot, gold=gold_subplot, subplot_name=subplot_key)
                    subplot_score += key_score
                    # Prefix keys with subplot name
                    for k, v in key_dict.items():
                        output_dict[f"{subplot_key}_{k}"] = v
                    if error_msg:
                        errors.append(error_msg)

                # Subplot score is 1.0 only if all keys match
                subplot_scores[subplot_key] = 1.0 if subplot_score == float(subplot_total_keys) else 0.0

            # Calculate average score across subplots
            avg_score = sum(subplot_scores.values()) / len(subplot_scores) if subplot_scores else 0.0
            return (avg_score, output_dict, {'subplot_scores': subplot_scores, 'errors': errors})
        else:
            # Single subplot case (original logic) - returns 1.0 if all match, 0.0 otherwise
            keys_compare = fig_keys if fig_keys is not None else gold.keys()
            total_keys = len(keys_compare)
            scores = 0.0

            for key in keys_compare:
                score, key_dict, error_msg = cls.compare_key(key, result=result, gold=gold)
                scores += score
                output_dict.update(key_dict)
                if error_msg:
                    errors.append(error_msg)

            # Single subplot: 1.0 if all keys match, 0.0 otherwise
            final_score = 1.0 if scores == float(total_keys) else 0.0
            return (final_score, output_dict, {'subplot_scores': {'single': final_score}, 'errors': errors})
        
def compare_image(output_file_name: str, gold_file_name: str, calculate_columns: list[str] = ['type']) -> Dict[str, Any]:
    """
    @args:
        output_file_name(str): the pred image file
        gold_file_name(str): the gold image file
        option(dict): the configuration dictionary

    Scoring logic:
        - For each subplot: score = 1 if both info AND data match, else 0
        - Final score = average across all subplots
    """
    keys_compare = calculate_columns
    meaning_parts = []

    result_image = output_file_name
    gold_image = gold_file_name
    result_json = os.path.splitext(result_image)[0] + ".json"
    gold_json = os.path.splitext(gold_image)[0] + ".json"
    result_npy = os.path.splitext(result_image)[0] + ".npy"
    gold_npy = os.path.splitext(gold_image)[0] + ".npy"

    output = {}

    issize = 'figsize' in keys_compare
    meaning_parts.append("compare image pixel data")
    if issize:
        meaning_parts.append("with exact figsize check")
    else:
        meaning_parts.append("with image resizing for comparison")

    image_score, img_dict = ImageTest.test_image(results=[result_image], golds=[gold_image],
            iscolor=False,
            issize=issize)
    output.update(img_dict)

    if image_score:
        output['score'] = 1.0
        output['errors'] = []
        output['meaning'] = f"Image pixel comparison: {' AND '.join(meaning_parts)}"
        return output

    numpy_result = ImageTest.test_numpy(result_np=result_npy, gold_np=gold_npy)
    numpy_subplot_matches = numpy_result.get('subplot_matches', {'single': False})
    errors = numpy_result.get('errors', [])

    meaning_parts = []
    if keys_compare is not None:
        if keys_compare:
            meaning_parts.append(f"compare figure properties: {', '.join(keys_compare)}")
        else:
            meaning_parts.append("no figure properties to compare")
    else:
        meaning_parts.append("compare all figure properties")

    if gold_json and result_json:
        info_score, info_dict, info_detail = ImageTest.test_info(result_js=result_json, gold_js=gold_json, fig_keys=keys_compare)
        output.update(info_dict)
        info_subplot_scores = info_detail.get('subplot_scores', {'single': 0.0})
        info_errors = info_detail.get('errors', [])
        errors.extend(info_errors)

        # Combine: each subplot scores 1 if BOTH numpy AND info match
        combined_subplot_scores = {}
        all_subplot_keys = set(numpy_subplot_matches.keys()) | set(info_subplot_scores.keys())

        for subplot_key in all_subplot_keys:
            numpy_match = numpy_subplot_matches.get(subplot_key, False)
            info_match = info_subplot_scores.get(subplot_key, 0.0) == 1.0
            # Subplot scores 1 only if both match
            combined_subplot_scores[subplot_key] = 1.0 if (numpy_match and info_match) else 0.0

        # Final score = average across subplots
        final_score = sum(combined_subplot_scores.values()) / len(combined_subplot_scores) if combined_subplot_scores else 0.0
        output['score'] = final_score
        output['subplot_scores'] = combined_subplot_scores
        output['errors'] = errors

        if final_score == 1.0:
            output['meaning'] = f"Image comparison passed: numpy data match AND figure properties match ({', '.join(meaning_parts)})"
        else:
            matched_count = sum(combined_subplot_scores.values())
            total_count = len(combined_subplot_scores)
            output['meaning'] = f"Image comparison: {' AND '.join(meaning_parts)} ({matched_count}/{total_count} subplots matched)"
        return output
    else:
        # No JSON, only numpy comparison
        final_score = sum(numpy_subplot_matches.values()) / len(numpy_subplot_matches) if numpy_subplot_matches else 0.0
        output['score'] = final_score
        output['subplot_scores'] = numpy_subplot_matches
        output['errors'] = errors
        matched_count = sum(numpy_subplot_matches.values())
        total_count = len(numpy_subplot_matches)
        output['meaning'] = f"Image comparison: numpy data check ({matched_count}/{total_count} subplots matched)"
        return output