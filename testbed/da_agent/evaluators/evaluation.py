"""
Evaluation Module - Evaluator for scoring agent outputs.

This module evaluates agent performance:
- Compares output files against gold standard
- Executes evaluation functions dynamically
- Calculates scores based on comparison results
- Tracks trajectory information (actions, token usage)

Reference: https://github.com/yiyihum/da-code/tree/main/da_agent/evaluators/evaluation.py
"""

import logging
import os, json
from typing import Callable, Any
from typing import List, Dict
from pathlib import Path
import sys, jsonlines
here=Path(__file__).absolute()
sys.path.append(str(here.parent))
from metrics import *
from tqdm import tqdm
from da_agent.envs.utils import timeout
import re
import traceback

Metric = Callable[[Any, Any], float]

class Evaluator:

    def __init__(self, output_dir: str, gold_dir: str, timeout_seconds: int = 10):
        self.output_dir = output_dir
        self.gold_dir = gold_dir
        self.timeout_second = timeout_seconds

    def get_result_file(self, results: List, dir: str, isgold: bool):
        results = results if isinstance(results, list)\
            else [results]
        if 'number' in results[0].keys():
            return 'number', [results[0]['number']] 
        result_files = []
        for result in results:
            multi = result.get("multi", False)
            files = result['file'] if isinstance(result['file'], list) \
                else [result['file']]
            if multi:
                files = [os.path.join(dir, file) for file in files] if not isgold \
                    else [os.path.join(dir, os.path.basename(file)) for file in files]
                result_files.append(files)
            else:
                for file in files:
                    file = file if not isgold else os.path.basename(file)
                    # if not os.path.exists(os.path.join(dir, file)):
                    #     print(f"File not found : {os.path.join(dir, file)}")
                    result_files.append(os.path.join(dir, file))
        return 'file', result_files

    def _get_eval_config_info(self, eval_config: Dict[str, Any]):
        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        # self.evaluator = task_config["evaluator"]
        id = eval_config['id']
        output_id_dir = os.path.join(self.output_dir, id)

        result_file = os.path.join(output_id_dir, 'dabench', 'result.json')
        
        if not os.path.exists(result_file):
            print(f"File {result_file} not found")
            return id, False, None, None
        trajectory_info = self._get_trajectory_info_from_json(result_file)

        gold_id_dir = os.path.join(self.gold_dir, id)

        if 'smolagents' in self.gold_dir:
            code_file = os.path.join(output_id_dir, 'code.ipynb')
            code_content = {}
            try:
                with open(code_file, 'r', encoding='utf-8') as f:
                    import nbformat
                    notebook = nbformat.read(f, as_version=4)
                    cells = []
                    for idx, cell in enumerate(notebook.cells):
                        if cell.source and cell.source.strip():
                            cell_type = 'code' if cell.cell_type == 'code' else 'markdown'
                            cells.append({
                                'id': idx,
                                'type': cell_type,
                                'content': cell.source
                            })
                    code_content[os.path.basename(code_file)] = cells
            except Exception as e:
                logging.warning(f"Failed to read code file {code_file}: {e}")
            if code_content:
                trajectory_info['code'] = code_content

        config = {'domain': eval_config['domain'].split('/')[0]}
        gold_file_name = eval_config['gold_file_name'] if isinstance(eval_config['gold_file_name'], list) \
            else [eval_config['gold_file_name']]
        gold_file_dict = {file_name: os.path.join(gold_id_dir, os.path.basename(file_name)) for file_name in gold_file_name}
        output_file_name = eval_config['output_file_name'] if isinstance(eval_config['output_file_name'], list) \
            else [eval_config['output_file_name']]
        output_file_dict = {file_name: os.path.join(output_id_dir, os.path.basename(file_name)) for file_name in output_file_name}

        eval_func = eval_config["eval_func"] \
            if isinstance(eval_config["eval_func"], list)\
            else [eval_config["eval_func"]]
        exe_eval_func = []
        for func in eval_func:
            for k, v in gold_file_dict.items():
                pattern = rf"(['\"]){k}\1"
                matches = list(re.finditer(pattern, func))
                if len(matches) > 1:
                    raise ValueError(f"Found multiple matches for {k}")
                func = re.sub(pattern, rf"\1{v}\1", func)
            for k, v in output_file_dict.items():
                pattern = rf"(['\"]){k}\1"
                matches = list(re.finditer(pattern, func))
                if len(matches) > 1:
                    raise ValueError(f"Found multiple matches for {k}")
                func = re.sub(rf"(['\"]){k}\1", rf"\1{v}\1", func)
            exe_eval_func.append(func)
        
        
        return id, True, trajectory_info, (config, exe_eval_func)
    
    def _get_trajectory_info_from_json(self,result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)
        trajectory = result["trajectory"]
        actions = []
        if 'smolagents' in result_file:
            for step in trajectory:
                if not 'model_output_message' in step:
                    continue
                if 'plan' in step:
                    actions.append({
                        "action": "Plan",
                        "content": step['plan'],
                        "token_usage": step.get('token_usage'),
                        "timing": step.get('timing')
                    })
                elif 'code_action' in step:
                    actions.append({
                        "action": "Code",
                        "content": len(step['code_action'].split('\n')) if step['code_action'] else 0,
                        "error": step.get('error'),
                        "token_usage": step.get('token_usage'),
                        "timing": step.get('timing')
                    })
                else:
                    logging.warning(f"Unknown step format in smolagents trajectory: {step}")
        else:
            for i, step in enumerate(trajectory):
                if i+1 < len(trajectory):
                    observation = trajectory[i+1]["observation"]
                    if "executed successfully. No output." in observation:
                        observation = "execution succeeded"
                    elif observation.startswith("Failed to parse action from your response,"):
                        observation = "action parse failed"
                    elif observation.startswith("ERROR:") or "Traceback (" in observation or \
                        observation.startswith("bash: -c: line ") or "Error: " in observation:
                        observation = "error message"
                    elif "Warning:" in observation:
                        observation = "warning message"
                    else:
                        observation = "standard output"
                else:
                    observation = ""

                actions.append({
                    "action": step["action"],
                    "content": step.get("code", "") if not step['action'].startswith('Python') else len(step.get("code", "").split('\n')),
                    "observation": observation,
                    "token_usage": step.get('usage'),
                    "timing": step.get('timing')
                })
                    
        info = {"finished": result["finished"],
                "result": result["result"],
                "added_files": result["result_files"]["added_files"],
                "changed_files": result["result_files"]["changed_files"],
                "actions": actions}
        return info


    def _get_result_file_from_json(self, output_id_dir, result_file, is_plot=False):
        pattern = r'\b(?:[\w/\-_]+/)?([\w\-_]+(\.\w+)+)\b'
        filenames = re.findall(pattern, result_file)
        if not filenames:
            return []
        filenames = [filename[0] for filename in filenames]
        result_file = [os.path.join(output_id_dir, file) for file in filenames]
        
        if is_plot:
            result_file += [os.path.join(output_id_dir,"dabench/plot.json"), os.path.join(output_id_dir,"dabench/result.npy")]
            result_file = [result_file]
        return result_file
    
    def evaluate(self, env_config: Dict|str):
        """
        Evaluate task
        """
        if isinstance(env_config, str):
            if not os.path.exists(env_config) or not os.path.isfile(env_config):
                raise ValueError('File Path Error: Please provide a right file path')
            if env_config.endswith('.json'):
                with open(env_config, 'r') as f:
                    env_configs = json.load(f)
            elif env_config.endswith('.jsonl'):
                with jsonlines.open(env_config, 'r') as js:
                    env_configs = [config_eval for config_eval in js]
            else:
                raise ValueError('File Type Error: Please Upload json or jsonl file')
            env_configs = env_configs if isinstance(env_configs, list) else [env_configs]
        elif isinstance(eval_config, dict):
            env_configs = [env_config] 
       
        eval_results = []
        pbar = tqdm(total=len(env_configs))

        for eval_config in env_configs:
            
            id, exist, trajectory_info, eval_info = self._get_eval_config_info(eval_config)
            pbar.set_description(f"Processing Task id: {id}")
            pbar.update(1)
            if not exist:
                print(f"Result of Task {id} does not exist!")
                continue
            
            (config, eval_func) = eval_info
            domain = config['domain']
            if trajectory_info["finished"] == False:
                eval_results.append({"id": id, "total_score": 0.0, **trajectory_info, 'domain': domain})
                continue

            try:
                with timeout(self.timeout_second,"Action execution time exceeded!"):
                    scores = []
                    info = []

                    for func in eval_func:
                        # Capture logging.warning
                        class WarningCaptureHandler(logging.Handler):
                            def __init__(self):
                                super().__init__()
                                self.warnings = []

                            def emit(self, record):
                                if record.levelno >= logging.WARNING:
                                    self.warnings.append(self.format(record))

                        # Create handler and add to root logger
                        warning_handler = WarningCaptureHandler()
                        warning_handler.setLevel(logging.WARNING)
                        formatter = logging.Formatter('%(levelname)s - %(message)s')
                        warning_handler.setFormatter(formatter)

                        root_logger = logging.getLogger()
                        original_level = root_logger.level
                        root_logger.addHandler(warning_handler)
                        root_logger.setLevel(min(logging.WARNING, original_level))

                        try:
                            result = eval(func)
                        except FileNotFoundError as e:
                            logging.error(f"File not found! Error: {e}")
                            scores.append(0.0)
                            continue
                        finally:
                            # Remove handler and restore
                            root_logger.removeHandler(warning_handler)
                            root_logger.setLevel(original_level)

                        if isinstance(result, dict):
                            scores.append(result.get('score', 0.0))
                            # Add captured warnings to result
                            # if warning_handler.warnings:
                            #     result['warnings'] = warning_handler.warnings
                            info.append(result)
                        else:
                            scores.append(result)
                            # For non-dict results, we still want to capture warnings
                            if warning_handler.warnings:
                                result_dict = {
                                    'score': result,
                                    # 'warnings': warning_handler.warnings
                                }
                                info.append(result_dict)
            except Exception as e:
                logging.error(f"Error in task {id}: {e}")
                traceback.print_exc()
                scores.append(0.0)
                info.append({"score": 0.0, "errors": [str(e)]})
            
            scores = [score if isinstance(score, float) or isinstance(score, int) else 0.0 for score in scores]
            total_score = sum(scores) / len(scores)
            eval_results.append({"id": id, "total_score": total_score, **trajectory_info,
                                  'info': info, 'domain': domain})
        return eval_results