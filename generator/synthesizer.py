"""
Main Synthesizer

Integrated workflow with PathSampler:
1. PathSampler samples a skill path
2. Synthesizer attempts to generate an evaluation case
3. On success: tell PathSampler which skills were actually used (penalize them)
4. On failure: don't penalize unused skills
5. PathSampler re-samples with adjusted weights
6. Continue until max_cases reached

Steps within each synthesis:
[Step 1] Exploring Domain - load and analyze dataset files
[Step 2] Building Skill Tree - LLM selects main skill based on dataset context
[Step 3] Generating Solution Steps (with Integrated Applicability Check)
        - Generate main step → Check applicability → (Regenerate if needed)
        - For each auxiliary skill: Generate → Check → Handle failures
[Step 4] Synthesizing Question - based on main skill and answer type
[Step 5] Validating Question - checklist scoring
[Step 6] Evaluating Difficulty - detect revealed skills
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Any, Tuple

from data_classes import Skill, SkillTree, SolutionStep, SolutionPath, EvalCase, DomainDataset
from dataset_loader import (
    DatasetLoader,
    METADATA_FILES,
    build_files_summary,
    explore_domain_dataset,
    format_file_info_line,
)
from skill_tree import SkillTreeBuilder
from step_generator import StepGenerator
from question_validator import QuestionValidator, print_validation_result
from sample_graph import PathSampler
import sys
sys.path.append('..')
from utils.config import DATASETS_DIR, OUTPUT_DIR, QWEN_MODEL, MAX_FILES_PER_CASE, MIN_STEPS_PER_CASE
from utils.llm_client import QwenClient

class EvalCaseSynthesizer:
    """Main synthesizer class"""
    
    def __init__(
        self,
        llm_client: QwenClient,
        min_files: int = 5,
    ):
        self.llm = llm_client
        self.min_files = min_files
        self.skill_tree_builder = SkillTreeBuilder(llm_client)
        
        # Question validator
        self.question_validator = QuestionValidator(llm_client, min_files=min_files)
        
    @staticmethod
    def _load_domain_connections(domain: str) -> Optional[str]:
        """Load domain_connections.txt for a domain if it exists."""
        conn_path = os.path.join(DATASETS_DIR, domain, "domain_connections.txt")
        if os.path.exists(conn_path):
            with open(conn_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    @staticmethod
    def _extract_referenced_files(question: str, files_info: List[Dict]) -> List[str]:
        """
        Extract file names from question text by matching against known file pool.
        Matches on both full path and basename (case-insensitive).
        Excludes internal metadata files (e.g. domain_connections.txt).
        """
        referenced = []
        question_lower = question.lower()
        for f in files_info:
            name = f['name']
            basename = os.path.basename(name)
            if basename in METADATA_FILES:
                continue
            if basename.lower() in question_lower or name.lower() in question_lower:
                referenced.append(name)
        return referenced

    # Keyword groups → human-readable target format label
    _INFRA_FORMAT_MAP = [
        (['sqlite', 'CREATE TABLE', 'to_sql', 'conn.cursor'],
         'a SQLite database'),
        (['sqlalchemy', 'engine.connect', 'create_engine'],
         'a SQL database (via SQLAlchemy)'),
        (['CREATE DATABASE', 'INSERT INTO', '.execute('],
         'a SQL database'),
    ]

    @staticmethod
    def _detect_infra_adaptations(solution_path: SolutionPath) -> str:
        """Scan pipeline steps for infrastructure adaptation patterns.

        Returns a prompt section if any step converts raw files into a different
        format (e.g. CSV -> SQLite) so the question can naturally incorporate the
        conversion requirement.  Returns empty string when no adaptation found.
        """
        adaptations = []
        detected_formats: list[str] = []

        for step in solution_path.steps:
            text = (step.description or '') + '\n' + (step.code_snippet or '')
            text_lower = text.lower()

            for keywords, fmt_label in EvalCaseSynthesizer._INFRA_FORMAT_MAP:
                if any(kw.lower() in text_lower for kw in keywords):
                    adaptations.append(
                        f"- Step {step.step_number} [{step.skill.name}]: "
                        f"{step.description}"
                    )
                    if fmt_label not in detected_formats:
                        detected_formats.append(fmt_label)
                    break

        if not adaptations:
            return ""

        fmt_str = detected_formats[0] if len(detected_formats) == 1 \
            else " / ".join(detected_formats)

        example_phrases = {
            'a SQLite database': (
                '"…load the CSV files into a SQLite database and use SQL '
                'queries to analyse…"'
            ),
            'a SQL database (via SQLAlchemy)': (
                '"…ingest the data into a SQL database through SQLAlchemy '
                'and perform the analysis with SQL…"'
            ),
            'a SQL database': (
                '"…store the raw data in a SQL database before running the '
                'aggregation queries…"'
            ),
        }
        embed_example = example_phrases.get(
            detected_formats[0],
            f'"…convert the data files into {fmt_str} for the analysis…"',
        )

        return (
            "\n## Infrastructure Requirement "
            "(embed naturally in the task description)\n"
            "The solution pipeline converts raw data files into "
            f"{fmt_str} for analysis.\n\n"
            "Detected adaptations in the pipeline:\n"
            + "\n".join(adaptations)
            + "\n\n"
            f"Natural embedding example: {embed_example}\n\n"
            "Mention the target infrastructure as part of the analytical "
            "workflow — do NOT present it as a separate conversion "
            "instruction.  NEVER use vague phrases like \"a suitable "
            "format\", \"an appropriate format\", or \"a format suitable "
            "for analysis\".\n"
        )

    _COMPLEX_ALGORITHM_PATTERNS = [
        re.compile(r'(?i)\b(?:random\s*forest|decision\s*tree|gradient\s*boost(?:ing)?|xgboost|lgbm|lightgbm|catboost)\b'),
        re.compile(r'(?i)\b(?:linear\s*regression|logistic\s*regression|ridge|lasso|elastic\s*net)\b'),
        re.compile(r'(?i)\b(?:svm|svr|svc|support\s*vector)\b'),
        re.compile(r'(?i)\b(?:k[\s-]?means|dbscan|hierarchical\s*cluster|agglomerative)\b'),
        re.compile(r'(?i)\b(?:pca|t[\s-]?sne|umap|dimensionality\s*reduction|singular\s*value\s*decomp)\b'),
        re.compile(r'(?i)\b(?:arima|sarima|prophet|exponential\s*smoothing|time\s*series\s*(?:forecast|decompos))\b'),
        re.compile(r'(?i)\b(?:neural\s*network|mlp|(?:deep\s*learn)|cnn|rnn|lstm|gru|transformer|bert|gpt)\b'),
        re.compile(r'(?i)\b(?:tf[\s-]?idf|word2vec|fasttext|word\s*embedding|sentiment\s*analy)\b'),
        re.compile(r'(?i)\b(?:hypothesis\s*test|chi[\s-]?square|t[\s-]?test|anova|mann[\s-]?whitney|kolmogorov|shapiro|kruskal)\b'),
        re.compile(r'(?i)\b(?:cross[\s-]?valid|k[\s-]?fold|bootstrap|bagging|boosting|stacking|ensemble)\b'),
        re.compile(r'(?i)\b(?:markov|bayesian|monte\s*carlo|expectation[\s-]?maximization)\b'),
        re.compile(r'(?i)\b(?:interpolat|extrapolat|spline|kernel\s*density|kde)\b'),
        re.compile(r'(?i)\b(?:dax|power\s*bi|olap|mdx|tabular\s*model)\b'),
        re.compile(r'(?i)\b(?:fourier|wavelet|signal\s*process|spectral\s*analy)\b'),
        re.compile(r'(?i)\b(?:anomaly\s*detect|outlier\s*detect|isolation\s*forest|one[\s-]?class\s*svm)\b'),
        re.compile(r'(?i)\b(?:association\s*rule|apriori|fp[\s-]?growth|market\s*basket)\b'),
        re.compile(r'(?i)\b(?:network\s*analy|graph\s*algorithm|pagerank|centrality|community\s*detect)\b'),
        re.compile(r'(?i)\b(?:survival\s*analy|kaplan[\s-]?meier|cox\s*regression|hazard)\b'),
        re.compile(r'(?i)\b(?:genetic\s*algorithm|evolutionary|simulated\s*anneal|particle\s*swarm)\b'),
        re.compile(r'(?i)\b(?:silhouette|elbow\s*method|davies[\s-]?bouldin|calinski)\b'),
        re.compile(r'(?i)\b(?:granger\s*causality|cointegrat|augmented\s*dickey|stationarity)\b'),
        re.compile(r'(?i)\b(?:hidden\s*markov|viterbi|em\s*algorithm)\b'),
        re.compile(r'(?i)\b(?:collaborative\s*filter|matrix\s*factoriz|recommender|content[\s-]?based\s*filter)\b'),
        re.compile(r'(?i)\b(?:naive\s*bayes|gaussian\s*process|kernel\s*regression)\b'),
    ]

    @staticmethod
    def _is_complex_algorithm(skill_name: str, skill_description: str, code_snippet: str) -> bool:
        """Check if a step involves a complex algorithm that warrants an explanation."""
        combined = f"{skill_name} {skill_description or ''} {code_snippet or ''}"
        return any(pat.search(combined) for pat in EvalCaseSynthesizer._COMPLEX_ALGORITHM_PATTERNS)

    def _generate_algorithm_explanations(
        self,
        solution_path: SolutionPath,
        rare_skill_names: List[str] = None,
    ) -> Dict[int, str]:
        """Generate annotator-friendly algorithm explanations for complex/rare steps.

        Returns a dict mapping step_number -> explanation string.
        Explanations are generated in a single batched LLM call for efficiency.
        """
        rare_set = set(rare_skill_names or [])
        steps_needing_explanation = []

        for step in solution_path.steps:
            is_rare = step.skill.name in rare_set
            is_complex = self._is_complex_algorithm(
                step.skill.name,
                step.skill.description or '',
                step.code_snippet or '',
            )
            if is_rare or is_complex:
                steps_needing_explanation.append(step)

        if not steps_needing_explanation:
            return {}

        steps_block = "\n\n".join(
            f"### Step {step.step_number}: [{step.skill.name}]\n"
            f"- Skill definition: {step.skill.description or 'N/A'}\n"
            f"- Step description: {step.description}\n"
            f"- Code snippet:\n```python\n{(step.code_snippet or '')[:800]}\n```"
            for step in steps_needing_explanation
        )
        step_ids = [s.step_number for s in steps_needing_explanation]

        prompt = f"""You are a senior data scientist. The following data analysis pipeline contains
steps that involve complex algorithms or rare techniques that annotators may not
be familiar with. Please provide a concise explanation for each step to help
annotators understand.

{steps_block}

## Requirements
Write a short explanation (100-200 words each) for each step above, covering:
1. What algorithm/technique does this step use? Explain its principle in plain language
2. The meaning of key parameters in the code
3. What are the inputs and outputs of this step

Return in JSON format, where the key is the step number (integer) and the value
is the corresponding explanation text:
{{
    {', '.join(f'"{sid}": "Explanation for Step {sid}"' for sid in step_ids)}
}}
"""
        result = self.llm.call_json(prompt)
        if not result:
            return {}

        explanations = {}
        for step in steps_needing_explanation:
            key_variants = [step.step_number, str(step.step_number)]
            for k in key_variants:
                if k in result:
                    explanations[step.step_number] = result[k]
                    break
        return explanations

    _ML_MODEL_PATTERNS = [
        re.compile(r'(RandomForest(?:Regressor|Classifier))\s*\(([^)]*)\)'),
        re.compile(r'(LinearRegression)\s*\(([^)]*)\)'),
        re.compile(r'(LogisticRegression)\s*\(([^)]*)\)'),
        re.compile(r'(DecisionTree(?:Regressor|Classifier))\s*\(([^)]*)\)'),
        re.compile(r'(GradientBoosting(?:Regressor|Classifier))\s*\(([^)]*)\)'),
        re.compile(r'(XGB(?:Regressor|Classifier))\s*\(([^)]*)\)'),
        re.compile(r'(LGBMRegressor|LGBMClassifier)\s*\(([^)]*)\)'),
        re.compile(r'(SVR|SVC)\s*\(([^)]*)\)'),
        re.compile(r'(KNeighbors(?:Regressor|Classifier))\s*\(([^)]*)\)'),
        re.compile(r'(Ridge|Lasso|ElasticNet)\s*\(([^)]*)\)'),
        re.compile(r'(MLPRegressor|MLPClassifier)\s*\(([^)]*)\)'),
    ]

    _SPLIT_PATTERN = re.compile(
        r'train_test_split\s*\([^)]*?test_size\s*=\s*([\d.]+)[^)]*?random_state\s*=\s*(\d+)'
    )

    _SEED_PATTERNS = [
        re.compile(r'np\.random\.seed\s*\(\s*(\d+)\s*\)'),
        re.compile(r'random\.seed\s*\(\s*(\d+)\s*\)'),
        re.compile(r'torch\.manual_seed\s*\(\s*(\d+)\s*\)'),
    ]

    @staticmethod
    def _detect_ml_specifications(solution_path: SolutionPath) -> str:
        """Scan pipeline steps for ML model training patterns.

        Extracts model types with hyperparameters, train/test split params,
        and random seeds from ``code_snippet`` and ``description`` fields.
        Returns a prompt section instructing the LLM to include these
        specifications in the generated question, or an empty string when
        no ML training pattern is detected.
        """
        models: list[dict] = []
        splits: list[dict] = []
        seeds: list[str] = []

        for step in solution_path.steps:
            text = (step.description or '') + '\n' + (step.code_snippet or '')

            for pat in EvalCaseSynthesizer._ML_MODEL_PATTERNS:
                for m in pat.finditer(text):
                    model_name = m.group(1)
                    params_str = m.group(2).strip()
                    models.append({"name": model_name, "params": params_str})

            for m in EvalCaseSynthesizer._SPLIT_PATTERN.finditer(text):
                splits.append({
                    "test_size": m.group(1),
                    "random_state": m.group(2),
                })

            for pat in EvalCaseSynthesizer._SEED_PATTERNS:
                for m in pat.finditer(text):
                    seed_expr = m.group(0)
                    if seed_expr not in seeds:
                        seeds.append(seed_expr)

        if not models and not splits and not seeds:
            return ""

        lines: list[str] = []
        lines.append(
            "\n## ML Model Specification "
            "(embed naturally in the task description)"
        )
        lines.append(
            "The solution pipeline relies on specific ML models and "
            "parameters.  To keep the answer deterministic and reproducible, "
            "every specification listed below MUST appear in your question — "
            "but woven into a sentence that describes the analytical step it "
            "belongs to, NOT as a bullet-point dump or parenthetical list."
        )

        if models:
            seen = set()
            for md in models:
                key = f"{md['name']}({md['params']})"
                if key in seen:
                    continue
                seen.add(key)
                param_hint = f" with {md['params']}" if md['params'] else ""
                lines.append(
                    f"- Specification: {md['name']}({md['params']})\n"
                    f"  Natural embedding example: \"…fit a {md['name']}"
                    f"{param_hint} to predict …\""
                )

        if splits:
            for sp in splits:
                pct = int(float(sp['test_size']) * 100)
                lines.append(
                    f"- Specification: test_size={sp['test_size']}, "
                    f"random_state={sp['random_state']}\n"
                    f"  Natural embedding example: \"…reserving {pct}% of the "
                    f"data for testing (random_state={sp['random_state']}) …\""
                )

        if seeds:
            for s in seeds:
                seed_val = re.search(r'\d+', s)
                val = seed_val.group(0) if seed_val else '42'
                lines.append(
                    f"- Specification: {s}\n"
                    f"  Natural embedding example: \"…set the random seed to "
                    f"{val} for reproducibility …\""
                )

        lines.append(
            "\nDo NOT list these as a parenthetical dump or bullet points. "
            "Weave each specification into a sentence that describes the "
            "analytical step it belongs to.\n"
            'NEVER say "build a model" or "train a model" without specifying '
            "which model and its parameters.\n"
        )
        return "\n".join(lines)

    _TOOL_REQUIREMENT_MAP = [
        {
            "pattern": re.compile(r"DAX|Power\s*BI", re.IGNORECASE),
            "constraint_template": (
                "The question MUST require the solver to write a DAX expression "
                "or use Power BI as the analysis tool."
            ),
            "question_hint": (
                '"…create a DAX measure in Power BI that computes the '
                'year-over-year growth rate for each product category…"'
            ),
        },
        {
            "pattern": re.compile(
                r"Version\s*Control|Dependency\s*Management|Package\s*Install",
                re.IGNORECASE,
            ),
            "constraint_template": (
                "The question MUST instruct the solver to install a specific "
                "Python package (pip install …) before performing the analysis."
            ),
            "question_hint": (
                '"…after installing the <package> library, use it to parse '
                'the raw log files and extract…"'
            ),
        },
        {
            "pattern": re.compile(r"\bSQL\b|Query\s*Execution", re.IGNORECASE),
            "constraint_template": (
                "The question MUST require the solver to use SQL queries to "
                "retrieve or transform the data."
            ),
            "question_hint": (
                '"…query the data using SQL to retrieve all customers whose '
                'total spend exceeds the regional median…"'
            ),
        },
        {
            "pattern": re.compile(
                r"Histogram|Plot|Visualization|Bar\s*Chart|Scatter|Heatmap",
                re.IGNORECASE,
            ),
            "constraint_template": (
                "The question MUST ask the solver to generate a specific "
                "visualization (chart/plot) and report a concrete value "
                "readable from the chart."
            ),
            "question_hint": (
                '"…produce a histogram of delivery times and report which '
                'bin has the highest frequency…"'
            ),
        },
        {
            "pattern": re.compile(r"Clustering|K-?Means|DBSCAN|Hierarchical", re.IGNORECASE),
            "constraint_template": (
                "The question MUST specify which clustering algorithm to use "
                "and any key parameters (e.g., number of clusters)."
            ),
            "question_hint": (
                '"…identify optimal customer segments via K-Means clustering, '
                'testing k from 2 to 10 and selecting by Silhouette Score…"'
            ),
        },
        {
            "pattern": re.compile(r"Time\s*Series|ARIMA|Forecasting|Seasonal", re.IGNORECASE),
            "constraint_template": (
                "The question MUST require the solver to apply a specific "
                "time-series analysis or forecasting method."
            ),
            "question_hint": (
                '"…apply an ARIMA model to forecast the next 12 months of '
                'monthly revenue, reporting the predicted total…"'
            ),
        },
        {
            "pattern": re.compile(
                r"Text\s*Preprocessing|Tokeniz|NLP|Sentiment|TF-?IDF",
                re.IGNORECASE,
            ),
            "constraint_template": (
                "The question MUST require the solver to perform text "
                "preprocessing or NLP analysis on the data."
            ),
            "question_hint": (
                '"…tokenize the customer reviews, compute TF-IDF scores, and '
                'identify the top 10 terms most associated with negative '
                'sentiment…"'
            ),
        },
        {
            "pattern": re.compile(r"Transformer|BERT|GPT|LLM|Hugging\s*Face", re.IGNORECASE),
            "constraint_template": (
                "The question MUST require the solver to use a Transformer-based "
                "model for the analysis."
            ),
            "question_hint": (
                '"…use a pre-trained BERT model to classify each support '
                'ticket into one of the predefined categories and report '
                'the macro-F1 score…"'
            ),
        },
    ]

    @staticmethod
    def _detect_joinable_keys(files_info: List[Dict]) -> str:
        """Detect joinable keys between selected files.

        Analyzes column names across all selected files and identifies
        shared columns that serve as potential join keys between file pairs.
        Returns a prompt section instructing the LLM to embed join key hints
        in the question text so annotators can understand the data relationships.
        """
        file_columns: Dict[str, set] = {}
        for f in files_info:
            name = os.path.basename(f['name'])
            ft = f.get('file_type', '')
            cols: set = set()
            if ft in ('csv', 'excel', 'tabular', 'parquet', 'dbf'):
                cols = set(f.get('columns', []))
            elif ft in ('json', 'yaml'):
                cols = set(f.get('top_level_keys', []))
            elif ft == 'sqlite':
                for table in f.get('tables', []):
                    for col_info in table.get('columns', []):
                        col_name = col_info.get('name', '') if isinstance(col_info, dict) else str(col_info)
                        if col_name:
                            cols.add(col_name)
            elif ft == 'geo':
                cols = set(f.get('columns', []))
            if cols:
                file_columns[name] = cols

        if len(file_columns) < 2:
            return ""

        join_pairs: list[tuple] = []
        file_names = sorted(file_columns.keys())
        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                f1, f2 = file_names[i], file_names[j]
                shared = file_columns[f1] & file_columns[f2]
                if shared:
                    join_pairs.append((f1, f2, sorted(shared)))

        if not join_pairs:
            return ""

        lines = [
            "\n## Join Key Hints (MUST append as the LAST sentence of the question)",
            "The following file pairs share common columns that can be used as join keys.",
            "You MUST append a closing sentence at the very end of the question that",
            "summarises which files can be linked and on which columns, so annotators",
            "can quickly understand the data relationships.\n",
        ]
        for f1, f2, shared_cols in join_pairs:
            if len(shared_cols) <= 5:
                cols_str = ", ".join(f'"{c}"' for c in shared_cols)
            else:
                cols_str = ", ".join(f'"{c}"' for c in shared_cols[:5])
                cols_str += f" ... ({len(shared_cols)} columns total)"
            lines.append(f"- {f1} ↔ {f2}: can be joined on {cols_str}")

        lines.append(
            "\nFormatting requirement: add the join-key hint as the LAST sentence of "
            "the question (after the main analytical task). Use a pattern like:\n"
            '"Hint: players.csv and teams.csv can be linked via the "team_id" column; '
            'teams.csv and stadiums.csv can be linked via the "stadium_id" column."\n'
            "Do NOT scatter join-key hints throughout the question body — keep them "
            "all together in this single closing sentence.\n"
        )

        return "\n".join(lines)

    @staticmethod
    def _detect_tool_requirements(solution_path: SolutionPath) -> str:
        """Scan skill names in *solution_path* for tool/platform constraints.

        Returns a prompt section that instructs the LLM to mention the
        detected tool requirements explicitly in the generated question.
        Returns an empty string when no tool-type skill is found.
        """
        constraints: list[dict] = []
        seen_patterns: set[int] = set()

        for step in solution_path.steps:
            skill_name = step.skill.name or ""
            step_text = skill_name + " " + (step.description or "")
            for idx, mapping in enumerate(EvalCaseSynthesizer._TOOL_REQUIREMENT_MAP):
                if idx in seen_patterns:
                    continue
                if mapping["pattern"].search(step_text):
                    seen_patterns.add(idx)
                    constraints.append({
                        "step_number": step.step_number,
                        "skill_name": skill_name,
                        "constraint": mapping["constraint_template"],
                        "hint": mapping["question_hint"],
                    })

        if not constraints:
            return ""

        lines: list[str] = [
            "\n## Tool Requirements (MUST include in the question)",
            "The solution pipeline uses specific tools/platforms. Your question "
            "MUST explicitly require the solver to use these tools so the "
            "question cannot be answered without them:",
        ]
        for c in constraints:
            lines.append(
                f"- Step {c['step_number']} [{c['skill_name']}]: "
                f"{c['constraint']}"
            )
            lines.append(f"  Hint: {c['hint']}")

        lines.append(
            "\nState the tool/method as part of the analytical task, not as a "
            "standalone instruction. The question should read as if the "
            "requester already knows which tool they want, not as if they are "
            "writing a recipe.\n"
        )
        return "\n".join(lines)

    def synthesize_question(
        self,
        solution_path: SolutionPath,
        exploration: Dict,
        answer_type: str,
        suggestions: List[str] = None,
        min_files: int = None,
        required_files: List[str] = None,
    ) -> Optional[str]:
        """
        Synthesize a concise evaluation question.
        """
        effective_min = min_files if min_files is not None else self.min_files
        main_step = solution_path.steps[-1]
        main_skill = main_step.skill
        answer_desc = getattr(main_step, '_answer_description', '')
        file_names = [f['name'] for f in exploration['files_info']]
        
        # Build file summary
        files_summary = build_files_summary(exploration['files_info'], detailed=False)
        
        # Build explicit file name list from code_snippets (verified usage only)
        all_known = [f['name'] for f in exploration['files_info']]
        verified_files: set = set()
        for step in solution_path.steps:
            verified_files.update(
                StepGenerator._extract_files_from_code(step.code_snippet or '', all_known)
            )
        if not verified_files:
            verified_files = set(file_names)

        # Force-add required files so the question prompt always references them
        if required_files:
            known_set = set(all_known)
            verified_files.update(f for f in required_files if f in known_set)

        verified_basenames = sorted(set(os.path.basename(n) for n in verified_files))
        required_files_section = (
            "## Required File References\n"
            "Your question MUST explicitly mention these file names (use the short names below).\n"
            "These are the files actually loaded in the solution code:\n"
            + "\n".join(f"- {bn}" for bn in verified_basenames)
            + "\n\nDo NOT use vague references like 'the given datasets' or 'the data files'. "
            "Each file must be mentioned by its exact name so the solver knows which files to load.\n"
        )
        file_basenames = verified_basenames
        
        # Build solution steps description (include code_snippet for context)
        steps_parts = []
        for i, step in enumerate(solution_path.steps):
            part = f"{i+1}. [{step.skill.name}]: {step.description}"
            if step.code_snippet:
                part += f"\n   Code:\n{step.code_snippet}"
            steps_parts.append(part)
        steps_desc = "\n\n".join(steps_parts)
        
        # Build skills list
        auxiliary_skills = [step.skill.name for step in solution_path.steps[:-1]]
        skills_section = f"- Main skill: {main_skill.name}"
        if auxiliary_skills:
            skills_section += f"\n- Auxiliary skills: {', '.join(auxiliary_skills)}"
        
        # Build suggestions section for retry
        suggestions_section = ""
        if suggestions:
            suggestions_text = "\n".join(f"- {s}" for s in suggestions[:3])
            suggestions_section = f"\n## Previous Attempt Failed - Please Fix These Issues\n{suggestions_text}\n"
        
        domain_conn_section = ""
        if hasattr(self, '_current_domain_connections') and self._current_domain_connections:
            domain_conn_section = (
                "\n## Domain Data Connection Guide (for your reference — DO NOT reveal in question)\n"
                + self._current_domain_connections + "\n"
            )

        infra_adaptation_section = self._detect_infra_adaptations(solution_path)
        ml_spec_section = self._detect_ml_specifications(solution_path)
        tool_req_section = self._detect_tool_requirements(solution_path)
        join_key_section = self._detect_joinable_keys(exploration['files_info'])

        if join_key_section:
            join_key_rule = (
                "9. **Append a join-key hint as the LAST sentence of the question.** "
                "After the main analytical task, add one closing sentence that lists "
                "which files can be linked and on which columns (see \"Join Key Hints\" "
                "section above). Example ending: 'Hint: players.csv and teams.csv can "
                "be linked via the \"team_id\" column; teams.csv and stadiums.csv can "
                "be linked via the \"stadium_id\" column.'"
            )
            join_key_prohibition = ""
        else:
            join_key_rule = ""
            join_key_prohibition = (
                "- Mention shared columns, join keys, or how to align/merge files "
                "(e.g., do NOT say \"use the column 'Year' to merge/align the data\") "
                "— the agent should discover cross-file relationships on its own\n"
            )

        prompt = f"""You are a data science expert creating evaluation questions for testing AI agents.

## Domain: {exploration['domain']}

{required_files_section}

## Files (detailed)
{files_summary}

{domain_conn_section}
## Solution Steps (these steps solve the problem - DO NOT reveal them in the question)
{steps_desc}

## Skills Used (DO NOT mention these by name in the question)
{skills_section}
Note: Any tool/platform requirements listed separately in "Tool Requirements" below MUST be mentioned in the question. Only hide the skill *names* listed above.

## Final Output
- What the analysis produces: {main_step.output_description}
- Answer type: {answer_type}
- Answer description: {answer_desc}
{suggestions_section}{infra_adaptation_section}{ml_spec_section}{tool_req_section}{join_key_section}
## Rules for creating good evaluation questions

### The question MUST:
1. Specify concrete conditions with exact values (e.g., "released in 1995", "budget over 20,000,000", "studied at least 6 hours")
2. Specify exactly what to compute or output (e.g., "calculate the average rating", "count the number of...", "report the R-squared value rounded to 4 decimal places")
3. Produce deterministic, verifiable answer - not an open-ended analysis
4. **Explicitly name the data files the solver needs** — use exact file names like "{file_basenames[0] if file_basenames else 'example.csv'}" in the question text. NEVER use vague phrases like "the given datasets", "the provided data", or "the data files". The question must reference at least {effective_min} files by name. (Available files: {', '.join(file_basenames)})
5. Connect files using shared columns, domain relationships, or comparison logic
6. Prefer questions that require the agent to discover and leverage cross-file relationships
7. The question should be DIFFICULT but grounded in a realistic analytical scenario — frame the question as a plausible business or research task that a data analyst might actually encounter. Increase complexity through multi-condition filters, cross-file integration, and multi-step computations rather than by piling on unrelated techniques. The question should read like a challenging real-world assignment, not an artificial stress test
8. Begin the question with a brief (1-2 sentence) real-world scenario that establishes WHO needs the analysis and WHY (e.g., "A regional agriculture bureau is evaluating crop suitability across provinces to optimize subsidy allocation. Using …"). The scenario must feel natural — not a contrived wrapper — and should logically motivate the analytical task that follows
{join_key_rule}
### The question MUST NOT:
- Use vague language like "analyze", "explore", "summarize findings", "investigate factors"
- Use generic file references like "the given datasets" or "the data" — always name specific files
- Ask for open-ended analysis, recommendations, or subjective interpretation
- Reveal intermediate steps
{join_key_prohibition}
Respond in JSON format only:
{{
    "question": "the evaluation question — must open with the scenario, then state the analytical task with explicit file names",
    "files_referenced": ["list of file names explicitly mentioned in the question"],
    "expected_answer_type": "{answer_type}",
    "answer_hint": "the exact expected answer or format"
}}
"""
        
        result = self.llm.call_json(prompt)
        
        if result and 'question' in result:
            return result['question']
        
        return f"Analyze the data in {file_names[0]} and report the key findings."
    
    def synthesize(
        self,
        sampled_path,  # SampledPath object with skills and examples
        domain_dataset: DomainDataset,
        min_skills: int = 2,
        target_steps: int = None,
        min_files: int = None,
        rare_skill_names: List[str] = None,
        required_files: List[str] = None,
    ) -> Tuple[Optional[EvalCase], List[str], Dict[str, Any]]:
        """
        Main synthesis algorithm with dynamic skill selection.

        min_files: if set, overrides self.min_files for this call (e.g. per-domain random).
        required_files: list of file names (relative paths within domain) that MUST
            appear in the generated eval case's data_sources and solution pipeline.
        
        Returns:
            Tuple of (EvalCase or None, used_skill_ids, synthesis_log)
            synthesis_log is always returned (even on failure) for diagnostics.
        """
        skills = sampled_path.skills
        synthesis_log: Dict[str, Any] = {
            "sampled_skills": [{"id": s.id, "name": s.name} for s in skills],
        }
        
        # ==================== [Step 1] Exploring Domain ====================
        print(f"\n{'='*60}")
        print(f"[Step 1] Exploring Domain: {domain_dataset.domain}")
        print(f"{'='*60}")
        
        exploration = explore_domain_dataset(domain_dataset)
        
        # Save the full files_info before Step 3 filters it, so we can
        # match file references in the question against the complete pool.
        self._full_exploration_files_info = list(exploration['files_info'])
        
        base_min = min_files if min_files is not None else self.min_files
        effective_min_files = min(
            base_min,
            len(exploration['files_info']),
            MAX_FILES_PER_CASE
        )
        
        self.question_validator.min_files = effective_min_files
        self.question_validator._build_criteria()
        
        for f in exploration['files_info']:
            print(f"    {format_file_info_line(f)}")
        file_types = exploration.get('file_types', {})
        tabular_count = file_types.get('csv', 0) + file_types.get('excel', 0)
        if tabular_count > 0:
            print(f"  Total rows (tabular): {exploration['total_rows']}")
            print(f"  Has missing values: {exploration.get('has_missing_values', False)}")
        if exploration.get('has_non_tabular'):
            non_tabular = {k: v for k, v in file_types.items() if k not in ('csv', 'excel')}
            print(f"  Non-tabular files: {non_tabular}")
        subfolders = exploration.get('subfolders', {})
        for folder, info in subfolders.items():
            label = folder if folder else "(root)"
            print(f"  Subfolder [{label}]: {info['files']}")
        
        synthesis_log["exploration"] = {
            "domain": exploration['domain'],
            "num_files": exploration['num_files'],
            "file_types": file_types,
            "total_rows": exploration.get('total_rows', 0),
            "has_missing_values": exploration.get('has_missing_values', False),
            "files": [f['name'] for f in exploration['files_info']],
            "effective_min_files": effective_min_files,
        }
        
        # Validate required_files exist in this domain
        if required_files:
            known_files = {f['name'] for f in exploration['files_info']}
            invalid_required = [f for f in required_files if f not in known_files]
            if invalid_required:
                print(f"  [Warning] Required files not found in domain: {invalid_required}")
                print(f"  Available files: {sorted(known_files)}")
                required_files = [f for f in required_files if f in known_files]
                if not required_files:
                    print(f"  [Error] No valid required files remain, aborting")
                    synthesis_log["outcome"] = "failed"
                    synthesis_log["failure_reason"] = "required_files_not_found"
                    return None, [], synthesis_log
            print(f"  Required files: {required_files}")
            synthesis_log["required_files"] = required_files
        
        # ==================== [Step 2] Building Skill Tree ====================
        print(f"\n{'='*60}")
        print(f"[Step 2] Building Skill Tree ({len(skills)} skills)")
        print(f"{'='*60}")
        
        skill_tree = self.skill_tree_builder.build_skill_tree(skills, exploration=exploration)
        
        if not skill_tree.main_skill:
            print("  [Error] Failed to determine main skill")
            synthesis_log["skill_tree"] = {"error": "failed_to_determine_main_skill"}
            synthesis_log["outcome"] = "failed"
            synthesis_log["failure_reason"] = "no_main_skill"
            return None, [], synthesis_log
        
        print(f"  Main skill (selected by LLM): {skill_tree.main_skill.name}")
        print(f"  Auxiliary skills ({len(skill_tree.auxiliary_skills)}):")
        for s in skill_tree.auxiliary_skills:
            print(f"    - {s.name}")
        
        if sampled_path.examples:
            print(f"  StackOverflow examples: {len(sampled_path.examples)} available")
            for ex in sampled_path.examples[:3]:
                print(f"    - {ex.id}: covers {', '.join(ex.covered_skills[:2])}...")

        synthesis_log["skill_tree"] = {
            "main_skill": {"id": skill_tree.main_skill.id, "name": skill_tree.main_skill.name},
            "auxiliary_skills": [
                {"id": s.id, "name": s.name} for s in skill_tree.auxiliary_skills
            ],
            "num_examples": len(sampled_path.examples) if sampled_path.examples else 0,
        }

        # ==================== [Step 3] Generate Steps with Integrated Applicability Check ====================
        print(f"\n{'='*60}")
        print(f"[Step 3] Generating Solution Steps (with Applicability Check)")
        print(f"{'='*60}")
        
        self._current_domain_connections = self._load_domain_connections(domain_dataset.domain)
        if self._current_domain_connections:
            print(f"  Loaded domain connection guide ({len(self._current_domain_connections)} chars)")
        domain_connections = self._current_domain_connections
        
        step_generator = StepGenerator(
            self.llm,
            sampled_path=sampled_path,
            min_files=effective_min_files,
            full_exploration=exploration,
            domain_connections=domain_connections,
            rare_skill_names=rare_skill_names,
            required_files=required_files,
        )
        
        solution_path, answer_type, skipped_skills, blacklisted_skills = step_generator.create_solution_path(
            main_skill=skill_tree.main_skill,
            auxiliary_skills=skill_tree.auxiliary_skills,
            exploration=exploration,
            target_steps=target_steps,
        )
        
        synthesis_log["step_generation"] = {
            "trace": step_generator.trace,
            "answer_type": answer_type,
        }
        
        # Collect final selected files from incremental accumulation
        exploration['selected_files'] = list(step_generator._selected_files)
        exploration['files_info'] = [
            f for f in exploration['files_info']
            if f['name'] in step_generator._selected_files
        ]
        
        # Check if we got enough steps
        if not solution_path.steps:
            print(f"\n  [Error] Failed to generate any valid steps")
            if blacklisted_skills:
                print(f"  Blacklisted skills: {[s.name for s in blacklisted_skills]}")
            synthesis_log["outcome"] = "failed"
            synthesis_log["failure_reason"] = "no_valid_steps"
            return None, [], synthesis_log
        
        if len(solution_path.steps) < MIN_STEPS_PER_CASE:
            print(f"\n  [Error] Only {len(solution_path.steps)} steps generated (minimum {MIN_STEPS_PER_CASE}), aborting")
            synthesis_log["outcome"] = "failed"
            synthesis_log["failure_reason"] = f"too_few_steps ({len(solution_path.steps)} < {MIN_STEPS_PER_CASE})"
            return None, [], synthesis_log
        
        print(f"\n  Answer type: {answer_type}")
        print(f"  Generated {len(solution_path.steps)} steps:")
        for step in solution_path.steps:
            marker = "★" if step.skill.id == skill_tree.main_skill.id else "○"
            print(f"    {step.step_number}. {marker} [{step.skill.name}]")
            desc = step.description[:70] + "..." if len(step.description) > 70 else step.description
            print(f"       {desc}")
        
        if skipped_skills:
            print(f"  Skipped skills ({len(skipped_skills)}):")
            for s in skipped_skills:
                print(f"    - {s.name}")
        
        if blacklisted_skills:
            print(f"  Blacklisted skills ({len(blacklisted_skills)}):")
            for s in blacklisted_skills:
                print(f"    - {s.name}")
        
        # ==================== [Step 4] Synthesizing Question ====================
        print(f"\n{'='*60}")
        print(f"[Step 4] Synthesizing Question")
        print(f"{'='*60}")
        
        question = self.synthesize_question(
            solution_path, exploration, answer_type,
            min_files=effective_min_files, required_files=required_files,
        )
        print(f"  Question: {question}")
        
        synthesis_log["question_synthesis"] = {
            "initial_question": question,
        }
        
        # ==================== [Step 5] Validating Question ====================
        print(f"\n{'='*60}")
        print(f"[Step 5] Validating Question")
        print(f"{'='*60}")
        
        max_retries = 3
        
        def regenerate_question(suggestions):
            return self.synthesize_question(
                solution_path, exploration, answer_type, suggestions,
                min_files=effective_min_files, required_files=required_files,
            )
        
        validated_question, validation_result = self.question_validator.validate_with_retry(
            question=question,
            exploration=exploration,
            solution_path=solution_path,
            answer_type=answer_type,
            regenerate_fn=regenerate_question,
            max_retries=max_retries,
            rare_skill_names=rare_skill_names,
        )
        
        print_validation_result(validation_result)
        
        synthesis_log["question_validation"] = {
            "passed": validation_result['passed'],
            "final_score": validation_result['score'],
            "max_score": validation_result['max_score'],
            "attempt_history": validation_result.get('attempt_history', []),
        }
        
        if validated_question is None:
            print(f"\n  [Warning] Question validation failed after {max_retries} retries")
            print(f"  This skill sequence may not be suitable for generating quality questions")
            synthesis_log["outcome"] = "failed"
            synthesis_log["failure_reason"] = "question_validation_failed"
            return None, [], synthesis_log
        
        question = validated_question
        print(f"  Final question: {question}")
        
        # ==================== [Step 5.5] Reconcile data_sources with question ====================
        # Extract files actually referenced in the question text
        all_files_info = self._full_exploration_files_info or exploration['files_info']
        question_referenced_files = self._extract_referenced_files(question, all_files_info)
        step_selected_files = exploration.get('selected_files', [f['name'] for f in exploration['files_info']])
        
        # Filter out ghost file names (LLM-fabricated intermediates not in the actual file pool)
        real_file_names = {f['name'] for f in all_files_info}
        step_selected_files = [f for f in step_selected_files if f in real_file_names]
        
        # Reconcile: data_sources = union of question-referenced and step-selected,
        # but prefer question-referenced as the primary source
        if question_referenced_files:
            # Files in question but not in steps — still needed by solver
            extra_in_question = [f for f in question_referenced_files if f not in step_selected_files]
            # Files in steps but not in question — may be needed as implicit dependencies
            extra_in_steps = [f for f in step_selected_files if f not in question_referenced_files]
            
            # Final data_sources: all files referenced in question + step-selected files
            reconciled_sources = list(dict.fromkeys(question_referenced_files + step_selected_files))
            
            if extra_in_question:
                print(f"  [Reconcile] Files in question but not in steps: {extra_in_question}")
            if extra_in_steps:
                print(f"  [Reconcile] Files in steps but not in question: {extra_in_steps}")
            
            synthesis_log["data_sources_reconciliation"] = {
                "question_referenced": question_referenced_files,
                "step_selected": step_selected_files,
                "extra_in_question": extra_in_question,
                "extra_in_steps": extra_in_steps,
                "final_sources": reconciled_sources,
            }
        else:
            reconciled_sources = step_selected_files
            print(f"  [Reconcile] Warning: no file names detected in question text, falling back to step-selected files")
            synthesis_log["data_sources_reconciliation"] = {
                "question_referenced": [],
                "step_selected": step_selected_files,
                "warning": "no_files_in_question",
                "final_sources": reconciled_sources,
            }
        
        # Ensure required files are in data_sources
        if required_files:
            for rf in required_files:
                if rf not in reconciled_sources:
                    reconciled_sources.append(rf)
                    print(f"  [Required] Force-added required file to data_sources: {rf}")
        
        # Collect actually used skill IDs (for PathSampler feedback)
        used_skill_ids = [step.skill.id for step in solution_path.steps]
        
        used_skills = [
            step.skill.name for step in solution_path.steps
            if step.skill.id != skill_tree.main_skill.id
        ] + [skill_tree.main_skill.name]
        
        # ==================== Generate Algorithm Explanations ====================
        algo_explanations = self._generate_algorithm_explanations(
            solution_path, rare_skill_names=rare_skill_names,
        )
        if algo_explanations:
            print(f"  Generated algorithm explanations for {len(algo_explanations)} step(s)")
            synthesis_log["algorithm_explanations"] = {
                str(k): v[:80] + "..." if len(v) > 80 else v
                for k, v in algo_explanations.items()
            }

        # ==================== Create EvalCase ====================
        synthesis_log["outcome"] = "success"

        pipeline_entries = []
        for step in solution_path.steps:
            entry = {
                "step_id": step.step_number,
                "dependent_step_ids": step.dependent_step_ids,
                "skill": step.skill.name,
                "description": step.description,
                "code_snippet": step.code_snippet or "",
            }
            if step.step_number in algo_explanations:
                entry["algorithm_explanation"] = algo_explanations[step.step_number]
            pipeline_entries.append(entry)

        eval_case = EvalCase(
            question=question,
            data_sources=reconciled_sources,
            skills=used_skills,
            domain=exploration['domain'],
            pipeline=pipeline_entries,
        )
        
        print(f"\n{'='*60}")
        print(f"[Done] Case synthesized successfully!")
        print(f"  Actually used skills: {[s.skill.name for s in solution_path.steps]}")
        print(f"{'='*60}")
        
        return eval_case, used_skill_ids, synthesis_log


def _next_round_dir(base_dir):
    """Create and return the next ``roundN`` directory under *base_dir*."""
    os.makedirs(base_dir, exist_ok=True)
    existing_nums = []
    for name in os.listdir(base_dir):
        m = re.fullmatch(r"round(\d+)", name)
        if m and os.path.isdir(os.path.join(base_dir, name)):
            existing_nums.append(int(m.group(1)))
    next_num = max(existing_nums, default=0) + 1
    round_dir = os.path.join(base_dir, f"round{next_num}")
    os.makedirs(round_dir, exist_ok=True)
    return round_dir


def _append_trace(output_dir: str, log_entry: Dict[str, Any]) -> None:
    """Append a synthesis trace entry (success or failure) to the trace JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    trace_file = os.path.join(output_dir, "synthesis_trace.jsonl")
    with open(trace_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + '\n')


def _append_domain_case(output_dir: str, domain: str, case) -> None:
    """Append an EvalCase to the domain-specific eval_cases JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"eval_cases_({domain}).jsonl")
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(case.to_dict(), ensure_ascii=False) + '\n')


def _append_domain_trace(output_dir: str, domain: str, log_entry: Dict[str, Any]) -> None:
    """Append a synthesis trace entry to the domain-specific trace JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"synthesis_trace_({domain}).jsonl")
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + '\n')


def main():
    """Main entry point - integrated with PathSampler for dynamic sampling"""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description='Synthesize evaluation cases')
    parser.add_argument('--path_record', type=str, default='../skill_cluster/data/sample_paths.jsonl',
                        help='Path to record sampled paths (default: ../skill_cluster/data/sample_paths.jsonl)')
    parser.add_argument('--datasets_dir', type=str, default=DATASETS_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--domain', type=str, default=None)
    parser.add_argument('--max_cases', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=10, help='Max total steps (skills) per path')
    parser.add_argument('--min_skills', type=int, default=2)
    parser.add_argument('--min_files', type=int, default=5,
                        help='Minimum number of files that eval cases should reference (default: 5)')
    parser.add_argument('--required_file', type=str, default=None, action='append',
                        help='File (relative path within domain) that MUST be included in '
                             'every generated eval case. Can be specified multiple times.')
    args = parser.parse_args()
    
    required_files = args.required_file  # None or list of strings
    
    print("=" * 60)
    print("Evaluation Case Synthesizer (Integrated with PathSampler)")
    print("=" * 60)
    
    # Initialize PathSampler
    print(f"\n[Init] Initializing PathSampler...")
    path_sampler = PathSampler(
        path_record_file=args.path_record,
        alpha=1.0,
        beta=1.0,
        random_prob=0.1
    )
    print(f"  Loaded {len(path_sampler.top_node_indices)} skills")
    print(f"  Loaded {len(path_sampler.example_dict)} examples")
    
    print(f"\n[Init] Initializing Synthesizer...")
    llm = QwenClient(model=QWEN_MODEL)
    loader = DatasetLoader(args.datasets_dir, llm_client=llm)
    synthesizer = EvalCaseSynthesizer(llm, min_files=args.min_files)
    
    domains = loader.get_all_domains()
    print(f"  Available domains: {domains}")
    
    # Each run writes to a new roundN directory
    round_dir = _next_round_dir(args.output_dir)
    print(f"\n[Output] Writing results to {round_dir}")

    # Main loop: sample -> synthesize -> feedback
    print(f"\n[Synthesize] Generating up to {args.max_cases} cases...")
    results = []
    attempt = 0
    max_attempts = args.max_cases * 3  # Allow some failures
    
    while len(results) < args.max_cases and attempt < max_attempts:
        attempt += 1
        print(f"\n{'='*60}")
        print(f"[Attempt {attempt}] Sampling and synthesizing (success: {len(results)}/{args.max_cases})")
        print(f"{'='*60}")
        
        # Step 1: Sample a path from PathSampler
        print(f"\n[Sample] Sampling skill path (total_steps={args.max_steps})...")
        path_nodes = path_sampler.sample(max_steps=args.max_steps - 1)
        path_ids = [n.id for n in path_nodes]
        
        print(f"  Sampled path ({len(path_nodes)} skills):")
        for n in path_nodes:
            print(f"    - {n.name}")
        
        # Step 2: Sample evidence (examples) for the path
        evidence = path_sampler.sample_evidence(path_ids, max_examples=10)
        print(f"  Sampled {len(evidence)} examples as evidence")
        
        # Step 3: Create SampledPath object for synthesizer
        # Define simple data classes inline
        class SkillExample:
            def __init__(self, id, covered_skills, steps, title="", answer=""):
                self.id = id
                self.covered_skills = covered_skills
                self.steps = steps
                self.title = title
                self.answer = answer
            
            def get_text(self):
                parts = []
                if self.title:
                    parts.append(f"**Problem**: {self.title}")
                if self.answer:
                    parts.append(f"**Solution**:\n{self.answer}")
                parts.append("**Steps**:\n" + "\n".join([f"- {s['text']}" for s in self.steps]))
                return "\n\n".join(parts)
        
        class SampledPath:
            def __init__(self, path_id, skills, examples):
                self.path_id = path_id
                self.skills = skills
                self.examples = examples
            
            def get_examples_for_skill(self, skill_name, max_examples=3):
                """Get examples that cover a specific skill."""
                relevant = [ex for ex in self.examples if skill_name in ex.covered_skills]
                result = []
                for ex in relevant[:max_examples]:
                    result.append(f"### Example {ex.id}\n{ex.get_text()}")
                return "\n\n".join(result)
            
        # Convert to Skill objects (with descriptions from skill-descriptions.jsonl), deduplicated
        skills = []
        seen_skill_ids = set()
        for node in path_nodes:
            if node.id in seen_skill_ids:
                continue
            seen_skill_ids.add(node.id)
            skill = Skill(
                id=node.id,
                name=node.name,
                description=path_sampler.get_skill_description(node.id)
            )
            skills.append(skill)
        
        # Convert evidence to SkillExample objects
        examples = []
        for ex_id, steps in evidence:
            ex = path_sampler.example_dict[ex_id]
            covered_skills = [s.name for s in path_sampler.get_example_covered_skills(ex, path_ids)]
            example = SkillExample(
                id=ex_id,
                covered_skills=covered_skills,
                steps=[{"step_id": step.id, "text": step.text} for step in steps],
                title=ex.title,
                answer=ex.answer,
            )
            examples.append(example)
        
        sampled_path = SampledPath(
            path_id=attempt,
            skills=skills,
            examples=examples,
        )
        
        # Step 4: Select domain and load dataset
        if args.domain:
            domain = args.domain
        else:
            domain = random.choice(domains)
        
        print(f"\n[Dataset] Loading domain: {domain}")
        domain_dataset = loader.load_domain(domain)
        
        if not domain_dataset or not domain_dataset.files:
            print("  Failed to load dataset, skipping...")
            continue
        
        # Step 5: Attempt synthesis
        try:
            llm.reset_token_count()
            t_start = time.time()
            
            rare_skill_names = [
                s.name for s in skills if path_sampler.is_rare_skill(s.id)
            ]

            case, used_skill_ids, syn_log = synthesizer.synthesize(
                sampled_path,
                domain_dataset,
                min_skills=args.min_skills,
                rare_skill_names=rare_skill_names or None,
                required_files=required_files,
            )
            
            elapsed = time.time() - t_start
            token_usage = llm.get_token_count()
            
            # Enrich the synthesis_log with timing and token info
            syn_log["attempt_number"] = attempt
            syn_log["elapsed_seconds"] = round(elapsed, 2)
            syn_log["token_usage"] = token_usage
            syn_log["domain"] = domain
            
            if case:
                case.synthesis_time_seconds = round(elapsed, 2)
                case.token_usage = token_usage
                results.append(case)
                print(f"\n✓ Generated case {len(results)}")
                print(f"  Time: {elapsed:.1f}s | Tokens: {token_usage['total_tokens']} "
                      f"(prompt: {token_usage['prompt_tokens']}, completion: {token_usage['completion_tokens']})")
                
                # Step 6: Tell PathSampler which skills were ACTUALLY used
                print(f"\n[Feedback] Updating PathSampler with {len(used_skill_ids)} used skills")
                
                used_evidence = []
                for ex_id, steps in evidence:
                    relevant_steps = [s for s in steps if any(skill_id in s.skills for skill_id in used_skill_ids)]
                    if relevant_steps:
                        used_evidence.append((ex_id, relevant_steps))
                
                path_sampler.add_path(used_skill_ids, used_evidence)
                print(f"  Updated weights for skills: {[path_sampler.nodes_id_dict[sid].name for sid in used_skill_ids if sid in path_sampler.nodes_id_dict]}")

                _append_domain_case(round_dir, domain, case)
            else:
                print(f"\n✗ Synthesis failed, not updating PathSampler weights")
                print(f"  Time: {elapsed:.1f}s | Tokens: {token_usage['total_tokens']} "
                      f"(prompt: {token_usage['prompt_tokens']}, completion: {token_usage['completion_tokens']})")
            
            # Save trace for every attempt (success or failure)
            _append_trace(round_dir, syn_log)
            _append_domain_trace(round_dir, domain, syn_log)
                
        except Exception as e:
            elapsed = time.time() - t_start
            token_usage = llm.get_token_count()
            print(f"\n✗ Error: {e}")
            print(f"  Time: {elapsed:.1f}s | Tokens: {token_usage['total_tokens']}")
            import traceback
            traceback.print_exc()
            error_log = {
                "attempt_number": attempt,
                "domain": domain,
                "outcome": "error",
                "failure_reason": str(e),
                "elapsed_seconds": round(elapsed, 2),
                "token_usage": token_usage,
            }
            _append_trace(round_dir, error_log)
            _append_domain_trace(round_dir, domain, error_log)
    
    # Save results
    output_file = os.path.join(round_dir, "eval_cases.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for case in results:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + '\n')
    
    trace_file = os.path.join(round_dir, "synthesis_trace.jsonl")
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"  Attempts: {attempt}")
    print(f"  Generated: {len(results)} cases")
    print(f"  Saved to: {output_file}")
    print(f"  Trace log: {trace_file}")
    
    # Print statistics
    path_sampler.print_node_statistics(topk=10)
    path_sampler.print_example_statistics(topk=10)
    
    if results:
        print(f"\n{'='*60}")
        print("Sample Cases:")
        for i, case in enumerate(results[:2]):
            print(f"\n--- Case {i+1} ---")
            print(f"Domain: {case.domain}")
            print(f"Skills: {case.skills}")
            print(f"Question: {case.question}")


if __name__ == "__main__":
    main()
