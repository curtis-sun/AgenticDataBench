"""
Solution Step Generator

Generates solution steps with DYNAMIC skill selection and immediate applicability check:
1. Generate main skill step → Check applicability → (Regenerate if needed)
2. For each auxiliary skill:
   - Select next skill → Generate step → Check applicability
   - If not applicable: determine cause and handle accordingly
3. Use StackOverflow examples from skill_paths.txt for guidance

Failure causes:
- skill_not_applicable: Blacklist the skill
- dependency_failed: Skip (caused by previous step failure)
- step_generation_error: Regenerate the step
"""

import os
import re
import time
from typing import List, Dict, Tuple, Any
from data_classes import Skill, SolutionStep, SolutionPath
from dataset_loader import build_files_summary, format_file_info_line
from auto_profiler import AutoProfiler
import sys
sys.path.append('..')
from utils.llm_client import QwenClient


# ---------------------------------------------------------------------------
# Static pre-filter: reject skills fundamentally incompatible with the data
# before any LLM calls.  Each entry is (pattern, condition_fn, reason).
# condition_fn receives the exploration dict and returns True when the skill
# should be REJECTED.
# ---------------------------------------------------------------------------


# Each entry: (pattern, condition_fn, reason, name_only[, exclude_pattern])
# When name_only=True the pattern is matched ONLY against the skill name;
# when False it is matched against "name + description".  This prevents
# false positives where a description casually mentions "streaming" or
# "real-time" while the skill itself is about something else (e.g.
# "Statistical Calculations and Descriptive Statistics").
# Optional exclude_pattern: if the same text also matches this pattern, the
# skill is NOT rejected (whitelisting file-streaming, chunked-reading, etc.).
_STATIC_SKILL_FILTERS: List[Tuple] = [
    # Category 1: GUI/Platform tools — always reject in a Python + static-file env
    # NOTE: Power BI and DAX skills are intentionally NOT filtered here.
    # They can be tested via Python DAX/Power BI tooling (pyadomd, pytabular,
    # sempy, etc.) to evaluate whether agents can call these tools correctly.
    (re.compile(r'(?i)\bpower\s*query\b'),
     lambda _: True,
     "requires Power Query (GUI tool)", False),
    (re.compile(r'(?i)\btableau\b'),
     lambda _: True,
     "requires Tableau platform", False),
    (re.compile(r'(?i)\bexcel\s*(?:macro|vba)\b'),
     lambda _: True,
     "requires Excel VBA (GUI tool)", False),

    # Category 2: Streaming / Real-time — name_only=True to avoid false positives
    # (many unrelated skills mention "streaming" or "real-time" in descriptions)
    (re.compile(r'(?i)\breal[\s-]?time\b.*\b(?:data|handling|processing|streaming)\b'),
     lambda _: True,
     "requires real-time/streaming data source, all files are static", True),
    (re.compile(r'(?i)\bstreaming\b'),
     lambda _: True,
     "requires streaming data source, all files are static", True,
     re.compile(r'(?i)\b(?:file|chunk|batch|read|large\s*data)\b')),

    # Category 3: Scale-dependent — reject when dataset is too small
    (re.compile(r'(?i)\b(?:spark|hadoop|mapreduce)\b'),
     lambda exp: exp.get('total_rows', 0) < 1_000_000,
     "dataset too small for distributed computing framework", False),
    (re.compile(r'(?i)\b(?:parallel|distributed)\s*(?:computing|processing)\b'),
     lambda exp: exp.get('total_rows', 0) < 1_000_000,
     "dataset too small for parallel/distributed processing", True),
    (re.compile(r'(?i)\bmemory\s*(?:management|efficien)'),
     lambda exp: exp.get('total_rows', 0) < 500_000,
     "dataset too small to require memory optimization", True),

    # Category 4: Tool-format mismatch
    (re.compile(r'(?i)\bexcel\s*(?:formula|function|array)\b'),
     lambda exp: 'excel' not in exp.get('file_types', {}),
     "requires Excel files but none present in dataset", False),
]


def check_skill_data_compatibility(
    skill_name: str,
    skill_description: str,
    exploration: dict,
) -> Tuple[bool, str]:
    """Static pre-filter: reject skills fundamentally incompatible with the data.

    Returns (is_compatible, reason).  reason is non-empty only when rejected.
    """
    full_text = f"{skill_name} {skill_description or ''}"
    for entry in _STATIC_SKILL_FILTERS:
        pattern, condition_fn, reason, name_only = entry[:4]
        exclude_pattern = entry[4] if len(entry) > 4 else None
        text = skill_name if name_only else full_text
        if pattern.search(text) and condition_fn(exploration):
            if exclude_pattern and exclude_pattern.search(text):
                continue
            return False, reason
    return True, ""


class StepGenerator:
    """
    Generate solution steps with dynamic skill selection and applicability checking.
    
    Key features:
    - Start from main skill (final output)
    - Generate and check each step immediately
    - Insert steps at correct positions based on data flow
    - Handle failures with retry or blacklist
    - Use StackOverflow examples for guidance
    """
    
    def __init__(
        self,
        llm_client: QwenClient,
        sampled_path = None,  # SampledPath object containing skills and examples
        min_files: int = 5,
        full_exploration: dict = None,
        domain_connections: str = None,
        rare_skill_names: List[str] = None,
        required_files: List[str] = None,
    ):
        """
        Initialize StepGenerator.
        
        Args:
            llm_client: LLM client for generating steps
            sampled_path: Object containing skills and examples (has get_examples_for_skill method)
            min_files: Minimum number of files the analysis should use
            full_exploration: Complete domain exploration dict (all files); used
                for incremental file selection — each step picks from the pool.
            domain_connections: Text describing how subfolders and files in this
                domain are connected. Loaded from datasets/<domain>/domain_connections.txt.
            rare_skill_names: Names of rare/low-frequency skills that should be
                judged leniently during applicability checks.
            required_files: File names (relative paths within domain) that MUST be
                included in the generated eval case. Pre-seeded into selected files
                so they appear in prompt context from the start.
        """
        self.llm = llm_client
        self.sampled_path = sampled_path
        self.min_files = min_files
        self._full_exploration = full_exploration
        self._domain_connections = domain_connections
        self._rare_skill_names: List[str] = rare_skill_names or []
        self._required_files: List[str] = required_files or []
        self._selected_files: set = set(self._required_files)
        self._trace: List[Dict[str, Any]] = []
    
    def _log(self, event: str, **data) -> None:
        """Record a trace event with timestamp."""
        entry = {"event": event, "timestamp": time.time()}
        entry.update(data)
        self._trace.append(entry)
    
    @property
    def trace(self) -> List[Dict[str, Any]]:
        return list(self._trace)
    
    def _get_examples_for_skill(self, skill_name: str) -> Tuple[str, List[str]]:
        """Get relevant examples for a specific skill.

        Returns:
            (formatted_text, example_ids) where example_ids lists the
            StackOverflow example IDs that were included.
        """
        if self.sampled_path:
            relevant = [
                ex for ex in self.sampled_path.examples
                if skill_name in ex.covered_skills
            ][:3]
            if relevant:
                text_parts = [f"### Example {ex.id}\n{ex.get_text()}" for ex in relevant]
                return "\n\n".join(text_parts), [ex.id for ex in relevant]
        return "", []

    @staticmethod
    def _is_description_low_quality(description: str, skill_name: str) -> bool:
        """Check if a step description is just the skill name repeated with no real content."""
        desc_norm = re.sub(r'[^a-z0-9 ]', ' ', description.lower()).split()
        skill_norm = re.sub(r'[^a-z0-9 ]', ' ', skill_name.lower()).split()
        if not desc_norm:
            return True
        overlap = len(set(desc_norm) & set(skill_norm))
        return overlap / len(set(desc_norm)) >= 0.85

    def _enrich_description(self, step: SolutionStep) -> None:
        """If a step's description is just the skill name, ask the LLM to expand it."""
        if not self._is_description_low_quality(step.description, step.skill.name):
            return

        prompt = f"""The following step description is too generic — it merely repeats the skill name.
Rewrite it into one concrete sentence that explains WHAT the step does with which data.

Skill: {step.skill.name}
Current description: {step.description}
Code snippet (for context):
```python
{(step.code_snippet or '')[:600]}
```

Respond in JSON: {{"description": "improved one-sentence description"}}"""

        result = self.llm.call_json(prompt)
        if result and result.get('description'):
            new_desc = result['description']
            if not self._is_description_low_quality(new_desc, step.skill.name):
                step.description = new_desc
                self._log("description_enriched", skill=step.skill.name,
                          old=step.description, new=new_desc)

    # ---- Incremental file-selection helpers ----

    @staticmethod
    def _extract_files_from_code(code_snippet: str, known_files: List[str]) -> List[str]:
        """Extract files actually read in code_snippet by matching against known_files.

        Scans for pd.read_csv/read_excel/read_json/read_parquet/read_table,
        open(), and similar I/O calls.  Returns the subset of *known_files*
        whose basename (or full relative path) appears inside a detected read
        operation.
        """
        import os
        if not code_snippet or not known_files:
            return []

        read_patterns = [
            re.compile(r'''pd\.read_(?:csv|excel|json|parquet|table|fwf|feather|hdf|pickle|stata|sas|spss|orc)\s*\(\s*(?:filepath_or_buffer\s*=\s*)?(['"])(.*?)\1'''),
            re.compile(r'''read_(?:csv|excel|json|parquet|table|fwf|feather|hdf|pickle|stata|sas|spss|orc)\s*\(\s*(?:filepath_or_buffer\s*=\s*)?(['"])(.*?)\1'''),
            re.compile(r'''open\s*\(\s*(['"])(.*?)\1'''),
            re.compile(r'''(?:load_workbook|openpyxl\.load_workbook)\s*\(\s*(['"])(.*?)\1'''),
            re.compile(r'''(?:gpd\.read_file|geopandas\.read_file)\s*\(\s*(['"])(.*?)\1'''),
        ]

        extracted_paths: set[str] = set()
        for pat in read_patterns:
            for m in pat.finditer(code_snippet):
                extracted_paths.add(m.group(2))

        known_basenames = {os.path.basename(f): f for f in known_files}

        matched: list[str] = []
        seen: set[str] = set()
        for ep in extracted_paths:
            ep_base = os.path.basename(ep)
            if ep in known_files and ep not in seen:
                matched.append(ep)
                seen.add(ep)
            elif ep_base in known_basenames and known_basenames[ep_base] not in seen:
                full = known_basenames[ep_base]
                matched.append(full)
                seen.add(full)

        return matched

    def _update_selected_files(self, step: SolutionStep) -> None:
        """Reconcile LLM-reported files_used with actual file reads in code_snippet.

        Overwrites ``step._files_used`` with the programmatically verified list
        so that downstream logic (question prompt, min-file checks) only counts
        files that are genuinely loaded in the code.
        """
        known_files = [f['name'] for f in self._get_all_files_info()]
        code_files = self._extract_files_from_code(
            step.code_snippet or '', known_files,
        )

        llm_reported = set(getattr(step, '_files_used', None) or [])
        code_set = set(code_files)

        reconciled = code_set | (llm_reported & set(known_files))
        if code_set:
            reconciled = code_set

        step._files_used = list(reconciled)
        self._selected_files.update(reconciled)

    def _get_all_files_info(self) -> list:
        """Return the full list of file-info dicts from the exploration."""
        if self._full_exploration:
            return self._full_exploration.get('files_info', [])
        return []

    def _build_selected_files_section(self) -> str:
        """Build a detailed description of already-selected files."""
        if not self._selected_files:
            return "(No files selected yet — this is the first step.)\n"
        all_info = self._get_all_files_info()
        selected_info = [f for f in all_info if f['name'] in self._selected_files]
        if not selected_info:
            return "(No files selected yet.)\n"
        return build_files_summary(selected_info, detailed=True, max_files=25)

    def _build_pool_files_section(self) -> str:
        """Build a brief one-line-per-file description of not-yet-selected files."""
        all_info = self._get_all_files_info()
        pool_info = [f for f in all_info if f['name'] not in self._selected_files]
        if not pool_info:
            return "(All files have been selected.)\n"
        lines = [format_file_info_line(f) for f in pool_info]
        return "\n".join(lines) + "\n"

    def _build_required_files_constraint(self) -> str:
        """Build a prompt constraint for mandatory file inclusion."""
        if not self._required_files:
            return ""
        files_list = "\n".join(f"  - {f}" for f in self._required_files)
        return (
            "\nMANDATORY FILE REQUIREMENT: The following file(s) MUST be loaded "
            "and meaningfully used in the analysis workflow. Your step MUST "
            "incorporate these files — load them with pd.read_csv() or the "
            "appropriate reader and integrate them into the analysis:\n"
            f"{files_list}\n"
        )

    @staticmethod
    def _build_subfolder_shared_columns_section(exploration: Dict) -> str:
        """Build a section listing shared columns (join keys) within each subfolder.

        Uses pattern detection to compress repetitive column names (e.g. hundreds
        of date columns like '2010-01', '2010-02', ...) into a single summary line,
        preventing token explosions for wide-format datasets.
        """
        subfolders = exploration.get("subfolders", {})
        if not subfolders:
            return ""

        lines = []
        for folder, info in subfolders.items():
            shared = info.get("shared_columns", {})
            if not shared:
                continue
            label = folder if folder else "(root)"
            n_files = len(info.get("files", []))

            all_in_all = all(len(fnames) == n_files for fnames in shared.values())
            if all_in_all and n_files > 2:
                col_summary = AutoProfiler._detect_column_patterns(
                    sorted(shared.keys())
                )['summary']
                lines.append(
                    f"### Subfolder: {label}\n"
                    f"All {n_files} files share the same schema. "
                    f"Columns: {col_summary}"
                )
            else:
                lines.append(f"### Subfolder: {label}")
                col_names = sorted(shared.keys())
                patterns = AutoProfiler._detect_column_patterns(col_names)
                for col in patterns['standalone']:
                    fnames = shared[col]
                    if len(fnames) == n_files:
                        lines.append(f"- {col} → all {n_files} files")
                    elif len(fnames) > 5:
                        lines.append(f"- {col} → {len(fnames)} of {n_files} files")
                    else:
                        lines.append(f"- {col} → {', '.join(fnames)}")
                for p in patterns['patterns']:
                    sample_col = p['range'][0]
                    sample_fnames = shared[sample_col]
                    lines.append(
                        f"- {p['pattern']} ({p['count']} cols, {p['range'][0]} ~ {p['range'][1]}) "
                        f"→ {len(sample_fnames)} files"
                    )
            lines.append("")

        if not lines:
            return ""
        return "## Shared Columns Within Subfolders (potential join keys)\n\n" + "\n".join(lines)

    def _build_file_sections_for_prompt(self, exploration: Dict) -> str:
        """
        Build the two-section file description for prompts.

        When full_exploration is available (incremental mode):
          - Selected Files: detailed info for files already used
          - Available File Pool: brief one-line summaries for unused files
          - Shared Columns within each subfolder (auto-detected join keys)
          - Domain Connection Guide (if available)

        Falls back to the legacy single-section layout otherwise.
        """
        if not self._full_exploration:
            return (
                f"- Files:\n"
                f"{build_files_summary(exploration['files_info'], detailed=True, max_files=25)}\n"
            )

        parts = []
        parts.append("## Selected Files (detailed — use these in your step)\n")
        parts.append(self._build_selected_files_section())

        parts.append("\n## Available File Pool (you MAY recruit additional files from here)\n")
        parts.append(self._build_pool_files_section())

        shared_section = self._build_subfolder_shared_columns_section(exploration)
        if shared_section:
            parts.append("\n" + shared_section)

        if self._domain_connections:
            parts.append("\n## Domain Data Connection Guide\n")
            parts.append("The following describes how the subfolders and files in this domain "
                         "are connected. Use this to design cross-file analyses.\n\n")
            parts.append(self._domain_connections)

        return "\n".join(parts)

    @staticmethod
    def _build_format_constraint(exploration: Dict) -> str:
        """Build a constraint section reminding LLM of available data formats."""
        file_types = exploration.get('file_types', {})
        if not file_types:
            return ""
        types_list = ", ".join(f"{ft.upper()} ({cnt})" for ft, cnt in file_types.items())
        return (
            f"\n## Data Format Constraint\n"
            f"Available file types in this domain: {types_list}\n"
            f"The generated step MUST be practically executable starting from these "
            f"file types. If the skill normally requires specific infrastructure "
            f"(e.g., a database for SQL, a specific data platform), you MAY include "
            f"a data conversion/loading sub-step (e.g., loading CSV files into a "
            f"SQLite database) as part of this step, as long as:\n"
            f"  1. The conversion can be done programmatically in Python\n"
            f"  2. The conversion is a natural part of applying this skill\n"
            f"  3. The code_snippet includes the full conversion + operation logic\n"
            f"However, if the skill fundamentally requires interactive/GUI tools "
            f"(e.g., Power BI dashboards, Tableau, Excel macros) that CANNOT be "
            f"replicated in a Python script, the step is NOT feasible.\n"
        )
    
    def _check_step_applicability(
        self, 
        step: SolutionStep, 
        exploration: Dict, 
        previous_steps: List[SolutionStep] = None,
        following_steps: List[SolutionStep] = None,
        is_main_step: bool = False
    ) -> Dict:
        """
        Check if a solution step is applicable to this dataset and fits in the workflow.
        
        Args:
            step: The step to check
            exploration: Dataset exploration info
            previous_steps: Steps that come BEFORE this step (provide input context)
            following_steps: Steps that come AFTER this step (depend on this step's output)
            is_main_step: If True, skip the output-necessity check (main step is
                always necessary as it produces the final answer)
        
        Returns:
            Dict with:
            - applicable: bool
            - reason: str
            - failure_cause: "skill_not_applicable" | "dependency_failed" | "step_generation_error" | "output_not_consumed" | None
        """
        # Build previous steps description for context (include code so LLM
        # can see what files/variables each step actually produces)
        previous_steps_desc = "None (this is the first step)"
        if previous_steps:
            parts = []
            for i, s in enumerate(previous_steps):
                part = (
                    f"- Step {i+1} [{s.skill.name}]: {s.description}\n"
                    f"  Output: {s.output_description}"
                )
                if s.code_snippet:
                    snippet = s.code_snippet
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "\n  # ... (truncated)"
                    part += f"\n  Code:\n  ```python\n  {snippet}\n  ```"
                parts.append(part)
            previous_steps_desc = "\n".join(parts)
        
        # Build following steps description (include code for consistency)
        following_steps_desc = "None (this is the last step)"
        if following_steps:
            parts = []
            for s in following_steps:
                part = f"- [{s.skill.name}]: {s.description}\n  Input needed: {s.input_data}"
                if s.code_snippet:
                    snippet = s.code_snippet
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "\n  # ... (truncated)"
                    part += f"\n  Code:\n  ```python\n  {snippet}\n  ```"
                parts.append(part)
            following_steps_desc = "\n".join(parts)
        
        file_types = exploration.get('file_types', {})
        file_types_desc = ", ".join(f"{ft}: {cnt}" for ft, cnt in file_types.items()) if file_types else "unknown"

        is_rare = bool(self._rare_skill_names and step.skill.name in self._rare_skill_names)

        if is_main_step:
            output_necessity_criterion = ""
            output_necessity_failure_cause = ""
        elif is_rare:
            output_necessity_criterion = f"""
5. **Output Necessity (LENIENT — rare skill)**: The skill "{step.skill.name}"
   is a rare/low-frequency skill. For rare skills, apply lenient evaluation:
   - PASS if the step's output has any plausible (even indirect) connection
     to the workflow's final goal or enriches the analysis in any way.
   - FAIL only if the output is completely isolated and has zero relevance
     to the overall workflow objective.
   Rare skills are inherently harder to integrate tightly into a pipeline;
   loose coupling and indirect contributions are acceptable."""
            output_necessity_failure_cause = """
- "output_not_consumed": The step's output is completely isolated with zero
  relevance to the workflow (applies only for egregious cases with rare skills)"""
        else:
            output_necessity_criterion = """
5. **Output Necessity**: The step's output must be ACTUALLY CONSUMED by at least
   one following step. Examine each following step's input_data, code, and
   description to determine if any of them depend on what this step produces.
   If the step produces output (e.g., a visualization, a summary, a side
   analysis) that no following step uses as input, it is a "dead-end" —
   technically executable but unnecessary for reaching the final answer.
   Dead-end steps FAIL this criterion.
   Note: simply being "useful" or "informative" is NOT enough — the output must
   be a concrete, required input for a subsequent step in the workflow."""
            output_necessity_failure_cause = """
- "output_not_consumed": The step can execute, but its output is not used by any
  following step — it is a dead-end in the workflow"""

        prompt = f"""You are a data science expert. Given the following dataset and a solution step,
determine if this step can be executed AND fits properly in the workflow.

## Dataset
- Domain: {exploration['domain']}
- Number of Files: {exploration['num_files']}
- Available File Types: {file_types_desc}
- Total Rows: {exploration['total_rows']}
- Has Missing Values: {exploration.get('has_missing_values', False)}

## Files Details:
{self._build_file_sections_for_prompt(exploration) if self._full_exploration else build_files_summary(exploration['files_info'], detailed=True, max_files=25)}

## Previous Steps (provide input to this step)
{previous_steps_desc}

## Current Step to Check
- Skill: {step.skill.name}
- Skill Definition: {step.skill.description or 'N/A'}
- Description: {step.description}
- Input Data: {step.input_data}
- Expected Output: {step.output_description}
- Code Snippet: {step.code_snippet if step.code_snippet else 'N/A'}

## Following Steps (depend on this step's output)
{following_steps_desc}

## Applicability Criteria (check ALL of the following)

1. **Format & Infrastructure Match**: If the skill requires specific infrastructure
   (e.g., databases, BI platforms):
   - **Adaptable**: If the step's code_snippet includes a legitimate programmatic
     conversion (e.g., loading CSV into SQLite for SQL operations, creating an
     in-memory database), AND the skill is then genuinely applied (not just
     trivially wrapping pandas calls), it IS applicable.
   - **Power BI / DAX skills**: These are APPLICABLE when the code_snippet uses
     actual Power BI / DAX Python tooling (e.g., `pyadomd`, `pytabular`,
     `sempy`/`semantic-link`, `powerbiclient`, `ssas_api`) to load data into
     a tabular model, define DAX measures/calculated columns/relationships,
     and execute DAX queries. The code must genuinely invoke these tools —
     reimplementing DAX logic using plain pandas is NOT acceptable for
     Power BI / DAX skills, as the goal is to test tool-calling ability.
   - **Not adaptable**: If the skill fundamentally requires interactive/GUI tools
     (e.g., Tableau dashboards, Excel VBA macros) that
     cannot be replicated in a Python script, it is NOT applicable. Mark as
     "skill_not_applicable".
   - **Trivially wrapped**: If the step just renames basic pandas operations to
     sound like the skill (e.g., calling df.merge() "SQL JOIN", or using pandas
     groupby but calling it "DAX SUMMARIZE") without actually invoking the
     skill's native tooling, it is NOT applicable.

2. **Content & Domain Match**: The step must operate on data characteristics that
   actually exist in the files (e.g., the right column types, temporal fields for
   time-based analysis, categorical fields for grouping). Check whether the files'
   actual content supports the operation described.

3. **Data Flow Fit**: The step must be able to receive proper input from previous
   steps (if any) and provide proper output for following steps (if any).

4. **File Reference Validity**: The code_snippet must ONLY read files that are
   EITHER (a) listed in the Files Details section above, OR (b) produced/saved
   by a previous step's code (check the Code blocks in Previous Steps above).
   A reference to an intermediate file name (e.g. 'merged_*.csv') is ONLY an
   error if it was NOT created by any previous step. In-memory variables
   (DataFrames, etc.) passed from previous steps are always valid references.

5. **File Usage Verification**: The step's code_snippet must actually call a
   file-reading function (e.g. pd.read_csv(), pd.read_excel(), open(), etc.)
   for EACH file it claims in files_used. Simply mentioning a file name in a
   comment, variable name, or string that is not passed to a read function
   does NOT count as genuine usage. If files_used lists files that are not
   truly loaded in the code, flag this as "step_generation_error".
{output_necessity_criterion}
## Failure Causes
- "skill_not_applicable": The skill cannot be meaningfully applied given the
  domain's file types, data content, or infrastructure
- "dependency_failed": The step cannot get required input from previous steps
- "step_generation_error": The skill is applicable, but the step description has
  errors (e.g., references non-existent columns or files){output_necessity_failure_cause}

Respond in JSON format only:
{{
    "applicable": true/false,
    "reason": "explanation",
    "failure_cause": "skill_not_applicable" or "dependency_failed" or "step_generation_error" or "output_not_consumed" or null
}}
"""
        result = self.llm.call_json(prompt)
        return result or {"applicable": False, "reason": "Failed to get LLM response", "failure_cause": "step_generation_error"}
    
    def _remap_dependencies_after_insert(
        self,
        steps: List[SolutionStep],
        insert_idx: int
    ) -> None:
        """
        Remap all dependent_step_ids after a new step is inserted at insert_idx.
        
        During generation, dependent_step_ids are 0-based indices into the steps list.
        When a new step is inserted at insert_idx, all indices >= insert_idx in
        existing steps' dependencies must shift by +1. The new step's own dependencies
        (which reference the OLD list's indices) also need the same remapping.
        
        Example:
            Before: [S0, S1, S2] with S2.deps=[1]
            Insert at idx=1: [S0, NEW, S1, S2]
            S2.deps=[1] → [2] (S1 moved from index 1 to 2)
            NEW.deps (from LLM, old indices) get same remapping
        """
        for step in steps:
            step.dependent_step_ids = [
                (dep + 1) if dep >= insert_idx else dep
                for dep in step.dependent_step_ids
            ]
    
    def generate_steps_dynamic(
        self,
        main_skill: Skill,
        auxiliary_skills: List[Skill],
        exploration: Dict,
        target_steps: int = None,
    ) -> Tuple[List[SolutionStep], str, List[Skill], List[Skill]]:
        """
        Generate solution steps with dynamic skill selection and immediate applicability check.
        
        Args:
            target_steps: If set, stop adding auxiliary skills once len(steps) >= target_steps.
                          This allows over-sampling skills while keeping the case within the
                          desired step count range.
        
        Returns:
            Tuple of (steps in execution order, answer_type, skipped_skills, blacklisted_skills)
        """
        max_regenerate_attempts = 3
        blacklisted_skills = []
        
        # Step 1: Generate main skill step (defer applicability check until full path is built)
        print(f"  [1/5] Generating main skill step: {main_skill.name}")
        main_step, answer_type = self._generate_main_step(main_skill, exploration)
        self._enrich_description(main_step)
        self._update_selected_files(main_step)
        
        self._log(
            "main_step_generated",
            skill=main_skill.name,
            skill_id=main_skill.id,
            description=main_step.description,
            answer_type=answer_type,
            code_snippet=main_step.code_snippet,
            input_data=main_step.input_data,
            output_description=main_step.output_description,
            files_used=list(getattr(main_step, '_files_used', [])),
        )
        
        # Start with main step
        steps = [main_step]
        skipped_skills = []
        dependency_failed_skills = []  # Skills that failed due to dependency, retry later
        
        # Step 2: Process each auxiliary skill
        print(f"  [2/5] Processing {len(auxiliary_skills)} auxiliary skills")
        
        for skill in auxiliary_skills:
            # Early stop: we have enough steps for the target step count
            if target_steps and len(steps) >= target_steps:
                print(f"    ⏹ Reached target step count ({target_steps}), stopping")
                self._log("early_stop", reason="target_steps_reached",
                          target_steps=target_steps, current_steps=len(steps))
                break

            # Skip if already blacklisted
            if skill.name in [b.name for b in blacklisted_skills]:
                print(f"    ✗ Skipping: {skill.name} (blacklisted)")
                self._log("skill_skipped", skill=skill.name, reason="already_blacklisted")
                continue

            # Static pre-filter: reject skills incompatible with the data
            compatible, reject_reason = check_skill_data_compatibility(
                skill.name, skill.description or '', exploration
            )
            if not compatible:
                print(f"    ✗ Static filter: {skill.name} ({reject_reason})")
                blacklisted_skills.append(skill)
                self._log("skill_static_filtered", skill=skill.name, reason=reject_reason)
                continue

            # Generate step for this skill
            print(f"    Processing: {skill.name}")
            print(f"      Generating step...", end=" ")
            new_steps = self._generate_and_insert_step(skill, steps, exploration, main_skill=main_skill)
            
            # Find the newly added step
            new_step = None
            insert_position = -1
            for i, step in enumerate(new_steps):
                if step.skill.id == skill.id:
                    new_step = step
                    insert_position = i
                    break
            
            if new_step is None:
                print(f"failed")
                print(f"      ✗ Skipping (failed to generate step)")
                self._log("aux_step_generation_failed", skill=skill.name)
                skipped_skills.append(skill)
                continue
            
            print(f"done (position {insert_position})")
            
            # Check applicability with previous and following steps as context
            previous_steps_ctx = new_steps[:insert_position] if insert_position > 0 else None
            following_steps_ctx = new_steps[insert_position + 1:] if insert_position + 1 < len(new_steps) else None
            
            print(f"      Checking applicability...", end=" ")
            check_result = self._check_step_applicability(
                new_step, exploration, previous_steps_ctx, following_steps_ctx
            )
            
            if check_result.get('applicable', False):
                # Success - update steps and track files
                steps = new_steps
                self._enrich_description(new_step)
                self._update_selected_files(new_step)
                if insert_position + 1 < len(steps):
                    next_step_name = steps[insert_position + 1].skill.name
                    print(f"✓ (before {next_step_name})")
                else:
                    print(f"✓ (at end)")
                self._log(
                    "aux_step_accepted",
                    skill=skill.name,
                    position=insert_position,
                    description=new_step.description,
                    code_snippet=new_step.code_snippet,
                    input_data=new_step.input_data,
                    output_description=new_step.output_description,
                    files_used=list(getattr(new_step, '_files_used', [])),
                    referenced_examples=list(getattr(new_step, '_referenced_examples', [])),
                    applicability=check_result,
                )
            else:
                # Failed - handle based on failure cause
                failure_cause = check_result.get('failure_cause', 'step_generation_error')
                reason_text = check_result.get('reason', 'Unknown')
                print(f"✗ ({failure_cause})")
                print(f"      Reason: {reason_text}")
                
                self._log(
                    "aux_step_rejected",
                    skill=skill.name,
                    failure_cause=failure_cause,
                    reason=reason_text,
                    description=new_step.description,
                )
                
                if failure_cause == 'skill_not_applicable':
                    print(f"      → Blacklisting this skill")
                    blacklisted_skills.append(skill)
                    self._log("skill_blacklisted", skill=skill.name, reason=reason_text)
                
                elif failure_cause == 'dependency_failed':
                    print(f"      → Will retry later")
                    dependency_failed_skills.append(skill)
                    self._log("skill_deferred", skill=skill.name, reason=reason_text)
                
                elif failure_cause == 'output_not_consumed':
                    print(f"      → Skipping (dead-end: output not consumed by downstream steps)")
                    skipped_skills.append(skill)
                    self._log("skill_skipped", skill=skill.name,
                              reason="output_not_consumed: " + reason_text)
                
                else:  # step_generation_error
                    # Try regenerating with failure feedback
                    regenerated = False
                    for attempt in range(max_regenerate_attempts):
                        print(f"      Regenerating (attempt {attempt + 1}/{max_regenerate_attempts})...", end=" ")
                        failure_reason = check_result.get('reason', '')
                        new_steps = self._generate_and_insert_step(
                            skill, steps, exploration, previous_failure=failure_reason,
                            main_skill=main_skill,
                        )
                        
                        # Find the new step again
                        new_step = None
                        insert_position = -1
                        for i, step in enumerate(new_steps):
                            if step.skill.id == skill.id:
                                new_step = step
                                insert_position = i
                                break
                        
                        if new_step is None:
                            print(f"failed")
                            self._log("aux_step_regeneration_failed", skill=skill.name, attempt=attempt + 1)
                            continue
                        
                        previous_steps_ctx = new_steps[:insert_position] if insert_position > 0 else None
                        following_steps_ctx = new_steps[insert_position + 1:] if insert_position + 1 < len(new_steps) else None
                        check_result = self._check_step_applicability(
                            new_step, exploration, previous_steps_ctx, following_steps_ctx
                        )
                        
                        if check_result.get('applicable', False):
                            steps = new_steps
                            self._enrich_description(new_step)
                            self._update_selected_files(new_step)
                            print(f"✓")
                            self._log(
                                "aux_step_regenerated_accepted",
                                skill=skill.name,
                                attempt=attempt + 1,
                                position=insert_position,
                                description=new_step.description,
                                code_snippet=new_step.code_snippet,
                                referenced_examples=list(getattr(new_step, '_referenced_examples', [])),
                                applicability=check_result,
                            )
                            regenerated = True
                            break
                        else:
                            print(f"✗")
                            self._log(
                                "aux_step_regeneration_rejected",
                                skill=skill.name,
                                attempt=attempt + 1,
                                failure_cause=check_result.get('failure_cause'),
                                reason=check_result.get('reason', ''),
                            )
                    
                    if not regenerated:
                        print(f"      → Skipping after {max_regenerate_attempts} failed attempts")
                        skipped_skills.append(skill)
                        self._log("skill_skipped", skill=skill.name,
                                  reason=f"failed_after_{max_regenerate_attempts}_regeneration_attempts")
        
        # Step 2.5: Retry dependency_failed skills
        if dependency_failed_skills:
            print(f"\n  [2.5/3] Retrying {len(dependency_failed_skills)} deferred skills")
            self._log("deferred_retry_start", skills=[s.name for s in dependency_failed_skills])
            
            for skill in dependency_failed_skills:
                print(f"    Retrying: {skill.name}")
                print(f"      Generating step...", end=" ")
                
                new_steps = self._generate_and_insert_step(skill, steps, exploration, main_skill=main_skill)
                
                # Find the newly added step
                new_step = None
                insert_position = -1
                for i, step in enumerate(new_steps):
                    if step.skill.id == skill.id:
                        new_step = step
                        insert_position = i
                        break
                
                if new_step is None:
                    print(f"failed")
                    self._log("deferred_retry_failed", skill=skill.name, reason="generation_failed")
                    skipped_skills.append(skill)
                    continue
                
                print(f"done (position {insert_position})")
                
                previous_steps_ctx = new_steps[:insert_position] if insert_position > 0 else None
                following_steps_ctx = new_steps[insert_position + 1:] if insert_position + 1 < len(new_steps) else None
                
                print(f"      Checking applicability...", end=" ")
                check_result = self._check_step_applicability(
                    new_step, exploration, previous_steps_ctx, following_steps_ctx
                )
                
                if check_result.get('applicable', False):
                    steps = new_steps
                    self._enrich_description(new_step)
                    self._update_selected_files(new_step)
                    if insert_position + 1 < len(steps):
                        next_step_name = steps[insert_position + 1].skill.name
                        print(f"✓ (before {next_step_name})")
                    else:
                        print(f"✓ (at end)")
                    self._log(
                        "deferred_retry_accepted",
                        skill=skill.name,
                        position=insert_position,
                        description=new_step.description,
                        referenced_examples=list(getattr(new_step, '_referenced_examples', [])),
                        applicability=check_result,
                    )
                else:
                    failure_cause = check_result.get('failure_cause', 'unknown')
                    reason_text = check_result.get('reason', 'Unknown')
                    print(f"✗ ({failure_cause})")
                    print(f"      Reason: {reason_text}")
                    self._log("deferred_retry_rejected", skill=skill.name,
                              failure_cause=failure_cause, reason=reason_text)
                    skipped_skills.append(skill)
        
        # Step 3: Validate main step after full path construction
        print(f"\n  [3/5] Checking main skill applicability after path construction...", end=" ")
        main_idx = None
        for i, s in enumerate(steps):
            if s.skill.id == main_skill.id:
                main_idx = i
                break
        if main_idx is None:
            print("✗")
            print("        [Error] Main step missing from workflow")
            return [], answer_type, skipped_skills, blacklisted_skills

        previous_steps_ctx = steps[:main_idx] if main_idx > 0 else None
        following_steps_ctx = steps[main_idx + 1:] if main_idx + 1 < len(steps) else None
        check_result = self._check_step_applicability(
            steps[main_idx], exploration, previous_steps_ctx, following_steps_ctx,
            is_main_step=True
        )

        if check_result.get('applicable', False):
            print("✓ Applicable")
            self._log("main_step_revalidated", applicable=True,
                       referenced_examples=list(getattr(steps[main_idx], '_referenced_examples', [])),
                       applicability=check_result)
        else:
            failure_cause = check_result.get('failure_cause', 'step_generation_error')
            reason_text = check_result.get('reason', 'Unknown')
            print(f"✗ Not applicable ({failure_cause})")
            print(f"        Reason: {reason_text}")
            self._log("main_step_revalidated", applicable=False,
                       referenced_examples=list(getattr(steps[main_idx], '_referenced_examples', [])),
                       failure_cause=failure_cause, reason=reason_text)

            if failure_cause == 'skill_not_applicable':
                print("        [Error] Main skill not applicable, cannot continue")
                self._log("main_step_abort", reason="skill_not_applicable")
                return [], answer_type, skipped_skills, blacklisted_skills + [main_skill]

            regenerated = False
            for attempt in range(max_regenerate_attempts):
                print(f"        Regenerating main step (attempt {attempt + 1}/{max_regenerate_attempts})...")
                failure_reason = check_result.get('reason', '')
                regenerated_main, answer_type = self._generate_main_step(
                    main_skill,
                    exploration,
                    previous_failure=failure_reason,
                    upstream_steps=previous_steps_ctx
                )
                steps[main_idx] = regenerated_main

                previous_steps_ctx = steps[:main_idx] if main_idx > 0 else None
                following_steps_ctx = steps[main_idx + 1:] if main_idx + 1 < len(steps) else None
                check_result = self._check_step_applicability(
                    steps[main_idx], exploration, previous_steps_ctx, following_steps_ctx,
                    is_main_step=True
                )
                if check_result.get('applicable', False):
                    print("        ✓ Regenerated successfully")
                    self._log("main_step_regenerated", attempt=attempt + 1,
                              success=True, description=regenerated_main.description,
                              referenced_examples=list(getattr(regenerated_main, '_referenced_examples', [])))
                    regenerated = True
                    break
                else:
                    self._log("main_step_regenerated", attempt=attempt + 1,
                              success=False, reason=check_result.get('reason', ''))

            if not regenerated:
                print(f"        [Error] Main step still not applicable after {max_regenerate_attempts} attempts")
                self._log("main_step_abort",
                          reason=f"failed_after_{max_regenerate_attempts}_regeneration_attempts")
                return [], answer_type, skipped_skills, blacklisted_skills

        # Step 4: Post-generation minimum file count check
        all_known = [f['name'] for f in self._get_all_files_info()]
        verified_files: set = set()
        for s in steps:
            verified_files.update(
                self._extract_files_from_code(s.code_snippet or '', all_known)
            )

        if len(verified_files) < self.min_files:
            unused_pool = [f for f in all_known if f not in verified_files]
            print(f"\n  [4/5] File count check: {len(verified_files)} verified < "
                  f"{self.min_files} required. Unused pool: {len(unused_pool)} files")
            self._log("min_files_check_start",
                      verified=len(verified_files), required=self.min_files,
                      unused_pool_size=len(unused_pool))

            if unused_pool:
                best_idx = self._pick_best_step_for_file_recruitment(steps)
                target_step = steps[best_idx]
                print(f"    Requesting step {best_idx} [{target_step.skill.name}] "
                      f"to recruit more files...")

                new_code = self._recruit_files_into_step(
                    target_step, unused_pool, exploration, steps, best_idx
                )
                if new_code:
                    target_step.code_snippet = new_code
                    self._update_selected_files(target_step)
                    # Re-verify after recruitment
                    verified_files = set()
                    for s in steps:
                        verified_files.update(
                            self._extract_files_from_code(s.code_snippet or '', all_known)
                        )
                    print(f"    After recruitment: {len(verified_files)} verified files")
                    self._log("min_files_recruitment_done",
                              verified_after=len(verified_files))
                else:
                    print(f"    Recruitment failed, proceeding with {len(verified_files)} files")
                    self._log("min_files_recruitment_failed")

        # Step 4.5: Required files verification
        if self._required_files:
            missing_required = [f for f in self._required_files if f not in verified_files]
            if missing_required:
                print(f"\n  [Required Files] {len(missing_required)} required file(s) "
                      f"not yet used in code: {[os.path.basename(f) for f in missing_required]}")
                self._log("required_files_missing", missing=missing_required)
                best_idx = self._pick_best_step_for_file_recruitment(steps)
                target_step = steps[best_idx]
                print(f"    Recruiting into step {best_idx} [{target_step.skill.name}]...")
                new_code = self._recruit_files_into_step(
                    target_step, missing_required, exploration, steps, best_idx
                )
                if new_code:
                    target_step.code_snippet = new_code
                    self._update_selected_files(target_step)
                    verified_files = set()
                    for s in steps:
                        verified_files.update(
                            self._extract_files_from_code(s.code_snippet or '', all_known)
                        )
                    still_missing = [f for f in self._required_files if f not in verified_files]
                    if still_missing:
                        print(f"    Still missing after recruitment: "
                              f"{[os.path.basename(f) for f in still_missing]}")
                        self._log("required_files_still_missing", still_missing=still_missing)
                    else:
                        print(f"    All required files now included in code ✓")
                        self._log("required_files_recruitment_done")
                else:
                    print(f"    Recruitment failed for required files")
                    self._log("required_files_recruitment_failed", missing=missing_required)
            else:
                print(f"\n  [Required Files] All required files verified in code ✓")
                self._log("required_files_verified")

        # Step 5: Finalize step order and numbering
        print(f"  [5/5] Finalizing step order ({len(steps)} steps, {len(skipped_skills)} skipped, {len(blacklisted_skills)} blacklisted)")
        steps = self._finalize_steps(steps)
        
        self._log(
            "steps_finalized",
            num_steps=len(steps),
            skipped_skills=[s.name for s in skipped_skills],
            blacklisted_skills=[s.name for s in blacklisted_skills],
            final_steps=[
                {
                    "step_number": s.step_number,
                    "skill": s.skill.name,
                    "description": s.description,
                    "dependent_step_ids": s.dependent_step_ids,
                }
                for s in steps
            ],
        )
        
        return steps, answer_type, skipped_skills, blacklisted_skills
    
    def _generate_main_step(
        self,
        main_skill: Skill,
        exploration: Dict,
        previous_failure: str = None,
        upstream_steps: List[SolutionStep] = None
    ) -> Tuple[SolutionStep, str]:
        """
        Generate the main skill step and determine answer type.
        This is the FINAL step that produces the answer.
        """
        # Get examples relevant to main skill
        examples_text, examples_ids = self._get_examples_for_skill(main_skill.name)
        so_section = (
            f"\n## Reference Examples (from StackOverflow)\n"
            f"NOTE: These examples are ONLY for illustrating the TECHNIQUE/METHOD.\n"
            f"Do NOT copy column names, file names, or variable names from these examples.\n"
            f"You MUST use the actual column/file names from the Dataset section above.\n\n"
            f"{examples_text}"
        ) if examples_text else ""
        
        failure_section = ""
        if previous_failure:
            failure_section = f"""
## Previous Attempt Failed - You MUST Avoid These Mistakes
{previous_failure}
Re-generate the step, making sure to ONLY use column names and file names that
actually exist in the dataset description above. Double-check every column reference.
"""
        
        skill_desc_section = ""
        if main_skill.description:
            skill_desc_section = f"- Description: {main_skill.description}\n"

        upstream_section = ""
        if upstream_steps:
            upstream_parts = []
            for i, s in enumerate(upstream_steps):
                part = (
                    f"- Step {i+1} [{s.skill.name}]\n"
                    f"  Description: {s.description}\n"
                    f"  Output: {s.output_description}\n"
                    f"  Input used: {s.input_data}"
                )
                if s.code_snippet:
                    snippet = s.code_snippet
                    if len(snippet) > 600:
                        snippet = snippet[:600] + "\n  # ... (truncated)"
                    part += f"\n  Code:\n  ```python\n  {snippet}\n  ```"
                upstream_parts.append(part)
            upstream_desc = "\n".join(upstream_parts)
            upstream_section = f"""
## Upstream Steps Already Exist (MUST be consumed by this final step)
{upstream_desc}

CRITICAL RULES for writing the final step:
1. You MUST consume the output variables produced by the upstream steps above.
   Read their Code blocks to see exactly what variables/DataFrames they produce.
2. Do NOT reload or re-read raw CSV/Excel files that were already loaded and
   transformed by upstream steps. Use the transformed outputs instead.
3. Do NOT repeat operations (imputation, encoding, normalization, merging) that
   upstream steps already performed.
4. Your code_snippet, input_data, and step_description must all be consistent
   with each other AND with the upstream steps' actual outputs.
5. If upstream step code produces variable `X`, your code must use `X` — not
   reload the same file and recreate it from scratch.
"""

        format_constraint = self._build_format_constraint(exploration)

        prompt = f"""You are a data science expert. Generate the FINAL step of a data analysis task.
This step uses the main skill and produces the final answer.

## Main Skill
- Name: {main_skill.name}
{skill_desc_section}- This skill should be the PRIMARY focus of the final step
{so_section}
---
{failure_section}
## Dataset (USE ONLY THESE ACTUAL COLUMNS AND FILES)
- Domain: {exploration['domain']}

{self._build_file_sections_for_prompt(exploration)}
{format_constraint}
## Infrastructure Adaptation (READ THIS if the skill involves specific tools/platforms)

Some skills (e.g., SQL querying, database operations, etc.) normally require
specific infrastructure that may not exist in the raw data files.

If the assigned skill falls into this category:
1. FIRST assess whether the skill can be adapted to work with the available
   files through a programmatic conversion step (e.g., load CSVs into SQLite,
   then use SQL; convert shapefile attributes to a database table)
2. If YES: include the conversion as part of your code_snippet. The step
   should both convert the data AND apply the skill's core technique
3. If NO (the skill fundamentally requires interactive/GUI tools like
   Tableau dashboards, Excel VBA macros, etc. that cannot run in a Python
   script): indicate this by setting step_description to "NOT_FEASIBLE: [reason]"

### Power BI / DAX Skills — Use Real Tools, NOT Pandas Substitutes
When the skill involves Power BI or DAX, you MUST generate code that uses
actual Power BI / DAX Python tooling. Do NOT reimplement DAX logic using
plain pandas — the goal is to test whether an agent can correctly call
Power BI / DAX related tools.

Recommended approach:
1. Load data files (CSV/Excel) into a local tabular model or in-memory
   structure that supports DAX queries
2. Define relationships, measures, and calculated columns using DAX syntax
3. Execute DAX queries to produce the final result

Available Python libraries (use the most appropriate ones):
- `pyadomd`: Execute DAX queries against Analysis Services / Power BI datasets
- `pytabular`: Python wrapper for Tabular Object Model (TOM); create/modify
  tabular models, define measures, relationships, calculated columns
- `semantic-link` (`sempy`): Microsoft Fabric / Power BI integration;
  `sempy.fabric.evaluate_dax()`, `sempy.fabric.read_table()`
- `powerbiclient`: Embed and interact with Power BI reports from Python
- `ssas_api`: Local SSAS Tabular instance management

Example patterns for code_snippet:
```python
# Pattern A: Use pytabular to build a model and evaluate DAX
import pytabular as p
model = p.Tabular(CONNECTION_STRING)
result = model.query("EVALUATE SUMMARIZECOLUMNS(...)")

# Pattern B: Use pyadomd to execute DAX queries
from pyadomd import Pyadomd
conn = Pyadomd(CONNECTION_STRING)
conn.cursor().execute("EVALUATE CALCULATETABLE(...)")

# Pattern C: Use sempy for Fabric/Power BI service
import sempy.fabric as fabric
result = fabric.evaluate_dax(dataset="...", dax_string="EVALUATE ...")
```

The code_snippet should demonstrate the FULL workflow:
- Loading raw data into the tabular model (or connecting to a dataset)
- Defining DAX measures / calculated columns / relationships as needed
- Executing DAX queries to produce verifiable results

This adaptation must be GENUINE — the code must actually invoke Power BI / DAX
tooling, not just wrap pandas operations in DAX-sounding variable names.

{upstream_section}
## Task
Generate the final analysis step that:
1. Uses {main_skill.name} as the main technique
2. Produces a clear, verifiable answer
3. ONLY references columns and files listed in the Dataset section above
4. Maximizes analytical complexity to challenge AI agents — real-world plausibility is NOT a priority; intricate multi-file, multi-condition pipelines are preferred even if they seem contrived

REQUIREMENT: Your analysis MUST use data from at least {self.min_files} different files
from the dataset. Design a multi-file analysis that combines/compares data across files
using shared columns or logical relationships. Single-file analysis is NOT acceptable
when multiple related files are available.
Whenever possible, recruit additional files from the Available File Pool that are
relevant to this skill and can be joined with already-selected files.
{self._build_required_files_constraint()}
CROSS-FOLDER PREFERENCE: If the Domain Data Connection Guide above explicitly
describes how data in different subfolders can be connected, then, guided by the
skill tree and the overall analysis goal, try to design the workflow so that it
uses data from multiple subfolders in this domain.
You do NOT need to mechanically force every subfolder to appear; prioritize
relevant and technically meaningful cross-folder joins or comparisons that
increase the analytical difficulty of the task, while still remaining executable
on the given datasets.

CRITICAL: Do NOT invent file names. Your code_snippet must ONLY call pd.read_csv() (or
similar) on files listed in the Dataset section above. Use in-memory variables from
upstream steps instead of fabricating intermediate CSV names like 'cleaned_*.csv' or
'processed_*.csv'.

Choose whatever type best fits the analysis goal.

Respond in JSON format only:
{{
    "step_description": "detailed description of what this final step does",
    "input_data": "what data/columns this step needs",
    "output_description": "what the final answer looks like",
    "code_snippet": "Python code for this step",
    "answer_type": "the output type (e.g., number, dataframe, visualization, etc.)",
    "answer_description": "what exactly the answer represents",
    "files_used": ["list of file names this step uses"]
}}
"""
        result = self.llm.call_json(prompt)
        
        if result:
            answer_type = result.get('answer_type', 'number')
            step = SolutionStep(
                step_number=0,  # Will be assigned later
                skill=main_skill,
                description=result.get('step_description', f"Apply {main_skill.name}"),
                code_snippet=result.get('code_snippet', ''),
                input_data=result.get('input_data', 'dataset'),
                output_description=result.get('output_description', 'final answer'),
                expected_output_type=answer_type,
                dependent_step_ids=[]  # Main step's dependencies will be set after auxiliary steps are added
            )
            step._answer_description = result.get('answer_description', '')
            step._files_used = result.get('files_used', [])
            step._referenced_examples = examples_ids
            return step, answer_type
        else:
            # Fallback
            step = SolutionStep(
                step_number=0,
                skill=main_skill,
                description=f"Apply {main_skill.name} to get final result",
                code_snippet='',
                input_data='dataset',
                output_description='analysis result',
                expected_output_type='number',
                dependent_step_ids=[]
            )
            step._answer_description = ''
            step._files_used = []
            step._referenced_examples = examples_ids
            return step, 'number'
    
    def _generate_and_insert_step(
        self,
        new_skill: Skill,
        existing_steps: List[SolutionStep],
        exploration: Dict,
        previous_failure: str = None,
        main_skill: Skill = None,
    ) -> List[SolutionStep]:
        """
        Generate a new step for the given skill and insert it at the correct position.
        
        Args:
            new_skill: The skill to generate a step for
            existing_steps: Current list of steps in the workflow
            exploration: Dataset exploration info
            previous_failure: Reason the previous attempt was rejected (for retry feedback)
            main_skill: The main (final-answer) skill — auxiliary steps must be
                inserted before it so that it always stays last.
            
        Returns:
            New list of steps with the new step inserted at the appropriate position.
            May also modify existing steps to create proper data flow.
        """
        # Locate the main step so we can keep it as the final step
        main_step_idx = len(existing_steps) - 1  # default: assume last
        if main_skill and existing_steps:
            for i, s in enumerate(existing_steps):
                if s.skill.id == main_skill.id:
                    main_step_idx = i
                    break

        # Describe existing steps with indices for reference (include code so LLM
        # can see actual variable names, libraries used, and output objects)
        existing_steps_parts = []
        for i, s in enumerate(existing_steps):
            part = (
                f"[{i}] {s.skill.name}\n"
                f"    Description: {s.description}\n"
                f"    Input: {s.input_data}\n"
                f"    Output: {s.output_description}"
            )
            if s.code_snippet:
                snippet = s.code_snippet
                if len(snippet) > 600:
                    snippet = snippet[:600] + "\n    # ... (truncated)"
                part += f"\n    Code:\n    ```python\n    {snippet}\n    ```"
            existing_steps_parts.append(part)
        existing_steps_desc = "\n".join(existing_steps_parts)
        likely_final_idx = main_step_idx
        likely_final_skill = existing_steps[likely_final_idx].skill.name if existing_steps else "N/A"
        
        # Get examples for this specific skill
        examples_text, examples_ids = self._get_examples_for_skill(new_skill.name)
        so_section = (
            f"\n## Reference Examples for {new_skill.name}\n"
            f"NOTE: These examples are ONLY for illustrating the TECHNIQUE/METHOD.\n"
            f"Do NOT copy column names, file names, or variable names from these examples.\n"
            f"You MUST use the actual column/file names from the Dataset section above.\n\n"
            f"{examples_text}"
        ) if examples_text else ""
        
        failure_section = ""
        if previous_failure:
            failure_section = f"""
## Previous Attempt Failed - You MUST Avoid These Mistakes
{previous_failure}
Re-generate the step, making sure to ONLY use column names and file names that
actually exist in the dataset description above. Double-check every column reference.
"""
        
        skill_desc_section = ""
        if new_skill.description:
            skill_desc_section = f"- Description: {new_skill.description}\n"

        format_constraint = self._build_format_constraint(exploration)

        prompt = f"""You are a data science expert. Add a step to an existing workflow at the CORRECT position.

## New Skill to Add
- Name: {new_skill.name}
{skill_desc_section}{so_section}
---
{failure_section}
## Dataset (USE ONLY THESE ACTUAL COLUMNS AND FILES)
- Domain: {exploration['domain']}

{self._build_file_sections_for_prompt(exploration)}
{format_constraint}
## Infrastructure Adaptation (READ THIS if the skill involves specific tools/platforms)

Some skills (e.g., SQL querying, database operations, etc.) normally require
specific infrastructure that may not exist in the raw data files.

If the assigned skill falls into this category:
1. FIRST assess whether the skill can be adapted to work with the available
   files through a programmatic conversion step (e.g., load CSVs into SQLite,
   then use SQL; convert shapefile attributes to a database table)
2. If YES: include the conversion as part of your code_snippet. The step
   should both convert the data AND apply the skill's core technique
3. If NO (the skill fundamentally requires interactive/GUI tools like
   Tableau dashboards, Excel VBA macros, etc. that cannot run in a Python
   script): indicate this by setting step_description to "NOT_FEASIBLE: [reason]"

### Power BI / DAX Skills — Use Real Tools, NOT Pandas Substitutes
When the skill involves Power BI or DAX, you MUST generate code that uses
actual Power BI / DAX Python tooling. Do NOT reimplement DAX logic using
plain pandas — the goal is to test whether an agent can correctly call
Power BI / DAX related tools.

Recommended approach:
1. Load data files (CSV/Excel) into a local tabular model or in-memory
   structure that supports DAX queries
2. Define relationships, measures, and calculated columns using DAX syntax
3. Execute DAX queries to produce the final result

Available Python libraries (use the most appropriate ones):
- `pyadomd`: Execute DAX queries against Analysis Services / Power BI datasets
- `pytabular`: Python wrapper for TOM; create/modify tabular models, define
  measures, relationships, calculated columns
- `semantic-link` (`sempy`): Microsoft Fabric / Power BI integration;
  `sempy.fabric.evaluate_dax()`, `sempy.fabric.read_table()`
- `powerbiclient`: Embed and interact with Power BI reports from Python
- `ssas_api`: Local SSAS Tabular instance management

The code_snippet should demonstrate the FULL workflow:
- Loading raw data into the tabular model (or connecting to a dataset)
- Defining DAX measures / calculated columns / relationships as needed
- Executing DAX queries to produce verifiable results

This adaptation must be GENUINE — the code must actually invoke Power BI / DAX
tooling, not just wrap pandas operations in DAX-sounding variable names.

## Existing Steps (in execution order, index 0 = first to execute)
{existing_steps_desc}

Likely FINAL step currently in workflow:
- Index: [{likely_final_idx}]
- Skill: {likely_final_skill}
- This step should typically consume upstream outputs, not provide inputs to earlier preprocessing steps.

## Task
1. Create a new step using {new_skill.name}
2. Determine WHERE to insert this step in the workflow:
   - It should come BEFORE any step that depends on its output
   - It should come AFTER any step whose output it needs
3. Identify which existing steps this new step depends on (uses their output)
4. If needed, modify existing steps to use this new step's output

## Important
- The new step does NOT have to be at the beginning!
- Think carefully about the logical data flow
- Example: If step [2] needs the output of the new step, insert at position 2
- ONLY reference columns and files listed in the Dataset section above
- Dependencies must be backward-only:
  - depends_on_steps can only include steps that execute earlier than the new step
  - If insert_before_index = k, then each dependency must be < k
  - If insert_before_index = 0, depends_on_steps MUST be []
- Do NOT reference any intermediate variable unless it is produced by one of depends_on_steps
- Never claim to use an upstream merged/processed dataframe if that dataframe is not yet created
- Do NOT invent intermediate file names (e.g. 'cleaned_*.csv', 'processed_*.csv', 'merged_*.csv').
  Your code should read raw files from the Dataset section OR consume in-memory variables
  produced by depends_on steps. Never call pd.read_csv() on a file that does not appear in
  the Dataset section above.
- Code consistency rules:
  - Your code_snippet must only import libraries that are standard, widely available, and not deprecated
  - Every variable your code reads must either be loaded from a raw file in this step or produced by a depends_on step (check the Code blocks above)
  - Your output variables must be saved/returned so downstream steps can use them
  - If the workflow already uses a specific ML framework (e.g., TensorFlow/Keras, Scikit-Learn), stay consistent with that framework unless there is a strong reason to switch

IMPORTANT: Whenever possible, recruit additional files from the Available File Pool
that are relevant to this skill and can be joined with already-selected files.
Prefer loading/joining new files to increase the breadth of data sources used in the
overall workflow.
{self._build_required_files_constraint()}
CROSS-FOLDER PREFERENCE: If the Domain Data Connection Guide above explicitly
describes how data in different subfolders can be connected, then, guided by the
skill tree and the overall analysis goal, try to design the workflow so that it
uses data from multiple subfolders in this domain.
You do NOT need to mechanically force every subfolder to appear; prioritize
relevant and technically meaningful cross-folder joins or comparisons that
increase the analytical difficulty of the task, while still remaining executable
on the given datasets.

Respond in JSON format only:
{{
    "new_step": {{
        "step_description": "what this step does",
        "input_data": "what data this step needs",
        "output_description": "what this step produces",
        "code_snippet": "Python code for this step",
        "depends_on_steps": [<list of indices of existing steps this new step depends on, e.g. [0, 1] if it needs output from step 0 and 1>],
        "files_used": ["list of file names this step reads or operates on"]
    }},
    "insert_before_index": <integer index of the step this should come BEFORE, or -1 to append at end>,
    "insert_before_skill": "name of the existing step this should come before (for verification)",
    "reason": "why this position makes sense in the data flow",
    "modify_existing": [
        {{
            "step_index": <integer index of step to modify>,
            "skill_name": "name of step to modify",
            "new_input_data": "updated input_data (or null)",
            "new_description": "updated description (or null)",
            "new_output_description": "updated output_description (or null)",
            "new_code_snippet": "updated Python code (or null)"
        }}
    ]
}}
"""
        result = self.llm.call_json(prompt)
        
        if result and 'new_step' in result:
            new_step_data = result['new_step']
            new_step = SolutionStep(
                step_number=0,
                skill=new_skill,
                description=new_step_data.get('step_description', f"Apply {new_skill.name}"),
                code_snippet=new_step_data.get('code_snippet', ''),
                input_data=new_step_data.get('input_data', 'dataset'),
                output_description=new_step_data.get('output_description', 'processed data'),
                expected_output_type='dataframe',
                dependent_step_ids=new_step_data.get('depends_on_steps', [])
            )
            new_step._files_used = new_step_data.get('files_used', [])
            new_step._referenced_examples = examples_ids

            # Determine insertion position
            insert_idx = result.get('insert_before_index', 0)
            insert_skill_name = result.get('insert_before_skill', '')
            insert_reason = result.get('reason', '')
            
            # Validate and find correct insertion index
            if insert_idx == -1:
                # Append at end — but before the main step so it stays last
                insert_idx = main_step_idx
            elif insert_idx < 0 or insert_idx > len(existing_steps):
                # Invalid index, try to find by skill name
                insert_idx = 0
                for i, step in enumerate(existing_steps):
                    if insert_skill_name and (
                        step.skill.name.lower() in insert_skill_name.lower() or 
                        insert_skill_name.lower() in step.skill.name.lower()
                    ):
                        insert_idx = i
                        break
            
            # Never insert after the main step — it must stay last
            if insert_idx > main_step_idx:
                insert_idx = main_step_idx
            
            # Apply modifications to existing steps
            modifications = result.get('modify_existing', [])
            if modifications:
                for mod in modifications:
                    if not mod:
                        continue
                    
                    # Try to find step by index first, then by name
                    target_step = None
                    mod_idx = mod.get('step_index')
                    skill_name = mod.get('skill_name', '')
                    
                    if mod_idx is not None and 0 <= mod_idx < len(existing_steps):
                        target_step = existing_steps[mod_idx]
                    else:
                        for step in existing_steps:
                            if step.skill.name.lower() in skill_name.lower() or skill_name.lower() in step.skill.name.lower():
                                target_step = step
                                break
                    
                    if target_step:
                        if mod.get('new_input_data'):
                            target_step.input_data = mod['new_input_data']
                        if mod.get('new_description'):
                            target_step.description = mod['new_description']
                        if mod.get('new_output_description'):
                            target_step.output_description = mod['new_output_description']
                        if mod.get('new_code_snippet'):
                            target_step.code_snippet = mod['new_code_snippet']
            
            # Insert at the determined position and remap dependencies
            result = existing_steps[:insert_idx] + [new_step] + existing_steps[insert_idx:]
            self._remap_dependencies_after_insert(result, insert_idx)
            return result
        else:
            # Fallback: create basic step and insert at beginning
            new_step = SolutionStep(
                step_number=0,
                skill=new_skill,
                description=f"Apply {new_skill.name} to prepare data",
                code_snippet='',
                input_data='dataset',
                output_description='processed data',
                expected_output_type='dataframe',
                dependent_step_ids=[]
            )
            new_step._files_used = []
            new_step._referenced_examples = examples_ids
            result = [new_step] + existing_steps
            self._remap_dependencies_after_insert(result, 0)
            return result
    
    # ---- Post-generation file recruitment helpers ----

    @staticmethod
    def _pick_best_step_for_file_recruitment(steps: List[SolutionStep]) -> int:
        """Choose the step most suitable for incorporating additional files.

        Heuristic: prefer earlier data-loading / merging / preprocessing steps
        that already read files (they are the easiest to extend).  Falls back to
        the first step.
        """
        merge_keywords = {'merge', 'join', 'concat', 'combine', 'load', 'read', 'import'}
        best_idx, best_score = 0, -1
        for i, step in enumerate(steps):
            text = ((step.description or '') + ' ' + (step.skill.name or '')).lower()
            score = sum(1 for kw in merge_keywords if kw in text)
            if step.code_snippet and 'read_csv' in step.code_snippet:
                score += 2
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _recruit_files_into_step(
        self,
        step: SolutionStep,
        unused_files: List[str],
        exploration: Dict,
        all_steps: List[SolutionStep],
        step_idx: int,
    ) -> str | None:
        """Ask the LLM to rewrite a step's code_snippet to load additional files.

        Returns the new code_snippet on success, or None.
        """
        all_info = self._get_all_files_info()
        unused_info = [f for f in all_info if f['name'] in unused_files]
        unused_summary = "\n".join(format_file_info_line(f) for f in unused_info[:10])

        prompt = f"""You are a data science expert. The current solution step below does not use
enough data files.  Rewrite ONLY the code_snippet so that it ALSO loads and
meaningfully uses at least {self.min_files - len(self._selected_files)} more
files from the "Additional Files" list.

## Current Step
- Skill: {step.skill.name}
- Description: {step.description}
- Code:
```python
{step.code_snippet}
```

## Additional Files (pick from these — load them with pd.read_csv / pd.read_excel etc.)
{unused_summary}

## Domain: {exploration['domain']}

Rules:
1. Keep ALL existing logic intact — only ADD new file loads and incorporate
   them (merge, concat, compare, etc.) in a meaningful way.
2. Use the EXACT file names shown above.
3. Return ONLY the updated Python code as a JSON string — no explanation.

Respond in JSON:
{{"code_snippet": "updated Python code"}}
"""
        result = self.llm.call_json(prompt)
        if result and result.get('code_snippet'):
            return result['code_snippet']
        return None

    def _finalize_steps(self, steps: List[SolutionStep]) -> List[SolutionStep]:
        """
        Finalize step order, assign step numbers, and adjust dependencies.
        
        Instead of discarding all dependency info and re-inferring from scratch,
        this preserves the dependencies established during step generation
        (which had full context: dataset info, code snippets, insertion position)
        and only applies index remapping, sanity checks, and light LLM validation.
        
        Pipeline:
        1. Assign 1-based step numbers
        2. Convert 0-based index deps → 1-based step numbers, filter invalid refs
        3. Gap-fill: ensure every non-first step has at least one dependency
        4. Light LLM validation: only fix obvious errors, don't regenerate
        """
        if not steps:
            return steps
        
        # Pass 1: Assign step numbers (1-based)
        for i, step in enumerate(steps):
            step.step_number = i + 1
        
        # Pass 2: Convert 0-based indices to 1-based step numbers and clean up
        for step in steps:
            converted = []
            for dep in step.dependent_step_ids:
                step_num = dep + 1  # 0-based index → 1-based step number
                # Must be: positive, within range, and strictly before current step
                if 1 <= step_num < step.step_number:
                    converted.append(step_num)
            step.dependent_step_ids = sorted(set(converted))  # deduplicate and sort
        
        # Pass 3: Gap-fill
        # Step 1 must have no deps. Any other step with no deps gets a default
        # dependency on its immediate predecessor (since the generation phase
        # placed it at this position for a reason, the previous step's output
        # is the most likely input).
        for step in steps:
            if step.step_number == 1:
                step.dependent_step_ids = []
            elif not step.dependent_step_ids:
                step.dependent_step_ids = [step.step_number - 1]
        
        # Pass 4: Light LLM validation (only fix clear errors)
        steps = self._validate_dependencies_with_llm(steps)
        
        return steps
    
    def _validate_dependencies_with_llm(self, steps: List[SolutionStep]) -> List[SolutionStep]:
        """
        Light validation of the dependency graph using LLM.
        
        Unlike the old approach which regenerated ALL dependencies from scratch
        using only step descriptions (losing generation-phase context), this method:
        - Shows the LLM the EXISTING dependency graph
        - Asks it to ONLY fix clear errors (e.g., missing obvious data flow links)
        - Preserves correct dependencies from the generation phase
        - Gracefully falls back to current deps if LLM fails
        """
        if not self.llm or len(steps) <= 1:
            return steps
        
        steps_description = "\n".join([
            f"Step {step.step_number}: [{step.skill.name}]\n"
            f"  Description: {step.description}\n"
            f"  Input: {step.input_data}\n"
            f"  Output: {step.output_description}\n"
            f"  Current dependencies: Step {step.dependent_step_ids}"
            for step in steps
        ])
        
        prompt = f"""Review the dependency graph for these data analysis solution steps.
The dependencies were determined during step generation with full dataset context.
Your job is to check for OBVIOUS errors only — do NOT restructure the entire graph.

## Solution Steps (in execution order)
{steps_description}

## Validation Rules
1. Step 1 must have no dependencies (it reads raw data)
2. Each step should depend on steps whose output it actually needs as input
3. No step may depend on itself or on later steps
4. Every step (except Step 1) must have at least one dependency
5. If a step's input clearly comes from a specific earlier step's output, that link should exist

## Important
- The current dependencies are MOSTLY CORRECT — they were set with full context
- Only suggest changes if there is a CLEAR mismatch between a step's input and its dependencies
- Do NOT restructure the graph just because you would have done it differently
- When in doubt, KEEP the existing dependency

Respond in JSON:
{{
    "is_valid": true/false,
    "fixes": [
        {{
            "step_id": <step number to fix>,
            "corrected_dependencies": [<corrected list of step numbers>],
            "reason": "brief explanation of the clear error"
        }}
    ]
}}

If the graph looks reasonable, return {{"is_valid": true, "fixes": []}}.
"""
        result = self.llm.call_json(prompt)
        
        if result and result.get('fixes'):
            for fix in result['fixes']:
                step_id = fix.get('step_id')
                new_deps = fix.get('corrected_dependencies', [])
                reason = fix.get('reason', '')
                
                if not step_id:
                    continue
                
                for step in steps:
                    if step.step_number == step_id:
                        # Validate the proposed fix before applying
                        valid_deps = sorted(set(
                            d for d in new_deps
                            if isinstance(d, int) and 1 <= d < step_id
                        ))
                        
                        # Safety: step 1 must keep empty deps
                        if step_id == 1:
                            valid_deps = []
                        
                        # Safety: non-first steps must have at least one dep
                        if step_id > 1 and not valid_deps:
                            valid_deps = [step_id - 1]
                        
                        step.dependent_step_ids = valid_deps
                        print(f"        Dep fix: Step {step_id} → {valid_deps} ({reason})")
                        self._log("dependency_fix", step_id=step_id,
                                  new_deps=valid_deps, reason=reason)
                        break
        
        return steps
    
    def create_solution_path(
        self,
        main_skill: Skill,
        auxiliary_skills: List[Skill],
        exploration: Dict,
        target_steps: int = None,
    ) -> Tuple[SolutionPath, str, List[Skill], List[Skill]]:
        """
        Create a complete solution path with dynamic skill selection and applicability checking.
        
        Returns:
            Tuple of (SolutionPath, answer_type, skipped_skills, blacklisted_skills)
        """
        steps, answer_type, skipped_skills, blacklisted_skills = self.generate_steps_dynamic(
            main_skill, auxiliary_skills, exploration, target_steps=target_steps,
        )
        
        solution_path = SolutionPath(steps=steps, main_skill=main_skill)
        return solution_path, answer_type, skipped_skills, blacklisted_skills
