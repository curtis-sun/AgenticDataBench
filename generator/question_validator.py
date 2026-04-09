"""
Question Validator

Validates the quality of generated evaluation questions.
Uses a checklist-based scoring approach (not subjective LLM scoring).

Scoring:
- 10 criteria total (5 required + 5 optional), each worth 1 point
  (4 module-level REQUIRED + 1 dynamic multi_file added at runtime)
- Pass conditions: ALL required criteria must pass AND total score >= 6
- If failed, retry question generation (max 3 attempts)
"""

import os
from typing import Dict, List, Tuple, Optional
from data_classes import SolutionPath, EvalCase
import sys
sys.path.append('..')
from utils.llm_client import QwenClient


# Required criteria - ALL must pass
REQUIRED_CRITERIA = [
    {
        "id": "clear_answer_type", 
        "description": "Question has a clear, verifiable answer type",
        "check": "Can you determine what type of answer is expected? Is it unambiguous?"
    },
    {
        "id": "uses_dataset",
        "description": "Question references actual dataset files or relevant columns",
        "check": "Does the question mention specific file names from the dataset? Note: a short file name (e.g. 'temp.csv') counts as a match if any dataset file ends with that name (e.g. 'SomeFolder/temp.csv')."
    },
    {
        "id": "verifiable_answer",
        "description": "Answer is objectively verifiable (not subjective judgment)",
        "check": "Can the answer be verified programmatically? Is it objective, not opinion-based?"
    },
    {
        "id": "requires_core_steps",
        "description": "Solution steps are relevant to answering the question",
        "check": """Check if the skills listed in 'Solution Steps' are relevant to answering
the question.

Evaluation rules:
1. Data preprocessing steps (e.g., handling missing values, type conversion,
   data cleaning, normalization, validation) are ALWAYS considered relevant
   as long as they operate on files/data that appear in the pipeline. These
   are standard good practice in data analysis and should NOT cause a FAIL.
2. For non-preprocessing skills, ask: "Does this skill address a meaningful
   aspect of the problem or support the overall analysis pipeline?"
3. FAIL only if: a skill is completely unrelated to the question AND operates
   on data that has no connection to the rest of the pipeline.
4. PASS if: each skill either directly contributes to the answer OR is a
   reasonable preprocessing / supporting step within the data pipeline."""
    },
]

# Optional criteria - bonus points
OPTIONAL_CRITERIA = [
    {
        "id": "concise",
        "description": "Question is concise (7 sentences or fewer)",
        "check": "Count the number of sentences. Is it 7 or fewer?"
    },
    {
        "id": "no_steps_exposed",
        "description": "Question asks about WHAT to find, not HOW to compute it",
        "check": """Determine whether the question describes WHAT to find or tells HOW to compute it.

Default is PASS. FAIL only if the question contains explicit PROCEDURAL instructions:
- Sequential operations: "First merge X with Y, then group by Z, then calculate..."
- Explicit API/method calls: "Use pd.merge()", "Apply one-hot encoding"
- Data manipulation that dictates implementation: "Join on column A", "Pivot the table"
- Revealing join keys: "Merge on column X", "Link files using the 'id' field"
- Multi-step pipelines spelled out in order

Describing the desired result (metrics, output format, conditions, scope) is NOT procedural.
e.g. "Create a DataFrame that includes..." or "save to a CSV" describe WHAT, not HOW."""
    },
    {
        "id": "analytical_complexity",
        "description": "Question demands non-trivial analytical reasoning",
        "check": "Does the question require multi-step reasoning, cross-file data integration, or complex computations? Simple single-step lookups or aggregations should FAIL. The question should be grounded in a plausible analytical scenario while still being challenging enough to genuinely test an AI agent's data analysis capabilities."
    },
    {
        "id": "clear_language",
        "description": "Question is clearly worded and unambiguous in what it asks for",
        "check": "Is the question phrased clearly enough that a skilled data analyst would know exactly what to compute? The scenario should feel realistic and well-motivated, not artificial or contrived. Both clarity and plausibility matter."
    },
    {
        "id": "unambiguous",
        "description": "Question has clear, unambiguous wording",
        "check": "Is there only one way to interpret what is being asked? No confusion?"
    },
]

# Minimum total score to pass (in addition to all required passing)
PASS_THRESHOLD = 6


class QuestionValidator:
    """
    Validates generated evaluation questions using checklist-based scoring.
    
    Total: 10 criteria (5 required + 5 optional), each worth 1 point.
    Pass threshold: 6 points minimum.
    
    Additionally performs a programmatic file-reference check (not LLM-based)
    to ensure the question explicitly names the required data files.
    """
    
    def __init__(self, llm_client: QwenClient, min_files: int = 5):
        self.llm = llm_client
        self.min_files = min_files
        self._build_criteria()
    
    @staticmethod
    def _count_file_references(question: str, file_names: List[str]) -> List[str]:
        """
        Programmatically detect which files from file_names are mentioned in
        the question text (case-insensitive match on basename or full path).
        """
        question_lower = question.lower()
        matched = []
        for name in file_names:
            basename = os.path.basename(name).lower()
            if basename in question_lower or name.lower() in question_lower:
                matched.append(name)
        return matched
    
    def _build_criteria(self):
        """Build criteria lists, injecting current min_files into multi_file criterion."""
        self.required_criteria = REQUIRED_CRITERIA + [
            {
                "id": "multi_file",
                "description": f"Question explicitly references at least {self.min_files} different files",
                "check": f"Count the distinct file names mentioned in the question. Are there at least {self.min_files}? File names include .csv, .xlsx, .json, .parquet, etc."
            }
        ]
        self.optional_criteria = OPTIONAL_CRITERIA
        self.all_criteria = self.required_criteria + self.optional_criteria
        self.pass_threshold = PASS_THRESHOLD

    def _inject_rare_skill_leniency(self, rare_skill_names: List[str]):
        """Replace the requires_core_steps check text with a lenient version
        that forgives loose integration of rare/low-frequency skills."""
        rare_list = ", ".join(rare_skill_names)
        lenient_check = f"""Check if the skills listed in 'Solution Steps' are necessary to answer
the question.

The following skills are RARE / low-frequency and should be judged leniently:
  [{rare_list}]

Evaluation rules:
1. For every NON-RARE (common) skill, ask: "Is this skill clearly relevant to
   answering the question?"  If a common skill is completely unrelated to the
   question, FAIL.
2. For RARE skills listed above, be lenient: even if a rare skill's integration
   with the main pipeline seems loose or its connection to the question is
   indirect, do NOT fail on that basis alone.  Rare skills are inherently harder
   to weave tightly into a pipeline, and minor integration gaps are acceptable.
3. FAIL only if: a COMMON skill is obviously redundant or entirely unrelated to
   the question, OR the overall pipeline has no meaningful connection to the
   question at all.
4. PASS if: every common skill addresses a relevant aspect of the question, and
   rare skills at least have a plausible (even if loose) role in the solution."""
        patched = [
            {**c, "check": lenient_check} if c["id"] == "requires_core_steps" else c
            for c in self.required_criteria
        ]
        self.required_criteria = patched
        self.all_criteria = self.required_criteria + self.optional_criteria
    
    def validate(
        self,
        question: str,
        exploration: Dict,
        solution_path: SolutionPath,
        answer_type: str,
        applicability_result: Dict = None,
        rare_skill_names: List[str] = None,
    ) -> Dict:
        """
        Validate a question against all criteria.
        
        Pass conditions:
        - ALL required criteria must pass
        - Total score must be >= 6
        
        Args:
            applicability_result: Result from ApplicabilityChecker (used in validation)
        
        Returns:
            Dict with validation results
        """
        # Dynamically adjust requires_core_steps criterion for rare skills
        if rare_skill_names:
            self._inject_rare_skill_leniency(rare_skill_names)

        # Build context for LLM (include both full paths and basenames for matching)
        full_paths = [f['name'] for f in exploration['files_info']]
        basenames = set(os.path.basename(p) for p in full_paths)
        file_names = full_paths + [b for b in sorted(basenames) if b not in full_paths]
        all_columns = exploration.get('all_columns', [])[:30]
        
        # Build solution steps description (include code_snippet for accurate judgment)
        steps_parts = []
        for s in solution_path.steps:
            part = f"Step {s.step_number}: [{s.skill.name}] {s.description}"
            if s.code_snippet:
                part += f"\n  Code:\n{s.code_snippet}"
            steps_parts.append(part)
        steps_desc = "\n\n".join(steps_parts)
        
        # Add applicability info if available
        applicability_info = ""
        if applicability_result:
            applicability_info = f"\n## Skill Applicability\n{applicability_result['applicable_count']}/{applicability_result['total_steps']} skills are applicable to the dataset."
        
        # Validate all criteria in one LLM call
        results = self._check_all_criteria(
            question=question,
            context={
                "domain": exploration['domain'],
                "files": file_names,
                "columns": all_columns,
                "answer_type": answer_type,
                "solution_steps": steps_desc,
                "applicability_info": applicability_info
            }
        )
        
        # Programmatic override: verify file references in question text
        matched_files = self._count_file_references(question, full_paths)
        for r in results:
            if r['id'] == 'multi_file' and r['passed']:
                if len(matched_files) < self.min_files:
                    r['passed'] = False
                    r['reason'] = (
                        f"Programmatic check: only {len(matched_files)} file(s) "
                        f"detected in question text ({matched_files}), "
                        f"need at least {self.min_files}"
                    )
                    r['suggestion'] = (
                        f"The question must explicitly name at least {self.min_files} "
                        f"data files. Available files: {[os.path.basename(n) for n in full_paths]}. "
                        f"Do NOT use vague references like 'the given datasets'."
                    )
            if r['id'] == 'uses_dataset' and r['passed']:
                if len(matched_files) == 0:
                    r['passed'] = False
                    r['reason'] = "Programmatic check: no file names detected in question text"
                    r['suggestion'] = (
                        f"The question must reference specific file names. "
                        f"Available: {[os.path.basename(n) for n in full_paths]}"
                    )
        
        # Separate required and optional results
        required_ids = {c['id'] for c in self.required_criteria}
        required_results = [r for r in results if r['id'] in required_ids]
        optional_results = [r for r in results if r['id'] not in required_ids]
        
        # Calculate scores
        required_passed = sum(1 for r in required_results if r['passed'])
        optional_passed = sum(1 for r in optional_results if r['passed'])
        total_score = required_passed + optional_passed
        
        # Pass conditions: ALL required must pass AND total >= 6
        all_required_passed = required_passed == len(self.required_criteria)
        passed = all_required_passed and total_score >= self.pass_threshold
        
        # Collect suggestions for failed criteria
        suggestions = []
        for r in results:
            if not r['passed'] and r.get('suggestion'):
                suggestions.append(f"[{r['id']}] {r['suggestion']}")
        
        # Restore original criteria if we injected rare-skill leniency
        if rare_skill_names:
            self._build_criteria()

        return {
            "passed": passed,
            "all_required_passed": all_required_passed,
            "required_score": required_passed,
            "required_total": len(self.required_criteria),
            "optional_score": optional_passed,
            "optional_total": len(self.optional_criteria),
            "score": total_score,
            "max_score": len(self.all_criteria),
            "threshold": self.pass_threshold,
            "results": results,
            "suggestions": suggestions
        }
    
    def _check_all_criteria(
        self,
        question: str,
        context: Dict
    ) -> List[Dict]:
        """
        Check question against all criteria using LLM.
        """
        # Build criteria checklist for prompt
        criteria_list = "\n".join([
            f"{i+1}. [{c['id']}] {c['description']}\n   Check: {c['check']}"
            for i, c in enumerate(self.all_criteria)
        ])
        
        prompt = f"""You are evaluating the quality of a data science evaluation question.

## Question to Evaluate
"{question}"

## Context
- Domain: {context['domain']}
- Dataset files: {context['files']}
- Available columns (sample): {context['columns'][:20]}
- Expected answer type: {context['answer_type']}

## Solution Steps (required to answer the question)
{context['solution_steps']}
{context.get('applicability_info', '')}

## Criteria Checklist ({len(self.all_criteria)} criteria, each worth 1 point)
First {len(self.required_criteria)} are REQUIRED (must all pass), next {len(self.optional_criteria)} are OPTIONAL (bonus points).

{criteria_list}

## Task
For EACH criterion, determine if the question PASSES or FAILS.

Respond in JSON format only:
{{
    "evaluations": [
        {{
            "id": "criterion_id",
            "passed": true/false,
            "reason": "brief explanation (1 sentence)",
            "suggestion": "how to fix if failed (or null if passed)"
        }},
        ... (10 items total)
    ]
}}
"""
        result = self.llm.call_json(prompt)
        
        if result and 'evaluations' in result:
            # Match results to criteria
            results = []
            eval_map = {e['id']: e for e in result['evaluations']}
            
            for c in self.all_criteria:
                if c['id'] in eval_map:
                    e = eval_map[c['id']]
                    results.append({
                        "id": c['id'],
                        "description": c['description'],
                        "passed": e.get('passed', False),
                        "reason": e.get('reason', ''),
                        "suggestion": e.get('suggestion')
                    })
                else:
                    # Criterion not evaluated, mark as failed
                    results.append({
                        "id": c['id'],
                        "description": c['description'],
                        "passed": False,
                        "reason": "Not evaluated by LLM",
                        "suggestion": "Please check this criterion manually"
                    })
            return results
        else:
            # LLM failed, mark all as failed
            return [
                {
                    "id": c['id'],
                    "description": c['description'],
                    "passed": False,
                    "reason": "LLM validation failed",
                    "suggestion": None
                }
                for c in self.all_criteria
            ]
    
    def validate_with_retry(
        self,
        question: str,
        exploration: Dict,
        solution_path: SolutionPath,
        answer_type: str,
        regenerate_fn,
        max_retries: int = 3,
        applicability_result: Dict = None,
        rare_skill_names: List[str] = None,
    ) -> Tuple[Optional[str], Dict]:
        """
        Validate question with retry logic.
        
        Pass conditions: ALL required must pass AND total >= 6
        
        Returns:
            Tuple of (final_question, validation_result)
            validation_result includes 'attempt_history' with all attempts.
            If all retries fail, returns (None, last_validation_result)
        """
        current_question = question
        attempt_history: List[Dict] = []
        
        for attempt in range(max_retries + 1):
            result = self.validate(
                question=current_question,
                exploration=exploration,
                solution_path=solution_path,
                answer_type=answer_type,
                applicability_result=applicability_result,
                rare_skill_names=rare_skill_names,
            )
            
            attempt_history.append({
                "attempt": attempt + 1,
                "question": current_question,
                "passed": result['passed'],
                "score": result['score'],
                "max_score": result['max_score'],
                "required_score": result['required_score'],
                "required_total": result['required_total'],
                "all_required_passed": result.get('all_required_passed', False),
                "suggestions": result.get('suggestions', []),
                "criteria_results": [
                    {"id": r['id'], "passed": r['passed'], "reason": r.get('reason', '')}
                    for r in result.get('results', [])
                ],
            })
            
            if result['passed']:
                result['attempt_history'] = attempt_history
                return current_question, result
            
            if attempt == max_retries:
                result['attempt_history'] = attempt_history
                return None, result
            
            failure_reasons = []
            if not result['all_required_passed']:
                failure_reasons.append(f"Required: {result['required_score']}/{result['required_total']}")
            if result['score'] < self.pass_threshold:
                failure_reasons.append(f"Total: {result['score']}/{result['max_score']} < {self.pass_threshold}")
            
            print(f"    Failed ({', '.join(failure_reasons)}), regenerating (attempt {attempt + 2}/{max_retries + 1})...")
            if result['suggestions']:
                print(f"    Top issues: {result['suggestions'][:2]}")
            
            current_question = regenerate_fn(result['suggestions'])
        
        result['attempt_history'] = attempt_history
        return None, result


def print_validation_result(result: Dict, verbose: bool = False):
    """Pretty print validation result."""
    status = "✓ PASSED" if result['passed'] else "✗ FAILED"
    
    # Show detailed score breakdown
    required_status = "✓" if result.get('all_required_passed', False) else "✗"
    print(f"  Validation: {status}")
    print(f"    Required: {required_status} {result.get('required_score', 0)}/{result.get('required_total', 5)} (must be all)")
    print(f"    Optional: {result.get('optional_score', 0)}/{result.get('optional_total', 5)}")
    print(f"    Total: {result['score']}/{result['max_score']} (threshold: {result['threshold']})")
    
    if verbose or not result['passed']:
        # Show failed criteria
        failed = [r for r in result['results'] if not r['passed']]
        if failed:
            print(f"  Failed criteria ({len(failed)}):")
            for r in failed[:5]:
                print(f"    ✗ [{r['id']}] {r['reason']}")
