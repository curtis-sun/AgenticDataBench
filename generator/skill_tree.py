"""
Skill Tree Module

Builds skill trees and uses LLM to determine main skill based on dataset context.
"""

import re
from typing import List, Dict, Tuple
from data_classes import Skill, SkillTree
from dataset_loader import format_file_info_line
from step_generator import check_skill_data_compatibility
import sys
sys.path.append('..')
from utils.llm_client import QwenClient


class SkillTreeBuilder:
    """Build skill trees using LLM to determine main skill"""
    
    def __init__(self, llm_client: QwenClient = None):
        self.llm = llm_client
    
    def select_main_skill_with_llm(
        self, 
        skills: List[Skill], 
        exploration: Dict
    ) -> Tuple[Skill, List[Skill], str]:
        """
        Use LLM to select the most appropriate main skill based on dataset characteristics.
        
        The main skill will be the FINAL step in the solution - it produces the final answer.
        
        Args:
            skills: List of candidate skills
            exploration: Dataset exploration info
            
        Returns:
            Tuple of (main_skill, auxiliary_skills, reason)
        """
        if not skills:
            return None, [], "No skills provided"
        
        if len(skills) == 1:
            return skills[0], [], "Only one skill available"
        
        if not self.llm:
            # Fallback to last skill if no LLM
            return skills[-1], skills[:-1], "No LLM available, using last skill"
        
        # Build skills description (include semantic description when available)
        skills_desc_parts = []
        for s in skills:
            line = f"- {s.name} (id: {s.id})"
            if s.description:
                line += f"\n  Description: {s.description}"
            skills_desc_parts.append(line)
        skills_desc = "\n".join(skills_desc_parts)
        
        # Build dataset summary
        files_desc = "\n".join([
            format_file_info_line(f)
            for f in exploration['files_info']
        ])
        
        prompt = f"""You are a data science expert. Select the BEST main skill for a data analysis task.

## Dataset
- Domain: {exploration['domain']}
- Total rows: {exploration['total_rows']}
- Has missing values: {exploration.get('has_missing_values', False)}

## Files
{files_desc}

## Available Skills (in sampled order)
{skills_desc}

## Task
Select ONE skill to be the MAIN SKILL. The main skill will be the FINAL step that produces the answer.

Consider:
1. Which skill is most suitable as the FINAL analysis step for this dataset?
2. Which skill can produce a clear, verifiable output?
3. Which skill makes sense as the culmination of a data analysis workflow?
4. The other skills will be used as preparatory/auxiliary steps leading to this main skill.

## Good main skills typically:
- Produce concrete, verifiable outputs (numbers, tables, visualizations, models, etc.)
- Are analytical in nature (not just data cleaning/preparation/setup)
- Make sense as the "answer-producing" step

## Avoid selecting skills that:
- Only perform intermediate operations (data loading, cleaning, environment setup)
- Don't produce a final verifiable result

Respond in JSON format only:
{{
    "main_skill_name": "exact name of the selected main skill",
    "output_type": "what type of output this skill produces",
    "reason": "why this skill is best as the final step"
}}
"""
        result = self.llm.call_json(prompt)
        
        if result and 'main_skill_name' in result:
            selected_name = result['main_skill_name']
            reason = result.get('reason', '')
            
            # Find matching skill
            for skill in skills:
                if skill.name.lower() == selected_name.lower():
                    auxiliary = [s for s in skills if s.id != skill.id]
                    return skill, auxiliary, reason
                if selected_name.lower() in skill.name.lower() or skill.name.lower() in selected_name.lower():
                    auxiliary = [s for s in skills if s.id != skill.id]
                    return skill, auxiliary, reason
        
        # Fallback to last skill
        return skills[-1], skills[:-1], "LLM selection failed, using last skill"
    
    _STOPWORDS = frozenset({"and", "or", "the", "a", "an", "of", "for", "in", "on", "with", "to"})

    @staticmethod
    def _deduplicate_skills(skills: List[Skill]) -> List[Skill]:
        """Remove semantically overlapping skills, keeping the first occurrence.

        Uses the overlap coefficient (shared / min-set-size) on content words
        to detect near-duplicate skill names like "Model Fitting and Evaluation"
        vs "Model Evaluation Metrics".
        """
        def _tokens(name: str) -> set:
            raw = set(re.sub(r'[^a-z0-9 ]', ' ', name.lower()).split())
            return raw - SkillTreeBuilder._STOPWORDS

        kept: List[Skill] = []
        for skill in skills:
            tokens_s = _tokens(skill.name)
            is_dup = False
            for existing in kept:
                tokens_e = _tokens(existing.name)
                min_size = min(len(tokens_s), len(tokens_e))
                if min_size == 0:
                    continue
                overlap = len(tokens_s & tokens_e) / min_size
                if overlap >= 0.80:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(skill)
        return kept

    def build_skill_tree(
        self, 
        skills: List[Skill], 
        exploration: Dict = None
    ) -> SkillTree:
        """
        Build skill tree with LLM-selected main skill.
        
        Args:
            skills: List of skills from the sampled path
            exploration: Dataset exploration info (required for LLM selection)
        
        Returns:
            SkillTree with main_skill and auxiliary_skills
        """
        # Deduplicate semantically overlapping skills before selection
        original_count = len(skills)
        skills = self._deduplicate_skills(skills)
        if len(skills) < original_count:
            removed = original_count - len(skills)
            print(f"  Deduplicated skills: removed {removed} overlapping skill(s)")

        # Static pre-filter: remove skills incompatible with the data environment
        if exploration:
            filtered = []
            for skill in skills:
                ok, reason = check_skill_data_compatibility(
                    skill.name, skill.description or '', exploration
                )
                if ok:
                    filtered.append(skill)
                else:
                    print(f"  ✗ Static filter removed: {skill.name} ({reason})")
            if len(filtered) < len(skills):
                print(f"  Static filter: removed {len(skills) - len(filtered)} incompatible skill(s)")
            skills = filtered

        if exploration and self.llm:
            # Use LLM to select main skill
            main_skill, auxiliary_skills, reason = self.select_main_skill_with_llm(
                skills, exploration
            )
            print(f"  LLM selected main skill: {main_skill.name}")
            print(f"    Reason: {reason[:80]}..." if len(reason) > 80 else f"    Reason: {reason}")
        else:
            # Fallback: last skill is main skill
            if not skills:
                main_skill = None
                auxiliary_skills = []
            elif len(skills) == 1:
                main_skill = skills[0]
                auxiliary_skills = []
            else:
                main_skill = skills[-1]
                auxiliary_skills = skills[:-1]
        
        return SkillTree(
            main_skill=main_skill,
            auxiliary_skills=auxiliary_skills,
            dependencies={}
        )

