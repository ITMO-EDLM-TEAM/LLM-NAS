# edlm_search/src/edlm_search/agent/candidate.py
from __future__ import annotations

from .llm_pipeline import LLMPipeline
from .problem import Problem
from ..devices import get_torch_device


class Candidate:
    """Represents a candidate solution, including code files and a descriptive idea."""

    def __init__(self, files: dict[str, str], idea: str, *, fix_attempts: int = 0):
        self.files: dict[str, str] = files
        self.idea = idea
        self.fix_attempts = fix_attempts

    @classmethod
    async def new_from_problem(cls, problem: Problem, llm_pipeline: LLMPipeline):
        """Create a new candidate by generating a solution for a given problem using an LLM."""
        device = get_torch_device('auto')
        idea, files = await llm_pipeline.generate_files_from_template(
                'new_candidate',
                problem=problem,
                previous_failure_message=None,
                torch_backend_name=device.type,
        )
        return Candidate(files=files, idea=idea)