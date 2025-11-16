# edlm_search/evaluator.py
from __future__ import annotations

from typing import Protocol
from typing import TYPE_CHECKING
from typing import runtime_checkable

if TYPE_CHECKING:
    from .candidate import Candidate
    from .runner import UnsafeRunner


@runtime_checkable
class Evaluator(Protocol):
    """A protocol for evaluators that calculate a score based on model predictions."""

    async def evaluate(self, runner: UnsafeRunner, candidate: Candidate) -> dict[str, float]:
        """Calculates and returns a score for a single epoch's predictions."""
        ...