# edlm_search/candidates_database.py
from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class CandidateRecord:
    """Result of a single candidate evaluation."""
    candidate_id: int
    idea: str
    metrics: dict[str, float]
    metadata: dict[str, object] = field(default_factory=dict)


class CandidateDatabase:
    """In-memory store of all evaluated candidates.

    This class is intentionally lightweight and keeps data both in
    a list of records and in a pandas.DataFrame for convenient
    downstream analysis.
    """

    def __init__(self) -> None:
        self._records: list[CandidateRecord] = []
        self._df = pd.DataFrame(
                columns=['candidate_id', 'idea', 'metrics', 'metadata'],
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a copy of the internal dataframe with all results."""
        return self._df.copy()

    def add_result(
            self,
            candidate_id: int,
            idea: str,
            metrics: Mapping[str, float],
            metadata: dict[str, object] | None = None,
    ) -> None:
        """Add a new evaluation result to the store.

        The method validates that at least one metric is provided and that
        all metric values are finite numeric scalars.
        """
        if not metrics:
            raise ValueError('Набор метрик пуст — результат оценки кандидата не может быть сохранён.')

        validated_metrics: dict[str, float] = {}
        for name, value in metrics.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                        f'Значение метрики {name!r} не может быть приведено к числу.'
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(
                        f'Значение метрики {name!r} должно быть конечным числом.'
                )
            validated_metrics[name] = numeric_value

        if metadata is None:
            metadata = {}

        record = CandidateRecord(
                candidate_id=candidate_id,
                idea=idea,
                metrics=validated_metrics,
                metadata=dict(metadata),
        )
        self._records.append(record)

        new_row = {
            'candidate_id': candidate_id,
            'idea': idea,
            'metrics': validated_metrics,
            'metadata': dict(metadata),
        }
        self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)

    def top_k_by_metric(self, metric_name: str, k: int) -> Iterable[CandidateRecord]:
        """Return top-k candidates sorted by the specified metric (ascending)."""
        if k <= 0:
            raise ValueError('Число кандидатов k должно быть положительным.')
        if not self._records:
            return []

        filtered = [
            record for record in self._records
            if metric_name in record.metrics
        ]
        if not filtered:
            return []

        sorted_records = sorted(filtered, key=lambda r: r.metrics[metric_name])
        return sorted_records[:k]