from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final
from typing import Iterable

import pandas as pd

from .candidate import Candidate


@dataclass
class CandidateRecord:
    """Структура для хранения результата оценки одного кандидата."""

    candidate_id: str
    candidate: Candidate
    idea: str
    metrics: dict[str, float]
    backend_name: str
    created_at: datetime


class CandidateDatabase:
    """Простая in-memory база данных для хранения и выборки кандидатов."""

    def __init__(self):
        """Инициализирует пустую базу кандидатов."""
        self._records: list[CandidateRecord] = []

    def add_result(
            self,
            candidate: Candidate,
            metrics: dict[str, float],
            backend_name: str,
    ) -> CandidateRecord:
        """
        Добавляет результат оценки кандидата в базу.

        Параметры
        ----------
        candidate : Candidate
            Экземпляр кандидата, сгенерированный LLM.
        metrics : dict[str, float]
            Словарь метрик качества модели.
        backend_name : str
            Имя используемого LLM-бэкенда (например, 'deepseek' или 'lmstudio').

        Возвращает
        ----------
        CandidateRecord
            Созданная запись о кандидате.
        """
        if not metrics:
            raise ValueError("Словарь метрик не может быть пустым.")

        candidate_id: Final[str] = f"cand-{len(self._records) + 1}"
        record = CandidateRecord(
                candidate_id=candidate_id,
                candidate=candidate,
                idea=candidate.idea,
                metrics=dict(metrics),
                backend_name=backend_name,
                created_at=datetime.utcnow(),
        )
        self._records.append(record)
        return record

    def list_records(self) -> list[CandidateRecord]:
        """
        Возвращает все записи о кандидатах.

        Возвращает
        ----------
        list[CandidateRecord]
            Список всех сохранённых записей.
        """
        return list(self._records)

    def top_k_by_metric(self, metric_name: str, k: int) -> list[CandidateRecord]:
        """
        Возвращает top-k кандидатов по указанной метрике (меньше — лучше).

        Параметры
        ----------
        metric_name : str
            Имя метрики, по которой будет происходить сортировка.
        k : int
            Количество лучших кандидатов для возврата.

        Возвращает
        ----------
        list[CandidateRecord]
            Отсортированный список из не более чем k кандидатов.

        Исключения
        ----------
        ValueError
            Если k меньше единицы или если нет ни одной записи с указанной метрикой.
        """
        if k < 1:
            raise ValueError("Параметр k должен быть не меньше 1.")

        filtered = [r for r in self._records if metric_name in r.metrics]
        if not filtered:
            raise ValueError(f"В базе нет кандидатов с метрикой '{metric_name}'.")

        sorted_records = sorted(filtered, key=lambda r: r.metrics[metric_name])
        return sorted_records[:k]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Возвращает содержимое базы кандидатов в виде DataFrame.

        Возвращает
        ----------
        pandas.DataFrame
            Таблица с основными полями кандидатов и метрик.
        """
        rows: list[dict] = []
        for record in self._records:
            row: dict[str, object] = {
                "candidate_id": record.candidate_id,
                "idea": record.idea,
                "backend_name": record.backend_name,
                "created_at": record.created_at,
            }
            for metric_name, metric_value in record.metrics.items():
                row[metric_name] = metric_value
            rows.append(row)
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        """
        Возвращает количество кандидатов в базе.

        Возвращает
        ----------
        int
            Число записей.
        """
        return len(self._records)

    def iter_records(self) -> Iterable[CandidateRecord]:
        """
        Возвращает итератор по всем записям в базе.

        Возвращает
        ----------
        Iterable[CandidateRecord]
            Итератор по записям.
        """
        return iter(self._records)