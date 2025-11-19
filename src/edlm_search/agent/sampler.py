from __future__ import annotations

from typing import Final

from .candidate import Candidate
from .candidates_database import CandidateDatabase
from .candidates_database import CandidateRecord
from .llm_pipeline import LLMPipeline
from .problem import Problem


class CandidateSampler:
    """Класс, отвечающий за генерацию новых кандидатов с помощью LLM."""

    def __init__(self, llm_pipeline: LLMPipeline, problem: Problem):
        """
        Инициализирует сэмплер кандидатов.

        Параметры
        ----------
        llm_pipeline : LLMPipeline
            Конвейер для взаимодействия с LLM.
        problem : Problem
            Описание задачи, на основе которого генерируются кандидаты.
        """
        self._llm_pipeline: Final[LLMPipeline] = llm_pipeline
        self._problem: Final[Problem] = problem

    async def create_initial_candidate(self) -> Candidate:
        """
        Создаёт нового кандидата с нуля на основе постановки задачи.

        Возвращает
        ----------
        Candidate
            Новый кандидат, сгенерированный из шаблона `new_candidate`.
        """
        candidate = await Candidate.new_from_problem(
                problem=self._problem,
                llm_pipeline=self._llm_pipeline,
        )
        return candidate

    async def crossover_candidates(
            self,
            parent_a: CandidateRecord,
            parent_b: CandidateRecord,
    ) -> Candidate:
        """
        Генерирует нового кандидата путём скрещивания двух родительских решений.

        Параметры
        ----------
        parent_a : CandidateRecord
            Первый родитель-кандидат.
        parent_b : CandidateRecord
            Второй родитель-кандидат.

        Возвращает
        ----------
        Candidate
            Новый кандидат, созданный на основе двух родителей.
        """
        idea, files = await self._llm_pipeline.generate_files_from_template(
                template_name="crossover_candidate",
                problem=self._problem,
                parent_a_idea=parent_a.idea,
                parent_b_idea=parent_b.idea,
                parent_a_metrics=parent_a.metrics,
                parent_b_metrics=parent_b.metrics,
                parent_a_files=parent_a.candidate.files,
                parent_b_files=parent_b.candidate.files,
        )
        return Candidate(files=files, idea=idea)

    def select_parents_for_crossover(
            self,
            database: CandidateDatabase,
            metric_name: str,
            top_k: int,
    ) -> tuple[CandidateRecord, CandidateRecord]:
        """
        Выбирает двух лучших родителей из базы для скрещивания по указанной метрике.

        Параметры
        ----------
        database : CandidateDatabase
            База данных с кандидатами и их метриками.
        metric_name : str
            Имя метрики, по которой выбираются родители.
        top_k : int
            Число лучших кандидатов, среди которых выбираются два родителя.

        Возвращает
        ----------
        tuple[CandidateRecord, CandidateRecord]
            Пара выбранных родительских кандидатов.

        Исключения
        ----------
        ValueError
            Если база не содержит достаточно кандидатов.
        """
        if len(database) < 2:
            raise ValueError("Для скрещивания требуется минимум два кандидата.")

        best_candidates = database.top_k_by_metric(metric_name=metric_name, k=top_k)
        if len(best_candidates) < 2:
            raise ValueError("Недостаточно кандидатов с указанной метрикой для скрещивания.")

        parent_a = best_candidates[0]
        parent_b = best_candidates[1]
        return parent_a, parent_b