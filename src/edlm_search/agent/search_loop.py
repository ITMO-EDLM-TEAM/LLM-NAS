from __future__ import annotations

import logging
from typing import Callable
from typing import Final

from .candidate import Candidate
from .candidates_database import CandidateDatabase
from .evaluator import Evaluator  # или другой Evaluator при расширении
from .runner import UnsafeRunner
from .sampler import CandidateSampler

logger = logging.getLogger(__name__)


class LLMBasedArchitectureSearch:
    """Основной цикл поиска архитектур с помощью LLM и внешнего раннера."""

    def __init__(
            self,
            evaluator: Evaluator,
            sampler: CandidateSampler,
            runner_factory: Callable[[], UnsafeRunner],
            backend_name: str,
            metric_name: str,
            max_candidates: int,
            top_k_for_crossover: int,
    ):
        """
        Инициализирует объект поиска архитектур.

        Параметры
        ----------
        evaluator : ETEvaluator
            Оценщик качества кандидатов для задачи ETT.
        sampler : CandidateSampler
            Сэмплер кандидатов, использующий LLM.
        runner_factory : Callable[[], UnsafeRunner]
            Фабрика, создающая новый экземпляр `UnsafeRunner` для каждого запуска.
        backend_name : str
            Имя используемого LLM-бэкенда (для логов и базы).
        metric_name : str
            Имя основной метрики, по которой ведётся отбор.
        max_candidates : int
            Максимальное количество кандидатов, которое будет оценено в цикле.
        top_k_for_crossover : int
            Количество лучших кандидатов, из которых выбираются родители для скрещивания.
        """
        if max_candidates < 1:
            raise ValueError("Параметр max_candidates должен быть не меньше 1.")
        if top_k_for_crossover < 2:
            raise ValueError("Параметр top_k_for_crossover должен быть не меньше 2.")

        self._evaluator: Final[Evaluator] = evaluator
        self._sampler: Final[CandidateSampler] = sampler
        self._runner_factory: Final[Callable[[], UnsafeRunner]] = runner_factory
        self._backend_name: Final[str] = backend_name
        self._metric_name: Final[str] = metric_name
        self._max_candidates: Final[int] = max_candidates
        self._top_k_for_crossover: Final[int] = top_k_for_crossover
        self._database: Final[CandidateDatabase] = CandidateDatabase()

    @property
    def database(self) -> CandidateDatabase:
        """
        Возвращает базу с результатами всех оценённых кандидатов.

        Возвращает
        ----------
        CandidateDatabase
            База данных кандидатов и их метрик.
        """
        return self._database

    async def _evaluate_single_candidate(self, candidate: Candidate) -> dict[str, float]:
        """
        Оценивает одного кандидата, создавая новый UnsafeRunner.

        Параметры
        ----------
        candidate : Candidate
            Кандидат, чьи файлы будут выполнены и оценены.

        Возвращает
        ----------
        dict[str, float]
            Словарь метрик качества для кандидата.
        """
        runner = self._runner_factory()
        logger.info(f"Запуск оценки кандидата с идеей: {candidate.idea}")
        metrics = await self._evaluator.evaluate(runner=runner, candidate=candidate)
        logger.info(f"Оценка кандидата с идеей '{candidate.idea}' завершена, метрики: {metrics}")
        return metrics

    async def run_search(self) -> CandidateDatabase:
        """
        Запускает полный цикл поиска архитектур с использованием LLM.

        Возвращает
        ----------
        CandidateDatabase
            База данных с результатами всех оценённых кандидатов.
        """
        logger.info(
                f"Старт поиска архитектур, планируется оценить {self._max_candidates} кандидатов."
        )

        while len(self._database) < self._max_candidates:
            logger.info(
                    f"Поиск кандидатов: текущий прогресс {len(self._database)} / "
                    f"{self._max_candidates} успешных кандидатов."
            )

            try:
                if len(self._database) < 2:
                    logger.info("Генерация нового кандидата без скрещивания.")
                    candidate = await self._sampler.create_initial_candidate()
                else:
                    parent_a, parent_b = self._sampler.select_parents_for_crossover(
                            database=self._database,
                            metric_name=self._metric_name,
                            top_k=self._top_k_for_crossover,
                    )
                    logger.info(
                            f"Выбраны родители для скрещивания: {parent_a.candidate_id} "
                            f"и {parent_b.candidate_id}."
                    )
                    candidate = await self._sampler.crossover_candidates(
                            parent_a=parent_a,
                            parent_b=parent_b,
                    )
                    logger.info(
                            f"Сгенерирован новый кандидат после скрещивания, идея: {candidate.idea}"
                    )
            except Exception as exc:
                error_message = (
                    f"Генерация кандидата завершилась ошибкой: {type(exc).__name__}: {exc}"
                )
                logger.exception(
                        f"Ошибка генерации кандидата: {error_message}"
                )
                self._sampler.register_failed_candidate(error_message)
                continue

            try:
                metrics = await self._evaluate_single_candidate(candidate=candidate)
            except Exception as exc:
                error_message = (
                    f"Оценка кандидата завершилась ошибкой: {type(exc).__name__}: {exc}"
                )
                logger.exception(
                        f"Ошибка оценки кандидата с идеей '{candidate.idea}': {error_message}"
                )
                self._sampler.register_failed_candidate(error_message)
                continue

            self._sampler.clear_failure_history()
            record = self._database.add_result(
                    candidate=candidate,
                    metrics=metrics,
                    backend_name=self._backend_name,
            )
            logger.info(
                    f"Кандидат с идеей '{candidate.idea}' сохранён в базе под идентификатором "
                    f"{record.candidate_id} с метриками {metrics}."
            )

        logger.info(
                f"Поиск архитектур завершён: успешно оценено {len(self._database)} кандидатов "
                f"из запланированных {self._max_candidates}."
        )
        return self._database