# edlm_search/ett_evaluator.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

__all__ = ['ETTM1Evaluator']

if TYPE_CHECKING:
    from .candidate import Candidate
    from .runner import UnsafeRunner

_LOGGER = logging.getLogger(__name__)


class ETTM1Evaluator:
    """
    Evaluator for LLM-generated forecasting pipelines working with the ETTm1 dataset.

    The evaluator is responsible for executing a candidate inside the sandboxed runner,
    collecting the validation predictions produced at the end of each epoch, and
    computing the chosen regression metric against ground-truth targets.

    In addition to the regression metric, the evaluator records:
      * total_energy_joules (if provided by the runner)
      * wall_time_seconds (elapsed wall-clock time between process start and last epoch result)
    """

    def __init__(
            self,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame,
            target_column: str,
            metric_name: str,
            num_epochs: int,
    ):
        if train_df.empty:
            raise ValueError('Обучающая выборка пуста, невозможно запустить оценку.')
        if valid_df.empty:
            raise ValueError('Валидационная выборка пуста, невозможно вычислить метрику.')
        if target_column not in train_df.columns:
            raise ValueError(f'Колонка {target_column!r} отсутствует в обучающей выборке.')
        if target_column not in valid_df.columns:
            raise ValueError(f'Колонка {target_column!r} отсутствует в валидационной выборке.')
        if num_epochs < 1:
            raise ValueError('Число эпох должно быть положительным.')

        self._train_df: pd.DataFrame = train_df.reset_index(drop=True).copy()
        self._valid_df: pd.DataFrame = valid_df.reset_index(drop=True).copy()
        self._target_column: str = target_column
        self._metric_name: str = metric_name
        self._num_epochs: int = num_epochs
        self._validation_targets: np.ndarray = np.asarray(
                self._valid_df[self._target_column].to_numpy(),
                dtype=np.float64,
        )

    async def evaluate(self, runner: UnsafeRunner, candidate: Candidate) -> dict[str, float]:
        _LOGGER.info(
                f'Запуск оценки кандидата "{candidate.idea}" на {self._num_epochs} эпох(и).'
        )
        run_args = {'num_epochs': self._num_epochs}
        latest_predictions: np.ndarray | None = None
        energy_joules: float | None = None
        process_start_time = None
        last_epoch_timestamp = None

        async for event in runner.run(
                candidate=candidate,
                train_df=self._train_df,
                validation_df=self._valid_df,
                run_args=run_args,
        ):
            if 'process_start_time' in event:
                process_start_time = event['process_start_time']
                _LOGGER.info(
                        f'Получено время старта процесса кандидата: {process_start_time.isoformat()}'
                )
            elif 'epoch_result' in event:
                predictions = np.asarray(event['epoch_result'], dtype=np.float64).reshape(-1)
                if predictions.size == 0:
                    raise RuntimeError('Кандидат вернул пустые предсказания на валидации.')
                latest_predictions = predictions
                last_epoch_timestamp = event.get('timestamp')
                _LOGGER.info(
                        f'Получены валидационные предсказания из {predictions.size} значений.'
                )
            elif 'total_energy_joules' in event:
                energy_joules = float(event['total_energy_joules'])
                _LOGGER.info(
                        f'Получена оценка энергопотребления {energy_joules:.4f} Дж.'
                )

        if latest_predictions is None:
            raise RuntimeError(
                    'Кандидат завершил работу, не выдав предсказаний на валидации.'
            )

        mse_value = self._compute_mse(latest_predictions)
        metrics: dict[str, float] = {self._metric_name: mse_value}

        if energy_joules is not None:
            metrics['total_energy_joules'] = energy_joules

        wall_time_seconds = 0.0
        if process_start_time is not None and last_epoch_timestamp is not None:
            wall_time_seconds = float(
                    (last_epoch_timestamp - process_start_time).total_seconds()
            )
            metrics['wall_time_seconds'] = wall_time_seconds

        _LOGGER.info(
                f'Оценка кандидата "{candidate.idea}" завершена: '
                f'{self._metric_name}={mse_value:.6f}, '
                f'energy={energy_joules if energy_joules is not None else 0.0:.4f} Дж, '
                f'wall_time={wall_time_seconds:.3f} c.'
        )
        return metrics

    def _compute_mse(self, predictions: np.ndarray) -> float:
        """
        Compute mean squared error between candidate predictions and validation targets.

        The method validates input length and checks that both predictions and targets
        contain only finite numeric values.
        """
        evaluation_count = min(predictions.size, self._validation_targets.size)
        if evaluation_count == 0:
            raise RuntimeError(
                    'Невозможно вычислить метрику: нет доступных значений целевой переменной.'
            )

        truncated_predictions = predictions[:evaluation_count]
        truncated_targets = self._validation_targets[:evaluation_count]

        if not np.isfinite(truncated_predictions).all():
            raise ValueError(
                    'Предсказания содержат некорректные значения (NaN или бесконечность).'
            )
        if not np.isfinite(truncated_targets).all():
            raise ValueError(
                    'Целевая переменная в валидационной выборке содержит некорректные значения.'
            )

        errors = truncated_predictions - truncated_targets
        mse_value = float(np.mean(errors * errors))
        return mse_value