# edlm_search/src/experiments/informer_runner.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from .types import ExperimentResult
from .types import InformerRunnerConfig


def run_informer_external(
        config: InformerRunnerConfig,
        dataset_name: str,
        model_name: str,
) -> ExperimentResult:
    """
    Запускает внешний скрипт Informer и считывает метрики из JSON-файла.

    Ожидается, что внешний скрипт:
      1. Принимает на вход путь к CSV-файлу с датасетом и путь к JSON для записи метрик.
      2. Запускает обучение/оценку модели Informer.
      3. По завершении записывает JSON вида {"mse": <float>, "mae": <float>, ...}.

    Метрики могут дополнительно включать:
      * 'total_runtime_seconds'
      * 'total_energy_joules'
      и другие числовые показатели.

    Параметры
    ----------
    config : InformerRunnerConfig
        Конфигурация запуска Informer: пути к скрипту, датасету и JSON с метриками.
    dataset_name : str
        Имя датасета (например, вариант семейства ETT).
    model_name : str
        Имя модели (например, 'informer-original').

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента с метриками, считанными из JSON.

    Исключения
    ----------
    FileNotFoundError
        Если скрипт Informer или CSV-файл не существуют.
    RuntimeError
        Если внешний скрипт вернул ненулевой код возврата или не создал JSON с метриками.
    ValueError
        Если JSON с метриками имеет некорректный формат.
    """
    if not os.path.exists(config.informer_script_path):
        raise FileNotFoundError(f'Скрипт Informer не найден: {config.informer_script_path}')
    if not os.path.exists(config.dataset_csv_path):
        raise FileNotFoundError(f'CSV-файл датасета не найден: {config.dataset_csv_path}')

    metrics_dir = os.path.dirname(config.metrics_json_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    command: list[str] = [
        sys.executable,
        config.informer_script_path,
        '--data-path',
        config.dataset_csv_path,
        '--metrics-path',
        config.metrics_json_path,
    ]
    command.extend(config.extra_args)

    completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
    )

    if completed.returncode != 0:
        stderr_text = completed.stderr.strip()
        stdout_text = completed.stdout.strip()
        raise RuntimeError(
                f'Скрипт Informer завершился с кодом {completed.returncode}. '
                f'stdout="{stdout_text}" stderr="{stderr_text}"'
        )

    if not os.path.exists(config.metrics_json_path):
        raise RuntimeError(
                f'После запуска Informer не найден файл с метриками: {config.metrics_json_path}'
        )

    with open(config.metrics_json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError('JSON с метриками должен быть объектом (dict).')

    metrics: dict[str, float] = {}
    extra_info: dict[str, Any] = {
        'timeout_seconds': float(config.timeout_seconds),
    }

    metrics_section = raw.get('metrics')
    artifacts_section = raw.get('artifacts')
    hyperparams_section = raw.get('hyperparams')

    if isinstance(metrics_section, dict):
        for key, value in metrics_section.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Значение метрики "{key}" не может быть приведено к float.') from exc
        if hyperparams_section is not None:
            extra_info['hyperparams'] = hyperparams_section
        if artifacts_section is not None:
            extra_info['artifacts'] = artifacts_section
    else:
        for key, value in raw.items():
            if key in ('metrics', 'artifacts', 'hyperparams'):
                continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Значение метрики "{key}" не может быть приведено к float.') from exc

    return ExperimentResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            extra_info=extra_info,
    )