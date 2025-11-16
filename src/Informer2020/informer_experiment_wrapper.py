# edlm_search/Informer2020/informer_experiment_wrapper.py
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import numpy as np


def _parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для запуска Informer-эксперимента.

    Ожидаемые аргументы:
      --data-path     путь к CSV-файлу ETTm1
      --metrics-path  путь к JSON-файлу с результатами
      --epochs        количество эпох обучения Informer
      все остальные аргументы пробрасываются далее в main_informer.py.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Путь к CSV-файлу с датасетом ETTm1.')
    parser.add_argument('--metrics-path', required=True, help='Путь к JSON-файлу с метриками.')
    parser.add_argument(
            '--epochs',
            type=int,
            required=False,
            default=10,
            help='Количество эпох обучения Informer (переопределяет конфиг по умолчанию, если нужно).',
    )
    args, extra = parser.parse_known_args()
    # Сохраняем дополнительные аргументы как атрибут, чтобы не ломать сигнатуру функции.
    setattr(args, 'extra_args', extra)
    return args


def _load_latest_metrics(results_root: str) -> dict[str, float]:
    """
    Находит последний файл metrics.npy в каталоге результатов и возвращает метрики.

    Параметры
    ----------
    results_root : str
        Корневой каталог, где Informer сохраняет результаты (обычно './results').

    Возвращает
    ----------
    dict[str, float]
        Словарь метрик с ключами 'mse', 'mae', 'rmse', 'mape', 'mspe'.

    Исключения
    ----------
    FileNotFoundError
        Если в каталоге результатов не найден ни один файл metrics.npy.
    ValueError
        Если формат файла metrics.npy некорректен.
    """
    root_path = Path(results_root)
    if not root_path.exists():
        raise FileNotFoundError(f'Каталог с результатами Informer не найден: {results_root}')

    latest_file: Path | None = None
    latest_mtime: float = -1.0

    for dirpath, _, filenames in os.walk(root_path):
        for name in filenames:
            if name != 'metrics.npy':
                continue
            candidate_path = Path(dirpath) / name
            mtime = candidate_path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = candidate_path

    if latest_file is None:
        raise FileNotFoundError(
                f'В каталоге результатов {results_root} не найдено ни одного файла metrics.npy.'
        )

    values = np.load(latest_file)
    if values.ndim != 1 or values.size < 5:
        raise ValueError(
                f'Ожидался одномерный массив длиной >=5 в файле {latest_file}, получено shape={values.shape}.'
        )

    mae = float(values[0])
    mse = float(values[1])
    rmse = float(values[2])
    mape = float(values[3])
    mspe = float(values[4])

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe,
    }


def _run_informer_and_get_metrics(
        data_path: str,
        num_epochs: int,
        extra_args: list[str],
) -> dict[str, float]:
    """
    Запускает обучение и оценку модели Informer и возвращает метрики.

    Параметры
    ----------
    data_path : str
        Полный путь к CSV-файлу с датасетом (например, ETTm1.csv).
    num_epochs : int
        Количество эпох обучения Informer.
    extra_args : list[str]
        Дополнительные аргументы командной строки, которые пробрасываются
        напрямую в main_informer.py (например, --model, --data, --features и т.д.).

    Возвращает
    ----------
    dict[str, float]
        Словарь метрик, например {"mse": ..., "mae": ..., ...}.

    Исключения
    ----------
    FileNotFoundError
        Если CSV-файл не существует.
    RuntimeError
        Если запуск main_informer.py завершился с ошибкой.
    """
    if num_epochs < 1:
        raise ValueError('Количество эпох должно быть положительным.')

    csv_path = Path(data_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV-файл датасета не найден: {csv_path}')

    root_path: Final[str] = str(csv_path.parent)
    data_filename: Final[str] = csv_path.name

    project_root = Path(__file__).resolve().parent

    command: list[str] = [
        sys.executable,
        'main_informer.py',
        '--model',
        'informer',
        '--data',
        'ETTm1',
        '--root_path',
        root_path,
        '--data_path',
        data_filename,
        '--train_epochs',
        str(num_epochs),
        '--itr',
        '1',
    ]
    command.extend(extra_args)

    completed = subprocess.run(
            command,
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
    )

    if completed.returncode != 0:
        stdout_text = completed.stdout.strip()
        stderr_text = completed.stderr.strip()
        raise RuntimeError(
                f'Запуск main_informer.py завершился с кодом {completed.returncode}. '
                f'stdout="{stdout_text}" stderr="{stderr_text}"'
        )

    results_dir = project_root / 'results'
    metrics = _load_latest_metrics(results_root=str(results_dir))
    return metrics


def _save_metrics(metrics_path: str, metrics: dict[str, float]) -> None:
    """
    Сохраняет метрики в JSON-файл.

    Параметры
    ----------
    metrics_path : str
        Путь к файлу, куда будут записаны метрики.
    metrics : dict[str, float]
        Словарь метрик.
    """
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Точка входа для запуска эксперимента Informer.

    Последовательность:
      1. Разбор аргументов.
      2. Запуск Informer и получение метрик.
      3. Сохранение метрик в JSON.
    """
    args = _parse_args()
    extra_args = list(getattr(args, 'extra_args', []))
    metrics = _run_informer_and_get_metrics(
            data_path=args.data_path,
            num_epochs=args.epochs,
            extra_args=extra_args,
    )
    _save_metrics(metrics_path=args.metrics_path, metrics=metrics)


if __name__ == '__main__':
    main()