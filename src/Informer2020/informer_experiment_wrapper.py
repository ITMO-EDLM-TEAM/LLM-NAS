# edlm_search/Informer2020/informer_experiment_wrapper.py
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

try:
    from zeus.monitor import ZeusMonitor
except ImportError:  # pragma: no cover
    ZeusMonitor = None  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для запуска Informer-эксперимента.

    Ожидаемые аргументы:
      --data-path     путь к CSV-файлу ETT-датасета (ETTh1, ETTh2, ETTm1, ETTm2 и т.п.)
      --metrics-path  путь к JSON-файлу с результатами
      --epochs        количество эпох обучения Informer
      все остальные аргументы пробрасываются далее в main_informer.py.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Путь к CSV-файлу с датасетом ETT.')
    parser.add_argument('--metrics-path', required=True, help='Путь к JSON-файлу с метриками.')
    parser.add_argument(
            '--epochs',
            type=int,
            required=False,
            default=10,
            help='Количество эпох обучения Informer (переопределяет конфиг по умолчанию, если нужно).',
    )
    args, extra = parser.parse_known_args()
    setattr(args, 'extra_args', extra)
    return args


def _resolve_dataset_csv_path(data_path: str) -> Path:
    """
    Разрешает значение параметра --data-path в существующий CSV-файл.

    Параметры
    ----------
    data_path : str
        Значение, переданное через --data-path. Может быть:
          * абсолютным или относительным путём к CSV-файлу;
          * именем датасета без расширения (например, 'ETTh1').

    Возвращает
    ----------
    pathlib.Path
        Абсолютный путь к существующему CSV-файлу.

    Исключения
    ----------
    FileNotFoundError
        Если ни один из кандидатов не существует.
    """
    initial_path = Path(data_path)
    if initial_path.is_file():
        return initial_path.resolve()

    dataset_name = initial_path.stem
    csv_filename = f'{dataset_name}.csv'

    repo_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = [
        repo_root / 'ETDataset' / 'ETT-small' / csv_filename,
        repo_root / 'ETDataset' / csv_filename,
        repo_root / 'data' / 'ETT' / csv_filename,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched_locations = ', '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
            f'CSV-файл датасета не найден для значения "{data_path}". '
            f'Пробованы пути: {searched_locations}'
    )


def _filter_extra_args(extra_args: list[str]) -> list[str]:
    """
    Удаляет из списка дополнительных аргументов флаг --data и его значение.

    Это нужно, чтобы имя датасета, вычисленное из CSV-файла, не конфликтовало
    с вручную переданным параметром --data.
    """
    filtered: list[str] = []
    skip_next = False

    for index, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if arg == '--data':
            if index + 1 < len(extra_args):
                skip_next = True
            continue
        filtered.append(arg)

    return filtered


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
        Путь к CSV-файлу с датасетом (например, '/path/to/ETTh1.csv') или имя датасета
        (например, 'ETTh1'), которое будет сопоставлено с типичным расположением ETT-данных.
    num_epochs : int
        Количество эпох обучения Informer.
    extra_args : list[str]
        Дополнительные аргументы командной строки, которые пробрасываются
        напрямую в main_informer.py (например, --features, --freq и т.д.).
        Параметр --data, если присутствует, игнорируется и вычисляется автоматически
        из имени CSV-файла.

    Возвращает
    ----------
    dict[str, float]
        Словарь метрик, например {"mse": ..., "mae": ..., "total_runtime_seconds": ..., ...}.

    Исключения
    ----------
    FileNotFoundError
        Если CSV-файл не существует ни по одному из ожидаемых путей.
    RuntimeError
        Если запуск main_informer.py завершился с ошибкой.
    ValueError
        Если num_epochs некорректно задан.
    """
    if num_epochs < 1:
        raise ValueError('Количество эпох должно быть положительным.')

    csv_path = _resolve_dataset_csv_path(data_path)
    root_path: Final[str] = str(csv_path.parent)
    data_filename: Final[str] = csv_path.name
    dataset_name: Final[str] = csv_path.stem

    project_root = Path(__file__).resolve().parent

    filtered_extra_args = _filter_extra_args(extra_args)

    command: list[str] = [
        sys.executable,
        'main_informer.py',
        '--model',
        'informer',
        '--data',
        dataset_name,
        '--root_path',
        root_path,
        '--data_path',
        data_filename,
        '--train_epochs',
        str(num_epochs),
        '--itr',
        '1',
    ]
    command.extend(filtered_extra_args)

    gpu_monitor = None
    if ZeusMonitor is not None:
        gpu_monitor = ZeusMonitor(gpu_indices=[0])
        gpu_monitor.begin_window('informer_external_run')

    start_time = time.perf_counter()
    completed = subprocess.run(
            command,
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
    )
    end_time = time.perf_counter()

    total_runtime_seconds = float(end_time - start_time)
    total_energy_joules = 0.0
    if gpu_monitor is not None:
        measurement = gpu_monitor.end_window('informer_external_run')
        total_energy_joules = float(measurement.total_energy)

    if completed.returncode != 0:
        stdout_text = completed.stdout.strip()
        stderr_text = completed.stderr.strip()
        raise RuntimeError(
                f'Запуск main_informer.py завершился с кодом {completed.returncode}. '
                f'stdout="{stdout_text}" stderr="{stderr_text}"'
        )

    results_dir = project_root / 'results'
    metrics = _load_latest_metrics(results_root=str(results_dir))
    metrics['total_runtime_seconds'] = total_runtime_seconds
    metrics['total_energy_joules'] = total_energy_joules
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
      2. Разрешение пути к CSV-файлу датасета.
      3. Запуск Informer и получение метрик.
      4. Сохранение метрик в JSON.
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