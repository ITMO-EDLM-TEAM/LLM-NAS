from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

try:
    from zeus.monitor import ZeusMonitor
except ImportError:  # pragma: no cover
    ZeusMonitor = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)

INFORMER_BUILTIN_DATASETS: Final[frozenset[str]] = frozenset(
        {
            'ETTh1',
            'ETTh2',
            'ETTm1',
            'ETTm2',
            'WTH',
            'ECL',
            'Solar',
        }
)


def _parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для запуска Informer-эксперимента.

    Ожидаемые аргументы:
      --data-path     путь к CSV-файлу ETT-датасета (любой вариант бенчмарка ETT)
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
          * именем датасета без расширения (например, именем варианта ETT-бенчмарка).

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


def _load_latest_metrics(
        results_root: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, Path]:
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

    setting_dir = latest_file.parent
    preds_path = setting_dir / 'pred.npy'
    trues_path = setting_dir / 'true.npy'

    if not preds_path.is_file() or not trues_path.is_file():
        raise FileNotFoundError(
                f'В каталоге {setting_dir} не найдены файлы pred.npy и true.npy.'
        )

    preds = np.load(preds_path)
    trues = np.load(trues_path)

    if preds.shape != trues.shape:
        raise ValueError(
                f'Формы pred.npy и true.npy не совпадают: {preds.shape} vs {trues.shape}.'
        )

    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe,
    }
    return metrics, trues, preds, setting_dir


def _extract_target_series(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует массивы предсказаний Informer к одномерным рядам по целевой переменной.

    Ожидаемые формы:
      * (N, L, C) — батчи, горизонт и число каналов;
      * (N, L) или (N,) — уже свернутые формы.

    Возвращает два одномерных массива одинаковой длины.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
                f'Формы массивов истинных и предсказанных значений не совпадают: '
                f'{y_true.shape} vs {y_pred.shape}.'
        )

    if y_true.ndim == 3:
        true_series = y_true[:, :, -1].reshape(-1)
        pred_series = y_pred[:, :, -1].reshape(-1)
        return true_series.astype(np.float64), pred_series.astype(np.float64)

    if y_true.ndim == 2:
        true_series = y_true.reshape(-1)
        pred_series = y_pred.reshape(-1)
        return true_series.astype(np.float64), pred_series.astype(np.float64)

    if y_true.ndim == 1:
        return y_true.astype(np.float64), y_pred.astype(np.float64)

    raise ValueError(f'Неожиданная форма массивов предсказаний: {y_true.shape}.')


def _find_best_alignment_start_index(
        full_series: np.ndarray,
        eval_series: np.ndarray,
) -> int:
    """
    Находит позицию в полном ряду full_series, с которой начинается подотрезок,
    наилучшим образом совпадающий с eval_series (по MSE).

    Возвращает индекс начала такого подотрезка.
    """
    full_length = int(full_series.shape[0])
    eval_length = int(eval_series.shape[0])

    if eval_length <= 0:
        raise ValueError('Длина eval_series должна быть положительной.')
    if eval_length > full_length:
        raise ValueError(
                f'Длина eval_series ({eval_length}) больше длины полного ряда ({full_length}).'
        )

    max_start = full_length - eval_length

    window_limit = 10000
    if max_start + 1 > window_limit:
        start_min = max_start - window_limit + 1
    else:
        start_min = 0

    best_start = 0
    best_score = float('inf')

    for start in range(start_min, max_start + 1):
        segment = full_series[start:start + eval_length]
        diff = segment - eval_series
        score = float(np.mean(diff * diff))
        if score < best_score:
            best_score = score
            best_start = start

    _LOGGER.info(
            f'Выбран стартовый индекс выравнивания {best_start} с MSE={best_score}.'
    )
    return best_start


def _run_informer_and_get_metrics(
        data_path: str,
        num_epochs: int,
        extra_args: list[str],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, str, Path]:
    """
    Запускает обучение и оценку модели Informer и возвращает метрики.

    Параметры
    ----------
    data_path : str
        Путь к CSV-файлу с датасетом (например, '/path/to/dataset.csv') или имя датасета
        (например, имя варианта семейства ETT), которое будет сопоставлено с типичным
        расположением ETT-данных.
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

    informer_data_name: Final[str]
    if dataset_name in INFORMER_BUILTIN_DATASETS:
        informer_data_name = dataset_name
    else:
        informer_data_name = 'custom'

    project_root = Path(__file__).resolve().parent

    filtered_extra_args = _filter_extra_args(extra_args)

    command: list[str] = [
        sys.executable,
        'main_informer.py',
        '--model',
        'informer',
        '--data',
        informer_data_name,
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
        try:
            gpu_monitor = ZeusMonitor(gpu_indices=[0])
            gpu_monitor.begin_window('informer_external_run')
        except Exception as exc:
            _LOGGER.warning(
                    f'ZeusMonitor could not be initialized for Informer external run, GPU metrics will be skipped: {exc}'
            )
            gpu_monitor = None

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
        try:
            measurement = gpu_monitor.end_window('informer_external_run')
            total_energy_joules = float(measurement.total_energy)
        except Exception as exc:
            _LOGGER.warning(
                    f'ZeusMonitor failed during Informer external run measurement, GPU metrics will be skipped: {exc}'
            )
            total_energy_joules = 0.0

    if completed.returncode != 0:
        stdout_text = completed.stdout.strip()
        stderr_text = completed.stderr.strip()
        raise RuntimeError(
                f'Запуск main_informer.py завершился с кодом {completed.returncode}. '
                f'stdout="{stdout_text}" stderr="{stderr_text}"'
        )

    results_dir = project_root / 'results'
    base_metrics, y_true_raw, y_pred_raw, setting_dir = _load_latest_metrics(
            results_root=str(results_dir)
    )

    base_metrics['total_runtime_seconds'] = total_runtime_seconds
    base_metrics['total_energy_joules'] = total_energy_joules

    return base_metrics, y_true_raw, y_pred_raw, dataset_name, setting_dir


def _create_experiment_directory(metrics_path: str, model_name: str, dataset_name: str) -> Path:
    """
    Create a dedicated directory for a single Informer experiment.

    The directory name encodes model, dataset and start time.
    """
    metrics_path_obj = Path(metrics_path).resolve()
    parent = metrics_path_obj.parent
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    slug = f'informer_{model_name}_{dataset_name}_{timestamp}'
    experiment_dir = parent / slug
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _build_valid_predictions_dataframe(
        csv_path: Path,
        dataset_name: str,
        y_true_full: np.ndarray,
        y_pred_full: np.ndarray,
) -> pd.DataFrame:
    """
    Строит DataFrame с датами, истинными и предсказанными значениями для валидационной части.

    Для выравнивания используется сопоставление стандартизованной валидационной части ряда
    и стандартизованного y_true из Informer, после чего предсказания де-нормализуются
    обратно в масштаб целевой переменной.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
                f'CSV-файл датасета для выравнивания предсказаний не найден: "{csv_path}".'
        )

    df = pd.read_csv(csv_path)

    if 'date' not in df.columns:
        raise ValueError(
                f'Колонка "date" отсутствует в датасете "{dataset_name}", выравнивание по времени невозможно.'
        )

    # Определяем целевую колонку
    target_column_env = os.getenv('TARGET_COLUMN', 'OT')
    target_column = target_column_env if target_column_env in df.columns else None
    if target_column is None:
        numeric_candidates: list[str] = []
        for column in df.columns:
            if column == 'date':
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_candidates.append(column)
        if not numeric_candidates:
            raise ValueError(
                    f'Не удалось определить числовую целевую колонку для датасета "{dataset_name}".'
            )
        target_column = numeric_candidates[-1]

    if target_column not in df.columns:
        raise ValueError(
                f'Целевая колонка "{target_column}" отсутствует в датасете "{dataset_name}".'
        )

    total_rows = int(df.shape[0])
    if total_rows <= 1:
        raise ValueError(
                f'Слишком мало строк в датасете "{dataset_name}" для построения валидационной части.'
        )

    # Читаем TRAIN_RATIO из окружения, чтобы согласовать разбиение с остальным пайплайном
    train_ratio_env = os.getenv('TRAIN_RATIO', '0.8')
    try:
        train_ratio = float(train_ratio_env)
    except ValueError as exc:
        raise ValueError(f'Некорректное значение TRAIN_RATIO="{train_ratio_env}".') from exc

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
                f'TRAIN_RATIO должно быть в диапазоне (0, 1), получено {train_ratio}.'
        )

    train_len = int(total_rows * train_ratio)
    if train_len <= 0 or train_len >= total_rows:
        raise ValueError(
                f'Некорректное значение train_len={train_len} для датасета длиной {total_rows}.'
        )

    train_values = df[target_column].iloc[:train_len].to_numpy(dtype=np.float64)
    valid_df = df.iloc[train_len:].copy()
    valid_target = valid_df[target_column].to_numpy(dtype=np.float64)
    valid_dates = pd.to_datetime(valid_df['date'])

    # Воспроизводим стандартную стандартизацию по train-части
    target_mean = float(np.mean(train_values))
    target_std = float(np.std(train_values))
    if target_std <= 0.0:
        target_std = 1.0

    valid_target_scaled = (valid_target - target_mean) / target_std

    # Извлекаем последнюю компоненту из true/pred Informer (обычно это целевой канал)
    series_true_scaled, series_pred_scaled = _extract_target_series(
            y_true=y_true_full,
            y_pred=y_pred_full,
    )

    if series_true_scaled.shape[0] != series_pred_scaled.shape[0]:
        raise ValueError(
                'Длины одномерных рядов y_true и y_pred после извлечения не совпадают.'
        )

    # Находим внутри длинного ряда Informer такой сегмент y_true, который лучше всего
    # совпадает с стандартизованной валидацией (valid_target_scaled)
    start_index = _find_best_alignment_start_index(
            full_series=series_true_scaled,
            eval_series=valid_target_scaled,
    )
    end_index = start_index + valid_target_scaled.shape[0]
    if end_index > series_pred_scaled.shape[0]:
        raise ValueError(
                f'Диапазон [{start_index}, {end_index}) выходит за пределы ряда предсказаний длиной '
                f'{series_pred_scaled.shape[0]}.'
        )

    aligned_pred_scaled = series_pred_scaled[start_index:end_index]
    aligned_pred = aligned_pred_scaled * target_std + target_mean

    if aligned_pred.shape[0] != valid_target.shape[0]:
        raise ValueError(
                f'После выравнивания длина предсказаний ({aligned_pred.shape[0]}) '
                f'не совпадает с длиной валидационной части ({valid_target.shape[0]}).'
        )

    predictions_df = pd.DataFrame(
            {
                'split': ['valid'] * valid_target.shape[0],
                'date': valid_dates.to_numpy(),
                'y_true': valid_target,
                'y_pred': aligned_pred.astype(np.float64),
            }
    )

    return predictions_df


def _compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute basic regression metrics for one-dimensional prediction and target series.

    The following metrics are returned:
      * mse
      * mae
      * rmse
      * mape
      * mspe
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
                f'Cannot compute metrics for arrays with different shapes: '
                f'{y_true.shape} vs {y_pred.shape}.'
        )
    if y_true.size == 0:
        raise ValueError('Cannot compute metrics for empty arrays.')

    y_true_flat = y_true.reshape(-1).astype(np.float64)
    y_pred_flat = y_pred.reshape(-1).astype(np.float64)

    diff = y_pred_flat - y_true_flat
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = diff / y_true_flat
        mape = float(np.mean(np.abs(ratio)))
        mspe = float(np.mean(np.square(ratio)))

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe,
    }


def _save_metrics(
        metrics_path: str,
        metrics: dict[str, float],
        dataset_name: str,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        setting_dir: Path,
        args: argparse.Namespace,
) -> None:
    """
    Сохраняет метрики в JSON-файл.

    Параметры
    ----------
    metrics_path : str
        Путь к файлу, куда будут записаны метрики.
    metrics : dict[str, float]
        Словарь метрик.
    """
    metrics_path_obj = Path(metrics_path).resolve()

    experiment_dir = _create_experiment_directory(
            metrics_path=metrics_path,
            model_name=model_name,
            dataset_name=dataset_name,
    )

    data_csv_path = Path(args.data_path).resolve()

    try:
        predictions_df = _build_valid_predictions_dataframe(
                csv_path=data_csv_path,
                dataset_name=dataset_name,
                y_true_full=y_true,
                y_pred_full=y_pred,
        )
        predictions_csv_path = experiment_dir / f'{model_name}_{dataset_name}_valid_predictions.csv'
        predictions_df.to_csv(predictions_csv_path, index=False)
        _LOGGER.info(
                f'CSV с выровненными предсказаниями Informer записан в "{predictions_csv_path}".'
        )
        y_true_flat = predictions_df['y_true'].to_numpy(dtype=np.float64)
        y_pred_flat = predictions_df['y_pred'].to_numpy(dtype=np.float64)
    except Exception as exc:
        _LOGGER.warning(
                f'Не удалось построить CSV с выровненными и де-нормализованными предсказаниями Informer: {exc}'
        )
        y_true_flat, y_pred_flat = _extract_target_series(
                y_true=y_true,
                y_pred=y_pred,
        )
        predictions_csv_path = experiment_dir / f'{model_name}_{dataset_name}_valid_predictions.csv'
        fallback_df = pd.DataFrame(
                {
                    'split': ['valid'] * int(y_true_flat.shape[0]),
                    'index': np.arange(int(y_true_flat.shape[0])),
                    'y_true': y_true_flat,
                    'y_pred': y_pred_flat,
                }
        )
        fallback_df.to_csv(predictions_csv_path, index=False)
        _LOGGER.info(
                f'CSV с предсказаниями Informer без выравнивания записан в "{predictions_csv_path}".'
        )

    recomputed_metrics = _compute_regression_metrics(
            y_true=y_true_flat,
            y_pred=y_pred_flat,
    )

    preds_npy_path = setting_dir / 'pred.npy'
    trues_npy_path = setting_dir / 'true.npy'

    hyperparams = {
        'train_epochs': int(getattr(args, 'epochs', 0)),
        'data_path': args.data_path,
        'extra_args': list(getattr(args, 'extra_args', [])),
    }

    metrics_copy: dict[str, float] = {}
    for key, value in metrics.items():
        if key in ('mse', 'mae', 'rmse', 'mape', 'mspe'):
            continue
        metrics_copy[key] = float(value)

    for key, value in recomputed_metrics.items():
        metrics_copy[key] = float(value)

    diagnostics = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'metrics': metrics_copy,
        'hyperparams': hyperparams,
        'artifacts': {
            'predictions_csv': str(predictions_csv_path),
            'predictions_npy': str(preds_npy_path),
            'targets_npy': str(trues_npy_path),
            'informer_results_root': str(setting_dir.parent),
        },
        'series_preview': {
            'y_true_min': float(np.min(y_true_flat)),
            'y_true_max': float(np.max(y_true_flat)),
            'y_pred_min': float(np.min(y_pred_flat)),
            'y_pred_max': float(np.max(y_pred_flat)),
            'num_points': int(y_true_flat.shape[0]),
        },
    }

    final_metrics_path = metrics_path_obj
    with final_metrics_path.open('w', encoding='utf-8') as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    _LOGGER.info(
            f'Informer diagnostics JSON written at "{final_metrics_path}" '
            f'for dataset="{dataset_name}", model="{model_name}".'
    )


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
    metrics, y_true, y_pred, dataset_name, setting_dir = _run_informer_and_get_metrics(
            data_path=args.data_path,
            num_epochs=args.epochs,
            extra_args=extra_args,
    )
    _save_metrics(
            metrics_path=args.metrics_path,
            metrics=metrics,
            dataset_name=dataset_name,
            model_name='informer',
            y_true=y_true,
            y_pred=y_pred,
            setting_dir=setting_dir,
            args=args,
    )


if __name__ == '__main__':
    main()