from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable
from typing import Final
from typing import TypeVar

import optuna
import pandas as pd
import psutil
import torch
from zeus.monitor import ZeusMonitor

from .datasets import load_ett_csv_dataset
from .types import ExperimentResult
from .types import InformerRunnerConfig
from .types import InformerSearchSpace
from .types import LSTMHyperParams
from .types import LSTMSearchSpace
from ..baselines import baseline_optuna
from ..baselines.informer_runner import run_informer_external
from ..baselines.lstm_runner import run_lstm_on_etth_dataset
from ..devices import get_torch_device

_logger = logging.getLogger(__name__)

T = TypeVar('T')


def _load_json_config_from_env(env_name: str) -> dict[str, object]:
    """
    Load JSON object from environment variable.

    The variable must be set, non-empty and contain a JSON object.
    """
    raw = os.getenv(env_name)
    if raw is None:
        raise ValueError(f'Environment variable "{env_name}" must be set.')
    raw_stripped = raw.strip()
    if not raw_stripped:
        raise ValueError(f'Environment variable "{env_name}" must not be empty.')
    try:
        parsed = json.loads(raw_stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
                f'Environment variable "{env_name}" must contain valid JSON.'
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
                f'Environment variable "{env_name}" must contain a JSON object.'
        )
    return parsed


def _extract_typed_list(
        config: dict[str, object],
        key: str,
        cast: Callable[[object], T],
        must_be_positive: bool,
        config_name: str,
) -> list[T]:
    """
    Extract and validate typed list from JSON config.

    The key must be present and map to a non-empty list.
    """
    if key not in config:
        raise ValueError(
                f'Key "{key}" is missing in {config_name}.'
        )

    value = config[key]
    if not isinstance(value, list):
        raise ValueError(
                f'Value for key "{key}" in {config_name} must be a list.'
        )
    if not value:
        raise ValueError(
                f'List for key "{key}" in {config_name} must not be empty.'
        )

    result: list[T] = []
    for item in value:
        try:
            converted = cast(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                    f'Value "{item}" in list for key "{key}" in {config_name} cannot be converted.'
            ) from exc
        if must_be_positive and converted <= 0:  # type: ignore[operator]
            raise ValueError(
                    f'All values for key "{key}" in {config_name} must be positive.'
            )
        result.append(converted)
    return result


def _build_lstm_search_space_from_env() -> LSTMSearchSpace:
    """
    Build LSTMSearchSpace instance from environment variable LSTM_SEARCH_SPACE_JSON.

    The JSON object must define all keys:
      * seq_len
      * pred_len
      * num_epochs
      * hidden_size
      * num_layers
      * learning_rate
      * batch_size
    """
    config_name = 'LSTM_SEARCH_SPACE_JSON'
    json_config = _load_json_config_from_env(config_name)

    allowed_keys = {
        'seq_len',
        'pred_len',
        'num_epochs',
        'hidden_size',
        'num_layers',
        'learning_rate',
        'batch_size',
    }
    unknown_keys = set(json_config.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(
                f'Unknown keys in {config_name}: {sorted(unknown_keys)}.'
        )

    missing_keys = allowed_keys - set(json_config.keys())
    if missing_keys:
        raise ValueError(
                f'Missing keys in {config_name}: {sorted(missing_keys)}.'
        )

    seq_len_values = _extract_typed_list(
            config=json_config,
            key='seq_len',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    pred_len_values = _extract_typed_list(
            config=json_config,
            key='pred_len',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    num_epochs_values = _extract_typed_list(
            config=json_config,
            key='num_epochs',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    hidden_size_values = _extract_typed_list(
            config=json_config,
            key='hidden_size',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    num_layers_values = _extract_typed_list(
            config=json_config,
            key='num_layers',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    learning_rate_values = _extract_typed_list(
            config=json_config,
            key='learning_rate',
            cast=lambda x: float(x),
            must_be_positive=True,
            config_name=config_name,
    )
    batch_size_values = _extract_typed_list(
            config=json_config,
            key='batch_size',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )

    search_space = LSTMSearchSpace(
            seq_len_values=seq_len_values,
            pred_len_values=pred_len_values,
            num_epochs_values=num_epochs_values,
            hidden_size_values=hidden_size_values,
            num_layers_values=num_layers_values,
            learning_rate_values=learning_rate_values,
            batch_size_values=batch_size_values,
    )
    return search_space


def _build_informer_search_space_from_env() -> InformerSearchSpace:
    """
    Build InformerSearchSpace instance from environment variable INFORMER_SEARCH_SPACE_JSON.

    The JSON object must define all keys:
      * d_model
      * n_heads
      * e_layers
      * d_layers
      * factor
      * dropout
      * learning_rate
      * batch_size
      * epochs
    """
    config_name = 'INFORMER_SEARCH_SPACE_JSON'
    json_config = _load_json_config_from_env(config_name)

    allowed_keys = {
        'd_model',
        'n_heads',
        'e_layers',
        'd_layers',
        'factor',
        'dropout',
        'learning_rate',
        'batch_size',
        'epochs',
    }
    unknown_keys = set(json_config.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(
                f'Unknown keys in {config_name}: {sorted(unknown_keys)}.'
        )

    missing_keys = allowed_keys - set(json_config.keys())
    if missing_keys:
        raise ValueError(
                f'Missing keys in {config_name}: {sorted(missing_keys)}.'
        )

    d_model_values = _extract_typed_list(
            config=json_config,
            key='d_model',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    n_heads_values = _extract_typed_list(
            config=json_config,
            key='n_heads',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    e_layers_values = _extract_typed_list(
            config=json_config,
            key='e_layers',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    d_layers_values = _extract_typed_list(
            config=json_config,
            key='d_layers',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    factor_values = _extract_typed_list(
            config=json_config,
            key='factor',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    dropout_values = _extract_typed_list(
            config=json_config,
            key='dropout',
            cast=lambda x: float(x),
            must_be_positive=True,
            config_name=config_name,
    )
    learning_rate_values = _extract_typed_list(
            config=json_config,
            key='learning_rate',
            cast=lambda x: float(x),
            must_be_positive=True,
            config_name=config_name,
    )
    batch_size_values = _extract_typed_list(
            config=json_config,
            key='batch_size',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )
    epochs_values = _extract_typed_list(
            config=json_config,
            key='epochs',
            cast=lambda x: int(x),
            must_be_positive=True,
            config_name=config_name,
    )

    search_space = InformerSearchSpace(
            d_model_values=d_model_values,
            n_heads_values=n_heads_values,
            e_layers_values=e_layers_values,
            d_layers_values=d_layers_values,
            factor_values=factor_values,
            dropout_values=dropout_values,
            learning_rate_values=learning_rate_values,
            batch_size_values=batch_size_values,
            epochs_values=epochs_values,
    )
    return search_space


def _run_lstm_with_resource_monitoring(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        hyperparams: LSTMHyperParams,
        dataset_name: str,
        model_name: str,
        artifacts_dir: str,
        device: torch.device,
) -> ExperimentResult:
    """
    Run LSTM experiment with resource monitoring and unified artifact saving.

    This helper:
      * measures wall time, CPU time, RSS and GPU energy;
      * calls run_lstm_on_etth_dataset to train and save artifacts;
      * enriches metrics and extra_info with resource statistics.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset.
    valid_df : pandas.DataFrame
        Validation dataset.
    hyperparams : LSTMHyperParams
        Hyperparameters for LSTM training.
    dataset_name : str
        Dataset name.
    model_name : str
        Model name used in reports and artifact filenames.
    artifacts_dir : str
        Directory for saving artifacts.
    device : torch.device
        Device used for model training and evaluation.
    """
    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    rss_before_bytes = process.memory_info().rss
    wall_start = time.perf_counter()

    monitor = None
    gpu_total_energy_joules = 0.0
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            current_device = torch.cuda.current_device()
            monitor = ZeusMonitor(gpu_indices=[current_device])
            monitor.begin_window('lstm_etth_experiment')
        except Exception as exc:
            _logger.warning(
                    f'ZeusMonitor could not be initialized for LSTM experiment, GPU metrics will be skipped: {exc}'
            )
            monitor = None

    dataset_name_final: Final[str] = dataset_name
    base_result = run_lstm_on_etth_dataset(
            train_df=train_df,
            valid_df=valid_df,
            hyperparams=hyperparams,
            dataset_name=dataset_name_final,
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            device=device,
    )

    if monitor is not None:
        try:
            measurement = monitor.end_window('lstm_etth_experiment')
            gpu_total_energy_joules = float(measurement.total_energy)
        except Exception as exc:
            _logger.warning(
                    f'ZeusMonitor failed during LSTM experiment measurement, GPU metrics will be skipped: {exc}'
            )
            gpu_total_energy_joules = 0.0

    wall_seconds_total = float(time.perf_counter() - wall_start)
    cpu_times_after = process.cpu_times()
    rss_after_bytes = process.memory_info().rss

    cpu_user_seconds_total = float(cpu_times_after.user - cpu_times_before.user)
    cpu_system_seconds_total = float(cpu_times_after.system - cpu_times_before.system)
    rss_mb_before = float(rss_before_bytes) / (1024.0 * 1024.0)
    rss_mb_after = float(rss_after_bytes) / (1024.0 * 1024.0)
    rss_mb_delta = rss_mb_after - rss_mb_before

    metrics = dict(base_result.metrics)
    if 'mse' in metrics:
        metrics['mse'] = float(metrics['mse'])
    metrics['wall_seconds_total'] = wall_seconds_total
    metrics['cpu_user_seconds_total'] = cpu_user_seconds_total
    metrics['cpu_system_seconds_total'] = cpu_system_seconds_total
    metrics['rss_mb_delta'] = rss_mb_delta
    metrics['gpu_total_energy_joules'] = gpu_total_energy_joules

    extra_info = dict(base_result.extra_info)
    extra_info['resource_monitoring'] = {
        'wall_seconds_total': wall_seconds_total,
        'cpu_user_seconds_total': cpu_user_seconds_total,
        'cpu_system_seconds_total': cpu_system_seconds_total,
        'rss_mb_before': rss_mb_before,
        'rss_mb_after': rss_mb_after,
        'rss_mb_delta': rss_mb_delta,
        'gpu_total_energy_joules': gpu_total_energy_joules,
    }

    result = ExperimentResult(
            model_name=base_result.model_name,
            dataset_name=base_result.dataset_name,
            metrics=metrics,
            extra_info=extra_info,
    )

    mse_value = float(result.metrics.get('mse', float('nan')))
    _logger.info(
            f'LSTM-эксперимент для датасета "{dataset_name}" завершён: '
            f'MSE={mse_value:.6f}, wall_seconds_total={wall_seconds_total:.3f}, '
            f'gpu_total_energy_joules={gpu_total_energy_joules:.3f}.'
    )
    return result


def _resolve_informer_csv_path(dataset_name: str, csv_path: str) -> str:
    """
    Resolve path to CSV for Informer experiments.

    Order:
      1) use the passed csv_path if it exists;
      2) use project_root/ETDataset/ETT-small/<dataset_name>.csv if it exists.

    Raises FileNotFoundError if nothing is found.
    """
    original_csv_path = csv_path
    csv_path_obj = Path(csv_path)

    if csv_path_obj.is_file():
        return str(csv_path_obj.resolve())

    project_root = Path(__file__).resolve().parent.parent
    fallback_csv = project_root / 'ETDataset' / 'ETT-small' / f'{dataset_name}.csv'
    if fallback_csv.is_file():
        _logger.info(
                f'Путь к CSV "{original_csv_path}" не найден, используется путь по умолчанию '
                f'"{fallback_csv}".'
        )
        return str(fallback_csv.resolve())

    raise FileNotFoundError(
            f'CSV-файл датасета для Informer не найден по пути "{original_csv_path}". '
            f'Также не удалось найти файл по ожидаемому пути "{fallback_csv}".'
    )


def _run_informer_with_resource_monitoring(
        dataset_name: str,
        csv_path: str,
        informer_script_path: str,
        metrics_json_path: str,
        extra_args: list[str],
        timeout_seconds: int,
        model_name: str,
) -> ExperimentResult:
    """
    Run Informer experiment via external wrapper with resource monitoring.

    This helper:
      * resolves CSV location;
      * builds InformerRunnerConfig;
      * measures wall time, CPU time, RSS and GPU energy;
      * calls run_informer_external which reads unified JSON diagnostics;
      * enriches metrics and extra_info with resource statistics and artifact paths.
    """
    resolved_csv_path = _resolve_informer_csv_path(dataset_name=dataset_name, csv_path=csv_path)

    informer_config = InformerRunnerConfig(
            dataset_csv_path=resolved_csv_path,
            metrics_json_path=metrics_json_path,
            informer_script_path=informer_script_path,
            extra_args=list(extra_args),
            timeout_seconds=timeout_seconds,
    )

    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    rss_before_bytes = process.memory_info().rss
    wall_start = time.perf_counter()

    monitor = None
    gpu_total_energy_joules = 0.0
    if torch.cuda.is_available():
        try:
            current_device = torch.cuda.current_device()
            monitor = ZeusMonitor(gpu_indices=[current_device])
            monitor.begin_window('informer_etth_experiment')
        except Exception as exc:
            _logger.warning(
                    f'ZeusMonitor could not be initialized for Informer experiment, GPU metrics will be skipped: {exc}'
            )
            monitor = None

    dataset_name_final: Final[str] = dataset_name
    base_result = run_informer_external(
            config=informer_config,
            dataset_name=dataset_name_final,
            model_name=model_name,
    )

    if monitor is not None:
        try:
            measurement = monitor.end_window('informer_etth_experiment')
            gpu_total_energy_joules = float(measurement.total_energy)
        except Exception as exc:
            _logger.warning(
                    f'ZeusMonitor failed during Informer experiment measurement, GPU metrics will be skipped: {exc}'
            )
            gpu_total_energy_joules = 0.0

    wall_seconds_total = float(time.perf_counter() - wall_start)
    cpu_times_after = process.cpu_times()
    rss_after_bytes = process.memory_info().rss

    cpu_user_seconds_total = float(cpu_times_after.user - cpu_times_before.user)
    cpu_system_seconds_total = float(cpu_times_after.system - cpu_times_before.system)
    rss_mb_before = float(rss_before_bytes) / (1024.0 * 1024.0)
    rss_mb_after = float(rss_after_bytes) / (1024.0 * 1024.0)
    rss_mb_delta = rss_mb_after - rss_mb_before

    metrics = dict(base_result.metrics)
    metrics['wall_seconds_total'] = wall_seconds_total
    metrics['cpu_user_seconds_total'] = cpu_user_seconds_total
    metrics['cpu_system_seconds_total'] = cpu_system_seconds_total
    metrics['rss_mb_delta'] = rss_mb_delta
    metrics['gpu_total_energy_joules'] = gpu_total_energy_joules

    extra_info = dict(base_result.extra_info)
    extra_info_resource = {
        'wall_seconds_total': wall_seconds_total,
        'cpu_user_seconds_total': cpu_user_seconds_total,
        'cpu_system_seconds_total': cpu_system_seconds_total,
        'rss_mb_before': rss_mb_before,
        'rss_mb_after': rss_mb_after,
        'rss_mb_delta': rss_mb_delta,
        'gpu_total_energy_joules': gpu_total_energy_joules,
    }
    extra_info['resource_monitoring'] = extra_info_resource
    extra_info['diagnostics_json_path'] = metrics_json_path

    artifacts_from_base = extra_info.get('artifacts')
    if isinstance(artifacts_from_base, dict):
        predictions_csv = artifacts_from_base.get('predictions_csv')
        if isinstance(predictions_csv, str):
            extra_info['predictions_csv_path'] = predictions_csv

    result = ExperimentResult(
            model_name=base_result.model_name,
            dataset_name=base_result.dataset_name,
            metrics=metrics,
            extra_info=extra_info,
    )

    mse_value = float(result.metrics.get('mse', float('nan')))
    _logger.info(
            f'Informer-эксперимент для датасета "{dataset_name}" завершён: '
            f'MSE={mse_value:.6f}, wall_seconds_total={wall_seconds_total:.3f}, '
            f'gpu_total_energy_joules={gpu_total_energy_joules:.3f}.'
    )
    return result


def run_lstm_etth_experiment(
        dataset_name: str,
        csv_path: str,
        max_rows: int,
        train_ratio: float,
        seq_len: int,
        pred_len: int,
        hidden_size: int,
        num_layers: int,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        model_name: str,
        artifacts_dir: str,
        device_type: str = 'auto',
) -> ExperimentResult:
    """
    Высокоуровневая функция запуска эксперимента с LSTM на ETT-датасетах (любой вариант бенчмарка ETT).

    Все параметры передаются явно и имеют значения по умолчанию, чтобы удобно вызывать
    функцию из ноутбука. Функция не привязана к конкретной структуре директории и
    использует только путь к CSV-файлу.

    В дополнение к традиционной метрике MSE функция собирает агрегированные
    показатели использования ресурсов и энергопотребления за весь эксперимент:

    * wall_seconds_total: общее время выполнения эксперимента (обучение + валидация), секунд;
    * cpu_user_seconds_total / cpu_system_seconds_total: суммарное пользовательское/системное CPU-время процесса;
    * rss_mb_before / rss_mb_after / rss_mb_delta: оценка использования оперативной памяти (Resident Set Size);
    * gpu_total_energy_joules: энергопотребление GPU, измеренное через ZeusMonitor (если доступно).

    Параметры
    ----------
    dataset_name : str
        Имя датасета (вариант семейства ETT).
    csv_path : str
        Полный путь к CSV-файлу с датасетом.
    max_rows : int
        Максимальное количество строк, которые загружаются из датасета (0 или меньше — без ограничения).
    train_ratio : float
        Доля обучающей выборки.
    seq_len : int
        Длина входной последовательности для LSTM.
    pred_len : int
        Длина горизонта прогноза.
    hidden_size : int
        Размер скрытого состояния LSTM.
    num_layers : int
        Количество слоёв LSTM.
    learning_rate : float
        Скорость обучения.
    batch_size : int
        Размер батча.
    num_epochs : int
        Количество эпох обучения.
    model_name : str
        Имя модели в отчёте результата.
    artifacts_dir : str
        Каталог, в который будут сохранены артефакты.
    device_type : str, optional
        Тип вычислительного устройства: "auto", "cpu", "cuda" или "mps".
        При значении "auto" выбирается CUDA, затем MPS, иначе CPU.

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента с метрикой MSE и дополнительными системными метриками.
    """
    _logger.info(
            f'Запуск LSTM-эксперимента для датасета "{dataset_name}" с CSV по пути "{csv_path}".'
    )

    device = get_torch_device(device_type=device_type)
    _logger.info(f'Для LSTM-эксперимента будет использовано устройство "{device.type}".')

    train_df, valid_df = load_ett_csv_dataset(
            csv_path=csv_path,
            max_rows=max_rows,
            train_ratio=train_ratio,
    )

    hyperparams = LSTMHyperParams(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
    )

    result = _run_lstm_with_resource_monitoring(
            train_df=train_df,
            valid_df=valid_df,
            hyperparams=hyperparams,
            dataset_name=dataset_name,
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            device=device,
    )
    return result


def run_lstm_optuna_etth_experiment(
        dataset_name: str,
        csv_path: str,
        max_rows: int,
        train_ratio: float,
        seq_len: int,
        pred_len: int,
        num_epochs: int,
        n_trials: int,
        target_column: str,
        model_name: str,
        artifacts_dir: str,
        device_type: str = 'auto',
) -> ExperimentResult:
    """
    Запускает поиск гиперпараметров LSTM с помощью Optuna и обучает лучшую модель.

    Результатом является ExperimentResult для лучшей конфигурации, при этом
    артефакты (предсказания и диагностика) сохраняются в каталоге artifacts_dir.

    Поисковое пространство гиперпараметров задаётся через переменную окружения
    LSTM_SEARCH_SPACE_JSON (JSON-объект). Все ключи
    seq_len, pred_len, num_epochs, hidden_size, num_layers, learning_rate, batch_size
    должны быть заданы явно и иметь непустые списки значений.

    Параметры
    ----------
    dataset_name : str
        Имя датасета (вариант семейства ETT).
    csv_path : str
        Путь к CSV-файлу датасета.
    max_rows : int
        Максимальное количество строк, которые загружаются из датасета (0 или меньше — без ограничения).
    train_ratio : float
        Доля обучающей выборки.
    seq_len : int
        Базовая длина входной последовательности (не используется в поисковом пространстве).
    pred_len : int
        Базовая длина горизонта прогноза (не используется в поисковом пространстве).
    num_epochs : int
        Базовое число эпох (не используется в поисковом пространстве).
    n_trials : int
        Число испытаний Optuna.
    target_column : str
        Имя целевой колонки (например, 'OT').
    model_name : str
        Имя модели.
    artifacts_dir : str
        Каталог для сохранения артефактов лучшей модели.
    device_type : str, optional
        Тип вычислительного устройства: "auto", "cpu", "cuda" или "mps".
        При значении "auto" выбирается CUDA, затем MPS, иначе CPU.

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента для лучшей конфигурации LSTM.
    """
    del seq_len
    del pred_len
    del num_epochs

    if n_trials < 1:
        raise ValueError('Параметр n_trials должен быть не меньше 1.')

    _logger.info(
            f'Запуск LSTM+Optuna для датасета "{dataset_name}" с {n_trials} испытаниями.'
    )

    device = get_torch_device(device_type=device_type)
    _logger.info(
            f'Для LSTM+Optuna-эксперимента будет использовано устройство "{device.type}".'
    )

    train_df, valid_df = load_ett_csv_dataset(
            csv_path=csv_path,
            max_rows=max_rows,
            train_ratio=train_ratio,
    )

    search_space = _build_lstm_search_space_from_env()
    _logger.info(
            f'LSTM search space for dataset "{dataset_name}": '
            f'seq_len={search_space.seq_len_values}, '
            f'pred_len={search_space.pred_len_values}, '
            f'num_epochs={search_space.num_epochs_values}, '
            f'hidden_size={search_space.hidden_size_values}, '
            f'num_layers={search_space.num_layers_values}, '
            f'learning_rate={search_space.learning_rate_values}, '
            f'batch_size={search_space.batch_size_values}.'
    )

    study = baseline_optuna.run_optuna_for_etth(
            train_df=train_df,
            valid_df=valid_df,
            search_space=search_space,
            n_trials=n_trials,
            target_column=target_column,
            device=device,
    )

    best_trial = study.best_trial
    hidden_size = int(best_trial.params['hidden_size'])
    num_layers = int(best_trial.params['num_layers'])
    learning_rate = float(best_trial.params['learning_rate'])
    batch_size = int(best_trial.params['batch_size'])
    best_seq_len = int(best_trial.params['seq_len'])
    best_pred_len = int(best_trial.params['pred_len'])
    best_num_epochs = int(best_trial.params['num_epochs'])

    hyperparams = LSTMHyperParams(
            seq_len=best_seq_len,
            pred_len=best_pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=best_num_epochs,
    )

    base_result = _run_lstm_with_resource_monitoring(
            train_df=train_df,
            valid_df=valid_df,
            hyperparams=hyperparams,
            dataset_name=dataset_name,
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            device=device,
    )

    extra_info = dict(base_result.extra_info)
    extra_info['optuna'] = {
        'best_value': float(study.best_value),
        'best_params': dict(best_trial.params),
        'n_trials': len(study.trials),
        'study_name': study.study_name,
    }

    result = ExperimentResult(
            model_name=base_result.model_name,
            dataset_name=base_result.dataset_name,
            metrics=dict(base_result.metrics),
            extra_info=extra_info,
    )

    _logger.info(
            f'LSTM+Optuna для датасета "{dataset_name}" завершён: '
            f'лучшая MSE={study.best_value:.6f}.'
    )
    return result


def run_informer_etth_experiment(
        dataset_name: str,
        csv_path: str,
        max_rows: int,
        train_ratio: float,
        informer_script_path: str,
        metrics_json_path: str,
        extra_args: list[str] | None,
        timeout_seconds: int,
        model_name: str,
) -> ExperimentResult:
    """
    Высокоуровневая функция запуска эксперимента с Informer на ETT-датасетах
    для использования в Jupyter.

    Для работы требуется внешний скрипт (например, informer_experiment_wrapper.py),
    который:
      * принимает флаги --data-path и --metrics-path;
      * выполняет обучение/оценку модели Informer;
      * сохраняет JSON с метриками в metrics_json_path.

    Помимо метрик качества, возвращаемых внешним скриптом, функция собирает
    агрегированные системные метрики по времени выполнения и использованию ресурсов:

    * wall_seconds_total: общее время работы внешнего скрипта;
    * cpu_user_seconds_total / cpu_system_seconds_total: изменение CPU-времени текущего процесса;
    * rss_mb_before / rss_mb_after / rss_mb_delta: оценка изменения объёма используемой памяти;
    * gpu_total_energy_joules: энергопотребление GPU в окне запуска Informer (ZeusMonitor).

    Параметры
    ----------
    dataset_name : str
        Имя датасета (вариант семейства ETT).
    csv_path : str
        Полный путь к CSV-файлу с датасетом.
    max_rows : int
        Максимальное количество строк, которые загружаются из датасета (0 или меньше — без ограничения).
        Параметр оставлен для единообразия интерфейса, но фактическое разбиение осуществляется
        внутри внешнего скрипта Informer.
    train_ratio : float
        Доля обучающей выборки. Может не использоваться напрямую, если логика разбиения
        реализована во внешнем скрипте.
    informer_script_path : str
        Путь к внешнему Python-скрипту, который запускает эксперименты Informer.
    metrics_json_path : str
        Путь к JSON-файлу, в который внешний скрипт запишет метрики.
    extra_args : list[str] | None
        Дополнительные аргументы командной строки для внешнего скрипта (например, параметры модели).
    timeout_seconds : int
        Максимальное время ожидания завершения внешнего скрипта в секундах.
    model_name : str
        Имя модели в отчёте результата.

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента с метриками Informer и дополнительными системными метриками.
    """
    del max_rows
    del train_ratio

    args_list: list[str] = []
    if extra_args is not None:
        args_list = list(extra_args)

    _logger.info(
            f'Запуск Informer-эксперимента для датасета "{dataset_name}" '
            f'с CSV по пути "{csv_path}".'
    )

    result = _run_informer_with_resource_monitoring(
            dataset_name=dataset_name,
            csv_path=csv_path,
            informer_script_path=informer_script_path,
            metrics_json_path=metrics_json_path,
            extra_args=args_list,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
    )
    return result


def _build_informer_optuna_args(
        base_extra_args: list[str],
        params: dict[str, float | int],
) -> list[str]:
    """
    Build CLI arguments list for Informer from base arguments and Optuna parameters.
    """
    args = list(base_extra_args)
    args.extend(
            [
                '--d_model',
                str(int(params['d_model'])),
                '--n_heads',
                str(int(params['n_heads'])),
                '--e_layers',
                str(int(params['e_layers'])),
                '--d_layers',
                str(int(params['d_layers'])),
                '--factor',
                str(int(params['factor'])),
                '--dropout',
                str(float(params['dropout'])),
                '--learning_rate',
                str(float(params['learning_rate'])),
                '--batch_size',
                str(int(params['batch_size'])),
                '--epochs',
                str(int(params['epochs'])),
            ]
    )
    return args


def run_informer_optuna_etth_experiment(
        dataset_name: str,
        csv_path: str,
        max_rows: int,
        train_ratio: float,
        informer_script_path: str,
        metrics_root_dir: str,
        base_extra_args: list[str] | None,
        timeout_seconds: int,
        model_name: str,
        n_trials: int,
) -> ExperimentResult:
    """
    Запускает поиск гиперпараметров Informer с помощью Optuna и обучает лучшую модель.

    Для каждого испытания запускается полный Informer-эксперимент с собственным
    JSON-диагностикой и артефактами. Лучшая конфигурация затем переобучается,
    и её результаты возвращаются в виде ExperimentResult.

    Поисковое пространство гиперпараметров задаётся через переменную окружения
    INFORMER_SEARCH_SPACE_JSON (JSON-объект). Все ключи
    d_model, n_heads, e_layers, d_layers, factor, dropout, learning_rate, batch_size, epochs
    должны быть заданы явно и иметь непустые списки значений.

    Параметры
    ----------
    dataset_name : str
        Имя датасета (вариант семейства ETT).
    csv_path : str
        Путь к CSV-файлу датасета.
    max_rows : int
        Максимальное количество строк (оставлено для единообразия интерфейса).
    train_ratio : float
        Доля обучающей выборки (оставлено для единообразия интерфейса).
    informer_script_path : str
        Путь к внешнему скрипту-обёртке Informer.
    metrics_root_dir : str
        Каталог, в котором будут созданы JSON-файлы диагностик для испытаний Optuna.
    base_extra_args : list[str] | None
        Базовый набор аргументов командной строки Informer, общих для всех испытаний.
    timeout_seconds : int
        Тайм-аут выполнения одного запуска Informer.
    model_name : str
        Имя модели.
    n_trials : int
        Число испытаний Optuna.

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента Informer для лучшей конфигурации.
    """
    del max_rows
    del train_ratio

    if n_trials < 1:
        raise ValueError('Параметр n_trials должен быть не меньше 1.')

    base_args: list[str] = []
    if base_extra_args is not None:
        base_args = list(base_extra_args)

    metrics_root_path = Path(metrics_root_dir).resolve()
    metrics_root_path.mkdir(parents=True, exist_ok=True)

    search_space = _build_informer_search_space_from_env()
    _logger.info(
            f'Informer search space for dataset "{dataset_name}": '
            f'd_model={search_space.d_model_values}, '
            f'n_heads={search_space.n_heads_values}, '
            f'e_layers={search_space.e_layers_values}, '
            f'd_layers={search_space.d_layers_values}, '
            f'factor={search_space.factor_values}, '
            f'dropout={search_space.dropout_values}, '
            f'learning_rate={search_space.learning_rate_values}, '
            f'batch_size={search_space.batch_size_values}, '
            f'epochs={search_space.epochs_values}.'
    )

    _logger.info(
            f'Запуск Informer+Optuna для датасета "{dataset_name}" с {n_trials} испытаниями.'
    )

    def objective(trial: optuna.Trial) -> float:
        params_for_cli: dict[str, float | int] = {
            'd_model': int(
                    trial.suggest_categorical('d_model', search_space.d_model_values)
            ),
            'n_heads': int(
                    trial.suggest_categorical('n_heads', search_space.n_heads_values)
            ),
            'e_layers': int(
                    trial.suggest_categorical('e_layers', search_space.e_layers_values)
            ),
            'd_layers': int(
                    trial.suggest_categorical('d_layers', search_space.d_layers_values)
            ),
            'factor': int(
                    trial.suggest_categorical('factor', search_space.factor_values)
            ),
            'dropout': float(
                    trial.suggest_categorical('dropout', search_space.dropout_values)
            ),
            'learning_rate': float(
                    trial.suggest_categorical(
                            'learning_rate', search_space.learning_rate_values
                    )
            ),
            'batch_size': int(
                    trial.suggest_categorical('batch_size', search_space.batch_size_values)
            ),
            'epochs': int(
                    trial.suggest_categorical('epochs', search_space.epochs_values)
            ),
        }

        trial_args = _build_informer_optuna_args(base_args, params_for_cli)
        trial_metrics_path = str(
                metrics_root_path / f'informer_optuna_trial_{trial.number}_diagnostics.json'
        )

        result = run_informer_etth_experiment(
                dataset_name=dataset_name,
                csv_path=csv_path,
                max_rows=0,
                train_ratio=0.5,
                informer_script_path=informer_script_path,
                metrics_json_path=trial_metrics_path,
                extra_args=trial_args,
                timeout_seconds=timeout_seconds,
                model_name=model_name,
        )
        mse = result.metrics.get('mse')
        if mse is None:
            raise RuntimeError('Informer-эксперимент не вернул метрику "mse".')
        return float(mse)

    study = optuna.create_study(
            direction='minimize',
            study_name=f'informer_etth_mse_{dataset_name}',
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_trial.params)
    best_args = _build_informer_optuna_args(base_args, best_params)
    best_metrics_path = str(metrics_root_path / 'informer_optuna_best_diagnostics.json')

    best_result = run_informer_etth_experiment(
            dataset_name=dataset_name,
            csv_path=csv_path,
            max_rows=0,
            train_ratio=0.5,
            informer_script_path=informer_script_path,
            metrics_json_path=best_metrics_path,
            extra_args=best_args,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
    )

    extra_info = dict(best_result.extra_info)
    extra_info['optuna'] = {
        'best_value': float(study.best_value),
        'best_params': best_params,
        'n_trials': len(study.trials),
        'study_name': study.study_name,
    }

    result = ExperimentResult(
            model_name=best_result.model_name,
            dataset_name=best_result.dataset_name,
            metrics=dict(best_result.metrics),
            extra_info=extra_info,
    )

    _logger.info(
            f'Informer+Optuna для датасета "{dataset_name}" завершён: '
            f'лучшая MSE={study.best_value:.6f}.'
    )
    return result