from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import psutil
import torch
from zeus.monitor import ZeusMonitor

from .datasets import load_ett_csv_dataset
from .informer_runner import run_informer_external
from .lstm_runner import run_lstm_on_etth_dataset
from .types import ExperimentResult
from .types import InformerRunnerConfig
from .types import LSTMHyperParams

_logger = logging.getLogger(__name__)


def run_lstm_etth_experiment(
        dataset_name: str,
        csv_path: str,
        max_rows: int = 10000,
        train_ratio: float = 0.8,
        seq_len: int = 96,
        pred_len: int = 24,
        hidden_size: int = 128,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 10,
        model_name: str = 'lstm-baseline',
) -> ExperimentResult:
    """
    Высокоуровневая функция запуска эксперимента с LSTM на ETTh-датасетах (ETTh1, ETTh2).

    Все параметры передаются явно и имеют значения по умолчанию, чтобы удобно вызывать
    функцию из ноутбука. Функция не привязана к конкретной структуре директорий и
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
        Имя датасета (например, 'ETTh1' или 'ETTh2').
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

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента с метрикой MSE и дополнительными системными метриками.
    """
    _logger.info(
            f'Запуск LSTM-эксперимента для датасета "{dataset_name}" с CSV по пути "{csv_path}".'
    )

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

    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    rss_before_bytes = process.memory_info().rss
    wall_start = time.perf_counter()

    monitor = None
    gpu_total_energy_joules = 0.0
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        monitor = ZeusMonitor(gpu_indices=[current_device])
        monitor.begin_window('lstm_etth_experiment')

    dataset_name_final: Final[str] = dataset_name
    base_result = run_lstm_on_etth_dataset(
            train_df=train_df,
            valid_df=valid_df,
            hyperparams=hyperparams,
            dataset_name=dataset_name_final,
            model_name=model_name,
    )

    if monitor is not None:
        measurement = monitor.end_window('lstm_etth_experiment')
        gpu_total_energy_joules = float(measurement.total_energy)

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


def run_informer_etth_experiment(
        dataset_name: str,
        csv_path: str,
        max_rows: int = 10000,
        train_ratio: float = 0.8,
        informer_script_path: str = './informer_experiment_wrapper.py',
        metrics_json_path: str = './informer_metrics/etth_metrics.json',
        extra_args: list[str] | None = None,
        timeout_seconds: int = 36000,
        model_name: str = 'informer-original',
) -> ExperimentResult:
    """
    Высокоуровневая функция запуска эксперимента с Informer на ETTh-датасетах (ETTh1, ETTh2)
    для использования в Jupyter.

    Для работы требуется внешний скрипт (например, из репозитория Informer2020),
    который:
      * принимает флаги --data-path и --metrics-path;
      * выполняет обучение/оценку модели Informer;
      * сохраняет метрики в указанный JSON-файл.

    Помимо метрик качества, возвращаемых внешним скриптом, функция собирает
    агрегированные системные метрики по времени выполнения и использованию ресурсов:

    * wall_seconds_total: общее время работы внешнего скрипта;
    * cpu_user_seconds_total / cpu_system_seconds_total: изменение CPU-времени текущего процесса;
    * rss_mb_before / rss_mb_after / rss_mb_delta: оценка изменения объёма используемой памяти;
    * gpu_total_energy_joules: энергопотребление GPU в окне запуска Informer (ZeusMonitor).

    Параметры
    ----------
    dataset_name : str
        Имя датасета (например, 'ETTh1' или 'ETTh2').
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

    original_csv_path = csv_path
    csv_path_obj = Path(csv_path)

    if not csv_path_obj.is_file():
        project_root = Path(__file__).resolve().parent.parent
        fallback_csv = project_root / 'ETDataset' / 'ETT-small' / f'{dataset_name}.csv'
        if fallback_csv.is_file():
            _logger.info(
                    f'Путь к CSV "{original_csv_path}" не найден, используется путь по умолчанию '
                    f'"{fallback_csv}".'
            )
            csv_path_obj = fallback_csv
        else:
            raise FileNotFoundError(
                    f'CSV-файл датасета для Informer не найден по пути "{original_csv_path}". '
                    f'Также не удалось найти файл по ожидаемому пути "{fallback_csv}".'
            )

    resolved_csv_path = str(csv_path_obj.resolve())

    _logger.info(
            f'Запуск Informer-эксперимента для датасета "{dataset_name}" '
            f'с CSV по пути "{resolved_csv_path}".'
    )

    args_list: list[str] = []
    if extra_args is not None:
        args_list = list(extra_args)

    informer_config = InformerRunnerConfig(
            dataset_csv_path=resolved_csv_path,
            metrics_json_path=metrics_json_path,
            informer_script_path=informer_script_path,
            extra_args=args_list,
            timeout_seconds=timeout_seconds,
    )

    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    rss_before_bytes = process.memory_info().rss
    wall_start = time.perf_counter()

    monitor = None
    gpu_total_energy_joules = 0.0
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        monitor = ZeusMonitor(gpu_indices=[current_device])
        monitor.begin_window('informer_etth_experiment')

    dataset_name_final: Final[str] = dataset_name
    base_result = run_informer_external(
            config=informer_config,
            dataset_name=dataset_name_final,
            model_name=model_name,
    )

    if monitor is not None:
        measurement = monitor.end_window('informer_etth_experiment')
        gpu_total_energy_joules = float(measurement.total_energy)

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
    extra_info.setdefault('resource_monitoring', extra_info_resource)
    if extra_info.get('resource_monitoring') is not extra_info_resource:
        extra_info['resource_monitoring'] = extra_info_resource

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