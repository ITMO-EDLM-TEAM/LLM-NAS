# edlm_search/src/experiments/experiment_api.py
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import edlm_search.baseline_optuna as baseline_optuna
import optuna
import pandas as pd
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


def _run_lstm_with_resource_monitoring(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        hyperparams: LSTMHyperParams,
        dataset_name: str,
        model_name: str,
        artifacts_dir: str,
) -> ExperimentResult:
    """
    Run LSTM experiment with resource monitoring and unified artifact saving.

    This helper:
      * measures wall time, CPU time, RSS and GPU energy;
      * calls run_lstm_on_etth_dataset to train and save artifacts;
      * enriches metrics and extra_info with resource statistics.
    """
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

    result = _run_lstm_with_resource_monitoring(
            train_df=train_df,
            valid_df=valid_df,
            hyperparams=hyperparams,
            dataset_name=dataset_name,
            model_name=model_name,
            artifacts_dir=artifacts_dir,
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
) -> ExperimentResult:
    """
    Запускает поиск гиперпараметров LSTM с помощью Optuna и обучает лучшую модель.

    Результатом является ExperimentResult для лучшей конфигурации, при этом
    артефакты (предсказания и диагностика) сохраняются в каталоге artifacts_dir.

    Параметры
    ----------
    dataset_name : str
        Имя датасета (например, 'ETTh1').
    csv_path : str
        Путь к CSV-файлу датасета.
    max_rows : int
        Максимальное количество строк, которые загружаются из датасета (0 или меньше — без ограничения).
    train_ratio : float
        Доля обучающей выборки.
    seq_len : int
        Длина входной последовательности.
    pred_len : int
        Длина горизонта прогноза.
    num_epochs : int
        Число эпох обучения в каждой попытке.
    n_trials : int
        Число испытаний Optuna.
    target_column : str
        Имя целевой колонки (например, 'OT').
    model_name : str
        Имя модели.
    artifacts_dir : str
        Каталог для сохранения артефактов лучшей модели.

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента для лучшей конфигурации LSTM.
    """
    if n_trials < 1:
        raise ValueError('Параметр n_trials должен быть не меньше 1.')

    _logger.info(
            f'Запуск LSTM+Optuna для датасета "{dataset_name}" с {n_trials} испытаниями.'
    )

    train_df, valid_df = load_ett_csv_dataset(
            csv_path=csv_path,
            max_rows=max_rows,
            train_ratio=train_ratio,
    )

    study = baseline_optuna.run_optuna_for_etth(
            train_df=train_df,
            valid_df=valid_df,
            seq_len=seq_len,
            pred_len=pred_len,
            num_epochs=num_epochs,
            n_trials=n_trials,
            target_column=target_column,
    )

    best_trial = study.best_trial
    hidden_size = int(best_trial.params['hidden_size'])
    num_layers = int(best_trial.params['num_layers'])
    learning_rate = float(best_trial.params['learning_rate'])
    batch_size = int(best_trial.params['batch_size'])

    hyperparams = LSTMHyperParams(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
    )

    base_result = _run_lstm_with_resource_monitoring(
            train_df=train_df,
            valid_df=valid_df,
            hyperparams=hyperparams,
            dataset_name=dataset_name,
            model_name=model_name,
            artifacts_dir=artifacts_dir,
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
    Высокоуровневая функция запуска эксперимента с Informer на ETTh-датасетах (ETTh1, ETTh2)
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

    Параметры
    ----------
    dataset_name : str
        Имя датасета (например, 'ETTh1').
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

    _logger.info(
            f'Запуск Informer+Optuna для датасета "{dataset_name}" с {n_trials} испытаниями.'
    )

    def objective(trial: optuna.Trial) -> float:
        params_for_cli: dict[str, float | int] = {
            'd_model': trial.suggest_int('d_model', 128, 512, step=64),
            'n_heads': trial.suggest_int('n_heads', 2, 8),
            'e_layers': trial.suggest_int('e_layers', 1, 3),
            'd_layers': trial.suggest_int('d_layers', 1, 3),
            'factor': trial.suggest_int('factor', 1, 5),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
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