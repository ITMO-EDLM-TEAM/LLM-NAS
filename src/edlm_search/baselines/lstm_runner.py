# edlm_search/src/experiments/lstm_runner.py
from __future__ import annotations

import json
from typing import Final

import numpy as np
import pandas as pd
import torch

from .baseline_optuna import _train_one_model
from .baseline_optuna import create_dataloaders_for_etth
from .baseline_optuna import get_last_run_diagnostics
from ..experiments.artifacts import ExperimentArtifactsManager
from ..experiments.artifacts import ExperimentDescriptor
from ..experiments.types import ExperimentResult
from ..experiments.types import LSTMHyperParams


def _create_artifacts_manager(
        artifacts_dir: str,
        model_name: str,
        dataset_name: str,
) -> ExperimentArtifactsManager:
    """
    Create ExperimentArtifactsManager instance for a single LSTM run.
    """
    descriptor = ExperimentDescriptor(
            model_name=model_name,
            dataset_name=dataset_name,
            experiment_kind='lstm',
    )
    manager = ExperimentArtifactsManager(root_dir=artifacts_dir, descriptor=descriptor)
    return manager


def _save_lstm_predictions(
        manager: ExperimentArtifactsManager,
        model_name: str,
        dataset_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> str | None:
    """
    Save validation predictions and targets to CSV inside the experiment directory.
    """
    if y_true.shape != y_pred.shape:
        return None

    filename = f'{model_name}_{dataset_name}_valid_predictions.csv'
    predictions_path = manager.build_path(filename)

    df_predictions = pd.DataFrame(
            {
                'y_true': y_true.reshape(-1),
                'y_pred': y_pred.reshape(-1),
            }
    )
    df_predictions.to_csv(predictions_path, index=False)
    return str(predictions_path)


def _build_lstm_diagnostics_payload(
        model_name: str,
        dataset_name: str,
        metrics: dict[str, float],
        hyperparams: LSTMHyperParams,
        diagnostics: dict[str, object],
        predictions_csv_path: str | None,
) -> dict[str, object]:
    """
    Build diagnostics JSON payload for a single LSTM run.
    """
    diagnostics_hyperparams: dict[str, float | int] = {
        'seq_len': int(hyperparams.seq_len),
        'pred_len': int(hyperparams.pred_len),
        'hidden_size': int(hyperparams.hidden_size),
        'num_layers': int(hyperparams.num_layers),
        'learning_rate': float(hyperparams.learning_rate),
        'batch_size': int(hyperparams.batch_size),
        'num_epochs': int(hyperparams.num_epochs),
    }

    artifacts_section: dict[str, object] = {}
    if predictions_csv_path is not None:
        artifacts_section['predictions_csv'] = predictions_csv_path

    metrics_copy: dict[str, float] = {}
    for key, value in metrics.items():
        metrics_copy[key] = float(value)

    diagnostics_payload: dict[str, object] = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'metrics': metrics_copy,
        'hyperparams': diagnostics_hyperparams,
        'artifacts': artifacts_section,
    }

    numeric_keys = {
        'train_time_seconds',
        'eval_time_seconds',
        'total_energy_joules',
        'peak_gpu_memory_mb',
        'max_rss_mb',
    }
    for key in numeric_keys:
        value = diagnostics.get(key)
        if isinstance(value, (int, float)):
            diagnostics_payload.setdefault('metrics', {})[key] = float(value)

    return diagnostics_payload


def run_lstm_on_etth_dataset(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        hyperparams: LSTMHyperParams,
        dataset_name: str,
        model_name: str,
        artifacts_dir: str,
        device: torch.device,
) -> ExperimentResult:
    """
    Запускает LSTM-модель на ETT-датасете и возвращает результат эксперимента.

    Помимо MSE, функция сохраняет дополнительные метрики (время, энергию, ресурсы) и
    предсказания на валидации в файлы внутри каталога artifacts_dir.

    Параметры
    ----------
    train_df : pandas.DataFrame
        Обучающая выборка.
    valid_df : pandas.DataFrame
        Валидационная выборка.
    hyperparams : LSTMHyperParams
        Гиперпараметры LSTM-модели и обучения.
    dataset_name : str
        Имя датасета.
    model_name : str
        Имя модели (например, 'lstm-baseline').
    artifacts_dir : str
        Каталог, в который будут сохранены метрики и предсказания.
    device : torch.device
        Устройство, на котором будет выполняться обучение и инференс модели.

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента с метрикой MSE на валидации и дополнительной информацией.
    """
    if not dataset_name:
        raise ValueError("Имя датасета не может быть пустым.")
    if not model_name:
        raise ValueError("Имя модели не может быть пустым.")

    manager = _create_artifacts_manager(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            dataset_name=dataset_name,
    )

    train_loader, valid_loader, feature_columns = create_dataloaders_for_etth(
            train_df=train_df,
            valid_df=valid_df,
            seq_len=hyperparams.seq_len,
            pred_len=hyperparams.pred_len,
            batch_size=hyperparams.batch_size,
            target_column='OT',
    )

    mse = _train_one_model(
            train_loader=train_loader,
            valid_loader=valid_loader,
            input_size=len(feature_columns),
            hidden_size=hyperparams.hidden_size,
            num_layers=hyperparams.num_layers,
            pred_len=hyperparams.pred_len,
            learning_rate=hyperparams.learning_rate,
            num_epochs=hyperparams.num_epochs,
            device=device,
    )

    diagnostics = get_last_run_diagnostics()

    metrics: dict[str, float] = {'mse': float(mse)}
    for key, value in diagnostics.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    y_pred = diagnostics.get('validation_predictions')
    y_true = diagnostics.get('validation_targets')

    predictions_csv_path: str | None = None
    if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
        predictions_csv_path = _save_lstm_predictions(
                manager=manager,
                model_name=model_name,
                dataset_name=dataset_name,
                y_true=y_true,
                y_pred=y_pred,
        )

    diagnostics_payload = _build_lstm_diagnostics_payload(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            hyperparams=hyperparams,
            diagnostics=diagnostics,
            predictions_csv_path=predictions_csv_path,
    )

    diagnostics_filename = f'{model_name}_{dataset_name}_diagnostics.json'
    diagnostics_json_path = manager.build_path(diagnostics_filename)
    with diagnostics_json_path.open('w', encoding='utf-8') as f:
        json.dump(diagnostics_payload, f, ensure_ascii=False, indent=2)

    extra_info: dict[str, float | str] = {
        'seq_len': float(hyperparams.seq_len),
        'pred_len': float(hyperparams.pred_len),
        'hidden_size': float(hyperparams.hidden_size),
        'num_layers': float(hyperparams.num_layers),
        'learning_rate': float(hyperparams.learning_rate),
        'batch_size': float(hyperparams.batch_size),
        'num_epochs': float(hyperparams.num_epochs),
        'predictions_csv_path': predictions_csv_path or '',
        'diagnostics_json_path': str(diagnostics_json_path),
        'artifacts_root_dir': str(manager.base_dir),
    }

    dataset_name_final: Final[str] = dataset_name

    return ExperimentResult(
            model_name=model_name,
            dataset_name=dataset_name_final,
            metrics=metrics,
            extra_info=extra_info,
    )