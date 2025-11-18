from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from edlm_search.baseline_optuna import _train_one_model
from edlm_search.baseline_optuna import create_dataloaders_for_etth
from edlm_search.baseline_optuna import get_last_run_diagnostics

from .types import ExperimentResult
from .types import LSTMHyperParams


def run_lstm_on_etth_dataset(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        hyperparams: LSTMHyperParams,
        dataset_name: str,
        model_name: str,
        artifacts_dir: str,
) -> ExperimentResult:
    """
    Запускает LSTM-модель на ETTh-датасете (ETTh1 или ETTh2) и возвращает результат эксперимента.

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
        Имя датасета (например, 'ETTh1' или 'ETTh2').
    model_name : str
        Имя модели (например, 'lstm-baseline').
    artifacts_dir : str
        Каталог, в который будут сохранены метрики и предсказания.

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента с метрикой MSE на валидации и дополнительной информацией.
    """
    if not dataset_name:
        raise ValueError("Имя датасета не может быть пустым.")
    if not model_name:
        raise ValueError("Имя модели не может быть пустым.")

    artifacts_path = Path(artifacts_dir).resolve()
    artifacts_path.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, feature_columns = create_dataloaders_for_etth(
            train_df=train_df,
            valid_df=valid_df,
            seq_len=hyperparams.seq_len,
            pred_len=hyperparams.pred_len,
            batch_size=hyperparams.batch_size,
            target_column='OT',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    predictions_path = artifacts_path / f'{model_name}_{dataset_name}_valid_predictions.csv'
    diagnostics_json_path = artifacts_path / f'{model_name}_{dataset_name}_diagnostics.json'

    y_pred = diagnostics.get('validation_predictions')
    y_true = diagnostics.get('validation_targets')
    if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
        if y_pred.shape == y_true.shape:
            df_predictions = pd.DataFrame(
                    {
                        'y_true': y_true.reshape(-1),
                        'y_pred': y_pred.reshape(-1),
                    }
            )
            df_predictions.to_csv(predictions_path, index=False)

    diagnostics_to_save: dict[str, object] = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'metrics': metrics,
        'hyperparams': {
            'seq_len': hyperparams.seq_len,
            'pred_len': hyperparams.pred_len,
            'hidden_size': hyperparams.hidden_size,
            'num_layers': hyperparams.num_layers,
            'learning_rate': hyperparams.learning_rate,
            'batch_size': hyperparams.batch_size,
            'num_epochs': hyperparams.num_epochs,
        },
        'artifacts': {
            'predictions_csv': str(predictions_path),
        },
    }

    with diagnostics_json_path.open('w', encoding='utf-8') as f:
        json.dump(diagnostics_to_save, f, ensure_ascii=False, indent=2)

    extra_info: dict[str, float | str] = {
        'seq_len': float(hyperparams.seq_len),
        'pred_len': float(hyperparams.pred_len),
        'hidden_size': float(hyperparams.hidden_size),
        'num_layers': float(hyperparams.num_layers),
        'learning_rate': float(hyperparams.learning_rate),
        'batch_size': float(hyperparams.batch_size),
        'num_epochs': float(hyperparams.num_epochs),
        'predictions_csv_path': str(predictions_path),
        'diagnostics_json_path': str(diagnostics_json_path),
    }

    dataset_name_final: Final[str] = dataset_name

    return ExperimentResult(
            model_name=model_name,
            dataset_name=dataset_name_final,
            metrics=metrics,
            extra_info=extra_info,
    )