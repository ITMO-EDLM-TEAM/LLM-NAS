from __future__ import annotations

import pandas as pd
import torch

from .types import ExperimentResult
from .types import LSTMHyperParams
from ..edlm_search.baseline_optuna import _create_dataloaders_for_ettm1
from ..edlm_search.baseline_optuna import _train_one_model


def run_lstm_on_ettm1(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        hyperparams: LSTMHyperParams,
        dataset_name: str,
        model_name: str,
) -> ExperimentResult:
    """
    Запускает LSTM-модель на датасете ETTm1 и возвращает результат эксперимента.

    Параметры
    ----------
    train_df : pandas.DataFrame
        Обучающая выборка.
    valid_df : pandas.DataFrame
        Валидационная выборка.
    hyperparams : LSTMHyperParams
        Гиперпараметры LSTM-модели и обучения.
    dataset_name : str
        Имя датасета (например, 'ETTm1').
    model_name : str
        Имя модели (например, 'lstm-baseline').

    Возвращает
    ----------
    ExperimentResult
        Результат эксперимента с метрикой MSE на валидации.
    """
    train_loader, valid_loader, feature_columns = _create_dataloaders_for_ettm1(
            train_df=train_df,
            valid_df=valid_df,
            seq_len=hyperparams.seq_len,
            pred_len=hyperparams.pred_len,
            batch_size=hyperparams.batch_size,
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

    metrics: dict[str, float] = {'mse': float(mse)}
    extra_info: dict[str, float] = {
        'seq_len': float(hyperparams.seq_len),
        'pred_len': float(hyperparams.pred_len),
        'hidden_size': float(hyperparams.hidden_size),
        'num_layers': float(hyperparams.num_layers),
        'learning_rate': float(hyperparams.learning_rate),
        'batch_size': float(hyperparams.batch_size),
        'num_epochs': float(hyperparams.num_epochs),
    }

    return ExperimentResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            extra_info=extra_info,
    )