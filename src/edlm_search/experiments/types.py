from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetConfig:
    """Описание датасета для эксперимента."""

    name: str
    root_dir: str
    csv_filename: str
    max_rows: int
    train_ratio: float


@dataclass
class LSTMHyperParams:
    """Гиперпараметры LSTM-модели для эксперимента."""

    seq_len: int
    pred_len: int
    hidden_size: int
    num_layers: int
    learning_rate: float
    batch_size: int
    num_epochs: int


@dataclass
class InformerRunnerConfig:
    """Конфигурация запуска эксперимента Informer через внешний скрипт."""

    dataset_csv_path: str
    metrics_json_path: str
    informer_script_path: str
    extra_args: list[str]
    timeout_seconds: int


@dataclass
class ExperimentResult:
    """Результат эксперимента для одной модели и одного датасета."""

    model_name: str
    dataset_name: str
    metrics: dict[str, float]
    extra_info: dict[str, Any]


@dataclass
class LSTMSearchSpace:
    """Search space configuration for LSTM + Optuna hyperparameter optimization."""

    seq_len_values: list[int]
    pred_len_values: list[int]
    num_epochs_values: list[int]
    hidden_size_values: list[int]
    num_layers_values: list[int]
    learning_rate_values: list[float]
    batch_size_values: list[int]


@dataclass
class InformerSearchSpace:
    """Search space configuration for Informer + Optuna hyperparameter optimization."""

    d_model_values: list[int]
    n_heads_values: list[int]
    e_layers_values: list[int]
    d_layers_values: list[int]
    factor_values: list[int]
    dropout_values: list[float]
    learning_rate_values: list[float]
    batch_size_values: list[int]
    epochs_values: list[int]