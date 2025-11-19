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