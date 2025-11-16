from __future__ import annotations

from typing import Final

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ETTSequenceDataset(Dataset):
    """Dataset для последовательностного прогнозирования по датасету ETTm1."""

    def __init__(
            self,
            df: pd.DataFrame,
            seq_len: int,
            pred_len: int,
            feature_columns: list[str],
            target_column: str,
    ):
        """
        Инициализирует датасет для временных рядов.

        Параметры
        ----------
        df : pandas.DataFrame
            Исходный датафрейм с признаками и целевой переменной.
        seq_len : int
            Длина входной последовательности.
        pred_len : int
            Длина прогнозируемого горизонта.
        feature_columns : list[str]
            Список имён колонок с признаками.
        target_column : str
            Имя колонки с целевой переменной.
        """
        if seq_len < 1 or pred_len < 1:
            raise ValueError("Параметры seq_len и pred_len должны быть положительными.")

        self._df: Final[pd.DataFrame] = df.reset_index(drop=True)
        self._seq_len: Final[int] = seq_len
        self._pred_len: Final[int] = pred_len
        self._feature_columns: Final[list[str]] = list(feature_columns)
        self._target_column: Final[str] = target_column

        max_index = len(self._df) - self._seq_len - self._pred_len + 1
        if max_index <= 0:
            raise ValueError("Размер датафрейма слишком мал для заданных seq_len и pred_len.")
        self._max_start_index: Final[int] = max_index

    def __len__(self) -> int:
        """
        Возвращает количество доступных сэмплов.

        Возвращает
        ----------
        int
            Число сэмплов в датасете.
        """
        return self._max_start_index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает один сэмпл (контекст и целевой отрезок).

        Параметры
        ----------
        index : int
            Индекс начальной позиции последовательности.

        Возвращает
        ----------
        tuple[torch.Tensor, torch.Tensor]
            Пара (x, y), где x — входная последовательность, y — целевой прогноз.
        """
        if index < 0 or index >= self._max_start_index:
            raise IndexError("Индекс вне диапазона датасета.")

        start = index
        mid = index + self._seq_len
        end = mid + self._pred_len

        context = self._df.iloc[start:mid]
        target = self._df.iloc[mid:end]

        x = torch.tensor(context[self._feature_columns].values, dtype=torch.float32)
        y = torch.tensor(target[self._target_column].values, dtype=torch.float32)
        return x, y


class SimpleLSTMForecaster(nn.Module):
    """Простой LSTM-модель для прогноза временного ряда."""

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            pred_len: int,
    ):
        """
        Инициализирует LSTM-модель.

        Параметры
        ----------
        input_size : int
            Размерность входного вектора признаков.
        hidden_size : int
            Размер скрытого состояния LSTM.
        num_layers : int
            Количество слоёв LSTM.
        pred_len : int
            Длина прогноза по времени.
        """
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or pred_len < 1:
            raise ValueError("Параметры модели должны быть положительными целыми числами.")

        self._input_size: Final[int] = input_size
        self._hidden_size: Final[int] = hidden_size
        self._num_layers: Final[int] = num_layers
        self._pred_len: Final[int] = pred_len

        self._lstm = nn.LSTM(
                input_size=self._input_size,
                hidden_size=self._hidden_size,
                num_layers=self._num_layers,
                batch_first=True,
        )
        self._head = nn.Linear(self._hidden_size, self._pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход модели.

        Параметры
        ----------
        x : torch.Tensor
            Входной тензор формы (batch_size, seq_len, input_size).

        Возвращает
        ----------
        torch.Tensor
            Выходной тензор формы (batch_size, pred_len).
        """
        output, _ = self._lstm(x)
        last_hidden = output[:, -1, :]
        prediction = self._head(last_hidden)
        return prediction


def _create_dataloaders_for_ettm1(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        batch_size: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Создаёт DataLoader'ы для обучения и валидации на ETTm1.

    Параметры
    ----------
    train_df : pandas.DataFrame
        Обучающий датафрейм.
    valid_df : pandas.DataFrame
        Валидационный датафрейм.
    seq_len : int
        Длина входной последовательности.
    pred_len : int
        Длина предсказания.
    batch_size : int
        Размер батча.

    Возвращает
    ----------
    tuple[DataLoader, DataLoader, list[str]]
        Обучающий и валидационный DataLoader, а также список колонок-признаков.
    """
    if "OT" not in train_df.columns:
        raise ValueError("В датафрейме отсутствует колонка 'OT'.")

    feature_columns = [c for c in train_df.columns if c not in ("date", "OT")]
    train_dataset = ETTSequenceDataset(
            df=train_df,
            seq_len=seq_len,
            pred_len=pred_len,
            feature_columns=feature_columns,
            target_column="OT",
    )
    valid_dataset = ETTSequenceDataset(
            df=valid_df,
            seq_len=seq_len,
            pred_len=pred_len,
            feature_columns=feature_columns,
            target_column="OT",
    )

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
    )
    valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
    )
    return train_loader, valid_loader, feature_columns


def _train_one_model(
        train_loader: DataLoader,
        valid_loader: DataLoader,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        pred_len: int,
        learning_rate: float,
        num_epochs: int,
        device: torch.device,
) -> float:
    """
    Обучает одну LSTM-модель и возвращает MSE на валидации.

    Параметры
    ----------
    train_loader : DataLoader
        Dataloader с обучающими данными.
    valid_loader : DataLoader
        Dataloader с валидационными данными.
    input_size : int
        Размерность входного вектора признаков.
    hidden_size : int
        Размер скрытого состояния LSTM.
    num_layers : int
        Количество слоёв LSTM.
    pred_len : int
        Длина прогноза по времени.
    learning_rate : float
        Скорость обучения.
    num_epochs : int
        Количество эпох обучения.
    device : torch.device
        Устройство для вычислений.

    Возвращает
    ----------
    float
        MSE на валидационном наборе.
    """
    model = SimpleLSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            pred_len=pred_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds, axis=0).reshape(-1)
    y_true = np.concatenate(all_targets, axis=0).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse


def run_optuna_for_ettm1(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        num_epochs: int,
        n_trials: int,
) -> optuna.Study:
    """
    Запускает оптимизацию гиперпараметров LSTM-модели для ETTm1 с помощью Optuna.

    Параметры
    ----------
    train_df : pandas.DataFrame
        Обучающие данные.
    valid_df : pandas.DataFrame
        Валидационные данные.
    seq_len : int
        Длина входной последовательности.
    pred_len : int
        Длина прогноза.
    num_epochs : int
        Количество эпох обучения для каждой попытки.
    n_trials : int
        Число испытаний Optuna.

    Возвращает
    ----------
    optuna.Study
        Объект исследования Optuna с результатами оптимизации.
    """
    if n_trials < 1:
        raise ValueError("Параметр n_trials должен быть не меньше 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        """
        Целевая функция Optuna, обучающая одну модель с выбранными гиперпараметрами.

        Параметры
        ----------
        trial : optuna.Trial
            Объект, предоставляющий параметры для текущего испытания.

        Возвращает
        ----------
        float
            Значение функции потерь (MSE) на валидации.
        """
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        train_loader, valid_loader, feature_columns = _create_dataloaders_for_ettm1(
                train_df=train_df,
                valid_df=valid_df,
                seq_len=seq_len,
                pred_len=pred_len,
                batch_size=batch_size,
        )

        mse = _train_one_model(
                train_loader=train_loader,
                valid_loader=valid_loader,
                input_size=len(feature_columns),
                hidden_size=hidden_size,
                num_layers=num_layers,
                pred_len=pred_len,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                device=device,
        )
        return mse

    study = optuna.create_study(
            direction="minimize",
            study_name="ettm1_lstm_mse",
    )
    study.optimize(objective, n_trials=n_trials)
    return study