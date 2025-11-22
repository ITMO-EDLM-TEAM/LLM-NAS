# edlm_search/src/edlm_search/baseline_optuna.py
from __future__ import annotations

import logging
import resource
import time
from typing import Final

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ..experiments.types import LSTMSearchSpace

logger = logging.getLogger(__name__)

try:
    from zeus.monitor import ZeusMonitor
except ImportError:  # pragma: no cover
    ZeusMonitor = None  # type: ignore[assignment]

_LAST_RUN_DIAGNOSTICS: dict[str, object] | None = None


def get_last_run_diagnostics() -> dict[str, object]:
    """
    Return diagnostics collected during the most recent LSTM training run.

    The dictionary may contain the following keys:
      * 'train_time_seconds' (float)
      * 'eval_time_seconds' (float)
      * 'total_energy_joules' (float)
      * 'peak_gpu_memory_mb' (float)
      * 'max_rss_mb' (float)
      * 'validation_predictions' (np.ndarray)
      * 'validation_targets' (np.ndarray)

    If training has not been run yet, an empty dict is returned.
    """
    if _LAST_RUN_DIAGNOSTICS is None:
        return {}
    return dict(_LAST_RUN_DIAGNOSTICS)


class ETTSequenceDataset(Dataset):
    """Dataset для последовательностного прогнозирования по семейству ETT-датасетов."""

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


def create_dataloaders_for_etth(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        batch_size: int,
        target_column: str,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Создаёт DataLoader'ы для обучения и валидации на ETT-датасетах произвольного варианта.

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
    target_column : str
        Имя колонки с целевой переменной (например, 'OT').

    Возвращает
    ----------
    tuple[DataLoader, DataLoader, list[str]]
        Обучающий и валидационный DataLoader, а также список колонок-признаков.
    """
    if target_column not in train_df.columns:
        raise ValueError(f"В обучающем датафрейме отсутствует колонка {target_column!r}.")
    if target_column not in valid_df.columns:
        raise ValueError(f"В валидационном датафрейме отсутствует колонка {target_column!r}.")

    feature_columns = [c for c in train_df.columns if c != "date"]

    if not feature_columns:
        raise ValueError("Список признаков пуст — невозможно построить датасет для обучения.")

    train_dataset = ETTSequenceDataset(
            df=train_df,
            seq_len=seq_len,
            pred_len=pred_len,
            feature_columns=feature_columns,
            target_column=target_column,
    )
    valid_dataset = ETTSequenceDataset(
            df=valid_df,
            seq_len=seq_len,
            pred_len=pred_len,
            feature_columns=feature_columns,
            target_column=target_column,
    )

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)

    if train_samples < batch_size:
        logger.warning(
                f'Размер обучающего набора ({train_samples}) меньше batch_size ({batch_size}). '
                f'Будет использован один неполный батч.'
        )
    if valid_samples < batch_size:
        logger.warning(
                f'Размер валидационного набора ({valid_samples}) меньше batch_size ({batch_size}). '
                f'Будет использован один неполный батч.'
        )

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
    )
    valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
    )
    return train_loader, valid_loader, feature_columns


def _extract_history_frame_for_autoregression(
        train_loader: DataLoader,
        seq_len: int,
        feature_columns: list[str],
        target_column: str,
) -> pd.DataFrame | None:
    """
    Сформировать контекст из обучающей выборки для авторегрессионного прогноза на валидации.
    """
    dataset = getattr(train_loader, 'dataset', None)
    if not isinstance(dataset, ETTSequenceDataset):
        return None

    if dataset._feature_columns != feature_columns:
        raise ValueError(
                'Наборы признаков обучающего и валидационного датасетов различаются, '
                'невозможно выполнить авторегрессионную оценку.'
        )
    if dataset._target_column != target_column:
        raise ValueError(
                'Целевые колонки обучающего и валидационного датасетов различаются, '
                'невозможно выполнить авторегрессионную оценку.'
        )

    df_train = dataset._df
    if len(df_train) < seq_len:
        raise ValueError(
                'Обучающий датасет слишком короткий для формирования авторегрессионного контекста.'
        )
    history_frame = df_train.iloc[-seq_len:].copy()
    return history_frame


def _build_autoregressive_predictions(
        model: SimpleLSTMForecaster,
        data_frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        seq_len: int,
        device: torch.device,
        history_frame: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Построить авторегрессионные предсказания по одному шагу вперёд для указанного датафрейма.

    Если передан history_frame, он используется как «тёплый старт» и обеспечивает seq_len
    последних известных наблюдений перед началом прогнозирования. Все последующие шаги
    выполняются строго на ранее предсказанных значениях целевой переменной.
    """
    if seq_len <= 0:
        raise ValueError("seq_len должен быть положительным целым числом для авторегрессии.")
    if target_column not in data_frame.columns:
        raise ValueError(
                f'В датафрейме отсутствует колонка таргета {target_column!r} для авторегрессии.'
        )
    if target_column not in feature_columns:
        raise ValueError(
                f'Колонка таргета {target_column!r} должна присутствовать в feature_columns '
                f'для авторегрессии.'
        )

    history_rows = 0
    frames_to_concat: list[pd.DataFrame] = []
    if history_frame is not None:
        required_columns = set(feature_columns)
        required_columns.add(target_column)
        missing_history_columns = [c for c in required_columns if c not in history_frame.columns]
        if missing_history_columns:
            raise ValueError(
                    'history_frame не содержит необходимые колонки: '
                    f'{sorted(missing_history_columns)}.'
            )
        frames_to_concat.append(history_frame.reset_index(drop=True))
        history_rows = len(history_frame)

    frames_to_concat.append(data_frame.reset_index(drop=True))
    combined_df = pd.concat(frames_to_concat, axis=0, ignore_index=True)

    combined_values = combined_df[feature_columns].to_numpy(dtype=np.float32)
    combined_targets = combined_df[target_column].to_numpy(dtype=np.float32)

    total_rows = int(combined_values.shape[0])
    if total_rows <= seq_len:
        raise ValueError(
                f'Для авторегрессии требуется больше строк, чем seq_len={seq_len}. '
                f'Текущий размер датафрейма: {total_rows}.'
        )

    target_feature_index = feature_columns.index(target_column)

    prediction_start_index = max(seq_len, history_rows)
    if prediction_start_index >= total_rows:
        raise ValueError(
                'Недостаточно наблюдений после контекста для авторегрессионного прогноза.'
        )

    context_buffer = combined_values[:seq_len].copy()
    for warm_index in range(seq_len, prediction_start_index):
        context_buffer[:-1] = context_buffer[1:]
        context_buffer[-1] = combined_values[warm_index]

    y_true_eval: list[float] = []
    y_pred_eval: list[float] = []

    model.eval()
    for current_index in range(prediction_start_index, total_rows):
        input_tensor = torch.from_numpy(context_buffer).unsqueeze(0).to(device)
        with torch.no_grad():
            model_output = model(input_tensor)

        step_prediction_tensor = model_output[0, 0]
        step_prediction = float(step_prediction_tensor.detach().cpu().item())
        y_true_value = float(combined_targets[current_index])

        if current_index >= history_rows:
            y_true_eval.append(y_true_value)
            y_pred_eval.append(step_prediction)

        context_buffer[:-1] = context_buffer[1:]
        next_row_features = combined_values[current_index].copy()
        next_row_features[target_feature_index] = step_prediction
        context_buffer[-1] = next_row_features

    if not y_true_eval:
        raise RuntimeError('Авторегрессионная оценка не вернула ни одного предсказания.')

    if history_rows >= seq_len and len(y_true_eval) != len(data_frame):
        raise RuntimeError(
                'Авторегрессионная оценка не покрыла все строки целевого датафрейма.'
        )

    y_true = np.asarray(y_true_eval, dtype=np.float32)
    y_pred = np.asarray(y_pred_eval, dtype=np.float32)
    return y_true, y_pred


def _evaluate_on_loader(
        model: SimpleLSTMForecaster,
        valid_loader: DataLoader,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate LSTM model on validation loader in teacher-forced mode.

    This function flattens all predictions and targets into one-dimensional arrays.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x_device = batch_x.to(device)
            outputs = model(batch_x_device)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())

    if not all_preds or not all_targets:
        raise RuntimeError('Validation loader did not produce any batches.')

    y_pred_eval = np.concatenate(all_preds, axis=0).reshape(-1)
    y_true_eval = np.concatenate(all_targets, axis=0).reshape(-1)
    return y_true_eval, y_pred_eval


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
    if num_epochs < 1:
        raise ValueError("Число эпох должно быть положительным.")
    if learning_rate <= 0.0:
        raise ValueError("Скорость обучения должна быть положительной.")

    global _LAST_RUN_DIAGNOSTICS

    model = SimpleLSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            pred_len=pred_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gpu_monitor = None
    if ZeusMonitor is not None and device.type == 'cuda' and torch.cuda.is_available():
        try:
            current_device_index = torch.cuda.current_device()
            gpu_monitor = ZeusMonitor(gpu_indices=[current_device_index])
            gpu_monitor.begin_window('lstm_train_eval')
        except Exception as exc:
            logger.warning(
                    f'ZeusMonitor could not be initialized, GPU energy metrics will be skipped: {exc}'
            )
            gpu_monitor = None

    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    train_start = time.perf_counter()
    for epoch_index in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        batch_count = 0
        for batch_x, batch_y in train_loader:
            batch_x_device = batch_x.to(device)
            batch_y_device = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x_device)
            loss = criterion(preds, batch_y_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss_sum += float(loss.item())
            batch_count += 1

        if batch_count > 0:
            avg_loss = epoch_loss_sum / float(batch_count)
            logger.info(
                    f'LSTM training epoch finished: epoch={epoch_index + 1}/{num_epochs}, '
                    f'avg_loss={avg_loss:.6f}.'
            )

    train_end = time.perf_counter()

    eval_start = time.perf_counter()

    y_true_eval, y_pred_eval = _evaluate_on_loader(
            model=model,
            valid_loader=valid_loader,
            device=device,
    )

    eval_end = time.perf_counter()

    energy_joules = 0.0
    if gpu_monitor is not None:
        try:
            measurement = gpu_monitor.end_window('lstm_train_eval')
            energy_joules = float(measurement.total_energy)
        except Exception as exc:
            logger.warning(
                    f'ZeusMonitor failed during measurement, GPU energy metric will be skipped: {exc}'
            )
            energy_joules = 0.0

    mse = float(np.mean((y_true_eval - y_pred_eval) ** 2))

    peak_gpu_memory_mb = 0.0
    if device.type == 'cuda' and torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_gpu_memory_mb = float(peak_bytes) / (1024.0 * 1024.0)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_mb = float(usage.ru_maxrss) / 1024.0

    train_time_seconds = float(train_end - train_start)
    eval_time_seconds = float(eval_end - eval_start)

    logger.info(
            f'LSTM training finished: train_time={train_time_seconds:.3f}s, '
            f'eval_time={eval_time_seconds:.3f}s, mse={mse:.6f}'
    )

    _LAST_RUN_DIAGNOSTICS = {
        'train_time_seconds': train_time_seconds,
        'eval_time_seconds': eval_time_seconds,
        'total_energy_joules': energy_joules,
        'peak_gpu_memory_mb': peak_gpu_memory_mb,
        'max_rss_mb': max_rss_mb,
        'validation_predictions': y_pred_eval,
        'validation_targets': y_true_eval,
    }

    return mse


def _validate_lstm_search_space(search_space: LSTMSearchSpace) -> None:
    """
    Validate LSTMSearchSpace instance before passing it to Optuna.
    """
    if not search_space.seq_len_values:
        raise ValueError('LSTM search space "seq_len_values" must not be empty.')
    for value in search_space.seq_len_values:
        if value <= 0:
            raise ValueError('All values in "seq_len_values" must be positive integers.')

    if not search_space.pred_len_values:
        raise ValueError('LSTM search space "pred_len_values" must not be empty.')
    for value in search_space.pred_len_values:
        if value <= 0:
            raise ValueError('All values in "pred_len_values" must be positive integers.')

    if not search_space.num_epochs_values:
        raise ValueError('LSTM search space "num_epochs_values" must not be empty.')
    for value in search_space.num_epochs_values:
        if value <= 0:
            raise ValueError('All values in "num_epochs_values" must be positive integers.')

    if not search_space.hidden_size_values:
        raise ValueError('LSTM search space "hidden_size_values" must not be empty.')
    for value in search_space.hidden_size_values:
        if value <= 0:
            raise ValueError('All values in "hidden_size_values" must be positive integers.')

    if not search_space.num_layers_values:
        raise ValueError('LSTM search space "num_layers_values" must not be empty.')
    for value in search_space.num_layers_values:
        if value <= 0:
            raise ValueError('All values in "num_layers_values" must be positive integers.')

    if not search_space.learning_rate_values:
        raise ValueError('LSTM search space "learning_rate_values" must not be empty.')
    for value in search_space.learning_rate_values:
        if value <= 0.0:
            raise ValueError('All values in "learning_rate_values" must be positive.')

    if not search_space.batch_size_values:
        raise ValueError('LSTM search space "batch_size_values" must not be empty.')
    for value in search_space.batch_size_values:
        if value <= 0:
            raise ValueError('All values in "batch_size_values" must be positive integers.')


def run_optuna_for_etth(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        search_space: LSTMSearchSpace,
        n_trials: int,
        target_column: str,
        device: torch.device,
) -> optuna.Study:
    """
    Запускает оптимизацию гиперпараметров LSTM-модели для ETT-датасетов с помощью Optuna.

    Параметры
    ----------
    train_df : pandas.DataFrame
        Обучающие данные.
    valid_df : pandas.DataFrame
        Валидационные данные.
    search_space : LSTMSearchSpace
        Search space for LSTM hyperparameters.
    n_trials : int
        Число испытаний Optuna.
    target_column : str
        Имя колонки целевой переменной (например, 'OT').
    device : torch.device
        Устройство, используемое для обучения и оценки модели.

    Возвращает
    ----------
    optuna.Study
        Объект исследования Optuna с результатами оптимизации.
    """
    if n_trials < 1:
        raise ValueError("Параметр n_trials должен быть не меньше 1.")

    _validate_lstm_search_space(search_space)

    logger.info(
            f'Optuna LSTM search will use device "{device.type}" '
            f'and {n_trials} trials.'
    )
    logger.info(
            f'LSTM search space: seq_len={search_space.seq_len_values}, '
            f'pred_len={search_space.pred_len_values}, '
            f'num_epochs={search_space.num_epochs_values}, '
            f'hidden_size={search_space.hidden_size_values}, '
            f'num_layers={search_space.num_layers_values}, '
            f'learning_rate={search_space.learning_rate_values}, '
            f'batch_size={search_space.batch_size_values}.'
    )

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
        seq_len = int(trial.suggest_categorical("seq_len", search_space.seq_len_values))
        pred_len = int(trial.suggest_categorical("pred_len", search_space.pred_len_values))
        num_epochs = int(
                trial.suggest_categorical("num_epochs", search_space.num_epochs_values)
        )
        hidden_size = int(
                trial.suggest_categorical("hidden_size", search_space.hidden_size_values)
        )
        num_layers = int(
                trial.suggest_categorical("num_layers", search_space.num_layers_values)
        )
        learning_rate = float(
                trial.suggest_categorical("learning_rate", search_space.learning_rate_values)
        )
        batch_size = int(
                trial.suggest_categorical("batch_size", search_space.batch_size_values)
        )

        train_loader, valid_loader, feature_columns = create_dataloaders_for_etth(
                train_df=train_df,
                valid_df=valid_df,
                seq_len=seq_len,
                pred_len=pred_len,
                batch_size=batch_size,
                target_column=target_column,
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
            study_name="etth_lstm_mse",
    )
    study.optimize(objective, n_trials=n_trials)
    return study