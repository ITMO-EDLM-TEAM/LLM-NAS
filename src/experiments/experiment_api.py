from __future__ import annotations

from typing import Final

from .datasets import load_ett_csv_dataset
from .informer_runner import run_informer_external
from .lstm_runner import run_lstm_on_etth_dataset
from .types import ExperimentResult
from .types import InformerRunnerConfig
from .types import LSTMHyperParams


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
        Результат эксперимента с метрикой MSE.
    """
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

    dataset_name_final: Final[str] = dataset_name
    result = run_lstm_on_etth_dataset(
            train_df=train_df,
            valid_df=valid_df,
            hyperparams=hyperparams,
            dataset_name=dataset_name_final,
            model_name=model_name,
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

    Параметры
    ----------
    dataset_name : str
        Имя датасета (например, 'ETTh1' или 'ETTh2').
    csv_path : str
        Полный путь к CSV-файлу с датасетом.
    max_rows : int
        Максимальное количество строк, которые загружаются из датасета (0 или меньше — без ограничения).
        Параметр добавлен для единообразия интерфейса, но на практике разбиение и выборка
        часто выполняются внутри скрипта Informer.
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
        Результат эксперимента с метриками Informer.
    """
    del max_rows  # Параметр оставлен для единообразия, но не используется здесь.
    del train_ratio

    args_list: list[str] = []
    if extra_args is not None:
        args_list = list(extra_args)

    informer_config = InformerRunnerConfig(
            dataset_csv_path=csv_path,
            metrics_json_path=metrics_json_path,
            informer_script_path=informer_script_path,
            extra_args=args_list,
            timeout_seconds=timeout_seconds,
    )

    dataset_name_final: Final[str] = dataset_name
    result = run_informer_external(
            config=informer_config,
            dataset_name=dataset_name_final,
            model_name=model_name,
    )
    return result