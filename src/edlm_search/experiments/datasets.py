from __future__ import annotations

import os

import pandas as pd


def load_ett_csv_dataset(csv_path: str, max_rows: int, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает датасет ETT-формата из CSV и разбивает его на обучающую и валидационную части.

    Параметры
    ----------
    csv_path : str
        Полный путь к CSV-файлу с датасетом (например, '/data/ETTm1.csv').
    max_rows : int
        Максимальное количество строк, которые нужно загрузить из файла.
        Если значение меньше или равно нулю, загружаются все строки.
    train_ratio : float
        Доля обучающей выборки в диапазоне (0, 1).

    Возвращает
    ----------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Пара (train_df, valid_df) с копиями данных.

    Исключения
    ----------
    FileNotFoundError
        Если CSV-файл не найден.
    ValueError
        Если train_ratio находится вне диапазона (0, 1) или после разбиения
        размер обучающей/валидационной выборки некорректен.
    """
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError('Параметр train_ratio должен быть в диапазоне (0, 1).')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Файл с датасетом не найден: {csv_path}')

    full_df = pd.read_csv(csv_path)
    if max_rows > 0:
        full_df = full_df.head(max_rows)

    train_size_float = float(len(full_df)) * float(train_ratio)
    train_size = int(train_size_float)
    if train_size <= 0 or train_size >= len(full_df):
        raise ValueError('После разбиения размер обучающей или валидационной выборки некорректен.')

    train_df = full_df.iloc[:train_size].copy()
    valid_df = full_df.iloc[train_size:].copy()
    return train_df, valid_df