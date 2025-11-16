# Overview

Проект решает задачу автоматического поиска архитектур моделей для прогноза временных рядов (датасеты семейства ETT:
ETTm1, ETTh1, ETTh2) с целевой метрикой MSE.

Сравниваются три подхода:

- LSTM + Optuna (baseline).
- Оригинальный Informer (внешний репозиторий Informer2020).
- Архитектуры, сгенерированные LLM-агентом под заданный контракт (main.py + конфиги), исполняемые в отдельном процессе.

## Что на входе и выходе

**Вход:**

- CSV-файлы ETT (`ETTm1.csv`, `ETTh1.csv`, `ETTh2.csv`) из внешнего репозитория ETDataset.
- Внешний код Informer2020, лежащий в `src/Informer2020`.
- Запущенный LLM endpoint (LM Studio или DeepSeek), сконфигурированный через переменные окружения.

**Выход:**

- Метрики MSE для:
    - LSTM + Optuna baseline.
    - Оригинального Informer по каждому датасету.
    - Лучшего кандидата из LLM-поиска.
- Графики сравнения MSE и динамики качества по кандидатам LLM.
- Таблица со всеми кандидатами LLM (pandas.DataFrame).

## Основной поток выполнения

Точка входа для демонстрации — `main.ipynb` в корне проекта.

### 1. Загрузка данных

Модуль: `experiments/datasets.py`

Функция:

- `load_ett_csv_dataset(csv_path, max_rows, train_ratio)`  
  Загружает ETT CSV, обрезает по `max_rows`, делит на train/validation по `train_ratio`.

В ноутбуке:

- Для `ETTm1`, `ETTh1`, `ETTh2` вызывается `load_ett_csv_dataset`.
- Результат: словари `train_dfs` и `valid_dfs` с pandas.DataFrame для каждого датасета.

### 2. LSTM + Optuna baseline (ETTm1)

Модуль: `edlm_search/baseline_optuna.py`

Основные компоненты:

- `ETTSequenceDataset` — Dataset для нарезки временных окон (контекст + target).
- `SimpleLSTMForecaster` — простая LSTM-модель с линейной головой для мультишагового прогноза.
- `_create_dataloaders_for_ettm1` — подготовка DataLoader’ов.
- `_train_one_model` — обучение одной LSTM и вычисление MSE.
- `run_optuna_for_ettm1` — циклы Optuna: подбирает `hidden_size`, `num_layers`, `learning_rate`, `batch_size`.

В ноутбуке:

- На `ETTm1` вызывается `run_optuna_for_ettm1(...)`.
- Результат: `optuna_study` и `optuna_best_mse`.

### 3. Запуск Informer (внешний проект)

Модули:

- `src/Informer2020/main_informer.py` — оригинальный тренинг-код Informer (внешний).
- `src/Informer2020/informer_experiment_wrapper.py` — обёртка над `main_informer.py`.
- `experiments/informer_runner.py` — внешний раннер Informer в виде функции `run_informer_external`.
- `experiments/experiment_api.py` — удобный API `run_informer_ettm1_experiment`.

Общий сценарий:

1. Ноутбук вызывает `run_informer_ettm1_experiment(...)` для каждого датасета (`ETTm1`, `ETTh1`, `ETTh2`).
2. `run_informer_ettm1_experiment` собирает `InformerRunnerConfig` и вызывает `run_informer_external`.
3. `run_informer_external` запускает `informer_experiment_wrapper.py` через `subprocess`, передавая:
    - `--data-path` (путь к CSV).
    - `--metrics-path` (куда сохранить JSON с метриками).
    - дополнительные аргументы (например, `--epochs`, `--data`).
4. `informer_experiment_wrapper.py`:
    - парсит аргументы,
    - вызывает `main_informer.py` с нужными параметрами,
    - находит последний `metrics.npy` в `Informer2020/results`,
    - конвертирует его в JSON с метриками и сохраняет.
5. `run_informer_external` читает JSON, возвращает `ExperimentResult`.

В ноутбуке:

- Результаты складываются в словарь `informer_results` по ключам `ETTm1`, `ETTh1`, `ETTh2`.

### 4. LLM-поиск архитектур

Модули:

- `edlm_search/problem.py` — загрузка постановки задачи из `examples/et/statement.md`.
- `edlm_search/llm_clients.py` — клиент к LLM (LM Studio / DeepSeek).
- `edlm_search/llm_pipeline.py` — работа с шаблонами Jinja и парсинг XML-подобного ответа.
- `edlm_search/sampler.py` — генерация кандидатов (новый и crossover).
- `edlm_search/runner.py` — безопасный запуск кода кандидата в отдельном процессе.
- `edlm_search/ett_evaluator.py` — исполнение кандидата и вычисление метрики.
- `edlm_search/search_loop.py` — главный цикл LLM-поиска.
- `edlm_search/candidates_database.py` — in-memory база результатов.

Поток:

1. **Загрузка постановки задачи**:  
   `Problem.from_directory("examples/et")` читает `statement.md`.

2. **Создание клиента LLM**:
    - Если `LLM_PROVIDER=lmstudio`, используется `LMStudioClient`.
    - Если `LLM_PROVIDER=deepseek`, используется `DeepSeekClient` (нужен API-ключ в переменной окружения).
    - Метод `create_pipeline` возвращает `LLMPipeline`.

3. **Sampler** (`CandidateSampler`):
    - `create_initial_candidate()` → вызывает `LLMPipeline.generate_files_from_template('new_candidate', ...)`.
    - LLM возвращает `<idea>...</idea>` и `<files><file path="...">...</file>...</files>`.
    - Parser создаёт объект `Candidate` с полями `idea` и `files`.

4. **Runner** (`UnsafeRunner`):
    - Создаёт временный каталог.
    - Сохраняет файлы кандидата (`main.py`, `model_config.json`, `training_args.json`).
    - Сохраняет `train.parquet` и `validation.parquet` из исходных DataFrame.
    - Запускает новый Python-процесс с `python -m edlm_search.runner --temp-dir ... --run-args ...`.
    - В дочернем процессе `_execute_candidate_script`:
        - импортирует `main.py`,
        - вызывает его функцию `main(train_data_path, valid_data_path, num_epochs)` как генератор,
        - по окончании каждой эпохи отправляет предсказания в родительский процесс через JSON,
        - при наличии GPU измеряет энергопотребление через `ZeusMonitor`.

5. **Evaluator** (`ETTM1Evaluator`):
    - Асинхронно читает события от `UnsafeRunner` (предсказания, энергию).
    - Использует последнюю последовательность предсказаний по валидации.
    - Сравнивает с истинными `OT` по валидационному ETTm1 (MSE).
    - Возвращает словарь метрик: `{"mse": ..., "total_energy_joules": ...}`.

6. **Search loop** (`LLMBasedArchitectureSearch`):
    - Создаёт первый кандидат через `sampler.create_initial_candidate()`, оценивает его и сохраняет в
      `CandidateDatabase`.
    - Затем цикл:
        - Берёт лучших кандидатов по `metric_name` из базы,
        - Выбирает двух родителей,
        - `sampler.crossover_candidates(...)` генерирует нового кандидата через шаблон `crossover_candidate.md.jinja`,
        - оценивает его через `evaluator`,
        - добавляет результат в базу.
    - Работает до достижения `max_candidates`.

Результат LLM-поиска:

- `CandidateDatabase` со всеми кандидатами, их идеями и метриками.
- Лучший кандидат по MSE выбирается через `top_k_by_metric("mse", k=1)`.

### 5. Сравнение и визуализация

В `main.ipynb`:

- Собирается таблица `comparison_df` с MSE для:
    - `lstm_optuna_best` (ETTm1),
    - `informer_original` (каждый датасет),
    - `llm_search_best_candidate` (ETTm1).
- Строятся:
    - bar-plot сравнения MSE между подходами,
    - график динамики MSE по номерам кандидатов в LLM-поиске.

## Как запустить

Минимальный сценарий:

1. Установить зависимости проекта (PyTorch, Optuna, OpenAI-клиент, Zeus, Jupyter, и т.д.).
2. Склонировать/подложить внешние репозитории:
    - ETDataset с файлами `ETTm1.csv`, `ETTh1.csv`, `ETTh2.csv` в каталог `./ETDataset`.
    - Informer2020 в `./src/Informer2020`.
3. Настроить LLM-endpoint:
    - Запустить LM Studio или другой OpenAI-совместимый сервер.
    - Прописать переменные окружения (см. `.example.env`).
4. Запустить Jupyter из корня проекта:
    - `jupyter lab` или `jupyter notebook`.
5. Открыть `main.ipynb` и выполнить все ячейки по порядку.

После выполнения будет сформирована таблица и графики для сравнения всех подходов по MSE.