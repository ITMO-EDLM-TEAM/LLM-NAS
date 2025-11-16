# Architecture

## Общая структура

- `src/`
    - `Informer2020/` — внешний репозиторий Informer (оригинальная модель и тренинг-код).
    - `ETDataset/` — внешние CSV-файлы ETT.
    - `experiments/` — высокоуровневые функции запуска экспериментов (LSTM и Informer).
    - `edlm_search/` — ядро LLM-поиска архитектур.
- `examples/et/` — постановка задачи (statement.md).
- `main.ipynb` — оркестрация всех экспериментов и визуализации.

Ниже подробно описаны только `experiments` и `edlm_search`.

---

## Папка `experiments`

### `experiments/__init__.py`

Экспортирует основные функции и тип:

- `run_lstm_ettm1_experiment`
- `run_informer_ettm1_experiment`
- `ExperimentResult`

Используется в `main.ipynb` для удобного импорта.

---

### `experiments/datasets.py`

Функция:

- `load_ett_csv_dataset(csv_path: str, max_rows: int, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]`  
  Назначение:
    - Проверить наличие CSV.
    - Загрузить датасет.
    - Обрезать по `max_rows` (если > 0).
    - Разделить на train/validation по `train_ratio`.
    - Вернуть копии DataFrame.

Используется:

- В `main.ipynb`.
- Может использоваться в других скриптах для единообразного разбиения ETT-данных.

---

### `experiments/types.py`

Dataclasses:

- `DatasetConfig`  
  Описание конфигурации датасета (имя, корень, файл, max_rows, train_ratio). В текущем коде не используется.
- `LSTMHyperParams`  
  Гиперпараметры LSTM-модели и обучения:
    - `seq_len`, `pred_len`, `hidden_size`, `num_layers`, `learning_rate`, `batch_size`, `num_epochs`.
- `InformerRunnerConfig`  
  Конфигурация запуска внешнего скрипта Informer:
    - путь к CSV, путь к JSON с метриками, путь к скрипту, список extra args, таймаут.
- `ExperimentResult`  
  Унифицированный результат эксперимента:
    - `model_name`, `dataset_name`, `metrics` (dict[str, float]), `extra_info` (dict[str, Any]).

---

### `experiments/lstm_runner.py`

Функция:

-
`run_lstm_on_ettm1(train_df, valid_df, hyperparams: LSTMHyperParams, dataset_name: str, model_name: str) -> ExperimentResult`

Назначение:

- Создаёт DataLoader’ы для ETTm1 через `_create_dataloaders_for_ettm1` из `edlm_search.baseline_optuna`.
- Определяет device (CPU/GPU).
- Обучает одну LSTM через `_train_one_model`.
- Оборачивает результат в `ExperimentResult`:
    - `metrics = {'mse': ...}`
    - `extra_info` — значения гиперпараметров.

Используется:

- В `experiments/experiment_api.py` через API-обёртку.

---

### `experiments/informer_runner.py`

Функция:

- `run_informer_external(config: InformerRunnerConfig, dataset_name: str, model_name: str) -> ExperimentResult`

Назначение:

- Проверить наличие скрипта Informer и CSV.
- Создать директорию под JSON с метриками, если её нет.
- Собрать команду вида:
    - `python informer_experiment_wrapper.py --data-path <csv> --metrics-path <json> [extra_args...]`
- Запустить команду через `subprocess.run` с таймаутом.
- Проверить код возврата и наличие JSON с метриками.
- Прочитать JSON, привести значения к float, поместить в `metrics`.
- Вернуть `ExperimentResult` с:
    - `model_name` — имя модели (например, `'informer-original'`),
    - `dataset_name` — имя датасета,
    - `metrics` — метрики Informer,
    - `extra_info` — например, `timeout_seconds`.

Используется:

- В `experiments/experiment_api.py`.

---

### `experiments/experiment_api.py`

Высокоуровневый API для запуска экспериментов.

Функции:

1. `run_lstm_ettm1_experiment(...) -> ExperimentResult`
    - Загружает ETT CSV через `load_ett_csv_dataset`.
    - Формирует `LSTMHyperParams`.
    - Вызывает `run_lstm_on_ettm1`.
    - Возвращает `ExperimentResult`.

2. `run_informer_ettm1_experiment(...) -> ExperimentResult`
    - Принимает путь к CSV и настройку внешнего скрипта Informer.
    - Формирует `InformerRunnerConfig` (dataset_csv_path, metrics_json_path, informer_script_path, extra_args,
      timeout_seconds).
    - Вызывает `run_informer_external`.
    - Возвращает `ExperimentResult`.

Используется:

- В `main.ipynb` для удобного запуска LSTM и Informer экспериментов.

---

## Папка `edlm_search`

### `edlm_search/__init__.py`

Пустой `__init__.py` делает пакет импортируемым (используется в импортах по всему проекту).

---

### `edlm_search/baseline_optuna.py`

Содержит baseline LSTM и поиск гиперпараметров Optuna.

Классы и функции:

- `ETTSequenceDataset(Dataset)`
    - Нарезает временной ряд из DataFrame в пары `(x, y)`:
        - `x` — окно длиной `seq_len` по признакам,
        - `y` — следующие `pred_len` значений целевой колонки.
    - Контролирует корректность размеров.

- `SimpleLSTMForecaster(nn.Module)`
    - LSTM с заданными `input_size`, `hidden_size`, `num_layers`.
    - Линейная голова на последнем скрытом состоянии для предсказания `pred_len` шагов.

- `_create_dataloaders_for_ettm1(train_df, valid_df, seq_len, pred_len, batch_size)`
    - Формирует feature-колонки (все кроме `date` и `OT`).
    - Создаёт `ETTSequenceDataset` для train и valid.
    - Возвращает `(train_loader, valid_loader, feature_columns)`.

- `_train_one_model(...) -> float`
    - Создаёт `SimpleLSTMForecaster`.
    - Обучает заданное число эпох.
    - Вычисляет MSE на валидации и возвращает его.

- `run_optuna_for_ettm1(train_df, valid_df, seq_len, pred_len, num_epochs, n_trials) -> optuna.Study`
    - Настраивает устройство (CPU/GPU).
    - Определяет `objective(trial)`, в котором:
        - семплируются `hidden_size`, `num_layers`, `learning_rate`, `batch_size`,
        - создаются DataLoader’ы,
        - обучается одна модель и считается MSE.
    - Запускает Optuna `study.optimize`.
    - Возвращает объект `Study`.

---

### `edlm_search/candidate.py`

Класс:

- `Candidate`
    - Связывает:
        - `files: dict[str, str]` — путь → содержимое файла (например, main.py, model_config.json, training_args.json).
        - `idea: str` — краткое текстовое описание идеи модели.

Методы:

- `@classmethod async new_from_problem(cls, problem: Problem, llm_pipeline: LLMPipeline)`
    - Использует `llm_pipeline.generate_files_from_template('new_candidate', problem=problem)`.
    - Получает `idea` и `files` от LLM.
    - Возвращает новый `Candidate`.

Используется:

- В `CandidateSampler` и в поисковом цикле как контейнер для решений.

---

### `edlm_search/candidates_database.py`

Dataclass:

- `CandidateRecord`
    - `candidate_id: str`
    - `candidate: Candidate`
    - `idea: str`
    - `metrics: dict[str, float]`
    - `backend_name: str`
    - `created_at: datetime`

Класс:

- `CandidateDatabase`  
  Методы:
    - `add_result(candidate, metrics, backend_name) -> CandidateRecord`  
      Добавляет запись, присваивает `candidate_id` вида `cand-<n>`.
    - `list_records() -> list[CandidateRecord]`  
      Возвращает копию списка всех записей.
    - `top_k_by_metric(metric_name, k) -> list[CandidateRecord]`  
      Фильтрует по наличию метрики, сортирует по возрастанию, возвращает top-k.
    - `to_dataframe() -> pd.DataFrame`  
      Строит таблицу с основными полями и всеми метриками.
    - `__len__()` — количество записей.
    - `iter_records()` — итератор по записям.

Используется:

- В `LLMBasedArchitectureSearch` для хранения результатов и выбора родителей для crossover.

---

### `edlm_search/ett_evaluator.py`

Класс:

- `ETTM1Evaluator`

Назначение:

- Выполняет оценку одного кандидата на ETTm1 через `UnsafeRunner`.

Основные шаги:

1. В конструкторе:
    - проверяет, что `train_df` и `valid_df` не пусты,
    - проверяет наличие target-колонки,
    - сохраняет целевой вектор по валидации.
2. Метод `async evaluate(runner: UnsafeRunner, candidate: Candidate) -> dict[str, float]`:
    - запускает `runner.run(candidate, train_df, valid_df, run_args={'num_epochs': ...})`.
    - читает события:
        - `epoch_result` — массив предсказаний (используется последняя итерация),
        - `zeus` — энергооценка (если есть).
    - выравнивает длины предсказаний и таргетов.
    - считает MSE.
    - возвращает словарь с метриками:
        - основная: `metric_name` (например, `"mse"`),
        - дополнительно: `'total_energy_joules'` при наличии.

Используется:

- В `main.ipynb` и `LLMBasedArchitectureSearch`.

---

### `edlm_search/evaluator.py`

Протокол:

- `Evaluator`  
  Описывает интерфейс:
    - `async def evaluate(self, runner: UnsafeRunner, candidate: Candidate) -> dict[str, float]`

Используется:

- В `LLMBasedArchitectureSearch` для абстракции от конкретного типа оценщика.

---

### `edlm_search/llm_clients.py`

Классы:

- `DeepSeekClient`
    - Оборачивает `AsyncOpenAI` с заданным `base_url`, `api_key`, `model_name`.
    - Метод `create_pipeline()` возвращает `LLMPipeline`.
- `LMStudioClient`
    - Тоже обёртка над `AsyncOpenAI`, но без реального API-ключа (LM Studio обычно работает локально).
    - Метод `create_pipeline()` возвращает `LLMPipeline`.

Используются:

- В `main.ipynb` для создания LLM-пайплайна.
- Абстрагируют конфигурацию LLM-провайдера и API.

---

### `edlm_search/llm_pipeline.py`

Исключение:

- `ModelOutputParseError` — выбрасывается при некорректном формате ответа модели.

Класс:

- `LLMPipeline`  
  Поля:
    - `_async_openai: AsyncOpenAI` — клиент.
    - `_model_name: str` — имя модели.

Методы:

- `_parse_xml_files(xml_string: str) -> dict[str, str]`
    - Парсит XML-подобный блок `<files>...<file path="...">...</file>...</files>`.
    - Возвращает словарь `path -> content`.
- `async generate_files_from_template(template_name: str, **kwargs) -> tuple[str, dict[str, str]]`
    - Загружает Jinja2-шаблон (из `prompts/`).
    - Рендерит prompt с контекстом (problem, родительские решения и т.п.).
    - Вызывает `chat.completions.create` у LLM.
    - Извлекает `<idea>...</idea>` и `<files>...</files>` из ответа.
    - Парсит файлы `_parse_xml_files`.
    - Возвращает `(idea, files)`.

Используется:

- В `Candidate.new_from_problem`.
- В `CandidateSampler.crossover_candidates`.

---

### `edlm_search/problem.py`

Класс:

- `Problem`
    - Поле `statement: str` — текст постановки задачи.

Методы:

- `__init__(statement: str)`
- `@classmethod from_directory(cls, dir: str)`
    - Читает файл `statement.md` в указанной директории.
    - Создаёт `Problem`.

Используется:

- В `main.ipynb` и в LLM-пайплайне как контекст задачи.

---

### `edlm_search/prompts/new_candidate.md.jinja`

Шаблон prompt’а для генерации **нового кандидата**.

Содержит:

- Формулировку задачи.
- Строгий контракт на файлы, которые должен сгенерировать LLM:
    - `main.py` (Dataset, Model, Trainer, main(...)).
    - `model_config.json`.
    - `training_args.json`.
- Требуемый XML-подобный формат ответа:
    - `<idea>...</idea>` и `<files><file path="...">...</file>...</files>`.

Используется:

- В `LLMPipeline.generate_files_from_template('new_candidate', ...)`.

---

### `edlm_search/prompts/crossover_candidate.md.jinja`

Шаблон prompt’а для генерации **нового кандидата на основе двух родителей**.

Содержит:

- Описание задачи.
- Идеи и метрики родителей.
- Исходные файлы родителей.
- Требование сделать улучшенный pipeline.
- Тот же контракт по файлам и XML-подобному формату, что и `new_candidate`.

Используется:

- В `LLMPipeline.generate_files_from_template('crossover_candidate', ...)`.
- Вызывается из `CandidateSampler.crossover_candidates`.

---

### `edlm_search/runner.py`

Исключение:

- `RunnerOutputParseError` — ошибки парсинга или некорректный выход дочернего процесса.

Протокол:

- `Runner` — описывает интерфейс:
    - `async run(candidate, train_df, validation_df, run_args)` — асинхронный генератор событий.
    - `stop()` — остановка процесса.

Класс:

- `UnsafeRunner`  
  Назначение:
    - Запустить код кандидата в отдельном Python-процессе, передав ему:
        - файлы кандидата,
        - данные train/validation в Parquet,
        - аргументы запуска.
    - Читать stdout и служебный канал JSON-событий.

Основные шаги:

1. Создаёт временную директорию, записывает туда:
    - файлы кандидата,
    - `train.parquet`, `validation.parquet`.
2. Формирует `run_args` (например, `train_data_path`, `valid_data_path`, `num_epochs`).
3. Создаёт pipe для служебных сообщений (`COMM_PIPE_FD`).
4. Стартует subprocess:
    - `python -m edlm_search.runner --temp-dir <tmp> --run-args <json>`.
5. Внутренний цикл:
    - читает stdout построчно,
    - читает служебные JSON-сообщения (`start`, `epoch_result`, `zeus`, `exception`),
    - при `epoch_result` -> `yield {'timestamp': ..., 'epoch_result': predictions}`,
    - при `zeus` -> `yield {'total_energy_joules': ...}`.

Метод:

- `stop()` — посылает `terminate()` дочернему процессу.

Внутренняя функция:

- `_execute_candidate_script(temp_dir: str, run_args: dict)`  
  Выполняется в дочернем процессе:
    - открывает pipe, определяет `send_event(...)`,
    - меняет `cwd` на temp_dir, добавляет его в `sys.path`,
    - импортирует `main.py` и выполняет его `main(**run_args)` как генератор,
    - после каждой эпохи вызывает `send_event('epoch_result', {...})`,
    - при наличии GPU измеряет энергию через `ZeusMonitor`,
    - при исключении отправляет сериализованное исключение в родителя и завершает процесс.

Блок `if __name__ == '__main__':`

- Парсит аргументы `--temp-dir`, `--run-args`.
- Десериализует `run_args`.
- Вызывает `_execute_candidate_script`.

---

### `edlm_search/sampler.py`

Класс:

- `CandidateSampler`  
  Поля:
    - `llm_pipeline: LLMPipeline`
    - `problem: Problem`

Методы:

- `async create_initial_candidate() -> Candidate`
    - Делегирует в `Candidate.new_from_problem(...)`.
- `async crossover_candidates(parent_a: CandidateRecord, parent_b: CandidateRecord) -> Candidate`
    - Вызывает `llm_pipeline.generate_files_from_template("crossover_candidate", ...)`.
    - Передаёт идеи, метрики и файлы родителей.
    - Возвращает новый `Candidate`.
-
`select_parents_for_crossover(database: CandidateDatabase, metric_name: str, top_k: int) -> tuple[CandidateRecord, CandidateRecord]`
    - Проверяет, что в базе минимум два кандидата.
    - Берёт `best_candidates = database.top_k_by_metric(metric_name, k=top_k)`.
    - Возвращает первых двух как родителей.

---

### `edlm_search/search_loop.py`

Класс:

- `LLMBasedArchitectureSearch`

Поля:

- `_evaluator: Evaluator`
- `_sampler: CandidateSampler`
- `_runner_factory: Callable[[], UnsafeRunner]`
- `_backend_name: str`
- `_metric_name: str`
- `_max_candidates: int`
- `_top_k_for_crossover: int`
- `_database: CandidateDatabase`

Методы:

- `database` (property) — доступ к внутренней базе кандидатов.

- `async _evaluate_single_candidate(candidate: Candidate) -> dict[str, float]`
    - Создаёт `runner = runner_factory()`.
    - Логирует запуск.
    - Вызывает `evaluator.evaluate(runner, candidate)`.
    - Логирует метрики.
    - Возвращает словарь метрик.

- `async run_search() -> CandidateDatabase`  
  Алгоритм:
    1. Логирует старт поиска.
    2. Создаёт `initial_candidate = await sampler.create_initial_candidate()`.
    3. Оценивает, сохраняет в `_database` через `add_result`.
    4. Пока `len(_database) < max_candidates`:
        - Логирует прогресс.
        - Выбирает родителей через `sampler.select_parents_for_crossover(...)`.
        - Логирует выбранных родителей.
        - Генерирует `child_candidate = await sampler.crossover_candidates(...)`.
        - Оценивает ребёнка.
        - Добавляет в `_database` через `add_result`.
        - Логирует успешное добавление.
    5. Логирует завершение и возвращает `_database`.

Используется:

- В `main.ipynb` как основной механизм LLM-поиска архитектур.