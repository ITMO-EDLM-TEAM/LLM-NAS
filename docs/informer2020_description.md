# Informer2020: архитектура, параметры и структура проекта

Этот документ описывает модель Informer, ее архитектуру, область применения, основные параметры (включая гиперпараметры,
подходящие для тюнинга с помощью Optuna), а также структуру репозитория `Informer2020`.

---

## 1. Назначение модели и область применения

**Informer** — это архитектура для долгосрочного прогнозирования временных рядов на основе Transformer-подобной модели с
ProbSparse Attention.

Основные цели:

- Прогнозирование длинных горизонтов (десятки и сотни шагов вперед).
- Работа с **многомерными** (multivariate) и **одномерными** (univariate) временными рядами.
- Повышение эффективности self-attention за счет вероятностного отбора «активных» запросов (ProbSparse).
- Сокращение длины последовательностей внутри энкодера за счет свёрточной дистилляции.

Типовые задачи:

- Прогнозирование энергопотребления (ETT, ECL).
- Прогнозирование погодных параметров (WTH).
- Прочие задачи долгосрочного предсказания временных рядов в формате «история → будущий отрезок».

---

## 2. Архитектура Informer

Модель реализована в `models/model.py` и состоит из следующих ключевых компонентов.

### 2.1. Входные данные и типы задач

В CLI (`main_informer.py`) параметр `--features` задает тип задачи:

- `M` — **multivariate → multivariate**
- `S` — **univariate → univariate**
- `MS` — **multivariate → univariate**

Соответствующие размеры входов/выходов (`enc_in`, `dec_in`, `c_out`) задаются через словарь `data_parser` в
`main_informer.py` в зависимости от выбранного датасета (`--data`).

### 2.2. DataEmbedding

Функция `DataEmbedding` (файл `models/embed.py`) объединяет три типа признаков:

1. **TokenEmbedding**
    - 1D-свёртка (`Conv1d`) по временной оси над входными признаками.
    - Фиксированное окно `kernel_size=3` с circular padding.
    - Преобразует вход `[..., c_in]` в `[..., d_model]`.

2. **PositionalEmbedding**
    - Классический синусо-косинусный позиционный эмбеддинг (как в Transformer).
    - Необучаемый буфер размерности `[max_len, d_model]`.

3. **TemporalEmbedding / TimeFeatureEmbedding**
    - Если `embed != "timeF"`: используется `TemporalEmbedding`, где каждая компонент календарного признака (час, день
      недели, день месяца, месяц, при необходимости минуты) кодируется через фиктивный или обучаемый `Embedding`.
    - Если `embed == "timeF"`: используется `TimeFeatureEmbedding`, который преобразует набор непрерывных временных
      признаков (нормализованных в диапазоне [-0.5, 0.5]) линейным слоем в пространство размерности `d_model`.

Итоговое представление:  
`value_embedding + positional_embedding + temporal_embedding`, далее `Dropout`.

### 2.3. Encoder

Реализован в `models/encoder.py`:

- **EncoderLayer**
    - Вход: `[B, L, D]`.
    - Блок внимания: `AttentionLayer` → (ProbAttention или FullAttention).
    - Резидуальное соединение + `LayerNorm`.
    - Двухслойная точечно-свёрточная FFN:
        - `Conv1d(d_model → d_ff)` → активация (ReLU или GELU) → `Conv1d(d_ff → d_model)`.
    - Еще одно резидуальное соединение + `LayerNorm`.

- **ConvLayer (distilling)**
    - Дополнительная 1D-свёртка + `BatchNorm1d` + `ELU` + `MaxPool1d` c `stride=2`.
    - Сокращает длину последовательности по времени, увеличивая эффективность.

- **Encoder**
    - Список `EncoderLayer` + (опционально) список `ConvLayer` (для дистилляции).
    - При включенной дистилляции (`distil=True`) после каждого слоя, кроме последнего, применяется `ConvLayer`.
    - Итоговое состояние нормализуется `LayerNorm`.

- **EncoderStack**
    - Используется в варианте `InformerStack`, когда несколько энкодеров с разной глубиной объединяются, а выходы
      конкатенируются по временной оси.

### 2.4. Attention: ProbSparse и Full

Реализовано в `models/attn.py`.

#### FullAttention

- Стандартное dot-product внимание:
    - `scores = QKᵀ / sqrt(d_k)`.
    - Опциональная **каузальная маска** (`TriangularCausalMask`) для авторегрессионных задач.
    - Softmax по последней размерности, затем умножение на `V`.

#### ProbAttention (ProbSparse Attention)

- Оценивает важность запросов и выбирает лишь top-u «активных» запросов.
- Основные шаги:
    - Сэмплирование подмножества ключей для оценки «разреженных» статистик.
    - Выбор `n_top = factor * ln(L_q)` запросов по измерению sparsity.
    - Расчет внимания только для выбранных запросов, остальные получают усреднённый контекст.
- Преимущества:
    - Снижает вычислительную сложность self-attention для длинных последовательностей.
    - Фокусируется на наиболее информативных временных позициях.

### 2.5. Decoder

Реализован в `models/decoder.py`:

- **DecoderLayer**
    - self-attention над выходом декодера (`x`), обычно каузальный (через маску).
    - cross-attention к выходу энкодера (`cross`).
    - Двухслойная FFN (через `Conv1d`, аналогично энкодеру).
    - Резидуальные связи + три `LayerNorm`.

- **Decoder**
    - Список `DecoderLayer` + опциональный `LayerNorm` в конце.
    - Работает поверх эмбеддингов декодера (выход `DataEmbedding` для `x_dec` и `x_mark_dec`).

### 2.6. Output projection

В `Informer` и `InformerStack`:

- После декодера применяется линейный слой `projection: d_model → c_out`.
- Возвращает только последние `pred_len` временных шагов: `dec_out[:, -pred_len:, :]`.

---

## 3. Формат данных и датасеты

Работа с данными реализована в `data/data_loader.py`.

### 3.1. Dataset_ETT_hour и Dataset_ETT_minute

- `Dataset_ETT_hour` — почасовые данные (ETTh1, ETTh2).
- `Dataset_ETT_minute` — минутные данные (ETTm1, ETTm2).
- Особенности:
    - Чтение CSV (`pandas.read_csv`).
    - Деление на `train/val/test` по фиксированным границам (12 месяцев обучения, 4 — валидация, 4 — тест).
    - Нормализация через `StandardScaler` (из `utils/tools.py`) по тренировочной части.
    - Генерация временных признаков через `utils.timefeatures.time_features`.
    - `__getitem__` возвращает:
        - `seq_x` — вход энкодера (`[seq_len, num_features]`).
        - `seq_y` — таргеты (concat(label_len + pred_len)).
        - `seq_x_mark` / `seq_y_mark` — временные эмбеддинги.

### 3.2. Dataset_Custom

- Гибкий датасет для произвольного CSV с колонкой `date` и таргетной колонкой `target`.
- Деление на train/val/test в пропорции примерно 70/10/20.
- Поддержка:
    - `features` ∈ {`M`, `S`, `MS`}.
    - Явный список колонок `cols` (если нужно ограничить признаки).

### 3.3. Dataset_Pred

- Специальный датасет для режима inference (когда нужно предсказать будущее).
- Использует последние `seq_len` точек ряда и дополнительно формирует временные метки для будущих `pred_len` шагов.

---

## 4. Обучение и оценка (Exp_Informer)

Основной тренировочный цикл реализован в `exp/exp_informer.py` (класс `Exp_Informer`, наследник `Exp_Basic`):

- `_build_model` — создаёт экземпляр `Informer` или `InformerStack` и, при необходимости, оборачивает в `DataParallel`.
- `_get_data(flag)` — инициализирует один из классов датасета и возвращает `Dataset` и `DataLoader`.
- `_select_optimizer` — Adam с lr из аргумента `--learning_rate`.
- `_select_criterion` — MSELoss (`nn.MSELoss`).
- `train(setting)`:
    - Цикл по эпохам:
        - Прямой проход → loss → шаг оптимизатора.
        - Поддержка AMP (`--use_amp`).
        - EarlyStopping (`utils.tools.EarlyStopping`) по валидационному MSE.
        - Снижение lr по схеме `adjust_learning_rate`.
    - По окончании загружает лучшее состояние модели (по валидационному loss) из
      `./checkpoints/<setting>/checkpoint.pth`.

- `test(setting)`:
    - Выполняет inference на тестовом наборе.
    - Считает метрики через `utils.metrics.metric` (MAE, MSE, RMSE, MAPE, MSPE).
    - Сохраняет `metrics.npy`, `pred.npy`, `true.npy` в `./results/<setting>/`.

- `predict(setting, load)`:
    - Режим прогнозирования на конце ряда (для `Dataset_Pred`).
    - Сохраняет `real_prediction.npy` в `./results/<setting>/`.

---

## 5. Основные параметры `main_informer.py`

Запуск базового эксперимента:

```bash
python -u main_informer.py   --model informer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --attn prob
```

Ниже краткое резюме ключевых параметров (не полный список, см. `main_informer.py`):

### 5.1. Данные и формат

| Параметр      | Назначение                                                                                                |
|---------------|-----------------------------------------------------------------------------------------------------------|
| `--data`      | Имя датасета (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `WTH`, `ECL`, `Solar`).                                 |
| `--root_path` | Каталог с данными (по умолчанию `./data/ETT/`).                                                           |
| `--data_path` | Имя CSV-файла (обычно подставляется автоматически из `data_parser`).                                      |
| `--features`  | Тип задачи: `M`, `S`, `MS`.                                                                               |
| `--target`    | Имя таргетной колонки (для `S` и `MS`). Автоматически берётся из `data_parser` для стандартных датасетов. |
| `--freq`      | Частота временных данных (`h`, `t`, `d`, `w`, `m` и т.п.).                                                |
| `--cols`      | Список колонок для использования в качестве признаков (опционально).                                      |

### 5.2. Архитектура модели

| Параметр             | Назначение                                                                        |
|----------------------|-----------------------------------------------------------------------------------|
| `--seq_len`          | Длина входной последовательности энкодера.                                        |
| `--label_len`        | Длина стартового отрезка для декодера (history в декодере).                       |
| `--pred_len`         | Длина горизонта прогнозирования.                                                  |
| `--enc_in`           | Размерность входа энкодера (число признаков).                                     |
| `--dec_in`           | Размерность входа декодера.                                                       |
| `--c_out`            | Размерность выхода (число прогнозируемых признаков).                              |
| `--d_model`          | Размерность внутренних представлений (эмбеддингов).                               |
| `--n_heads`          | Число голов внимания.                                                             |
| `--e_layers`         | Число слоёв энкодера (для `informer`).                                            |
| `--d_layers`         | Число слоёв декодера.                                                             |
| `--s_layers`         | Конфигурация глубин энкодера для `InformerStack` (строка вида `3,2,1`).           |
| `--d_ff`             | Размер скрытого слоя FFN (Conv1d) внутри encoder/decoder.                         |
| `--factor`           | Гиперпараметр ProbSparse attention: влияет на количество отбираемых запросов.     |
| `--attn`             | Тип внимания в энкодере: `prob` (Informer) или `full` (классический Transformer). |
| `--embed`            | Тип временных эмбеддингов: `timeF`, `fixed`, `learned`.                           |
| `--activation`       | Активация в FFN: `gelu` или `relu`.                                               |
| `--dropout`          | Доля dropout.                                                                     |
| `--distil`           | Использовать ли ConvLayer-дистилляцию в энкодере.                                 |
| `--mix`              | Использовать ли mix attention в декодере.                                         |
| `--output_attention` | Возвращать ли матрицы вниманий из энкодера.                                       |

### 5.3. Обучение

| Параметр          | Назначение                                                                              |
|-------------------|-----------------------------------------------------------------------------------------|
| `--batch_size`    | Размер batch.                                                                           |
| `--train_epochs`  | Количество эпох обучения.                                                               |
| `--learning_rate` | Начальное значение learning rate.                                                       |
| `--lradj`         | Схема адаптации lr (`type1`, `type2`).                                                  |
| `--patience`      | Patience для EarlyStopping.                                                             |
| `--itr`           | Сколько раз повторить эксперимент с заданными параметрами.                              |
| `--loss`          | Тип функции потерь (по умолчанию `mse`).                                                |
| `--use_amp`       | Включение AMP (automatic mixed precision).                                              |
| `--inverse`       | Обратное преобразование таргетов в исходное масштабирование при вычислении loss/метрик. |

### 5.4. Устройство и GPU

| Параметр          | Назначение                                             |
|-------------------|--------------------------------------------------------|
| `--use_gpu`       | Использовать ли GPU (если доступен).                   |
| `--gpu`           | Номер GPU для обучения.                                |
| `--use_multi_gpu` | Включить ли DataParallel.                              |
| `--devices`       | Список id устройств через запятую, например `0,1,2,3`. |
| `--num_workers`   | Число воркеров DataLoader.                             |

---

## 6. Гиперпараметры для тюнинга через Optuna

Ниже перечислены параметры, которые наиболее часто имеют смысл оптимизировать с помощью Optuna (или любого другого
HPO-инструмента). Конкретные диапазоны зависят от задач и ограничений по ресурсам.

### 6.1. Архитектура

Рекомендуемые кандидаты:

- `d_model` — размерность эмбеддингов, например:
    - поиск по значениям: {128, 256, 512, 768}.
- `n_heads` — число голов внимания:
    - {4, 6, 8}.
- `e_layers` — число слоёв энкодера:
    - {1, 2, 3, 4}.
- `d_layers` — число слоёв декодера:
    - {1, 2, 3}.
- `d_ff` — размер FFN:
    - {512, 1024, 2048, 4096}.
- `factor` — параметр ProbSparse:
    - целые значения из диапазона [3, 15].

### 6.2. Обучение и регуляризация

- `learning_rate`:
    - логарифмический поиск в диапазоне `[1e-5, 1e-3]`.
- `batch_size`:
    - значения: {16, 32, 64, 128}, ограничено памятью.
- `dropout`:
    - непрерывный диапазон `[0.0, 0.3]`.
- `train_epochs`:
    - целые значения (например, [5, 30]), при этом ранняя остановка не позволит сильно переобучиться.
- `patience`:
    - целые значения [3, 10].
- `lradj`:
    - категориальный выбор: `type1`, `type2` (или отключение при фиксации lr).

### 6.3. Структура входов

- `seq_len`:
    - дискретные значения в диапазоне [48, 720] (зависит от данных).
- `label_len`:
    - дискретные значения ≤ `seq_len`.
- `pred_len`:
    - задается задачей, но иногда можно оптимизировать несколько вариантов горизонта.
- `features`:
    - выбор между `M`, `S`, `MS`, если постановка задачи допускает несколько формулировок.

### 6.4. Дополнительные флаги

- `attn`:
    - `prob` vs `full` (на малых последовательностях full может быть конкурентоспособен).
- `embed`:
    - `timeF` vs `fixed` vs `learned`.
- `activation`:
    - `relu` vs `gelu`.
- `distil`:
    - включение/выключение дистилляции (можно как bool-переменная).
- `mix`:
    - включение/выключение mix attention в декодере.

Оптимизация может быть организована как внешний Python-скрипт, который:

1. Формирует набор аргументов для `main_informer.py`.
2. Запускает его через `subprocess`.
3. Читает `metrics.npy` или использует уже готовую обёртку `informer_experiment_wrapper.py` (см. ниже).

---

## 7. Обёртка `informer_experiment_wrapper.py`

Файл `informer_experiment_wrapper.py` предоставляет высокоуровневый интерфейс для:

1. Разбора аргументов:
    - `--data-path` — путь к CSV-файлу (или имя датасета).
    - `--metrics-path` — путь к JSON-файлу, куда будут сохранены метрики и артефакты.
    - `--epochs` — количество эпох обучения (проксируется в `--train_epochs`).

2. Разрешения пути к датасету:
    - `_resolve_dataset_csv_path` ищет CSV в типичных расположениях (`ETDataset/ETT-small`, `data/ETT` и т.д.).

3. Запуска эксперимента:
    - `_run_informer_and_get_metrics` формирует команду вызова `main_informer.py`, запускает её, измеряет время
      выполнения и (опционально) энергию GPU через ZeusMonitor.

4. Чтения результатов:
    - `_load_latest_metrics` ищет последний `metrics.npy` и соответствующие `pred.npy`/`true.npy`.

5. Сохранения итогового JSON:
    - `_save_metrics` записывает:
        - метрики (`mse`, `mae`, `rmse`, `mape`, `mspe`);
        - гиперпараметры (`train_epochs`, `data_path`, `extra_args`);
        - ссылки на артефакты (`predictions_csv`, `pred.npy`, `true.npy`).

Это делает обёртку удобной точкой интеграции с системами автоматического тюнинга (Optuna, Ray Tune и др.), поскольку:

- Вход: простой CLI.
- Выход: один JSON с метриками и путями к артефактам.

---

## 8. Структура репозитория `Informer2020`

Ниже логическая структура каталога `edlm_search/src/Informer2020` с описанием основных узлов.

```text
Informer2020/
├─ Dockerfile
├─ Makefile
├─ LICENSE
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ __init__.py
├─ informer_experiment_wrapper.py
├─ main_informer.py
├─ data/
│  ├─ __init__.py
│  └─ data_loader.py
├─ exp/
│  ├─ __init__.py
│  ├─ exp_basic.py
│  └─ exp_informer.py
├─ models/
│  ├─ __init__.py
│  ├─ attn.py
│  ├─ decoder.py
│  ├─ embed.py
│  ├─ encoder.py
│  └─ model.py
├─ scripts/
│  ├─ ETTh1.sh
│  ├─ ETTh2.sh
│  ├─ ETTm1.sh
│  └─ WTH.sh
├─ utils/
│  ├─ __init__.py
│  ├─ masking.py
│  ├─ metrics.py
│  ├─ timefeatures.py
│  └─ tools.py
├─ img/
│  ├─ informer.png
│  ├─ probsparse_intro.png
│  ├─ data.png
│  ├─ result_univariate.png
│  └─ result_multivariate.png
├─ checkpoints/        # создается во время обучения
└─ results/            # создается во время тестирования/предсказаний
```

### 8.1. Корневые файлы

- `Dockerfile` — образ на основе Miniconda + установка окружения из `environment.yml`.
- `Makefile`:
    - `make init` — сборка Docker-образа.
    - `make dataset` — загрузка ETT/ECL/WTH датасетов.
    - `make run_module module="..."` — запуск произвольного модуля внутри контейнера.
    - `make jupyter` — запуск Jupyter Lab внутри контейнера.
- `environment.yml` / `requirements.txt` — спецификация зависимостей (Python 3.6, torch 1.8.0 и др.).
- `README.md` — оригинальное описание проекта, примеры запуска и список параметров.

### 8.2. Пакет `data/`

- `data_loader.py`:
    - Реализация датасетов `Dataset_ETT_hour`, `Dataset_ETT_minute`, `Dataset_Custom`, `Dataset_Pred`.

### 8.3. Пакет `exp/`

- `exp_basic.py`:
    - Базовый класс `Exp_Basic` с общими полями (device, model).
- `exp_informer.py`:
    - Наследник `Exp_Basic` с полной логикой обучения/валидирования/теста/предсказаний для Informer.

### 8.4. Пакет `models/`

- `attn.py`:
    - `FullAttention`, `ProbAttention`, `AttentionLayer`.
- `decoder.py`:
    - `DecoderLayer`, `Decoder`.
- `embed.py`:
    - `PositionalEmbedding`, `TokenEmbedding`, `TemporalEmbedding`, `TimeFeatureEmbedding`, `DataEmbedding`.
- `encoder.py`:
    - `ConvLayer`, `EncoderLayer`, `Encoder`, `EncoderStack`.
- `model.py`:
    - Основные классы моделей: `Informer`, `InformerStack`.

### 8.5. Пакет `scripts/`

- Shell-скрипты для воспроизведения экспериментов на различных датасетах:
    - `ETTh1.sh`, `ETTh2.sh`, `ETTm1.sh`, `WTH.sh`.
- Каждый скрипт содержит набор типичных комбинаций `seq_len`, `label_len`, `pred_len`, `e_layers`, `d_layers` и т.д.

### 8.6. Пакет `utils/`

- `masking.py`:
    - Маски для внимания: `TriangularCausalMask`, `ProbMask`.
- `metrics.py`:
    - Метрики: `MAE`, `MSE`, `RMSE`, `MAPE`, `MSPE`, а также вспомогательные `RSE`, `CORR`.
- `timefeatures.py`:
    - Классы временных признаков и функция `time_features` для генерации временных эмбеддингов.
- `tools.py`:
    - `adjust_learning_rate` — изменение lr по эпохам.
    - `EarlyStopping` — ранняя остановка.
    - `dotdict` — словарь с доступом по точке.
    - `StandardScaler` — стандартная нормализация (mean/std) с поддержкой NumPy и PyTorch.

### 8.7. Служебные каталоги

- `checkpoints/`:
    - Содержит сохранённые веса моделей (`checkpoint.pth`) для разных конфигураций (`setting`).
- `results/`:
    - Для каждого `setting` содержит:
        - `metrics.npy` — массив метрик.
        - `pred.npy` — предсказания.
        - `true.npy` — истинные значения.
        - В режиме `predict` — `real_prediction.npy`.