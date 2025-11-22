### Шаблон `new_candidate.md.jinja`

Вызывается из:

- `Candidate.new_from_problem(...)`
- `_generate_initial_candidate(...)`

Во всех вызовах в шаблон попадают переменные:

- `problem`: объект `Problem`
    - `problem.statement`: полный текст постановки задачи (строка)
- `previous_failure_message`: `str | None`
    - `None` при первом запуске (`Candidate.new_from_problem`)
    - строка с описанием ошибки предыдущего кандидата при последующих генерациях
- `torch_backend_name`: `str`
    - имя backend’а PyTorch, берётся как `get_torch_device('auto').type`
    - примеры значений: `"cpu"`, `"cuda"` и т.п.

### Шаблон `fix_candidate.md.jinja`

Вызывается из:

- `_generate_fixed_candidate(...)`

В шаблон передаются:

- `problem`: объект `Problem`
    - `problem.statement`: текст задачи
- `previous_candidate_idea`: `str`
    - текстовое описание идеи предыдущего кандидата
- `previous_candidate_files`: `dict[str, str]`
    - ключ: путь к файлу (например, `"main.py"`, `"model_config.json"`, `"training_args.json"`)
    - значение: содержимое файла в виде строки
- `error_message`: `str`
    - отформатированное сообщение об ошибке из неуспешной оценки / запуска кандидата
- `torch_backend_name`: `str`
    - имя backend’а PyTorch (`get_torch_device('auto').type`)

### Шаблон `crossover_candidate.md.jinja`

Вызывается из:

- `_generate_crossover_candidate(...)`

В шаблон передаются:

- `problem`: объект `Problem`
    - `problem.statement`: текст задачи
- `parent_a_idea`: `str`
    - идея родителя A
- `parent_b_idea`: `str`
    - идея родителя B
- `parent_a_metrics`: `dict[str, float] | None`
    - метрики кандидата A, как словарь:
        - ключ: имя метрики (например, `"mse"`, `"total_energy_joules"`, `"wall_time_seconds"`)
        - значение: числовое значение `float`
- `parent_b_metrics`: `dict[str, float] | None`
    - аналогично `parent_a_metrics`, но для B
- `parent_a_files`: `dict[str, str]`
    - файлы кандидата A: путь → содержимое
- `parent_b_files`: `dict[str, str]`
    - файлы кандидата B: путь → содержимое
- `previous_failure_message`: `str | None`
    - описание последней ошибки при генерации/запуске предыдущего crossover-кандидата
- `torch_backend_name`: `str`
    - имя backend’а PyTorch (`get_torch_device('auto').type`)