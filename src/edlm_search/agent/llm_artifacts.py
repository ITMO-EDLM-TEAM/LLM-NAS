from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import asdict
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

logger = logging.getLogger(__name__)


class LLMAgentArtifactsWriter:
    """
    Responsible for persisting LLM agent search runs to disk in a structured way.

    The writer creates a dedicated directory for each experiment and stores:
      * experiment-level metadata and configuration;
      * per-dataset candidate summaries (CSV);
      * per-candidate directories with idea, metrics, metadata and code files.
    """

    def __init__(self, root_dir: str):
        if not isinstance(root_dir, str):
            raise ValueError('Parameter "root_dir" must be a string.')
        root_dir_stripped = root_dir.strip()
        if not root_dir_stripped:
            raise ValueError('Parameter "root_dir" must be a non-empty string.')

        self._root_path = Path(root_dir_stripped).resolve()
        self._root_path.mkdir(parents=True, exist_ok=True)

    @property
    def root_path(self) -> Path:
        """
        Return the absolute root path where all LLM agent artifacts are stored.
        """
        return self._root_path

    def create_experiment_directory(
            self,
            provider_config: object,
            search_config: object,
    ) -> Path:
        """
        Create and return a unique directory for a single LLM agent experiment.

        The directory name encodes:
          * start timestamp (to second resolution),
          * provider name,
          * model name,
          * primary metric name.

        If a directory with the same name already exists, an incremental numeric
        suffix is appended.
        """
        provider_name = _extract_attribute_as_slug(
                source=provider_config,
                attr_name='provider',
                default_slug='provider',
        )
        model_name = _extract_attribute_as_slug(
                source=provider_config,
                attr_name='model_name',
                default_slug='model',
        )
        metric_name = _extract_attribute_as_slug(
                source=search_config,
                attr_name='metric_name',
                default_slug='metric',
        )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        base_dir_name = f'{timestamp}_{provider_name}_{model_name}_{metric_name}'
        experiment_dir = self._build_unique_directory(base_dir_name)
        experiment_dir.mkdir(parents=True, exist_ok=False)
        logger.info(
                f'[LLM agent] Created artifacts directory "{experiment_dir}" '
                f'for provider="{provider_name}", model="{model_name}", metric="{metric_name}".'
        )
        return experiment_dir

    def save_run(
            self,
            experiment_dir: Path,
            run: object,
            dataset_configs: Sequence[object],
            target_column: str,
            experiment_started_at: datetime,
            experiment_finished_at: datetime,
    ) -> None:
        """
        Persist the full LLM agent run, including configs and candidates, into the experiment directory.

        The method writes:
          * experiment_metadata.json with configuration and global metrics;
          * candidates_summary.csv with a flat view across all datasets and candidates;
          * per-dataset directories under "datasets/" with per-candidate artifacts.
        """
        if not isinstance(experiment_dir, Path):
            raise ValueError('Parameter "experiment_dir" must be a pathlib.Path instance.')
        if not experiment_dir.exists():
            raise ValueError(f'Artifacts directory "{experiment_dir}" does not exist.')
        if experiment_started_at > experiment_finished_at:
            raise ValueError('Experiment start time must not be later than finish time.')

        results_by_dataset = _extract_results_by_dataset(run)
        best_metrics = _extract_best_metrics(run)

        provider_config_obj = getattr(run, 'provider_config', None)
        search_config_obj = getattr(run, 'search_config', None)

        provider_config_dict = _sanitize_provider_config_dict(
                _config_to_dict(provider_config_obj)
        )
        search_config_dict = _config_to_dict(search_config_obj)
        datasets_config_list = _convert_dataset_configs_to_dicts(dataset_configs)

        metadata = {
            'experiment_started_at': experiment_started_at.isoformat(),
            'experiment_finished_at': experiment_finished_at.isoformat(),
            'experiment_duration_seconds': float(
                    (experiment_finished_at - experiment_started_at).total_seconds()
            ),
            'target_column': target_column,
            'provider_config': provider_config_dict,
            'search_config': search_config_dict,
            'datasets': datasets_config_list,
            'best_metrics': dict(best_metrics),
        }

        metadata_path = experiment_dir / 'experiment_metadata.json'
        _write_json_file(metadata_path, metadata)

        _save_candidates_per_dataset(
                experiment_dir=experiment_dir,
                results_by_dataset=results_by_dataset,
        )

        summary_csv_path = experiment_dir / 'candidates_summary.csv'
        _write_candidates_summary_csv(
                csv_path=summary_csv_path,
                results_by_dataset=results_by_dataset,
        )

        logger.info(
                f'[LLM agent] Saved artifacts for global search to "{experiment_dir}".'
        )

    def _build_unique_directory(self, base_dir_name: str) -> Path:
        """
        Build a unique directory path under the root path by appending a numeric suffix if needed.
        """
        base_path = self._root_path / base_dir_name
        if not base_path.exists():
            return base_path

        index = 1
        while True:
            candidate = self._root_path / f'{base_dir_name}_{index:02d}'
            if not candidate.exists():
                return candidate
            index += 1


def build_llm_artifacts_root(root_env_var_name: str, fallback_relative_dir: str) -> Path:
    """
    Build root directory for LLM agent artifacts using environment variable or fallback path.

    The environment variable takes precedence if it is set and non-empty; otherwise
    the fallback path (relative to the current working directory) is used.
    """
    raw = os.getenv(root_env_var_name)
    if raw is not None and raw.strip():
        base_dir = raw.strip()
    else:
        base_dir = fallback_relative_dir

    root_path = Path(base_dir).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    logger.info(
            f'[LLM agent] Using artifacts root directory "{root_path}" '
            f'(env_var="{root_env_var_name}").'
    )
    return root_path


def _extract_attribute_as_slug(source: object, attr_name: str, default_slug: str) -> str:
    """
    Extract an attribute value from an object and convert it to a filesystem-safe slug.
    """
    if source is None:
        return default_slug
    raw_value = getattr(source, attr_name, None)
    if raw_value is None:
        return default_slug
    return _slugify(str(raw_value))


def _slugify(value: str) -> str:
    """
    Convert an arbitrary string into a filesystem-safe slug.

    Only ASCII letters, digits, '-' and '_' are preserved. All other characters
    are replaced with '-'. Consecutive separators are collapsed.
    """
    lowered = value.strip().lower()
    if not lowered:
        return 'value'

    replaced = lowered.replace(' ', '-')
    allowed_chars = []
    for ch in replaced:
        if ch.isalnum() or ch in ('-', '_'):
            allowed_chars.append(ch)
        else:
            allowed_chars.append('-')
    allowed = ''.join(allowed_chars)

    collapsed = re.sub(r'-+', '-', allowed)
    collapsed = re.sub(r'_+', '_', collapsed)
    collapsed = collapsed.strip('-_')
    return collapsed or 'value'


def _config_to_dict(config: object | None) -> dict[str, Any]:
    """
    Convert configuration object into a JSON-serializable dict.

    Dataclasses are converted via dataclasses.asdict; for other objects, public
    attributes from __dict__ are used. Unknown types result in an empty dict.
    """
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, '__dict__'):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith('_')
        }
    return {}


def _sanitize_provider_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """
    Redact sensitive fields such as API keys and tokens from provider configuration.
    """
    if not config:
        return {}

    sanitized = dict(config)
    sensitive_keys = {
        'api_key',
        'token',
        'access_token',
        'secret',
        'secret_key',
    }
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = '<redacted>'
    return sanitized


def _convert_dataset_configs_to_dicts(dataset_configs: Sequence[object]) -> list[dict[str, Any]]:
    """
    Convert dataset configuration objects to dictionaries.

    At minimum, the 'name' attribute is preserved if present.
    """
    result: list[dict[str, Any]] = []
    for cfg in dataset_configs:
        cfg_dict = _config_to_dict(cfg)
        if 'name' not in cfg_dict and hasattr(cfg, 'name'):
            cfg_dict['name'] = getattr(cfg, 'name')
        result.append(cfg_dict)
    return result


def _extract_results_by_dataset(run: object) -> Mapping[str, Iterable[object]]:
    """
    Extract mapping from dataset name to candidate records from the run object.
    """
    results = getattr(run, 'results', None)
    if results is None:
        return {}
    if not isinstance(results, Mapping):
        raise ValueError('Attribute "results" of run must be a mapping.')
    return results


def _extract_best_metrics(run: object) -> Mapping[str, float]:
    """
    Extract mapping from dataset name to best metric value from the run object.
    """
    best_metrics = getattr(run, 'best_metrics', None)
    if best_metrics is None:
        return {}
    if not isinstance(best_metrics, Mapping):
        raise ValueError('Attribute "best_metrics" of run must be a mapping.')
    return best_metrics


def _write_json_file(path: Path, content: Mapping[str, Any]) -> None:
    """
    Write a JSON mapping to disk with UTF-8 encoding and indentation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def _safe_relative_candidate_file_path(relative_path: str) -> Path:
    """
    Sanitize candidate file path to ensure it stays within the artifacts directory.

    Parent directory references ('..') are removed. If the resulting path is empty,
    a generic file name is used.
    """
    candidate = Path(relative_path)
    safe_parts = [part for part in candidate.parts if part not in ('.', '..')]
    if not safe_parts:
        return Path('file')
    return Path(*safe_parts)


def _extract_candidate_status(metrics: object, metadata: Mapping[str, Any]) -> str:
    """
    Determine high-level candidate status string for logging and artifact naming.
    """
    status_raw = metadata.get('status')
    if isinstance(status_raw, str) and status_raw.strip():
        return _slugify(status_raw)

    if metrics is None:
        return 'failed'
    return 'ok'


def _save_candidates_per_dataset(
        experiment_dir: Path,
        results_by_dataset: Mapping[str, Iterable[object]],
) -> None:
    """
    Save per-dataset and per-candidate artifacts into the experiment directory.
    """
    datasets_dir = experiment_dir / 'datasets'
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, records in results_by_dataset.items():
        dataset_slug = _slugify(str(dataset_name))
        dataset_dir = datasets_dir / dataset_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)

        candidates_root = dataset_dir / 'candidates'
        candidates_root.mkdir(parents=True, exist_ok=True)

        for record in records:
            candidate_id = int(getattr(record, 'candidate_id'))
            idea = getattr(record, 'idea', '')
            metrics = getattr(record, 'metrics', None)
            candidate_obj = getattr(record, 'candidate')
            metadata = getattr(record, 'metadata', {})

            if not isinstance(metadata, Mapping):
                metadata = {}

            status = _extract_candidate_status(metrics, metadata)
            candidate_dir_name = f'candidate_{candidate_id:03d}_{status}'
            candidate_dir = candidates_root / candidate_dir_name
            candidate_dir.mkdir(parents=True, exist_ok=True)

            idea_text = '' if idea is None else str(idea)
            idea_path = candidate_dir / 'idea.txt'
            idea_path.write_text(idea_text, encoding='utf-8')

            metrics_dict: dict[str, Any] = {}
            if isinstance(metrics, Mapping):
                metrics_dict = dict(metrics)

            metrics_path = candidate_dir / 'metrics.json'
            _write_json_file(metrics_path, metrics_dict)

            metadata_dict = dict(metadata)
            metadata_path = candidate_dir / 'metadata.json'
            _write_json_file(metadata_path, metadata_dict)

            files_mapping = getattr(candidate_obj, 'files', {})
            if isinstance(files_mapping, Mapping):
                files_root = candidate_dir / 'files'
                for relative_name, content in files_mapping.items():
                    safe_relative = _safe_relative_candidate_file_path(str(relative_name))
                    destination_path = files_root / safe_relative
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    destination_path.write_text(str(content), encoding='utf-8')


def _write_candidates_summary_csv(
        csv_path: Path,
        results_by_dataset: Mapping[str, Iterable[object]],
) -> None:
    """
    Write a flat CSV summary of all candidates across all datasets.

    The summary includes:
      * dataset and candidate identifiers;
      * high-level status;
      * primary metric name and value (first metric key);
      * energy and wall-time metrics if present;
      * repair attempts and token usage if present in metadata.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'dataset_name',
        'candidate_id',
        'status',
        'idea',
        'metric_primary_name',
        'metric_primary_value',
        'total_energy_joules',
        'wall_time_seconds',
        'fix_attempts',
        'total_input_tokens',
        'total_output_tokens',
    ]

    with csv_path.open('w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_name, records in results_by_dataset.items():
            dataset_name_str = str(dataset_name)
            for record in records:
                candidate_id = int(getattr(record, 'candidate_id'))
                idea = getattr(record, 'idea', '')
                idea_str = '' if idea is None else str(idea)

                metrics = getattr(record, 'metrics', None)
                metadata = getattr(record, 'metadata', {})

                primary_metric_name: str | None = None
                primary_metric_value: float | None = None
                total_energy: float | None = None
                wall_time: float | None = None

                if isinstance(metrics, Mapping) and metrics:
                    first_key = next(iter(metrics.keys()))
                    primary_metric_name = str(first_key)
                    primary_metric_raw = metrics.get(first_key)
                    if primary_metric_raw is not None:
                        primary_metric_value = float(primary_metric_raw)

                    energy_value = metrics.get('total_energy_joules')
                    if energy_value is not None:
                        total_energy = float(energy_value)

                    wall_time_value = metrics.get('wall_time_seconds')
                    if wall_time_value is not None:
                        wall_time = float(wall_time_value)

                if not isinstance(metadata, Mapping):
                    metadata = {}

                status = _extract_candidate_status(metrics, metadata)
                fix_attempts = metadata.get('fix_attempts')
                total_input_tokens = metadata.get('total_input_tokens')
                total_output_tokens = metadata.get('total_output_tokens')

                row = {
                    'dataset_name': dataset_name_str,
                    'candidate_id': candidate_id,
                    'status': status,
                    'idea': idea_str,
                    'metric_primary_name': primary_metric_name,
                    'metric_primary_value': primary_metric_value,
                    'total_energy_joules': total_energy,
                    'wall_time_seconds': wall_time,
                    'fix_attempts': fix_attempts,
                    'total_input_tokens': total_input_tokens,
                    'total_output_tokens': total_output_tokens,
                }
                writer.writerow(row)