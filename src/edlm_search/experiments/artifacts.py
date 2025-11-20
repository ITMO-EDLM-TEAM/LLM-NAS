from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Final

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentDescriptor:
    """Immutable description of a single experiment run."""

    model_name: str
    dataset_name: str
    experiment_kind: str


class ExperimentArtifactsManager:
    """
    Responsible for creating a unique directory for a single experiment run
    and writing artifacts into that directory.

    The manager generates a unique slug based on experiment kind, model and
    dataset names and the wall-clock start time. All artifacts for a single
    experiment must be stored under the same directory.
    """

    def __init__(self, root_dir: str, descriptor: ExperimentDescriptor) -> None:
        """
        Initialize artifacts manager.

        Parameters
        ----------
        root_dir : str
            Root directory where experiment subdirectories will be created.
        descriptor : ExperimentDescriptor
            Immutable description of the experiment (model, dataset, kind).
        """
        if not isinstance(root_dir, str):
            raise ValueError('root_dir must be a string.')
        normalized_root = root_dir.strip()
        if not normalized_root:
            raise ValueError('root_dir must not be empty.')

        if not descriptor.model_name:
            raise ValueError('descriptor.model_name must not be empty.')
        if not descriptor.dataset_name:
            raise ValueError('descriptor.dataset_name must not be empty.')
        if not descriptor.experiment_kind:
            raise ValueError('descriptor.experiment_kind must not be empty.')

        self._descriptor: Final[ExperimentDescriptor] = descriptor
        self._root_dir: Final[Path] = Path(normalized_root).resolve()

        started_at = datetime.now()
        self._started_at: Final[datetime] = started_at

        timestamp = started_at.strftime('%Y%m%d-%H%M%S')
        slug_parts = [
            descriptor.experiment_kind,
            descriptor.model_name,
            descriptor.dataset_name,
            timestamp,
        ]
        slug = self._build_slug(slug_parts)
        self._base_dir: Final[Path] = self._root_dir / slug
        self._ensure_directory(self._base_dir)

        _LOGGER.info(
                f'Artifacts directory created at "{self._base_dir}" '
                f'for dataset="{descriptor.dataset_name}", '
                f'model="{descriptor.model_name}", '
                f'kind="{descriptor.experiment_kind}".'
        )

    @staticmethod
    def _build_slug(parts: list[str]) -> str:
        """
        Build a filesystem-friendly slug from provided parts.

        Non-alphanumeric characters are replaced with underscores, repeated
        underscores are collapsed.
        """
        joined = '_'.join(parts)
        simplified = re.sub(r'[^a-zA-Z0-9_.-]+', '_', joined)
        compact = re.sub(r'_+', '_', simplified).strip('_')
        if not compact:
            return 'experiment'
        return compact

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        """Create the directory if it does not exist."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                    f'Failed to create artifacts directory at "{path}": {exc}'
            ) from exc

    @property
    def base_dir(self) -> Path:
        """Base directory where all artifacts for this experiment are stored."""
        return self._base_dir

    @property
    def descriptor(self) -> ExperimentDescriptor:
        """Descriptor describing this experiment run."""
        return self._descriptor

    @property
    def started_at(self) -> datetime:
        """Timestamp when the artifacts manager was created."""
        return self._started_at

    def build_path(self, relative_path: str) -> Path:
        """
        Build a path inside the experiment directory for the given relative path.

        Parent directories are created if they do not exist.
        """
        if not isinstance(relative_path, str):
            raise ValueError('relative_path must be a string.')
        normalized = relative_path.strip()
        if not normalized:
            raise ValueError('relative_path must not be empty.')

        path = self._base_dir / normalized
        parent = path.parent
        if not parent.exists():
            self._ensure_directory(parent)
        return path

    def write_json(
            self,
            relative_path: str,
            payload: dict[str, Any],
    ) -> Path:
        """
        Write a JSON payload inside the experiment directory.

        Returns the full path to the written file.
        """
        if not isinstance(payload, dict):
            raise ValueError('payload must be a dict.')

        path = self.build_path(relative_path)
        try:
            with path.open('w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise RuntimeError(f'Failed to write JSON artifact at "{path}": {exc}') from exc

        _LOGGER.info(f'JSON artifact written at "{path}".')
        return path

    def write_summary(
            self,
            metrics: dict[str, float],
            extra_info: dict[str, Any],
            context: dict[str, Any],
    ) -> Path:
        """
        Write high-level experiment summary JSON.

        The summary contains model and dataset identifiers, experiment kind,
        start timestamp, metrics, extra_info and optional context.
        """
        if not isinstance(metrics, dict):
            raise ValueError('metrics must be a dict.')

        summary: dict[str, Any] = {
            'model_name': self._descriptor.model_name,
            'dataset_name': self._descriptor.dataset_name,
            'experiment_kind': self._descriptor.experiment_kind,
            'started_at': self._started_at.isoformat(),
            'artifacts_dir': str(self._base_dir),
            'metrics': dict(metrics),
            'extra_info': dict(extra_info),
            'context': dict(context),
        }

        summary_path = self.write_json('experiment_summary.json', summary)
        _LOGGER.info(
                f'Experiment summary stored at "{summary_path}" '
                f'for dataset="{self._descriptor.dataset_name}", '
                f'model="{self._descriptor.model_name}", '
                f'kind="{self._descriptor.experiment_kind}".'
        )
        return summary_path