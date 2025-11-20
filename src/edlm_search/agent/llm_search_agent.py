from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Mapping
from typing import Protocol
from typing import Sequence

import pandas as pd

from .candidate import Candidate
from .ett_evaluator import ETTEvaluator
from .llm_clients import DeepSeekClient
from .llm_clients import LMStudioClient
from .llm_clients import OpenAILikeClient
from .llm_pipeline import LLMPipeline
from .problem import Problem
from .runner import UnsafeRunner
from ..devices import get_torch_device

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS: Final[int] = 5


class LLMProviderConfigProtocol(Protocol):
    """Structural protocol describing configuration for an LLM provider."""

    provider: str
    base_url: str
    model_name: str
    temperature: float
    top_p: float
    api_key: str | None


class LLMSearchConfigProtocol(Protocol):
    """Structural protocol describing search configuration for the LLM agent."""

    num_initial_candidates: int
    num_crossover_candidates: int
    num_epochs_per_candidate: int
    metric_name: str


class DatasetConfigProtocol(Protocol):
    """Structural protocol describing a dataset configuration used in LLM search."""

    name: str


@dataclass
class AgentCandidateRecord:
    """Result of a single LLM-generated candidate evaluation."""

    candidate_id: int
    idea: str
    metrics: Dict[str, float] | None
    candidate: Candidate
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class LLMSearchRun:
    """Result of a single LLM-generated candidate evaluation."""

    results: Dict[str, List[AgentCandidateRecord]]
    best_metrics: Dict[str, float]
    provider_config: LLMProviderConfigProtocol
    search_config: LLMSearchConfigProtocol


def create_llm_pipeline(provider_config: LLMProviderConfigProtocol) -> LLMPipeline:
    """Create LLMPipeline instance for the given provider configuration."""
    provider_lower = provider_config.provider.lower()
    if provider_lower == 'lmstudio':
        client = LMStudioClient(
                base_url=provider_config.base_url,
                model_name=provider_config.model_name,
                temperature=provider_config.temperature,
                top_p=provider_config.top_p,
        )
    elif provider_lower == 'deepseek':
        if provider_config.api_key is None or not provider_config.api_key:
            raise RuntimeError('API key must be provided for provider "deepseek".')
        client = DeepSeekClient(
                api_key=provider_config.api_key,
                base_url=provider_config.base_url,
                model_name=provider_config.model_name,
                temperature=provider_config.temperature,
                top_p=provider_config.top_p,
        )
    elif provider_lower == 'openai':
        if provider_config.api_key is None or not provider_config.api_key:
            raise RuntimeError('API key must be provided for provider "openai".')
        client = OpenAILikeClient(
                api_key=provider_config.api_key,
                base_url=provider_config.base_url,
                model_name=provider_config.model_name,
                temperature=provider_config.temperature,
                top_p=provider_config.top_p,
        )
    else:
        raise ValueError(
                f'Unsupported LLM provider "{provider_config.provider}". '
                f'Expected one of ["lmstudio", "deepseek", "openai"].'
        )
    pipeline = client.create_pipeline()
    logger.info(
            f'LLM pipeline created for provider="{provider_config.provider}", '
            f'model="{provider_config.model_name}", '
            f'base_url="{provider_config.base_url}", '
            f'temperature={provider_config.temperature}, '
            f'top_p={provider_config.top_p}'
    )
    return pipeline


def build_problem(statement_path: Path) -> Problem:
    """Load problem statement from a markdown file and create Problem instance."""
    if not statement_path.is_file():
        raise FileNotFoundError(f'Problem statement file not found at {statement_path}')
    statement_text = statement_path.read_text(encoding='utf-8')
    problem = Problem(statement=statement_text)
    logger.info(f'Problem statement loaded from {statement_path}')
    return problem


async def evaluate_single_candidate(
        candidate_index: int,
        candidate: Candidate,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_column: str,
        metric_name: str,
        num_epochs: int,
) -> AgentCandidateRecord:
    """
    Train and evaluate a single LLM-generated candidate on the ETT dataset.

    Parameters
    ----------
    candidate_index : int
        Index of the candidate within the current search run.
    candidate : Candidate
        Candidate whose files will be executed and evaluated.
    train_df : pandas.DataFrame
        Training dataframe.
    valid_df : pandas.DataFrame
        Validation dataframe.
    target_column : str
        Name of the target column in the ETT dataset.
    metric_name : str
        Name of the primary metric (for example, "mse").
    num_epochs : int
        Number of training epochs for the candidate.

    Returns
    -------
    AgentCandidateRecord
        Evaluation record with metrics and candidate reference.
    """
    evaluator = ETTEvaluator(
            train_df=train_df,
            valid_df=valid_df,
            target_column=target_column,
            metric_name=metric_name,
            num_epochs=num_epochs,
    )
    runner = UnsafeRunner()
    logger.info(
            f'[LLM agent] Evaluating candidate_id={candidate_index}, '
            f'idea="{candidate.idea[:80]}..."'
    )
    metrics = await evaluator.evaluate(runner=runner, candidate=candidate)
    logger.info(
            f'[LLM agent] Finished candidate_id={candidate_index}, '
            f'{metric_name}={metrics.get(metric_name, float("nan"))}'
    )
    record = AgentCandidateRecord(
            candidate_id=candidate_index,
            idea=candidate.idea,
            metrics=dict(metrics),
            candidate=candidate,
    )
    return record


async def _generate_initial_candidate(
        dataset_name: str,
        candidate_index: int,
        problem: Problem,
        llm_pipeline: LLMPipeline,
        previous_failure_message: str | None,
        torch_backend_name: str,
) -> Candidate:
    """
    Generate a brand new candidate using the `new_candidate` template.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset for logging.
    candidate_index : int
        Index of the candidate.
    problem : Problem
        Problem description instance.
    llm_pipeline : LLMPipeline
        LLM pipeline used for generation.
    previous_failure_message : str | None
        Description of the last failure, if any.
    torch_backend_name : str
        Name of the PyTorch backend to be passed into the template.

    Returns
    -------
    Candidate
        Newly generated candidate.
    """
    logger.info(
            f'[LLM agent] Generating initial candidate_id={candidate_index} '
            f'for dataset="{dataset_name}".'
    )
    idea, files = await llm_pipeline.generate_files_from_template(
            template_name='new_candidate',
            problem=problem,
            previous_failure_message=previous_failure_message,
            torch_backend_name=torch_backend_name,
    )
    candidate = Candidate(files=files, idea=idea)
    logger.info(
            f'[LLM agent] Initial candidate_id={candidate_index} for dataset="{dataset_name}" '
            f'generated successfully.'
    )
    return candidate


async def _generate_crossover_candidate(
        dataset_name: str,
        step_index: int,
        problem: Problem,
        llm_pipeline: LLMPipeline,
        previous_failure_message: str | None,
        torch_backend_name: str,
        parent_a: AgentCandidateRecord,
        parent_b: AgentCandidateRecord,
) -> Candidate:
    """
    Generate a new candidate by crossing over two parent candidates.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset for logging.
    step_index : int
        Sequential index of the crossover step.
    problem : Problem
        Problem description instance.
    llm_pipeline : LLMPipeline
        LLM pipeline used for generation.
    previous_failure_message : str | None
        Description of the last failure, if any.
    torch_backend_name : str
        Name of the PyTorch backend to be passed into the template.
    parent_a : AgentCandidateRecord
        First parent candidate.
    parent_b : AgentCandidateRecord
        Second parent candidate.

    Returns
    -------
    Candidate
        Newly generated crossover candidate.
    """
    logger.info(
            f'[LLM agent] Generating crossover candidate for dataset="{dataset_name}", '
            f'step={step_index}, parent_a_id={parent_a.candidate_id}, '
            f'parent_b_id={parent_b.candidate_id}.'
    )
    idea, files = await llm_pipeline.generate_files_from_template(
            template_name='crossover_candidate',
            problem=problem,
            parent_a_idea=parent_a.idea,
            parent_b_idea=parent_b.idea,
            parent_a_metrics=parent_a.metrics,
            parent_b_metrics=parent_b.metrics,
            parent_a_files=parent_a.candidate.files,
            parent_b_files=parent_b.candidate.files,
            previous_failure_message=previous_failure_message,
            torch_backend_name=torch_backend_name,
    )
    candidate = Candidate(files=files, idea=idea)
    logger.info(
            f'[LLM agent] Crossover candidate for dataset="{dataset_name}", '
            f'step={step_index} generated successfully.'
    )
    return candidate


async def _generate_fixed_candidate(
        dataset_name: str,
        candidate_index: int,
        problem: Problem,
        llm_pipeline: LLMPipeline,
        previous_candidate: Candidate,
        error_message: str,
        torch_backend_name: str,
) -> Candidate:
    """
    Generate a repaired candidate using the `fix_candidate` template.

    The repaired candidate is based on the previous candidate files and the error
    message produced during execution.
    """
    logger.info(
            f'[LLM agent] Generating repaired candidate for dataset="{dataset_name}", '
            f'candidate_id={candidate_index}.'
    )
    idea, files = await llm_pipeline.generate_files_from_template(
            template_name='fix_candidate',
            problem=problem,
            previous_candidate_idea=previous_candidate.idea,
            previous_candidate_files=previous_candidate.files,
            error_message=error_message,
            torch_backend_name=torch_backend_name,
    )
    candidate = Candidate(files=files, idea=idea)
    logger.info(
            f'[LLM agent] Repaired candidate for dataset="{dataset_name}", '
            f'candidate_id={candidate_index} generated successfully.'
    )
    return candidate


async def _run_single_candidate_flow(
        dataset_name: str,
        candidate_index: int,
        base_candidate: Candidate,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_column: str,
        metric_name: str,
        num_epochs: int,
        problem: Problem,
        llm_pipeline: LLMPipeline,
        torch_backend_name: str,
) -> AgentCandidateRecord:
    """
    Evaluate one candidate and optionally try to repair it on failure.

    Returns
    -------
    AgentCandidateRecord
        On success, a record with metrics. On failure, a record with metrics=None and error details in metadata.
    """
    current_candidate = base_candidate
    fix_attempts = 0
    last_error_message: str | None = None

    while fix_attempts < MAX_REPAIR_ATTEMPTS:
        if fix_attempts > 0:
            logger.info(
                    f'[LLM agent] Repair attempt {fix_attempts + 1}/{MAX_REPAIR_ATTEMPTS} '
                    f'for dataset="{dataset_name}", candidate_id={candidate_index}.'
            )
        else:
            logger.info(
                    f'[LLM agent] Evaluating initial candidate for dataset="{dataset_name}", '
                    f'candidate_id={candidate_index}.'
            )

        try:
            record = await evaluate_single_candidate(
                    candidate_index=candidate_index,
                    candidate=current_candidate,
                    train_df=train_df,
                    valid_df=valid_df,
                    target_column=target_column,
                    metric_name=metric_name,
                    num_epochs=num_epochs,
            )
            # Success: Add fix_attempts to metadata and return
            record.metadata['fix_attempts'] = fix_attempts
            if last_error_message:
                record.metadata['previous_error'] = last_error_message
            logger.info(
                    f'[LLM agent] Candidate evaluation succeeded for dataset="{dataset_name}", '
                    f'candidate_id={candidate_index} after {fix_attempts} repairs.'
            )
            return record

        except Exception as exc:
            last_error_message = (
                f'Candidate evaluation failed on dataset="{dataset_name}", '
                f'candidate_index={candidate_index}, error={type(exc).__name__}: {exc}'
            )
            logger.exception(f'[LLM agent] {last_error_message}')
            fix_attempts += 1

            if fix_attempts < MAX_REPAIR_ATTEMPTS:
                logger.info(
                        f'[LLM agent] Starting repair attempt {fix_attempts}/{MAX_REPAIR_ATTEMPTS} '
                        f'for dataset="{dataset_name}", candidate_id={candidate_index}.'
                )
                try:
                    current_candidate = await _generate_fixed_candidate(
                            dataset_name=dataset_name,
                            candidate_index=candidate_index,
                            problem=problem,
                            llm_pipeline=llm_pipeline,
                            previous_candidate=current_candidate,
                            error_message=last_error_message,
                            torch_backend_name=torch_backend_name,
                    )
                except Exception as repair_exc:
                    last_error_message = (
                        f'Candidate repair generation failed on dataset="{dataset_name}", '
                        f'candidate_index={candidate_index}, error={type(repair_exc).__name__}: {repair_exc}'
                    )
                    logger.exception(f'[LLM agent] {last_error_message}')
                    # If repair generation itself fails, no point in further attempts for this candidate
                    break
            else:
                logger.info(
                        f'[LLM agent] Max repair attempts ({MAX_REPAIR_ATTEMPTS}) reached for '
                        f'dataset="{dataset_name}", candidate_id={candidate_index}.'
                )
                break

    # If the loop finishes, it means all attempts failed or repair generation failed.
    logger.info(
            f'[LLM agent] Candidate permanently failed for dataset="{dataset_name}", '
            f'candidate_id={candidate_index}. Storing as unrepairable.'
    )
    return AgentCandidateRecord(
            candidate_id=candidate_index,
            idea=base_candidate.idea, # Use original idea for the failed record
            metrics=None,
            candidate=base_candidate, # Use original candidate for the failed record
            metadata={
                'fix_attempts': fix_attempts,
                'last_error': last_error_message,
                'status': 'failed_unrepairable',
            },
    )


async def run_llm_search_for_dataset(
        dataset_name: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        problem: Problem,
        llm_pipeline: LLMPipeline,
        config: LLMSearchConfigProtocol,
        target_column: str,
        torch_backend_name: str,
) -> List[AgentCandidateRecord]:
    """
    Run LLM-based architecture search for a single dataset with automatic repair.

    The function generates initial candidates, then crossover candidates, evaluates
    each of them, and performs a single repair attempt when evaluation fails.
    """
    records: List[AgentCandidateRecord] = []
    last_failure_message: str | None = None

    logger.info(
            f'[LLM agent] Dataset="{dataset_name}", '
            f'initial_candidates={config.num_initial_candidates}, '
            f'crossover_candidates={config.num_crossover_candidates}, '
            f'epochs_per_candidate={config.num_epochs_per_candidate}, '
            f'torch_backend="{torch_backend_name}"'
    )

    for index in range(config.num_initial_candidates):
        try:
            candidate = await _generate_initial_candidate(
                    dataset_name=dataset_name,
                    candidate_index=index,
                    problem=problem,
                    llm_pipeline=llm_pipeline,
                    previous_failure_message=last_failure_message,
                    torch_backend_name=torch_backend_name,
            )
        except Exception as exc:
            last_failure_message = (
                f'Initial candidate generation failed on dataset="{dataset_name}", '
                f'candidate_index={index}, error={type(exc).__name__}: {exc}'
            )
            logger.exception(f'[LLM agent] {last_failure_message}')
            continue

        record = await _run_single_candidate_flow(
                dataset_name=dataset_name,
                candidate_index=index,
                base_candidate=candidate,
                train_df=train_df,
                valid_df=valid_df,
                target_column=target_column,
                metric_name=config.metric_name,
                num_epochs=config.num_epochs_per_candidate,
                problem=problem,
                llm_pipeline=llm_pipeline,
                torch_backend_name=torch_backend_name,
        )

        if record.metrics is None:
            last_failure_message = record.metadata.get('last_error', None)
        else:
            last_failure_message = None
        records.append(record)

    for offset in range(config.num_crossover_candidates):
        successful_records = [r for r in records if r.metrics is not None]
        sorted_records = sorted(
                successful_records,
                key=lambda r: r.metrics.get(config.metric_name, float('inf')),
        )
        if len(sorted_records) < 2:
            logger.info(
                    f'[LLM agent] Not enough candidates for crossover on dataset="{dataset_name}".'
            )
            break

        parent_a = sorted_records[0]
        parent_b = sorted_records[1]
        logger.info(
                f'[LLM agent] Crossover step={offset + 1} on dataset="{dataset_name}", '
                f'parent_a_id={parent_a.candidate_id}, parent_b_id={parent_b.candidate_id}.'
        )

        try:
            child_candidate = await _generate_crossover_candidate(
                    dataset_name=dataset_name,
                    step_index=offset + 1,
                    problem=problem,
                    llm_pipeline=llm_pipeline,
                    previous_failure_message=last_failure_message,
                    torch_backend_name=torch_backend_name,
                    parent_a=parent_a,
                    parent_b=parent_b,
            )
        except Exception as exc:
            last_failure_message = (
                f'Crossover candidate generation failed on dataset="{dataset_name}", '
                f'step={offset + 1}, error={type(exc).__name__}: {exc}'
            )
            logger.exception(f'[LLM agent] {last_failure_message}')
            continue

        candidate_index = config.num_initial_candidates + offset
        child_record = await _run_single_candidate_flow(
                dataset_name=dataset_name,
                candidate_index=candidate_index,
                base_candidate=child_candidate,
                train_df=train_df,
                valid_df=valid_df,
                target_column=target_column,
                metric_name=config.metric_name,
                num_epochs=config.num_epochs_per_candidate,
                problem=problem,
                llm_pipeline=llm_pipeline,
                torch_backend_name=torch_backend_name,
        )

        if child_record.metrics is None:
            last_failure_message = child_record.metadata.get('last_error', None)
        else:
            last_failure_message = None
        records.append(child_record)

    logger.info(
            f'[LLM agent] Completed search on dataset="{dataset_name}", '
            f'total_successful_candidates={len(records)}'
    )
    return records


async def run_llm_search_for_all_datasets(
        dataset_configs: Sequence[DatasetConfigProtocol],
        train_dfs: Mapping[str, pd.DataFrame],
        valid_dfs: Mapping[str, pd.DataFrame],
        problem: Problem,
        llm_pipeline: LLMPipeline,
        config: LLMSearchConfigProtocol,
        target_column: str,
        provider_config: LLMProviderConfigProtocol,
) -> LLMSearchRun:
    """
    Run LLM-based architecture search with repair for all configured datasets.

    Parameters
    ----------
    dataset_configs : Sequence[DatasetConfigProtocol]
        Sequence of dataset configurations (must provide `.name`).
    train_dfs : Mapping[str, pandas.DataFrame]
        Mapping from dataset name to training dataframe.
    valid_dfs : Mapping[str, pandas.DataFrame]
        Mapping from dataset name to validation dataframe.
    problem : Problem
        Problem description instance.
    llm_pipeline : LLMPipeline
        LLM pipeline used for generation.
    config : LLMSearchConfigProtocol
        LLM search configuration (number of candidates, metric name, epochs).
    target_column : str
        Target column name in the ETT dataset.
    provider_config : LLMProviderConfigProtocol
        LLM provider configuration.

    Returns
    -------
    LLMSearchRun
        An object containing the search results, best metrics, and configurations.
    """
    llm_search_results: Dict[str, List[AgentCandidateRecord]] = {}
    llm_best_mse: Dict[str, float] = {}

    device = get_torch_device('auto')
    torch_backend_name = device.type

    logger.info(
            f'[LLM agent] Global search started for {len(dataset_configs)} datasets with '
            f'torch_backend="{torch_backend_name}".'
    )

    for cfg in dataset_configs:
        dataset_name = cfg.name
        train_df = train_dfs[dataset_name]
        valid_df = valid_dfs[dataset_name]

        logger.info(
                f'[LLM agent] Starting search on dataset="{dataset_name}".'
        )

        try:
            records = await run_llm_search_for_dataset(
                    dataset_name=dataset_name,
                    train_df=train_df,
                    valid_df=valid_df,
                    problem=problem,
                    llm_pipeline=llm_pipeline,
                    config=config,
                    target_column=target_column,
                    torch_backend_name=torch_backend_name,
            )
        except Exception as exc:
            logger.exception(
                    f'[LLM agent] Search failed on dataset="{dataset_name}" with '
                    f'error={type(exc).__name__}: {exc}'
            )
            llm_search_results[dataset_name] = []
            continue

        llm_search_results[dataset_name] = records
        successful_records_for_best = [r for r in records if r.metrics is not None]
        if not successful_records_for_best:
            logger.info(
                    f'[LLM agent] No successful candidates on dataset="{dataset_name}".'
            )
            continue
        best_record = min(
                successful_records_for_best,
                key=lambda r: r.metrics.get(config.metric_name, float('inf')),
        )
        best_mse_value = float(
                best_record.metrics.get(config.metric_name, float('nan'))
        )
        llm_best_mse[dataset_name] = best_mse_value
        logger.info(
                f'[LLM agent] Best candidate on dataset="{dataset_name}": '
                f'candidate_id={best_record.candidate_id}, '
                f'{config.metric_name}={best_mse_value}'
        )

    logger.info(
            f'[LLM agent] Global search finished for {len(dataset_configs)} datasets. '
            f'Successful_datasets={len(llm_best_mse)}.'
    )
    return LLMSearchRun(
            results=llm_search_results,
            best_metrics=llm_best_mse,
            provider_config=provider_config,
            search_config=config,
    )