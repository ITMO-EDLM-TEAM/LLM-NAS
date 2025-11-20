from __future__ import annotations

from abc import ABC
from typing import Final

from openai import AsyncOpenAI

from ..llm_pipeline import LLMPipeline

__all__ = ['BaseLLMClient']


class BaseLLMClient(ABC):
    """Общий базовый класс для клиентов OpenAI-совместимых LLM."""

    def __init__(
            self,
            *,
            async_client: AsyncOpenAI,
            model_name: str,
            temperature: float,
            top_p: float,
            provider: str,
    ) -> None:
        if async_client is None:
            raise ValueError('AsyncOpenAI client instance must be provided.')
        self._async_client: Final[AsyncOpenAI] = async_client
        self._model_name: Final[str] = self._validate_model_name(model_name)
        self._temperature: Final[float] = self._validate_temperature(temperature)
        self._top_p: Final[float] = self._validate_top_p(top_p)
        self._provider: Final[str] = provider

    @staticmethod
    def _validate_model_name(value: str) -> str:
        if not value or not value.strip():
            raise ValueError('Model name must be a non-empty string.')
        return value.strip()

    @staticmethod
    def _validate_temperature(value: float) -> float:
        temperature = float(value)
        if temperature < 0.0:
            raise ValueError('Temperature must be greater or equal to zero.')
        return temperature

    @staticmethod
    def _validate_top_p(value: float) -> float:
        top_p = float(value)
        if not 0.0 < top_p <= 1.0:
            raise ValueError('top_p must be within (0, 1].')
        return top_p

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_p(self) -> float:
        return self._top_p

    def create_pipeline(self) -> LLMPipeline:
        """Создаёт LLMPipeline с учётом настроек текущего клиента."""
        return LLMPipeline(
                async_openai=self._async_client,
                model_name=self._model_name,
                temperature=self._temperature,
                top_p=self._top_p,
                provider=self._provider,
                base_url=str(self._async_client.base_url),
        )