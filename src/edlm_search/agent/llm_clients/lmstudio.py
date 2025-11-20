from __future__ import annotations

from typing import Final

from openai import AsyncOpenAI

from .base import BaseLLMClient

__all__ = ['LMStudioClient']


class LMStudioClient(BaseLLMClient):
    """Клиент для работы с LM Studio (локальный OpenAI-совместимый endpoint)."""

    _API_KEY_PLACEHOLDER: Final[str] = 'lm-studio-placeholder-key'

    def __init__(
            self,
            base_url: str,
            model_name: str,
            temperature: float,
            top_p: float,
    ) -> None:
        """
        Инициализирует клиента LM Studio.

        Параметры
        ----------
        base_url : str
            Базовый URL локального OpenAI-совместимого API LM Studio.
        model_name : str
            Имя модели LM Studio, которая будет использоваться для генерации кода.
        temperature : float
            Значение параметра temperature для генераций.
        top_p : float
            Значение параметра top_p для генераций.
        """
        if not base_url or not base_url.strip():
            raise ValueError('LM Studio base URL must be provided.')

        self._base_url: Final[str] = base_url.strip()

        async_client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._API_KEY_PLACEHOLDER,
        )
        super().__init__(
                async_client=async_client,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                provider='lmstudio',
        )