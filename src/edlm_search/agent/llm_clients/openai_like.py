from __future__ import annotations

from typing import Final

from openai import AsyncOpenAI

from .base import BaseLLMClient

__all__ = ['OpenAILikeClient']


class OpenAILikeClient(BaseLLMClient):
    """Клиент для работы с любым OpenAI-совместимым облачным провайдером."""

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str,
            temperature: float,
            top_p: float,
    ) -> None:
        """
        Инициализирует OpenAI-совместимый клиент.

        Параметры
        ----------
        api_key : str
            Ключ доступа к выбранному провайдеру.
        base_url : str
            Базовый URL OpenAI-совместимого API.
        model_name : str
            Имя модели, которая будет использоваться для генерации кода.
        temperature : float
            Значение параметра temperature для генераций.
        top_p : float
            Значение параметра top_p для генераций.
        """
        if not api_key or not api_key.strip():
            raise ValueError('OpenAI-compatible API key must be provided.')
        if not base_url or not base_url.strip():
            raise ValueError('OpenAI-compatible base URL must be provided.')

        self._api_key: Final[str] = api_key.strip()
        self._base_url: Final[str] = base_url.strip()

        async_client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
        )
        super().__init__(
                async_client=async_client,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
        )