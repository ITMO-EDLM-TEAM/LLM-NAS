from __future__ import annotations

from typing import Final

from openai import AsyncOpenAI

from .llm_pipeline import LLMPipeline


class DeepSeekClient:
    """Клиент для работы с моделью DeepSeek через OpenAI-совместимый HTTP API."""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        Инициализирует клиента DeepSeek.

        Параметры
        ----------
        api_key : str
            Ключ доступа к API DeepSeek.
        base_url : str
            Базовый URL OpenAI-совместимого API DeepSeek.
        model_name : str
            Имя модели DeepSeek, которая будет использоваться для генерации кода.
        """
        self._api_key: Final[str] = api_key
        self._base_url: Final[str] = base_url
        self._model_name: Final[str] = model_name
        self._client: Final[AsyncOpenAI] = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
        )

    def create_pipeline(self) -> LLMPipeline:
        """
        Создаёт конвейер `LLMPipeline` для работы с DeepSeek.

        Возвращает
        ----------
        LLMPipeline
            Экземпляр конвейера, использующий текущий клиент и модель.
        """
        return LLMPipeline(async_openai=self._client, model_name=self._model_name)


class LMStudioClient:
    """Клиент для работы с LM Studio (локальный OpenAI-совместимый endpoint)."""

    def __init__(self, base_url: str, model_name: str):
        """
        Инициализирует клиента LM Studio.

        Параметры
        ----------
        base_url : str
            Базовый URL локального OpenAI-совместимого API LM Studio.
        model_name : str
            Имя модели LM Studio, которая будет использоваться для генерации кода.
        """
        self._base_url: Final[str] = base_url
        self._model_name: Final[str] = model_name
        # LM Studio обычно не требует API-ключа, поэтому передаём заглушку.
        self._client: Final[AsyncOpenAI] = AsyncOpenAI(
                base_url=self._base_url,
                api_key="lm-studio-placeholder-key",
        )

    def create_pipeline(self) -> LLMPipeline:
        """
        Создаёт конвейер `LLMPipeline` для работы с LM Studio.

        Возвращает
        ----------
        LLMPipeline
            Экземпляр конвейера, использующий текущий клиент и модель.
        """
        return LLMPipeline(async_openai=self._client, model_name=self._model_name)