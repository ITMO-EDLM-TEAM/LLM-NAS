import pathlib
import re
from inspect import cleandoc
from typing import Final

import tiktoken
from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import StrictUndefined
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

here = pathlib.Path(__file__).parent.resolve()
jinja_env = Environment(
        loader=FileSystemLoader(str(here / 'prompts')), undefined=StrictUndefined
)


class ModelOutputParseError(Exception):
    """Raised when the output from the language model is not in the expected format."""

    pass


class LLMPipeline:
    """Manages interactions with a language model to generate and parse responses."""

    def __init__(
            self,
            async_openai: AsyncOpenAI,
            model_name: str,
            temperature: float,
            top_p: float,
            provider: str,
            base_url: str,
    ):
        if async_openai is None:
            raise ValueError('Parameter "async_openai" must be provided.')
        if not model_name or not model_name.strip():
            raise ValueError('Parameter "model_name" must be a non-empty string.')
        temperature_value = float(temperature)
        if temperature_value < 0.0:
            raise ValueError('Parameter "temperature" must be non-negative.')
        top_p_value = float(top_p)
        if not 0.0 < top_p_value <= 1.0:
            raise ValueError('Parameter "top_p" must be within (0, 1].')

        self._async_openai = async_openai
        self._model_name: str = model_name.strip()
        self._temperature: float = temperature_value
        self._top_p: float = top_p_value
        self._provider: str = provider
        self._base_url: str = base_url
        self._prompt_tokens_total: int = 0
        self._completion_tokens_total: int = 0
        self._encoding: Final | None = self._init_encoding()

    @property
    def model_name(self) -> str:
        """Name of the LLM model used in this pipeline."""
        return self._model_name

    @property
    def temperature(self) -> float:
        """The temperature setting for the LLM."""
        return self._temperature

    @property
    def top_p(self) -> float:
        """The top_p setting for the LLM."""
        return self._top_p
    
    @property
    def provider(self) -> str:
        """The provider of the LLM."""
        return self._provider

    @property
    def base_url(self) -> str:
        """The base URL of the LLM API."""
        return self._base_url

    def _init_encoding(self):
        """
        Initialize tokenizer encoding for token counting.

        If the encoding cannot be created, None is returned and token counting
        will silently fall back to zero.
        """
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using the configured encoding.

        If encoding is not available, returns 0.
        """
        if self._encoding is None:
            return 0
        if not text:
            return 0
        return len(self._encoding.encode(text))

    @property
    def total_prompt_tokens(self) -> int:
        """
        Total number of prompt tokens sent to the LLM through this pipeline.
        """
        return self._prompt_tokens_total

    @property
    def total_completion_tokens(self) -> int:
        """
        Total number of completion tokens received from the LLM through this pipeline.
        """
        return self._completion_tokens_total

    def reset_token_counters(self) -> None:
        """
        Reset accumulated token usage statistics to zero.
        """
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0

    @staticmethod
    def _strip_markdown_code_fence(content: str) -> str:
        """
        Remove outer Markdown code fences (```lang ... ```) if present.

        The function searches for the first and the last line that start with
        triple backticks and returns the text between them. If no such pair of
        lines is found, the original content is returned unchanged.
        """
        if not content:
            return content

        lines = content.splitlines()
        first_fence_index: int | None = None
        last_fence_index: int | None = None

        for index, line in enumerate(lines):
            if line.lstrip().startswith('```'):
                first_fence_index = index
                break

        if first_fence_index is None:
            return content

        for index in range(len(lines) - 1, -1, -1):
            if lines[index].lstrip().startswith('```'):
                last_fence_index = index
                break

        if last_fence_index is None or last_fence_index <= first_fence_index:
            return content

        inner_lines = lines[first_fence_index + 1:last_fence_index]
        inner_text = '\n'.join(inner_lines)
        return inner_text.strip('\n')

    def _parse_xml_files(self, xml_string: str) -> dict[str, str]:
        """
        Parses an XML-like string containing file data and returns a dictionary.

        This parser is "forgiving" and uses regex to extract content,
        allowing special characters like '<' or '&' inside the file tags.

        Args
        -----
            xml_string: A string containing the data, expected to be wrapped in
                        <files>...</files> and contain <file path="...">...</file> tags.

        Returns
        -------
            A dictionary where keys are file paths and values are file contents.
        """
        files_dict: dict[str, str] = {}

        pattern = re.compile(r'<file path="([^"]+)">(.+?)</file>', re.DOTALL)
        matches = list(pattern.finditer(xml_string))

        if not matches:
            if '<files>' not in xml_string or '</files>' not in xml_string:
                raise ModelOutputParseError('failed to find <files>...</files> section')
            raise ModelOutputParseError('no <file ...> entries found within <files> section')

        for match in matches:
            path = match.group(1)
            content = match.group(2)

            if not path:
                raise ModelOutputParseError('found a file entry with no path')

            without_fence = self._strip_markdown_code_fence(content)
            cleaned_content = cleandoc(without_fence)

            if not cleaned_content.strip():
                raise ModelOutputParseError(f"content of file entry for '{path}' is empty")

            files_dict[path] = cleaned_content

        return files_dict

    async def generate_files_from_template(
            self, template_name: str, **kwargs
    ) -> tuple[str, dict[str, str], int, int]:
        """
        Generate files from a Jinja2 template rendered with the given context.

        Args:
            template_name: The name of the Jinja2 template to use.
            **kwargs: The context to render the template with.

        Returns
        -------
            A tuple containing the idea, a dictionary of file paths to file contents,
            the number of input tokens, and the number of output tokens.
        """
        template_name = template_name.removesuffix('.jinja').removesuffix('.md')
        prompt_template = jinja_env.get_template(f'{template_name}.md.jinja')

        prompt_content = prompt_template.render(**kwargs)
        prompt: ChatCompletionUserMessageParam = {
            'role': 'user',
            'content': prompt_content,
        }

        prompt_tokens = self._count_tokens(prompt_content)
        self._prompt_tokens_total += prompt_tokens

        response = await self._async_openai.chat.completions.create(
                messages=[prompt],
                model=self._model_name,
                temperature=self._temperature,
                top_p=self._top_p,
        )
        output = response.choices[0].message.content
        if output is None:
            raise ModelOutputParseError('language model returned empty content')

        completion_tokens = self._count_tokens(output)
        self._completion_tokens_total += completion_tokens

        print(output)
        assert output

        idea_match = re.search(r'<idea>(.*?)</idea>', output, re.DOTALL)
        if not idea_match:
            raise ModelOutputParseError('failed to find <idea>...</idea> section')
        idea = idea_match.group(1).strip()

        files_match = re.search(r'<files>(.*?)</files>', output, re.DOTALL)
        if not files_match:
            raise ModelOutputParseError('failed to find <files>...</files> section')
        files_content = files_match.group(1)

        files_dict = self._parse_xml_files(files_content)

        return idea, files_dict, prompt_tokens, completion_tokens