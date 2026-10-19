"""Gemini client configuration and structured response generation."""

import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class GeminiConfigurationError(RuntimeError):
    """Raised when the Gemini API key or model is not configured."""


class GeminiProviderError(RuntimeError):
    """Raised when Gemini cannot provide a usable response."""


class _ModelsClient(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse: ...


class _GeminiClient(Protocol):
    models: _ModelsClient


class GeminiProvider:
    """Call the configured Gemini model and request a JSON response."""

    def __init__(
        self,
        *,
        client: _GeminiClient | None = None,
        env_file: str | Path = DEFAULT_ENV_FILE,
    ) -> None:
        load_dotenv(dotenv_path=Path(env_file), override=False)

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        model = os.environ.get("GEMINI_MODEL", "").strip()
        if not api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is missing. Set it in the project .env file "
                "or environment."
            )
        if not model:
            raise GeminiConfigurationError(
                "GEMINI_MODEL is missing. Set it in the project .env file "
                "or environment."
            )

        self._model = model
        self._client = client if client is not None else genai.Client(api_key=api_key)

    def generate_json(
        self,
        prompt: str,
        response_schema: dict[str, object],
    ) -> str:
        """Request JSON conforming to the supplied response schema."""
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0,
                ),
            )
        except errors.APIError as error:
            raise GeminiProviderError(
                "Gemini request failed. Check the API configuration, quota, "
                "and network connection."
            ) from error

        if not response.text:
            raise GeminiProviderError("Gemini returned an empty response.")
        return response.text
