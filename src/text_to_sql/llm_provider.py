"""LLM provider configuration and structured response generation."""

import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from text_to_sql.app_paths import environment_file_path

DEFAULT_ENV_FILE = environment_file_path()
MAX_API_ATTEMPTS = 3
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 5.0


class ProviderConfigurationError(RuntimeError):
    """Raised when provider selection or configuration is invalid."""


class GeminiConfigurationError(ProviderConfigurationError):
    """Raised when the Gemini API key or model is not configured."""


class GeminiProviderError(RuntimeError):
    """Raised when Gemini cannot provide a usable response."""

    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        self.diagnostic = diagnostic
        detail = f" Diagnostic: {diagnostic}" if diagnostic else ""
        super().__init__(f"{message}{detail}")


class GeminiRateLimitError(GeminiProviderError):
    """Raised when Gemini rejects a request due to a rate or quota limit."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        quota_exhausted: bool = False,
        diagnostic: str | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.quota_exhausted = quota_exhausted
        super().__init__(message, diagnostic=diagnostic)


class GeminiQuotaError(GeminiRateLimitError):
    """Raised when Gemini explicitly reports exhausted quota."""


class OpenAIConfigurationError(ProviderConfigurationError):
    """Raised when OpenAI configuration is missing or invalid."""


class OpenAIProviderError(RuntimeError):
    """Raised when OpenAI cannot provide a usable response."""

    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        self.diagnostic = diagnostic
        detail = f" Diagnostic: {diagnostic}" if diagnostic else ""
        super().__init__(f"{message}{detail}")


class OpenAIRateLimitError(OpenAIProviderError):
    """Raised when OpenAI rejects a request due to a rate limit."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        diagnostic: str | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message, diagnostic=diagnostic)


class OpenAIQuotaError(OpenAIRateLimitError):
    """Raised when OpenAI reports an exhausted quota or billing limit."""


class OpenRouterConfigurationError(ProviderConfigurationError):
    """Raised when OpenRouter configuration is missing or invalid."""


class OpenRouterProviderError(RuntimeError):
    """Raised when OpenRouter cannot provide a usable response."""

    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        self.diagnostic = diagnostic
        detail = f" Diagnostic: {diagnostic}" if diagnostic else ""
        super().__init__(f"{message}{detail}")


class OpenRouterRateLimitError(OpenRouterProviderError):
    """Raised when OpenRouter rejects a request due to a rate or usage limit."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        diagnostic: str | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message, diagnostic=diagnostic)


class OpenRouterQuotaError(OpenRouterRateLimitError):
    """Raised when OpenRouter reports exhausted credits or a free-tier quota."""


class OpenRouterStructuredOutputError(OpenRouterProviderError):
    """Raised when the selected model endpoint does not support JSON Schema."""


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


class _ResponsesClient(Protocol):
    def create(
        self,
        *,
        model: str,
        input: str,
        text: dict[str, object],
    ) -> object: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesClient


class _ChatCompletionsClient(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        extra_body: dict[str, object],
    ) -> object: ...


class _ChatClient(Protocol):
    completions: _ChatCompletionsClient


class _OpenRouterClient(Protocol):
    chat: _ChatClient


class GeminiProvider:
    """Call the configured Gemini model and request a JSON response."""

    def __init__(
        self,
        *,
        client: _GeminiClient | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        configuration_path = (
            Path(env_file) if env_file is not None else environment_file_path()
        )
        load_dotenv(dotenv_path=configuration_path, override=False)

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

        self._api_key = api_key
        self._model = model
        self._client = client if client is not None else genai.Client(api_key=api_key)

    def generate_json(
        self,
        prompt: str,
        response_schema: dict[str, object],
    ) -> str:
        """Request JSON conforming to the supplied response schema."""
        for attempt in range(MAX_API_ATTEMPTS):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
            except errors.APIError as error:
                retry_after = _retry_after_seconds(error)
                if _is_quota_exhausted(error):
                    raise GeminiQuotaError(
                        _quota_message(retry_after),
                        retry_after_seconds=retry_after,
                        quota_exhausted=True,
                        diagnostic=_format_api_error(error, self._api_key),
                    ) from error
                delay = _retry_delay(error, attempt)
                if delay is not None and attempt + 1 < MAX_API_ATTEMPTS:
                    time.sleep(delay)
                    continue
                if getattr(error, "code", None) == 429:
                    raise GeminiRateLimitError(
                        _rate_limit_message(retry_after),
                        retry_after_seconds=retry_after,
                        diagnostic=_format_api_error(error, self._api_key),
                    ) from error
                raise GeminiProviderError(
                    "Gemini request failed.",
                    diagnostic=_format_api_error(error, self._api_key),
                ) from error
            except Exception as error:
                if type(error).__module__.split(".", 1)[0] != "httpx":
                    raise
                raise GeminiProviderError(
                    "Gemini network request failed.",
                    diagnostic=_redact_diagnostic(
                        f"{type(error).__name__}: {error}",
                        self._api_key,
                    ),
                ) from error
            break

        if not response.text:
            raise GeminiProviderError("Gemini returned an empty response.")
        return response.text


class OpenAIProvider:
    """Call the configured OpenAI model and request a JSON-schema response."""

    def __init__(
        self,
        *,
        client: _OpenAIClient | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        configuration_path = (
            Path(env_file) if env_file is not None else environment_file_path()
        )
        load_dotenv(dotenv_path=configuration_path, override=False)

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("AI_MODEL", "").strip()
        if not api_key:
            raise OpenAIConfigurationError(
                "OPENAI_API_KEY is missing. Set it in the project .env file "
                "or environment."
            )
        if not model:
            raise OpenAIConfigurationError(
                "AI_MODEL is missing. Set it in the project .env file "
                "or environment."
            )

        self._api_key = api_key
        self._model = model
        self._client = (
            client
            if client is not None
            else OpenAI(api_key=api_key, max_retries=0)
        )

    def generate_json(
        self,
        prompt: str,
        response_schema: dict[str, object],
    ) -> str:
        """Request JSON conforming to the supplied response schema."""
        try:
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "text_to_sql_response",
                        "schema": _openai_json_schema(response_schema),
                        "strict": True,
                    }
                },
            )
        except AuthenticationError as error:
            raise OpenAIProviderError(
                "OpenAI authentication failed (HTTP 401). Verify OPENAI_API_KEY "
                "and the API project's access.",
                diagnostic=_format_openai_error(error, self._api_key),
            ) from error
        except RateLimitError as error:
            retry_after = _openai_retry_after_seconds(error)
            diagnostic = _format_openai_error(error, self._api_key)
            if _is_openai_quota_exhausted(error):
                raise OpenAIQuotaError(
                    _openai_quota_message(),
                    retry_after_seconds=retry_after,
                    diagnostic=diagnostic,
                ) from error
            raise OpenAIRateLimitError(
                _openai_rate_limit_message(retry_after),
                retry_after_seconds=retry_after,
                diagnostic=diagnostic,
            ) from error
        except APIConnectionError as error:
            raise OpenAIProviderError(
                "OpenAI network request failed. Check the network connection.",
                diagnostic=_redact_openai_diagnostic(
                    f"{type(error).__name__}: {error}",
                    self._api_key,
                ),
            ) from error
        except APIStatusError as error:
            status_code = error.status_code
            diagnostic = _format_openai_error(error, self._api_key)
            if status_code == 404:
                message = (
                    "OpenAI model is unavailable (HTTP 404). Verify AI_MODEL "
                    "and the account's model access."
                )
            elif isinstance(status_code, int) and status_code >= 500:
                message = (
                    f"OpenAI is temporarily unavailable (HTTP {status_code}). "
                    "No automatic retry was made."
                )
            else:
                message = (
                    f"OpenAI request failed (HTTP {status_code}). Check the "
                    "configured model and request."
                )
            raise OpenAIProviderError(
                message,
                diagnostic=diagnostic,
            ) from error
        except OpenAIError as error:
            raise OpenAIProviderError(
                "OpenAI request failed.",
                diagnostic=_redact_openai_diagnostic(
                    f"{type(error).__name__}: {error}",
                    self._api_key,
                ),
            ) from error

        response_text = getattr(response, "output_text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise OpenAIProviderError("OpenAI returned an empty response.")
        return response_text


class OpenRouterProvider:
    """Request JSON-schema-constrained chat output from OpenRouter."""

    def __init__(
        self,
        *,
        client: _OpenRouterClient | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        configuration_path = (
            Path(env_file) if env_file is not None else environment_file_path()
        )
        load_dotenv(dotenv_path=configuration_path, override=False)

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        model = os.environ.get("AI_MODEL", "").strip()
        if not api_key:
            raise OpenRouterConfigurationError(
                "OPENROUTER_API_KEY is missing. Set it in the project .env file "
                "or environment."
            )
        if not model:
            raise OpenRouterConfigurationError(
                "AI_MODEL is missing. Set it in the project .env file "
                "or environment."
            )

        self._api_key = api_key
        self._model = model
        self._client = (
            client
            if client is not None
            else OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                max_retries=0,
            )
        )

    def generate_json(
        self,
        prompt: str,
        response_schema: dict[str, object],
    ) -> str:
        """Return the JSON text from a JSON-schema-constrained completion."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "text_to_sql_response",
                        "strict": True,
                        "schema": _openai_json_schema(response_schema),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )
        except AuthenticationError as error:
            raise OpenRouterProviderError(
                "OpenRouter authentication failed (HTTP 401). Verify "
                "OPENROUTER_API_KEY.",
                diagnostic=_format_openrouter_error(
                    error,
                    self._api_key,
                    requested_model=self._model,
                ),
            ) from error
        except RateLimitError as error:
            retry_after = _openai_retry_after_seconds(error)
            diagnostic = _format_openrouter_error(
                error,
                self._api_key,
                requested_model=self._model,
            )
            if _is_openrouter_quota_exhausted(error):
                raise OpenRouterQuotaError(
                    _openrouter_quota_message(),
                    retry_after_seconds=retry_after,
                    diagnostic=diagnostic,
                ) from error
            raise OpenRouterRateLimitError(
                _openrouter_rate_limit_message(retry_after),
                retry_after_seconds=retry_after,
                diagnostic=diagnostic,
            ) from error
        except APIConnectionError as error:
            raise OpenRouterProviderError(
                "OpenRouter network request failed. Check the network connection.",
                diagnostic=_redact_openai_diagnostic(
                    f"{type(error).__name__}: {error}",
                    self._api_key,
                ),
            ) from error
        except APIStatusError as error:
            status_code = error.status_code
            diagnostic = _format_openrouter_error(
                error,
                self._api_key,
                requested_model=self._model,
            )
            if _is_unsupported_structured_output(error, diagnostic):
                raise OpenRouterStructuredOutputError(
                    "The selected OpenRouter model endpoint does not support "
                    "the required JSON Schema output. Choose a model and "
                    "provider endpoint that advertise structured_outputs.",
                    diagnostic=diagnostic,
                ) from error
            if status_code == 404:
                message = (
                    "OpenRouter model or provider endpoint is unavailable "
                    "(HTTP 404). Check AI_MODEL and current model availability."
                )
            elif status_code == 402:
                message = (
                    "OpenRouter request requires credits or account access "
                    "(HTTP 402). Check account credits and model access."
                )
            elif isinstance(status_code, int) and status_code >= 500:
                message = (
                    f"OpenRouter is temporarily unavailable (HTTP {status_code}). "
                    "No automatic retry was made."
                )
            else:
                message = (
                    f"OpenRouter request failed (HTTP {status_code}). Check "
                    "AI_MODEL and the requested parameters."
                )
            raise OpenRouterProviderError(
                message,
                diagnostic=diagnostic,
            ) from error
        except APIResponseValidationError as error:
            response = getattr(error, "body", None)
            http_status = getattr(
                getattr(error, "response", None),
                "status_code",
                None,
            )
            diagnostic = _format_openrouter_response_diagnostic(
                response,
                self._api_key,
                self._model,
                http_status=http_status,
            )
            raise OpenRouterProviderError(
                "OpenRouter returned a response that does not match the "
                "expected Chat Completions format.",
                diagnostic=diagnostic,
            ) from error
        except OpenAIError as error:
            raise OpenRouterProviderError(
                "OpenRouter request failed.",
                diagnostic=_redact_openai_diagnostic(
                    f"{type(error).__name__}: {error}",
                    self._api_key,
                ),
            ) from error

        choices = _openrouter_field(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            diagnostic = _format_openrouter_response_diagnostic(
                response,
                self._api_key,
                self._model,
                http_status=200,
            )
            raise OpenRouterProviderError(
                "OpenRouter returned an empty completion (no response choices).",
                diagnostic=diagnostic,
            )

        choice = choices[0]
        diagnostic = _format_openrouter_response_diagnostic(
            response,
            self._api_key,
            self._model,
            http_status=200,
            choice=choice,
        )
        choice_error = _openrouter_field(choice, "error")
        if choice_error is not None:
            raise OpenRouterProviderError(
                "OpenRouter reported an upstream completion error.",
                diagnostic=diagnostic,
            )

        finish_reason = _openrouter_field(choice, "finish_reason")
        if finish_reason == "error":
            raise OpenRouterProviderError(
                "OpenRouter reported an upstream completion error "
                "(finish reason: error).",
                diagnostic=diagnostic,
            )
        if finish_reason == "length":
            raise OpenRouterProviderError(
                "OpenRouter completion was incomplete because the output "
                "reached its length limit.",
                diagnostic=diagnostic,
            )
        if finish_reason == "content_filter":
            raise OpenRouterProviderError(
                "OpenRouter completion was stopped by a content filter.",
                diagnostic=diagnostic,
            )

        message = _openrouter_field(choice, "message")
        if message is None:
            raise OpenRouterProviderError(
                "OpenRouter returned a choice without an assistant message.",
                diagnostic=diagnostic,
            )
        refusal = _openrouter_field(message, "refusal")
        if refusal:
            raise OpenRouterProviderError(
                "OpenRouter model refused to produce the required structured "
                "response.",
                diagnostic=diagnostic,
            )
        content = _openrouter_field(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise OpenRouterProviderError(
                "OpenRouter returned missing, empty, or non-text message "
                "content.",
                diagnostic=diagnostic,
            )
        return content


def create_provider(
    *,
    env_file: str | Path | None = None,
) -> GeminiProvider | OpenAIProvider | OpenRouterProvider:
    """Create the explicitly selected provider; OpenAI is the default."""
    configuration_path = (
        Path(env_file) if env_file is not None else environment_file_path()
    )
    load_dotenv(dotenv_path=configuration_path, override=False)
    selected_provider = os.environ.get("AI_PROVIDER", "openai").strip().casefold()
    if selected_provider == "openai":
        return OpenAIProvider(env_file=configuration_path)
    if selected_provider == "openrouter":
        return OpenRouterProvider(env_file=configuration_path)
    if selected_provider == "gemini":
        return GeminiProvider(env_file=configuration_path)
    raise ProviderConfigurationError(
        "AI_PROVIDER must be 'openai', 'openrouter', or 'gemini'."
    )


def _openai_json_schema(response_schema: dict[str, object]) -> dict[str, object]:
    """Translate the shared JSON schema to OpenAI strict-schema form."""
    type_names = {
        "OBJECT": "object",
        "STRING": "string",
        "BOOLEAN": "boolean",
        "INTEGER": "integer",
        "NUMBER": "number",
        "ARRAY": "array",
    }

    def convert(value: object) -> object:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value

        converted: dict[str, object] = {}
        for key, item in value.items():
            if key in {"nullable", "propertyOrdering"}:
                continue
            if key == "type" and isinstance(item, str):
                item = type_names.get(item, item)
                if value.get("nullable") is True:
                    item = [item, "null"]
            converted[key] = convert(item)

        if converted.get("type") == "object":
            converted["additionalProperties"] = False
        return converted

    converted_schema = convert(response_schema)
    if not isinstance(converted_schema, dict):
        raise ValueError("Response schema must be a JSON object.")
    return converted_schema


def _format_openai_error(error: APIStatusError, api_key: str) -> str:
    """Format OpenAI status details without exposing credentials."""
    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    message = getattr(error, "message", None)
    details = [type(error).__name__]
    if status_code is not None:
        details.append(f"HTTP {status_code}")
    if code:
        details.append(str(code))
    if message:
        details.append(str(message))
    return _redact_openai_diagnostic(": ".join(details), api_key)


def _format_openrouter_error(
    error: APIStatusError,
    api_key: str,
    *,
    requested_model: str | None = None,
) -> str:
    """Format OpenRouter's OpenAI-compatible error body without credentials."""
    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    details = [
        type(error).__name__,
        _format_openrouter_response_diagnostic(
            body,
            api_key,
            requested_model,
            http_status=status_code,
        ),
    ]
    sdk_message = _openrouter_field(error, "message")
    if isinstance(sdk_message, str) and sdk_message:
        details.append(
            f"sdk_message={_safe_openrouter_detail(sdk_message, api_key)}"
        )
    return _redact_openai_diagnostic(
        "; ".join(part for part in details if part),
        api_key,
    )


def _openrouter_field(value: object, name: str) -> object | None:
    """Read a field from a mapping, SDK model, or SDK model extra fields."""
    if isinstance(value, dict):
        return value.get(name)

    field_value = getattr(value, name, None)
    if field_value is not None:
        return field_value

    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    return None


def _format_openrouter_error_details(value: object, api_key: str) -> str:
    """Extract only structured error fields, not whole payloads or headers."""
    if value is None:
        return ""
    details = []
    for name in ("code", "type", "message"):
        field_value = _openrouter_field(value, name)
        if isinstance(field_value, (str, int, float)) and not isinstance(
            field_value,
            bool,
        ):
            details.append(
                f"{name}={_safe_openrouter_detail(field_value, api_key)}"
            )

    metadata = _openrouter_field(value, "metadata")
    if isinstance(metadata, dict):
        safe_metadata = []
        for name in (
            "provider_name",
            "provider",
            "upstream_status",
            "upstream_http_status",
            "error_code",
            "error_type",
        ):
            field_value = metadata.get(name)
            if isinstance(field_value, (str, int, float)) and not isinstance(
                field_value,
                bool,
            ):
                safe_metadata.append(
                    f"{name}={_safe_openrouter_detail(field_value, api_key)}"
                )
        if safe_metadata:
            details.append("metadata[" + ", ".join(safe_metadata) + "]")
    return ", ".join(details)


def _safe_openrouter_detail(value: object, api_key: str) -> str:
    """Bound and redact individual provider-supplied diagnostic values."""
    text = " ".join(str(value).split())
    text = _redact_openai_diagnostic(text, api_key)
    if len(text) > 300:
        return text[:297] + "..."
    return text


def _format_openrouter_response_diagnostic(
    response: object,
    api_key: str,
    requested_model: str | None,
    *,
    http_status: int | None = None,
    choice: object | None = None,
) -> str:
    """Build bounded diagnostics from an explicit allowlist of response fields."""
    parts = [
        f"HTTP {http_status}" if http_status is not None else None,
        f"requested_model={requested_model}" if requested_model else None,
    ]
    for field_name, label in (
        ("id", "response_id"),
        ("model", "returned_model"),
    ):
        value = _openrouter_field(response, field_name)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            parts.append(f"{label}={_safe_openrouter_detail(value, api_key)}")

    choices = _openrouter_field(response, "choices")
    if isinstance(choices, (list, tuple)):
        parts.append(f"choices_count={len(choices)}")

    if choice is not None:
        for field_name, label in (
            ("finish_reason", "finish_reason"),
            ("native_finish_reason", "native_finish_reason"),
        ):
            value = _openrouter_field(choice, field_name)
            if isinstance(value, (str, int, float)) and not isinstance(
                value,
                bool,
            ):
                parts.append(
                    f"{label}={_safe_openrouter_detail(value, api_key)}"
                )
        error_details = _format_openrouter_error_details(
            _openrouter_field(choice, "error"),
            api_key,
        )
        if error_details:
            parts.append(f"choice_error={error_details}")

    top_level_error = _openrouter_field(response, "error")
    error_details = _format_openrouter_error_details(top_level_error, api_key)
    if error_details:
        parts.append(f"response_error={error_details}")

    metadata = _openrouter_field(response, "metadata")
    if isinstance(metadata, dict):
        metadata_details = _format_openrouter_error_details(
            {"metadata": metadata},
            api_key,
        )
        if metadata_details:
            parts.append(metadata_details)

    usage = _openrouter_field(response, "usage")
    if usage is not None:
        usage_details = []
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            value = _openrouter_field(usage, field_name)
            if isinstance(value, int) and not isinstance(value, bool):
                usage_details.append(f"{field_name}={value}")
        if usage_details:
            parts.append("usage[" + ", ".join(usage_details) + "]")

    return _redact_openai_diagnostic(
        "; ".join(part for part in parts if part),
        api_key,
    )


def _is_unsupported_structured_output(
    error: APIStatusError,
    diagnostic: str,
) -> bool:
    if error.status_code not in {400, 422}:
        return False
    normalized = diagnostic.casefold().replace("_", " ")
    return any(
        marker in normalized
        for marker in (
            "response format",
            "json schema",
            "structured output",
            "structured outputs",
            "unsupported parameter",
            "parameter not supported",
            "no endpoints found",
        )
    )


def _is_openrouter_quota_exhausted(error: RateLimitError) -> bool:
    body = getattr(error, "body", None)
    details = _format_openrouter_error(error, "")
    diagnostic = f"{details} {body}".casefold()
    return any(
        marker in diagnostic
        for marker in (
            "insufficient_quota",
            "free-models-per-day",
            "free-models-per-min",
            "credits exhausted",
            "quota exceeded",
        )
    )


def _openrouter_quota_message() -> str:
    return (
        "OpenRouter credits or free-model usage quota is exhausted. Check the "
        "account's credits or free-tier limits before trying again."
    )


def _openrouter_rate_limit_message(retry_after_seconds: float | None) -> str:
    message = "OpenRouter rate limit reached or the selected endpoint is busy."
    if retry_after_seconds is not None:
        message += (
            " The provider suggests retrying in approximately "
            f"{_format_retry_delay(retry_after_seconds)}; this is not a guarantee."
        )
    return message + " No automatic retry was made."


def _redact_openai_diagnostic(diagnostic: str, api_key: str) -> str:
    """Remove API credentials and bearer tokens from SDK diagnostics."""
    if api_key:
        diagnostic = diagnostic.replace(api_key, "[REDACTED]")
    diagnostic = re.sub(
        r"(?i)\bsk-[0-9A-Za-z_-]{12,}\b",
        "[REDACTED_API_KEY]",
        diagnostic,
    )
    diagnostic = re.sub(
        r"(?i)([?&](?:key|api_key)=)[^&\s]+",
        r"\1[REDACTED]",
        diagnostic,
    )
    return re.sub(r"(?i)\bBearer\s+\S+", "******", diagnostic)


def _openai_retry_after_seconds(error: APIStatusError) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    return _parse_retry_delay(retry_after) if retry_after is not None else None


def _is_openai_quota_exhausted(error: RateLimitError) -> bool:
    body = getattr(error, "body", None)
    diagnostic = " ".join(
        str(value)
        for value in (
            getattr(error, "code", None),
            getattr(error, "type", None),
            getattr(error, "message", None),
            body,
        )
        if value is not None
    ).casefold()
    return any(
        marker in diagnostic
        for marker in (
            "insufficient_quota",
            "credit_balance_exhausted",
            "organization_spend_limit_exceeded",
            "project_spend_limit_exceeded",
            "organization_usage_limit_exceeded",
            "quota",
        )
    )


def _openai_quota_message() -> str:
    return (
        "OpenAI API quota or billing limit reached. Review API credits and "
        "project or organization usage limits; retrying will not restore access "
        "until the limit is resolved."
    )


def _openai_rate_limit_message(retry_after_seconds: float | None) -> str:
    message = "OpenAI API rate limit reached."
    if retry_after_seconds is not None:
        message += (
            " The provider suggests retrying in approximately "
            f"{_format_retry_delay(retry_after_seconds)}; this is not a guarantee."
        )
    return message + " No automatic retry was made."


def _retry_delay(error: errors.APIError, attempt: int) -> float | None:
    """Return a bounded retry delay only for known transient API responses."""
    status_code = getattr(error, "code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        return None

    if status_code == 429 and _is_quota_exhausted(error):
        return None
    if 500 <= status_code <= 599:
        return min(
            INITIAL_RETRY_DELAY_SECONDS * (2**attempt),
            MAX_RETRY_DELAY_SECONDS,
        )
    if status_code == 429:
        retry_after = _retry_after_seconds(error)
        if retry_after is not None and retry_after <= MAX_RETRY_DELAY_SECONDS:
            return retry_after
    return None


def _is_quota_exhausted(error: errors.APIError) -> bool:
    if getattr(error, "code", None) != 429:
        return False
    details = getattr(error, "details", None)
    diagnostic = " ".join(
        (
            str(getattr(error, "status", "")),
            str(getattr(error, "message", "")),
            _flatten_detail_text(details),
        )
    ).casefold()
    return any(
        marker in diagnostic
        for marker in (
            "quota",
            "daily limit",
            "free tier",
            "free-tier",
        )
    )


def _flatten_detail_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(
            [
                *(str(key) for key in value),
                *(_flatten_detail_text(item) for item in value.values()),
            ]
        )
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_detail_text(item) for item in value)
    return str(value) if value is not None else ""


def _quota_message(retry_after_seconds: float | None) -> str:
    message = "Gemini quota is exhausted."
    if retry_after_seconds is not None:
        message += (
            " The provider suggests retrying in approximately "
            f"{_format_retry_delay(retry_after_seconds)}. "
            "This is an estimate and does not guarantee the quota will reset "
            "at that time."
        )
    message += " No automatic retry was made."
    return message


def _rate_limit_message(retry_after_seconds: float | None) -> str:
    message = "Gemini rate limit reached."
    if retry_after_seconds is not None:
        message += (
            " The provider suggests retrying in approximately "
            f"{_format_retry_delay(retry_after_seconds)}. "
            "This is guidance, not a guarantee that the limit will clear then."
        )
    message += " No further automatic retry was made."
    return message


def _format_retry_delay(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    if rounded_seconds < 60:
        return f"{rounded_seconds} seconds"

    total_minutes = int(round(rounded_seconds / 60))
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remaining_minutes, 60)
    parts = []
    if days:
        parts.append(f"{days} day" + ("s" if days != 1 else ""))
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    return " and ".join(parts)


def _retry_after_seconds(error: errors.APIError) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after is not None:
            parsed = _parse_retry_delay(retry_after)
            if parsed is not None:
                return parsed

    return _find_retry_info(getattr(error, "details", None))


def _find_retry_info(value: object) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).casefold().replace("_", "").replace("-", "")
            if normalized_key in {"retrydelay", "retryafter"}:
                parsed = _parse_retry_delay(item)
                if parsed is not None:
                    return parsed
            nested = _find_retry_info(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_retry_info(item)
            if nested is not None:
                return nested
    return None


def _parse_retry_delay(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value >= 0 else None
    if isinstance(value, dict):
        seconds = value.get("seconds", 0)
        nanos = value.get("nanos", 0)
        if (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and isinstance(nanos, (int, float))
            and not isinstance(nanos, bool)
        ):
            delay = float(seconds) + float(nanos) / 1_000_000_000
            return delay if delay >= 0 else None
        return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    try:
        delay = float(text)
    except ValueError:
        duration = re.fullmatch(r"(\d+(?:\.\d+)?)s", text)
        if duration:
            return float(duration.group(1))
        try:
            retry_at = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    return delay if delay >= 0 else None


def _format_api_error(error: errors.APIError, api_key: str) -> str:
    """Format useful SDK status details without exposing response internals."""
    status_code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    message = getattr(error, "message", None)
    details = [
        type(error).__name__,
        f"HTTP {status_code}" if status_code is not None else None,
        str(status) if status else None,
        str(message) if message else None,
    ]
    diagnostic = ": ".join(part for part in details[:3] if part)
    if message:
        diagnostic = f"{diagnostic}: {message}" if diagnostic else str(message)
    return _redact_diagnostic(diagnostic, api_key)


def _redact_diagnostic(diagnostic: str, api_key: str) -> str:
    """Remove credential-like values from SDK messages before display."""
    if api_key:
        diagnostic = diagnostic.replace(api_key, "[REDACTED]")
    diagnostic = re.sub(
        r"(?i)\bAIza[0-9A-Za-z_-]{20,}\b",
        "[REDACTED_API_KEY]",
        diagnostic,
    )
    diagnostic = re.sub(
        r"(?i)([?&](?:key|api_key)=)[^&\s]+",
        r"\1[REDACTED]",
        diagnostic,
    )
    diagnostic = re.sub(
        r"(?i)\bBearer\s+\S+",
        "Bearer [REDACTED]",
        diagnostic,
    )
    return diagnostic
