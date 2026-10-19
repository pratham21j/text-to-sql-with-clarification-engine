"""Gemini client configuration and structured response generation."""

import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
MAX_API_ATTEMPTS = 3
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 5.0


class GeminiConfigurationError(RuntimeError):
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
            except httpx.HTTPError as error:
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
