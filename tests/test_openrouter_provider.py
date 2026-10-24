import json
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from text_to_sql import llm_provider
from text_to_sql.conversation_context import (
    answer_conversation_clarification,
    create_conversation,
    execute_conversation_query,
    process_user_question,
)
from text_to_sql.database import initialize_database
from text_to_sql.desktop_app import format_desktop_error
from text_to_sql.llm_provider import (
    OpenAIProvider,
    OpenRouterConfigurationError,
    OpenRouterProvider,
    OpenRouterProviderError,
    OpenRouterQuotaError,
    OpenRouterRateLimitError,
    OpenRouterStructuredOutputError,
    ProviderConfigurationError,
    create_provider,
)
from text_to_sql.sql_generator import InvalidModelResponseError, generate_sql


class _FakeCompletions:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.arguments = []

    def create(self, **arguments):
        self.arguments.append(arguments)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str) or outcome is None:
            message = SimpleNamespace(content=outcome, refusal=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )
        return outcome


class _FakeClient:
    def __init__(self, *outcomes):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(*outcomes)
        )


def _write_env(tmp_path, monkeypatch, contents):
    env_file = tmp_path / ".env"
    env_file.write_text(contents, encoding="utf-8")
    for name in (
        "AI_PROVIDER",
        "AI_MODEL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    return env_file


def _configured_provider(tmp_path, monkeypatch, client):
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        "OPENROUTER_API_KEY=test-openrouter-key\n"
        "AI_MODEL=xingchen-agi/xing4.0-29b-a4b:free\n",
    )
    return OpenRouterProvider(client=client, env_file=env_file)


def _api_error(error_type, status, message, body, headers=None):
    response = httpx.Response(
        status,
        headers=headers,
        request=httpx.Request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    )
    return error_type(message, response=response, body=body)


def test_openrouter_provider_requires_api_key_and_model(tmp_path, monkeypatch):
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        "AI_MODEL=example/model:free\n",
    )

    with pytest.raises(OpenRouterConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(env_file=env_file)

    env_file.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("AI_MODEL", raising=False)
    with pytest.raises(OpenRouterConfigurationError, match="AI_MODEL"):
        OpenRouterProvider(env_file=env_file)


def test_create_provider_selects_openrouter_and_uses_configured_model(
    tmp_path,
    monkeypatch,
):
    client = _FakeClient('{"result":"ok"}')
    client_arguments = {}

    def create_client(**arguments):
        client_arguments.update(arguments)
        return client

    env_file = _write_env(
        tmp_path,
        monkeypatch,
        "AI_PROVIDER=openrouter\n"
        "AI_MODEL=example/model:free\n"
        "OPENROUTER_API_KEY=test-openrouter-key\n",
    )
    monkeypatch.setattr(llm_provider, "OpenAI", create_client)

    provider = create_provider(env_file=env_file)

    assert isinstance(provider, OpenRouterProvider)
    assert provider.generate_json("prompt", {"type": "OBJECT"}) == (
        '{"result":"ok"}'
    )
    assert client_arguments == {
        "api_key": "test-openrouter-key",
        "base_url": "https://openrouter.ai/api/v1",
        "max_retries": 0,
    }
    assert client.chat.completions.arguments[0]["model"] == "example/model:free"


def test_openrouter_selection_does_not_change_openai_selection(
    tmp_path,
    monkeypatch,
):
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        "AI_PROVIDER=openai\n"
        "AI_MODEL=gpt-4.1-mini\n"
        "OPENAI_API_KEY=test-openai-key\n",
    )
    provider = create_provider(env_file=env_file)
    assert isinstance(provider, OpenAIProvider)


def test_openrouter_provider_uses_strict_schema_request(tmp_path, monkeypatch):
    client = _FakeClient('{"result":"ok"}')
    provider = _configured_provider(tmp_path, monkeypatch, client)
    schema = {
        "type": "OBJECT",
        "properties": {"result": {"type": "STRING"}},
        "required": ["result"],
    }

    response = provider.generate_json("prompt", schema)

    request = client.chat.completions.arguments[0]
    assert response == '{"result":"ok"}'
    assert request["messages"] == [{"role": "user", "content": "prompt"}]
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "text_to_sql_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    }
    assert request["extra_body"] == {
        "provider": {"require_parameters": True}
    }


def test_openrouter_authentication_error_is_actionable_and_redacts_key(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        AuthenticationError,
        401,
        "Invalid key test-openrouter-key",
        {"error": {"code": "invalid_api_key"}},
    )
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(api_error),
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="authentication failed",
    ) as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "test-openrouter-key" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "authentication failed" in format_desktop_error(caught.value)
    assert caught.value.__cause__ is api_error


def test_openrouter_authentication_diagnostic_includes_requested_model(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        AuthenticationError,
        401,
        "Invalid API key.",
        {"error": {"code": "invalid_api_key"}},
    )
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(api_error),
    )

    with pytest.raises(OpenRouterProviderError) as caught:
        provider.generate_json("private prompt text", {"type": "OBJECT"})

    assert "HTTP 401" in str(caught.value)
    assert "requested_model=xingchen-agi/xing4.0-29b-a4b:free" in str(caught.value)
    assert "private prompt text" not in str(caught.value)


def test_openrouter_rate_limit_reports_retry_hint_without_retrying(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        RateLimitError,
        429,
        "Requests are arriving too quickly.",
        {"error": {"code": "rate_limit_exceeded"}},
        {"Retry-After": "3"},
    )
    client = _FakeClient(api_error)
    provider = _configured_provider(tmp_path, monkeypatch, client)

    with pytest.raises(
        OpenRouterRateLimitError,
        match="rate limit",
    ) as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert caught.value.retry_after_seconds == 3
    assert "approximately 3 seconds" in str(caught.value)
    assert len(client.chat.completions.arguments) == 1


def test_openrouter_free_tier_quota_is_reported_without_retrying(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        RateLimitError,
        429,
        "Free model usage limit reached.",
        {"error": {"code": "free-models-per-day"}},
    )
    client = _FakeClient(api_error)
    provider = _configured_provider(tmp_path, monkeypatch, client)

    with pytest.raises(
        OpenRouterQuotaError,
        match="free-tier limits",
    ):
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert len(client.chat.completions.arguments) == 1


def test_openrouter_model_unavailable_error_identifies_configuration(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        APIStatusError,
        404,
        "No endpoints found for this model.",
        {"error": {"code": "model_not_found"}},
    )
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(api_error),
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="model or provider endpoint is unavailable",
    ):
        provider.generate_json("prompt", {"type": "OBJECT"})


def test_openrouter_unsupported_structured_output_is_actionable(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        APIStatusError,
        400,
        "Provider does not support response_format json_schema.",
        {"error": {"code": "unsupported_parameter"}},
    )
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(api_error),
    )

    with pytest.raises(
        OpenRouterStructuredOutputError,
        match="does not support the required JSON Schema",
    ):
        provider.generate_json("prompt", {"type": "OBJECT"})


def test_openrouter_temporary_service_failure_is_reported_without_retry(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        APIStatusError,
        503,
        "Service unavailable.",
        {"error": {"message": "Service unavailable."}},
    )
    client = _FakeClient(api_error)
    provider = _configured_provider(tmp_path, monkeypatch, client)

    with pytest.raises(
        OpenRouterProviderError,
        match="temporarily unavailable",
    ) as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "HTTP 503" in str(caught.value)
    assert len(client.chat.completions.arguments) == 1


@pytest.mark.parametrize("content", [None, "", "  "])
def test_openrouter_provider_rejects_empty_response(
    tmp_path,
    monkeypatch,
    content,
):
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(content),
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="missing, empty, or non-text message content",
    ) as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "choices_count=1" in str(caught.value)


def test_openrouter_provider_reports_empty_choices_with_safe_metadata(
    tmp_path,
    monkeypatch,
):
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(
            SimpleNamespace(
                id="gen-empty-choice",
                model="example/returned-model",
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=0,
                    total_tokens=12,
                ),
                error={
                    "code": 502,
                    "message": "Upstream failed for test-openrouter-key",
                    "metadata": {
                        "provider_name": "Example Provider",
                        "upstream_status": 502,
                        "raw": "must not be rendered",
                    },
                },
            )
        ),
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="empty completion",
    ) as caught:
        provider.generate_json("private prompt text", {"type": "OBJECT"})

    diagnostic = str(caught.value)
    assert "HTTP 200" in diagnostic
    assert "response_id=gen-empty-choice" in diagnostic
    assert "requested_model=xingchen-agi/xing4.0-29b-a4b:free" in diagnostic
    assert "returned_model=example/returned-model" in diagnostic
    assert "choices_count=0" in diagnostic
    assert "code=502" in diagnostic
    assert (
        "metadata[provider_name=Example Provider, upstream_status=502]"
        in diagnostic
    )
    assert "prompt_tokens=12" in diagnostic
    assert "test-openrouter-key" not in diagnostic
    assert "[REDACTED]" in diagnostic
    assert "must not be rendered" not in diagnostic
    assert "private prompt text" not in diagnostic


def test_openrouter_provider_reports_choice_error_and_provider_metadata(
    tmp_path,
    monkeypatch,
):
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(
            SimpleNamespace(
                id="gen-choice-error",
                model="example/returned-model",
                choices=[
                    SimpleNamespace(
                        finish_reason="error",
                        native_finish_reason="upstream_error",
                        error={
                            "code": 503,
                            "message": "Endpoint temporarily unavailable.",
                            "metadata": {
                                "provider_name": "Example Provider",
                                "upstream_status": 503,
                            },
                        },
                        message=SimpleNamespace(content=None, refusal=None),
                    )
                ],
            )
        ),
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="upstream completion error",
    ) as caught:
        provider.generate_json("private prompt text", {"type": "OBJECT"})

    diagnostic = str(caught.value)
    assert "response_id=gen-choice-error" in diagnostic
    assert "finish_reason=error" in diagnostic
    assert "native_finish_reason=upstream_error" in diagnostic
    assert "code=503" in diagnostic
    assert "provider_name=Example Provider" in diagnostic


def test_openrouter_provider_distinguishes_missing_assistant_message(
    tmp_path,
    monkeypatch,
):
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(
            SimpleNamespace(
                id="gen-no-message",
                model="example/returned-model",
                choices=[SimpleNamespace(message=None, finish_reason="stop")],
            )
        ),
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="without an assistant message",
    ) as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "response_id=gen-no-message" in str(caught.value)


def test_openrouter_provider_distinguishes_refusal_without_echoing_text(
    tmp_path,
    monkeypatch,
):
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(
            SimpleNamespace(
                id="gen-refusal",
                model="example/returned-model",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            refusal="refusal details should not be displayed",
                            content=None,
                        ),
                    )
                ],
            )
        ),
    )

    with pytest.raises(OpenRouterProviderError, match="model refused") as caught:
        provider.generate_json("private prompt text", {"type": "OBJECT"})

    assert "response_id=gen-refusal" in str(caught.value)
    assert "refusal details should not be displayed" not in str(caught.value)
    assert "private prompt text" not in str(caught.value)


@pytest.mark.parametrize(
    ("finish_reason", "message"),
    [
        ("length", "completion was incomplete"),
        ("content_filter", "stopped by a content filter"),
    ],
)
def test_openrouter_provider_classifies_incomplete_or_filtered_completions(
    tmp_path,
    monkeypatch,
    finish_reason,
    message,
):
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(
            SimpleNamespace(
                id="gen-incomplete",
                model="example/returned-model",
                choices=[
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        native_finish_reason="provider_finish",
                        message=SimpleNamespace(
                            refusal=None,
                            content='{"partial":',
                        ),
                    )
                ],
            )
        ),
    )

    with pytest.raises(OpenRouterProviderError, match=message) as caught:
        provider.generate_json("private prompt text", {"type": "OBJECT"})

    assert f"finish_reason={finish_reason}" in str(caught.value)
    assert "provider_finish" in str(caught.value)
    assert '{"partial":' not in str(caught.value)
    assert "private prompt text" not in str(caught.value)


def test_openrouter_provider_reports_sdk_response_shape_errors_safely(
    tmp_path,
    monkeypatch,
):
    response = httpx.Response(
        200,
        request=httpx.Request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    )
    validation_error = APIResponseValidationError(
        response=response,
        body={
            "id": "gen-invalid-shape",
            "model": "example/returned-model",
            "choices": [{"invalid": "shape"}],
        },
    )
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(validation_error),
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="does not match the expected Chat Completions format",
    ) as caught:
        provider.generate_json("private prompt text", {"type": "OBJECT"})

    assert "response_id=gen-invalid-shape" in str(caught.value)
    assert "returned_model=example/returned-model" in str(caught.value)
    assert "private prompt text" not in str(caught.value)


def test_malformed_openrouter_output_uses_existing_sql_generation_validation(
    tmp_path,
    monkeypatch,
):
    database_path = initialize_database(tmp_path / "academic.db")
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient("not JSON"),
    )

    with pytest.raises(InvalidModelResponseError, match="not valid JSON"):
        generate_sql(
            "Show students.",
            database_path,
            provider=provider,
        )


def test_openrouter_provider_works_with_clarification_and_conversation_context(
    tmp_path,
    monkeypatch,
):
    database_path = initialize_database(tmp_path / "academic.db")
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(
            json.dumps(
                {
                    "status": "clarification_needed",
                    "clarification_needed": True,
                    "clarification_question": (
                        "Should students be ranked by CGPA or examination marks?"
                    ),
                    "clarification_reason": "The ranking criterion is ambiguous.",
                    "resolved_question": None,
                    "context_action": "replace",
                }
            ),
            json.dumps(
                {
                    "status": "clear",
                    "clarification_needed": False,
                    "clarification_question": None,
                    "clarification_reason": None,
                    "resolved_question": "Show students ranked by CGPA.",
                    "context_action": "replace",
                }
            ),
            json.dumps(
                {
                    "status": "ready",
                    "generated_sql": (
                        "SELECT student_id, cgpa FROM students "
                        "ORDER BY cgpa DESC LIMIT 2"
                    ),
                    "ambiguity_reason": None,
                    "clarification_needed": False,
                }
            ),
        ),
    )
    conversation = create_conversation()

    pending = process_user_question(
        conversation,
        "Show the best students.",
        database_path,
        provider=provider,
    )
    resolved = answer_conversation_clarification(
        conversation,
        "Rank by CGPA.",
        database_path,
        provider=provider,
    )
    execution = execute_conversation_query(
        conversation,
        database_path,
        provider=provider,
    )

    assert pending.status == "clarification_needed"
    assert resolved.status == "clear"
    assert conversation.current_result.original_question == (
        "Show the best students."
    )
    assert conversation.current_result.clarification_history[0].clarification_answer == (
        "Rank by CGPA."
    )
    assert execution.sql.status == "ready"
    assert execution.execution.status == "success"
    assert execution.execution.row_count == 2
    prompts = [
        item["messages"][0]["content"]
        for item in provider._client.chat.completions.arguments
    ]
    assert "Original user question (preserve its intent)" in prompts[1]
    assert "Rank by CGPA." in prompts[1]
    assert "Original request: Show the best students." in prompts[2]


def test_openrouter_unknown_provider_name_does_not_fall_back(
    tmp_path,
    monkeypatch,
):
    env_file = _write_env(tmp_path, monkeypatch, "AI_PROVIDER=unknown\n")

    with pytest.raises(ProviderConfigurationError, match="AI_PROVIDER"):
        create_provider(env_file=env_file)
