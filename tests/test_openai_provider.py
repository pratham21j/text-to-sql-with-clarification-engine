import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, AuthenticationError, RateLimitError

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
    GeminiProvider,
    OpenAIConfigurationError,
    OpenAIProvider,
    OpenAIProviderError,
    OpenAIQuotaError,
    OpenAIRateLimitError,
    ProviderConfigurationError,
    create_provider,
)
from text_to_sql.sql_generator import InvalidModelResponseError, generate_sql


class _FakeResponses:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.arguments = []

    def create(self, **arguments):
        self.arguments.append(arguments)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str) or outcome is None:
            return SimpleNamespace(output_text=outcome)
        return outcome


class _FakeClient:
    def __init__(self, *outcomes):
        self.responses = _FakeResponses(*outcomes)


def _write_env(tmp_path, monkeypatch, contents):
    env_file = tmp_path / ".env"
    env_file.write_text(contents, encoding="utf-8")
    for name in (
        "AI_PROVIDER",
        "AI_MODEL",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    return env_file


def _configured_provider(tmp_path, monkeypatch, client):
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        "OPENAI_API_KEY=test-openai-key\nAI_MODEL=gpt-4.1-mini\n",
    )
    return OpenAIProvider(client=client, env_file=env_file)


def _api_error(error_type, status, message, body, headers=None):
    response = httpx.Response(
        status,
        headers=headers,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    return error_type(message, response=response, body=body)


def test_openai_provider_requires_api_key_and_model(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, monkeypatch, "AI_MODEL=gpt-4.1-mini\n")

    with pytest.raises(OpenAIConfigurationError, match="OPENAI_API_KEY"):
        OpenAIProvider(env_file=env_file)

    env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("AI_MODEL", raising=False)
    with pytest.raises(OpenAIConfigurationError, match="AI_MODEL"):
        OpenAIProvider(env_file=env_file)


def test_openai_is_default_provider_and_reads_model_from_environment(
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
        "OPENAI_API_KEY=test-openai-key\nAI_MODEL=gpt-4.1-mini\n",
    )
    monkeypatch.setattr(llm_provider, "OpenAI", create_client)

    provider = create_provider(env_file=env_file)

    assert isinstance(provider, OpenAIProvider)
    assert provider.generate_json("prompt", {"type": "OBJECT"}) == (
        '{"result":"ok"}'
    )
    assert client.responses.arguments[0]["model"] == "gpt-4.1-mini"
    assert client_arguments == {
        "api_key": "test-openai-key",
        "max_retries": 0,
    }


def test_provider_selection_keeps_gemini_explicit(tmp_path, monkeypatch):
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        "AI_PROVIDER=gemini\n"
        "GEMINI_API_KEY=test-gemini-key\n"
        "GEMINI_MODEL=gemini-test-model\n",
    )

    provider = create_provider(env_file=env_file)

    assert isinstance(provider, GeminiProvider)
    assert provider._model == "gemini-test-model"


def test_provider_selection_rejects_unknown_provider(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, monkeypatch, "AI_PROVIDER=groq\n")

    with pytest.raises(ProviderConfigurationError, match="AI_PROVIDER"):
        create_provider(env_file=env_file)


def test_openai_failure_does_not_fall_back_to_gemini(tmp_path, monkeypatch):
    api_error = _api_error(
        AuthenticationError,
        401,
        "Invalid API key.",
        {"error": {"code": "invalid_api_key"}},
    )
    client = _FakeClient(api_error)
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        "AI_PROVIDER=openai\n"
        "OPENAI_API_KEY=test-openai-key\n"
        "AI_MODEL=gpt-4.1-mini\n",
    )
    monkeypatch.setattr(llm_provider, "OpenAI", lambda **_kwargs: client)

    def forbidden_gemini_fallback(**_kwargs):
        raise AssertionError("Gemini must not be selected as a fallback.")

    monkeypatch.setattr(llm_provider, "GeminiProvider", forbidden_gemini_fallback)
    provider = create_provider(env_file=env_file)

    with pytest.raises(OpenAIProviderError, match="authentication failed"):
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert len(client.responses.arguments) == 1


def test_openai_provider_returns_mocked_json_and_converts_strict_schema(
    tmp_path,
    monkeypatch,
):
    client = _FakeClient('{"result":"ok"}')
    provider = _configured_provider(tmp_path, monkeypatch, client)
    schema = {
        "type": "OBJECT",
        "properties": {
            "result": {"type": "STRING", "nullable": True},
        },
        "required": ["result"],
        "propertyOrdering": ["result"],
    }

    response = provider.generate_json("prompt", schema)

    request = client.responses.arguments[0]
    assert response == '{"result":"ok"}'
    assert request["model"] == "gpt-4.1-mini"
    assert request["input"] == "prompt"
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"] == {
        "type": "object",
        "properties": {
            "result": {"type": ["string", "null"]},
        },
        "required": ["result"],
        "additionalProperties": False,
    }


def test_openai_authentication_error_is_actionable_and_redacts_key(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        AuthenticationError,
        401,
        "Invalid key test-openai-key",
        {"error": {"code": "invalid_api_key"}},
    )
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(api_error),
    )

    with pytest.raises(OpenAIProviderError, match="authentication failed") as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "test-openai-key" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "authentication failed" in format_desktop_error(caught.value)
    assert caught.value.__cause__ is api_error


def test_openai_rate_limit_reports_retry_hint_without_retrying(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        RateLimitError,
        429,
        "Requests are arriving too quickly.",
        {"error": {"code": "rate_limit_exceeded"}},
        {"Retry-After": "2"},
    )
    client = _FakeClient(api_error)
    provider = _configured_provider(tmp_path, monkeypatch, client)

    with pytest.raises(OpenAIRateLimitError, match="rate limit") as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert caught.value.retry_after_seconds == 2
    assert "approximately 2 seconds" in str(caught.value)
    assert len(client.responses.arguments) == 1


def test_openai_quota_errors_explain_that_retrying_will_not_help(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        RateLimitError,
        429,
        "Quota exhausted.",
        {"error": {"code": "insufficient_quota"}},
    )
    client = _FakeClient(api_error)
    provider = _configured_provider(tmp_path, monkeypatch, client)

    with pytest.raises(OpenAIQuotaError, match="quota or billing limit") as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "retrying will not restore access" in str(caught.value)
    assert len(client.responses.arguments) == 1
    assert "quota or billing limit" in format_desktop_error(caught.value)


def test_openai_temporary_service_failure_is_reported_without_retry(
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

    with pytest.raises(OpenAIProviderError, match="temporarily unavailable") as caught:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "HTTP 503" in str(caught.value)
    assert len(client.responses.arguments) == 1


def test_openai_unavailable_model_error_points_to_model_access(
    tmp_path,
    monkeypatch,
):
    api_error = _api_error(
        APIStatusError,
        404,
        "Model not found or access denied.",
        {"error": {"code": "model_not_found"}},
    )
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(api_error),
    )

    with pytest.raises(OpenAIProviderError, match="account's model access"):
        provider.generate_json("prompt", {"type": "OBJECT"})


@pytest.mark.parametrize("response", [None, "", "  "])
def test_openai_provider_rejects_empty_responses(tmp_path, monkeypatch, response):
    provider = _configured_provider(
        tmp_path,
        monkeypatch,
        _FakeClient(response),
    )

    with pytest.raises(OpenAIProviderError, match="empty response"):
        provider.generate_json("prompt", {"type": "OBJECT"})


def test_malformed_openai_output_uses_existing_sql_generation_validation(
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


def test_openai_provider_works_with_clarification_and_conversation_context(
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
    prompts = [item["input"] for item in provider._client.responses.arguments]
    assert "Original user question (preserve its intent)" in prompts[1]
    assert "Rank by CGPA." in prompts[1]
    assert "Original request: Show the best students." in prompts[2]
