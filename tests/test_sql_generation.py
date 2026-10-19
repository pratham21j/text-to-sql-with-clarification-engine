import sqlite3
import json
from types import SimpleNamespace

import pytest
import httpx
from google.genai import errors

from text_to_sql import llm_provider
from text_to_sql.database import initialize_database
from text_to_sql.llm_provider import (
    GeminiConfigurationError,
    GeminiProvider,
    GeminiProviderError,
    GeminiQuotaError,
)
from text_to_sql.sql_generator import (
    InvalidModelResponseError,
    generate_sql,
    get_database_schema,
)
from text_to_sql.sql_validator import SQLValidationError, validate_read_only_sql


class _FakeProvider:
    def __init__(self, response):
        self.response = response
        self.prompt = None
        self.response_schema = None

    def generate_json(self, prompt, response_schema):
        self.prompt = prompt
        self.response_schema = response_schema
        return self.response


class _FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.arguments = None

    def generate_content(self, **arguments):
        self.arguments = arguments
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.response)


class _SequenceModels:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **arguments):
        del arguments
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


class _FakeClient:
    def __init__(self, models):
        self.models = models


def _create_database(tmp_path):
    return initialize_database(tmp_path / "academic.db")


def _ready_response(generated_sql="SELECT student_id FROM students"):
    return json.dumps(
        {
            "status": "ready",
            "generated_sql": generated_sql,
            "ambiguity_reason": None,
            "clarification_needed": False,
        }
    )


def test_schema_is_retrieved_from_sqlite_tables_columns_and_foreign_keys(tmp_path):
    database_path = _create_database(tmp_path)

    schema = get_database_schema(database_path)

    tables = {table.name: table for table in schema.tables}
    assert set(tables) == {
        "courses",
        "departments",
        "enrollments",
        "examinations",
        "students",
    }
    student_columns = {column.name: column for column in tables["students"].columns}
    assert student_columns["student_id"].data_type == "INTEGER"
    assert student_columns["department_id"].data_type == "INTEGER"
    assert student_columns["student_id"].primary_key_position == 1
    assert any(
        relationship.from_table == "students"
        and relationship.from_column == "department_id"
        and relationship.to_table == "departments"
        and relationship.to_column == "department_id"
        for relationship in schema.relationships
    )


def test_generate_sql_uses_live_schema_and_returns_validated_result(tmp_path):
    database_path = _create_database(tmp_path)
    provider = _FakeProvider(_ready_response("SELECT student_id FROM students"))

    result = generate_sql(
        "Show all student IDs.",
        database_path=database_path,
        provider=provider,
    )

    assert result.status == "ready"
    assert result.generated_sql == "SELECT student_id FROM students"
    assert result.ambiguity_reason is None
    assert result.clarification_needed is False
    assert '"departments"' in provider.prompt
    assert '"from_column": "department_id"' in provider.prompt
    assert provider.response_schema["required"] == [
        "status",
        "generated_sql",
        "ambiguity_reason",
        "clarification_needed",
    ]
    assert "Never invent" in provider.prompt
    assert "Never execute SQL" in provider.prompt


def test_generate_sql_returns_clarification_without_sql(tmp_path):
    database_path = _create_database(tmp_path)
    provider = _FakeProvider(
        json.dumps(
            {
                "status": "clarification_needed",
                "generated_sql": None,
                "ambiguity_reason": "Specify how students should be ranked.",
                "clarification_needed": True,
            }
        )
    )

    result = generate_sql(
        "Show the best students.",
        database_path=database_path,
        provider=provider,
    )

    assert result.status == "clarification_needed"
    assert result.generated_sql is None
    assert result.clarification_needed is True


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"status":"ready"}',
        json.dumps(
            {
                "status": "ready",
                "generated_sql": "SELECT 1",
                "ambiguity_reason": None,
                "clarification_needed": "false",
            }
        ),
        json.dumps(
            {
                "status": "ready",
                "generated_sql": "SELECT 1",
                "ambiguity_reason": None,
                "clarification_needed": False,
                "unexpected": "field",
            }
        ),
        json.dumps(
            {
                "status": "clarification_needed",
                "generated_sql": "SELECT 1",
                "ambiguity_reason": "Ambiguous",
                "clarification_needed": True,
            }
        ),
    ],
)
def test_generate_sql_rejects_malformed_or_inconsistent_responses(
    tmp_path, response
):
    database_path = _create_database(tmp_path)

    with pytest.raises(InvalidModelResponseError):
        generate_sql(
            "List students.",
            database_path=database_path,
            provider=_FakeProvider(response),
        )


@pytest.mark.parametrize(
    "generated_sql",
    [
        "SELECT FROM students",
        "DELETE FROM students",
        "DROP TABLE students",
        "VALUES (1)",
        "SELECT student_id FROM students; SELECT student_id FROM courses",
        "WITH removed AS (DELETE FROM students RETURNING *) SELECT * FROM removed",
    ],
)
def test_generate_sql_rejects_invalid_unsafe_and_multiple_statements(
    tmp_path, generated_sql
):
    database_path = _create_database(tmp_path)

    with pytest.raises(SQLValidationError):
        generate_sql(
            "List students.",
            database_path=database_path,
            provider=_FakeProvider(_ready_response(generated_sql)),
        )


@pytest.mark.parametrize(
    "generated_sql",
    [
        "SELECT unknown_column FROM students",
        "SELECT student_id FROM missing_table",
        "SELECT student_id FROM students WHERE student_id = ?",
    ],
)
def test_generate_sql_rejects_unknown_schema_references_and_parameters(
    tmp_path, generated_sql
):
    database_path = _create_database(tmp_path)

    with pytest.raises(SQLValidationError):
        generate_sql(
            "List students.",
            database_path=database_path,
            provider=_FakeProvider(_ready_response(generated_sql)),
        )


def test_read_only_validation_prepares_sql_without_modifying_database(tmp_path):
    database_path = _create_database(tmp_path)
    with sqlite3.connect(database_path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    assert (
        validate_read_only_sql(
            "SELECT student_id FROM students",
            database_path=database_path,
        )
        == "SELECT student_id FROM students"
    )

    with sqlite3.connect(database_path) as connection:
        after = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    assert after == before


def test_generate_sql_rejects_an_empty_question_without_calling_provider(tmp_path):
    provider = _FakeProvider(_ready_response())

    with pytest.raises(ValueError, match="non-empty"):
        generate_sql("  ", database_path=tmp_path / "missing.db", provider=provider)

    assert provider.prompt is None


def test_gemini_provider_requires_api_key_and_model(tmp_path, monkeypatch):
    empty_env_file = tmp_path / "not-configured.env"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        GeminiProvider(env_file=empty_env_file)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(GeminiConfigurationError, match="GEMINI_MODEL"):
        GeminiProvider(env_file=empty_env_file)


def test_gemini_provider_loads_configuration_and_requests_json_schema(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    models = _FakeModels(response='{"result":"ok"}')
    provider = GeminiProvider(
        client=_FakeClient(models),
        env_file=env_file,
    )

    response = provider.generate_json(
        "prompt",
        {"type": "OBJECT", "properties": {"result": {"type": "STRING"}}},
    )

    assert response == '{"result":"ok"}'
    assert models.arguments["model"] == "test-model"
    assert models.arguments["contents"] == "prompt"
    assert models.arguments["config"].response_mime_type == "application/json"
    assert models.arguments["config"].response_schema["type"] == "OBJECT"
    assert models.arguments["config"].tools is None
    assert (
        models.arguments["config"].automatic_function_calling.disable is True
    )


def test_gemini_provider_preserves_api_diagnostic_and_redacts_credentials(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    api_error = errors.APIError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": (
                    "Quota exceeded for key=test-key; "
                    "private provider diagnostic"
                ),
            }
        },
    )
    provider = GeminiProvider(
        client=_FakeClient(_FakeModels(error=api_error)),
        env_file=env_file,
    )

    with pytest.raises(GeminiProviderError, match="HTTP 429") as error:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "RESOURCE_EXHAUSTED" in str(error.value)
    assert "Quota exceeded" in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert "test-key" not in str(error.value)
    assert "private provider diagnostic" in str(error.value)
    assert error.value.__cause__ is api_error


@pytest.mark.parametrize(
    ("status_code", "status", "message", "expected"),
    [
        (401, "UNAUTHENTICATED", "Invalid API credential.", "HTTP 401"),
        (429, "RESOURCE_EXHAUSTED", "Quota exhausted.", "HTTP 429"),
        (404, "NOT_FOUND", "Configured model is unavailable.", "HTTP 404"),
    ],
)
def test_gemini_provider_exposes_safe_api_diagnostics(
    tmp_path,
    monkeypatch,
    status_code,
    status,
    message,
    expected,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    api_error = errors.APIError(
        status_code,
        {"error": {"status": status, "message": message}},
    )
    provider = GeminiProvider(
        client=_FakeClient(_FakeModels(error=api_error)),
        env_file=env_file,
    )

    with pytest.raises(GeminiProviderError) as captured:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert expected in str(captured.value)
    assert status in str(captured.value)
    assert message in str(captured.value)
    assert captured.value.__cause__ is api_error


def test_gemini_provider_reports_network_diagnostics_and_preserves_cause(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    network_error = httpx.ConnectError(
        "Connection failed at https://example.test/?key=test-key"
    )
    provider = GeminiProvider(
        client=_FakeClient(_FakeModels(error=network_error)),
        env_file=env_file,
    )

    with pytest.raises(
        GeminiProviderError,
        match="Gemini network request failed",
    ) as captured:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert "ConnectError" in str(captured.value)
    assert "Connection failed" in str(captured.value)
    assert "test-key" not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)
    assert captured.value.__cause__ is network_error


def test_gemini_provider_retries_transient_server_failures_with_backoff(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    server_errors = [
        errors.APIError(
            503,
            {"error": {"status": "UNAVAILABLE", "message": "High demand."}},
        ),
        errors.APIError(
            503,
            {"error": {"status": "UNAVAILABLE", "message": "Still busy."}},
        ),
    ]
    models = _SequenceModels(*server_errors, '{"ok":true}')
    provider = GeminiProvider(client=_FakeClient(models), env_file=env_file)
    delays = []
    monkeypatch.setattr(llm_provider.time, "sleep", delays.append)

    response = provider.generate_json("prompt", {"type": "OBJECT"})

    assert response == '{"ok":true}'
    assert models.calls == 3
    assert delays == [0.5, 1.0]


def test_gemini_provider_stops_after_bounded_transient_retries(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    api_errors = [
        errors.APIError(
            503,
            {"error": {"status": "UNAVAILABLE", "message": f"failure {index}"}},
        )
        for index in range(3)
    ]
    models = _SequenceModels(*api_errors)
    provider = GeminiProvider(client=_FakeClient(models), env_file=env_file)
    delays = []
    monkeypatch.setattr(llm_provider.time, "sleep", delays.append)

    with pytest.raises(GeminiProviderError, match="failure 2") as captured:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert models.calls == llm_provider.MAX_API_ATTEMPTS == 3
    assert delays == [0.5, 1.0]
    assert captured.value.__cause__ is api_errors[-1]


@pytest.mark.parametrize(
    ("status_code", "status"),
    [(401, "UNAUTHENTICATED"), (404, "NOT_FOUND")],
)
def test_gemini_provider_does_not_retry_permanent_api_failures(
    tmp_path,
    monkeypatch,
    status_code,
    status,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    api_error = errors.APIError(
        status_code,
        {"error": {"status": status, "message": "Permanent failure."}},
    )
    models = _SequenceModels(api_error)
    provider = GeminiProvider(client=_FakeClient(models), env_file=env_file)
    delays = []
    monkeypatch.setattr(llm_provider.time, "sleep", delays.append)

    with pytest.raises(GeminiProviderError):
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert models.calls == 1
    assert delays == []


def test_gemini_provider_retries_rate_limit_only_after_server_retry_hint(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    rate_limit_error = errors.APIError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Too many requests.",
            }
        },
        response=httpx.Response(429, headers={"Retry-After": "2"}),
    )
    models = _SequenceModels(rate_limit_error, '{"ok":true}')
    provider = GeminiProvider(client=_FakeClient(models), env_file=env_file)
    delays = []
    monkeypatch.setattr(llm_provider.time, "sleep", delays.append)

    response = provider.generate_json("prompt", {"type": "OBJECT"})

    assert response == '{"ok":true}'
    assert models.calls == 2
    assert delays == [2.0]


def test_gemini_provider_does_not_retry_rate_limit_without_retry_hint(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    quota_error = errors.APIError(
        429,
        {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exhausted."}},
    )
    models = _SequenceModels(quota_error)
    provider = GeminiProvider(client=_FakeClient(models), env_file=env_file)
    delays = []
    monkeypatch.setattr(llm_provider.time, "sleep", delays.append)

    with pytest.raises(GeminiQuotaError):
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert models.calls == 1
    assert delays == []


def test_gemini_provider_reports_long_quota_retry_hint_without_waiting(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    quota_error = errors.APIError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded.",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "54000s",
                    }
                ],
            }
        },
    )
    models = _SequenceModels(quota_error)
    provider = GeminiProvider(client=_FakeClient(models), env_file=env_file)
    delays = []
    monkeypatch.setattr(llm_provider.time, "sleep", delays.append)

    with pytest.raises(GeminiQuotaError) as captured:
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert models.calls == 1
    assert delays == []
    assert captured.value.retry_after_seconds == 54000
    assert captured.value.quota_exhausted is True
    assert "approximately 15 hours" in str(captured.value)
    assert "does not guarantee the quota will reset" in str(captured.value)
    assert "No automatic retry was made." in str(captured.value)


def test_gemini_provider_does_not_retry_beyond_rate_limit_retry_after_cap(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    rate_limit_error = errors.APIError(
        429,
        {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Rate limited."}},
        response=httpx.Response(429, headers={"Retry-After": "30"}),
    )
    models = _SequenceModels(rate_limit_error)
    provider = GeminiProvider(client=_FakeClient(models), env_file=env_file)
    delays = []
    monkeypatch.setattr(llm_provider.time, "sleep", delays.append)

    with pytest.raises(GeminiProviderError):
        provider.generate_json("prompt", {"type": "OBJECT"})

    assert models.calls == 1
    assert delays == []


def test_gemini_provider_rejects_empty_response(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    provider = GeminiProvider(
        client=_FakeClient(_FakeModels(response=None)),
        env_file=env_file,
    )

    with pytest.raises(GeminiProviderError, match="empty response"):
        provider.generate_json("prompt", {"type": "OBJECT"})


def test_schema_retrieval_does_not_create_missing_database(tmp_path):
    database_path = tmp_path / "missing.db"

    with pytest.raises(sqlite3.OperationalError):
        get_database_schema(database_path)

    assert not database_path.exists()
