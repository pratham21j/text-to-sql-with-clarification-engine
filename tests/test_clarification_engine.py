import json
import sqlite3
from types import SimpleNamespace

import pytest
from google.genai import errors

from text_to_sql.clarification_engine import (
    ClarificationRequiredError,
    ClarificationResponseError,
    ClarificationResult,
    analyze_question,
    generate_sql_after_clarification,
    submit_clarification_answer,
)
from text_to_sql.database import initialize_database
from text_to_sql.llm_provider import GeminiProvider, GeminiProviderError
from text_to_sql.sql_generator import SQLGenerationResult


class _FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses) if isinstance(responses, list) else [responses]
        self.calls = []

    def generate_json(self, prompt, response_schema):
        self.calls.append((prompt, response_schema))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _database(tmp_path):
    return initialize_database(tmp_path / "academic.db")


def _response(
    status,
    *,
    clarification_needed,
    clarification_question=None,
    clarification_reason=None,
    resolved_question=None,
):
    return json.dumps(
        {
            "status": status,
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
            "clarification_reason": clarification_reason,
            "resolved_question": resolved_question,
        }
    )


def _clarification_needed_response(
    question="Which department should be used?",
    reason="A department was mentioned without identifying which one.",
    status="clarification_needed",
):
    return _response(
        status,
        clarification_needed=True,
        clarification_question=question,
        clarification_reason=reason,
    )


def test_clear_supported_question_does_not_need_clarification(tmp_path):
    database_path = _database(tmp_path)
    provider = _FakeProvider(
        _response(
            "clear",
            clarification_needed=False,
            resolved_question="Show students in the CSE department.",
        )
    )

    result = analyze_question(
        "Show all CSE students.",
        database_path,
        provider=provider,
    )

    assert result.status == "clear"
    assert result.clarification_needed is False
    assert result.clarification_question is None
    assert result.original_question == "Show all CSE students."
    assert result.resolved_question == "Show students in the CSE department."
    assert '"cgpa"' in provider.calls[0][0]
    assert '"marks"' in provider.calls[0][0]
    assert '"attendance"' not in provider.calls[0][0]


@pytest.mark.parametrize(
    ("question", "clarification_question", "reason"),
    [
        (
            "Show students with good marks.",
            "What marks threshold or grade should count as good?",
            "The phrase 'good marks' does not define a threshold.",
        ),
        (
            "Show students from a department.",
            "Which department should the students belong to?",
            "The department filter does not identify a department.",
        ),
        (
            "Show the best students.",
            "Should students be ranked by CGPA or examination marks?",
            "The ranking criterion is unspecified.",
        ),
        (
            "Show their courses.",
            "Which students does 'their' refer to?",
            "The reference has no subject in this request.",
        ),
    ],
)
def test_ambiguous_or_incomplete_questions_request_targeted_clarification(
    tmp_path, question, clarification_question, reason
):
    provider = _FakeProvider(
        _clarification_needed_response(clarification_question, reason)
    )

    result = analyze_question(
        question,
        _database(tmp_path),
        provider=provider,
    )

    assert result.status == "clarification_needed"
    assert result.clarification_needed is True
    assert result.clarification_question == clarification_question
    assert result.clarification_reason == reason
    assert result.resolved_question is None
    assert result.original_question == question


def test_request_for_missing_attendance_field_is_reported_as_unsupported(
    tmp_path,
):
    provider = _FakeProvider(
        _clarification_needed_response(
            "Attendance is not stored. Would you like to use CGPA or examination "
            "marks instead?",
            "The database has no attendance column; it stores cgpa and examination marks.",
            status="unsupported",
        )
    )

    result = analyze_question(
        "Show students with the best attendance.",
        _database(tmp_path),
        provider=provider,
    )

    assert result.status == "unsupported"
    assert result.clarification_needed is True
    assert "attendance" in result.clarification_reason.lower()
    assert "cgpa" in result.clarification_reason.lower()
    assert result.resolved_question is None
    assert '"attendance"' not in provider.calls[0][0]


def test_valid_answer_resolves_request_and_preserves_serializable_context(
    tmp_path,
):
    database_path = _database(tmp_path)
    provider = _FakeProvider(
        _response(
            "clear",
            clarification_needed=False,
            resolved_question=(
                "Show students whose examination marks are at least 80."
            ),
        )
    )
    original = "Show students with good marks."
    clarification = "What marks threshold or grade should count as good?"
    answer = "At least 80 out of 100."

    result = submit_clarification_answer(
        original,
        clarification,
        answer,
        database_path,
        provider=provider,
    )

    assert result.status == "clear"
    assert result.original_question == original
    assert result.clarification_question is None
    assert result.clarification_answer == answer
    assert result.resolved_question.endswith("at least 80.")
    assert len(result.clarification_history) == 1
    assert result.clarification_history[0].clarification_question == clarification
    assert result.clarification_history[0].clarification_answer == answer
    assert clarification in provider.calls[0][0]
    assert answer in provider.calls[0][0]

    serialized = json.dumps(result.to_dict())
    restored = ClarificationResult.from_dict(json.loads(serialized))
    assert restored == result
    assert ClarificationResult.from_dict(result.to_dict()) == result


def test_answer_that_does_not_resolve_ambiguity_requests_a_follow_up(tmp_path):
    database_path = _database(tmp_path)
    first_provider = _FakeProvider(_clarification_needed_response())
    first_result = analyze_question(
        "Show students with good marks.",
        database_path,
        provider=first_provider,
    )
    follow_up_question = "Which department should be included?"
    follow_up_answer = "Use a high threshold."
    next_provider = _FakeProvider(
        _clarification_needed_response(
            "What numeric examination mark should count as high?",
            "The answer does not provide a numeric marks threshold.",
        )
    )

    next_result = submit_clarification_answer(
        first_result.original_question,
        follow_up_question,
        follow_up_answer,
        database_path,
        provider=next_provider,
        clarification_history=first_result.clarification_history,
    )

    assert next_result.status == "clarification_needed"
    assert next_result.clarification_question == (
        "What numeric examination mark should count as high?"
    )
    assert next_result.original_question == first_result.original_question
    assert next_result.clarification_answer == follow_up_answer
    assert len(next_result.clarification_history) == 1
    assert follow_up_question in next_provider.calls[0][0]
    assert follow_up_answer in next_provider.calls[0][0]
    assert first_result.original_question in next_provider.calls[0][0]

    final_provider = _FakeProvider(
        _response(
            "clear",
            clarification_needed=False,
            resolved_question=(
                "Show students in the CSE department with examination marks "
                "at least 80."
            ),
        )
    )
    final_result = submit_clarification_answer(
        next_result.original_question,
        next_result.clarification_question,
        "At least 80 marks, from CSE.",
        database_path,
        provider=final_provider,
        clarification_history=next_result.clarification_history,
    )
    assert final_result.status == "clear"
    assert len(final_result.clarification_history) == 2
    assert final_result.clarification_history[0].clarification_answer == (
        "Use a high threshold."
    )
    assert final_result.clarification_history[1].clarification_answer == (
        "At least 80 marks, from CSE."
    )
    assert first_result.original_question in final_provider.calls[0][0]
    assert "Use a high threshold." in final_provider.calls[0][0]


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"status":"clear"}',
        _response(
            "unknown",
            clarification_needed=False,
            resolved_question="Some query",
        ),
        _response(
            "clear",
            clarification_needed=True,
            resolved_question="Show students.",
        ),
        _response(
            "clear",
            clarification_needed=False,
            resolved_question=None,
        ),
        _response(
            "clarification_needed",
            clarification_needed=True,
            clarification_question=" ",
            clarification_reason="Not clear",
        ),
        _response(
            "unsupported",
            clarification_needed=False,
            clarification_question="Choose a field",
            clarification_reason="Field is missing",
        ),
    ],
)
def test_malformed_or_inconsistent_model_responses_are_rejected(
    tmp_path, response
):
    with pytest.raises(ClarificationResponseError):
        analyze_question(
            "Show students.",
            _database(tmp_path),
            provider=_FakeProvider(response),
        )


def test_gemini_provider_errors_propagate_without_fallback(tmp_path):
    provider_error = GeminiProviderError(
        "Gemini request failed. Check the API configuration, quota, and network connection."
    )

    with pytest.raises(GeminiProviderError, match="request failed"):
        analyze_question(
            "Show CSE students.",
            _database(tmp_path),
            provider=_FakeProvider(provider_error),
        )


def test_api_error_is_not_converted_into_a_clear_result(tmp_path):
    provider = _FakeProvider(
        GeminiProviderError("Gemini request failed.")
    )

    with pytest.raises(GeminiProviderError):
        analyze_question(
            "Show students.",
            _database(tmp_path),
            provider=provider,
        )


def test_gemini_api_error_is_wrapped_and_propagates_from_clarification(
    tmp_path, monkeypatch
):
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "GEMINI_API_KEY=test-only-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    class _Models:
        def generate_content(self, **kwargs):
            del kwargs
            raise errors.APIError(
                429,
                {"error": {"message": "private diagnostic and secret"}},
            )

    provider = GeminiProvider(
        client=SimpleNamespace(models=_Models()),
        env_file=env_file,
    )

    with pytest.raises(GeminiProviderError) as error:
        analyze_question(
            "Show all CSE students.",
            _database(tmp_path),
            provider=provider,
        )

    assert "private diagnostic" not in str(error.value)
    assert "secret" not in str(error.value)


def test_sql_generation_is_blocked_until_clarification_is_resolved(tmp_path):
    pending = analyze_question(
        "Show the best students.",
        _database(tmp_path),
        provider=_FakeProvider(_clarification_needed_response()),
    )
    sql_provider = _FakeProvider(
        json.dumps(
            {
                "status": "ready",
                "generated_sql": "SELECT student_id FROM students",
                "ambiguity_reason": None,
                "clarification_needed": False,
            }
        )
    )

    with pytest.raises(ClarificationRequiredError):
        generate_sql_after_clarification(
            pending,
            database_path=tmp_path / "database-must-not-be-opened.db",
            provider=sql_provider,
        )

    assert sql_provider.calls == []


def test_resolved_clarification_integrates_with_sql_generator_and_keeps_intent(
    tmp_path,
):
    database_path = _database(tmp_path)
    result = submit_clarification_answer(
        "Show students with good marks.",
        "What numeric marks threshold should count as good?",
        "At least 80.",
        database_path,
        provider=_FakeProvider(
            _response(
                "clear",
                clarification_needed=False,
                resolved_question=(
                    "Show students with examination marks of at least 80."
                ),
            )
        ),
    )
    sql_provider = _FakeProvider(
        json.dumps(
            {
                "status": "ready",
                "generated_sql": (
                    "SELECT student_id FROM examinations WHERE marks >= 80"
                ),
                "ambiguity_reason": None,
                "clarification_needed": False,
            }
        )
    )

    sql_result = generate_sql_after_clarification(
        result,
        database_path,
        provider=sql_provider,
    )

    assert isinstance(sql_result, SQLGenerationResult)
    assert sql_result.status == "ready"
    assert sql_result.generated_sql == (
        "SELECT student_id FROM examinations WHERE marks >= 80"
    )
    prompt = sql_provider.calls[0][0]
    assert "Original request: Show students with good marks." in prompt
    assert "User answer: At least 80." in prompt
    assert "Resolved request: Show students with examination marks of at least 80." in prompt


def test_invalid_follow_up_inputs_do_not_call_provider(tmp_path):
    provider = _FakeProvider(_response(
        "clear",
        clarification_needed=False,
        resolved_question="Show students.",
    ))

    with pytest.raises(ValueError, match="Clarification answer"):
        submit_clarification_answer(
            "Show students.",
            "Which department?",
            " ",
            _database(tmp_path),
            provider=provider,
        )

    assert provider.calls == []


def test_database_schema_used_for_clarification_is_read_only(tmp_path):
    database_path = _database(tmp_path)
    with sqlite3.connect(database_path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    analyze_question(
        "Show all CSE students.",
        database_path,
        provider=_FakeProvider(
            _response(
                "clear",
                clarification_needed=False,
                resolved_question="Show all CSE students.",
            )
        ),
    )
    with sqlite3.connect(database_path) as connection:
        after = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    assert after == before
