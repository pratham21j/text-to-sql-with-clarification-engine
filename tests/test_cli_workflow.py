import json
import sys

import pytest

from scripts import generate_sql as cli
from text_to_sql.database import initialize_database
from text_to_sql.llm_provider import GeminiProviderError, GeminiQuotaError


def _clarification(
    question="Should the top 5 students be determined by highest CGPA or by "
    "highest average examination marks?",
    reason="The ranking criterion is ambiguous.",
):
    return {
        "status": "clarification_needed",
        "clarification_needed": True,
        "clarification_question": question,
        "clarification_reason": reason,
        "resolved_question": None,
        "context_action": "replace",
    }


def _clear(
    resolved_question="Show the 5 students with the highest CGPA.",
    *,
    context_action="refine",
):
    return {
        "status": "clear",
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_reason": None,
        "resolved_question": resolved_question,
        "context_action": context_action,
    }


def _sql_response():
    return {
        "status": "ready",
        "generated_sql": (
            "SELECT student_id, cgpa FROM students "
            "ORDER BY cgpa DESC LIMIT 5"
        ),
        "ambiguity_reason": None,
        "clarification_needed": False,
    }


class _MockProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []
        self.schemas = []

    def generate_json(self, prompt, response_schema):
        self.prompts.append(prompt)
        self.schemas.append(response_schema)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)


def _run_cli(monkeypatch, tmp_path, provider, answers):
    database_path = initialize_database(tmp_path / "academic.db")
    inputs = iter(answers)
    monkeypatch.setattr(cli, "create_provider", lambda: provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_sql.py",
            "--database",
            str(database_path),
            "Show 5 best students.",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))


def _json_output(output):
    return json.loads(output[output.index("{") :])


def test_cli_original_question_clarification_answer_generates_and_executes(
    monkeypatch,
    tmp_path,
    capsys,
):
    provider = _MockProvider(
        _clarification(),
        _clear(),
        _sql_response(),
    )
    _run_cli(
        monkeypatch,
        tmp_path,
        provider,
        ["highest CGPA"],
    )

    cli.main()

    captured = capsys.readouterr()
    payload = _json_output(captured.out)
    answer_prompt = provider.prompts[1]
    generation_prompt = provider.prompts[2]

    assert captured.out.startswith(
        "Should the top 5 students be determined by highest CGPA or by "
        "highest average examination marks?"
    )
    assert "Original user question (preserve its intent):\nShow 5 best students." in answer_prompt
    assert "User's answer to that question:\nhighest CGPA" in answer_prompt
    assert "Original request: Show 5 best students." in generation_prompt
    assert "User answer: highest CGPA" in generation_prompt
    assert "resolved_question" in provider.schemas[1]["required"]
    assert len(provider.prompts) == 3
    assert payload["clarification"]["original_question"] == "Show 5 best students."
    assert payload["clarification"]["clarification_answer"] == "highest CGPA"
    assert payload["sql"]["generated_sql"].endswith("LIMIT 5")
    assert payload["results"]["status"] == "success"
    assert payload["results"]["row_count"] == 5
    assert payload["conversation"]["clarification_status"] == "clear"
    assert [
        (event["role"], event["kind"], event["content"])
        for event in payload["conversation"]["history"][:3]
    ] == [
        ("user", "question", "Show 5 best students."),
        ("assistant", "clarification", _clarification()["clarification_question"]),
        ("user", "clarification_answer", "highest CGPA"),
    ]
    assert payload["conversation"]["history"][-1]["kind"] == "resolved_request"


def test_cli_additional_clarifications_keep_same_original_request(
    monkeypatch,
    tmp_path,
    capsys,
):
    second_question = "Should ties be included if students share the same CGPA?"
    provider = _MockProvider(
        _clarification(),
        _clarification(
            second_question,
            "Tie handling was not specified.",
        ),
        _clear(),
        _sql_response(),
    )
    _run_cli(
        monkeypatch,
        tmp_path,
        provider,
        ["highest CGPA", "Include ties"],
    )

    cli.main()

    payload = _json_output(capsys.readouterr().out)
    assert len(provider.prompts) == 4
    assert "Show 5 best students." in provider.prompts[1]
    assert "highest CGPA" in provider.prompts[2]
    assert second_question in provider.prompts[2]
    assert "Include ties" in provider.prompts[3]
    assert payload["clarification"]["original_question"] == "Show 5 best students."
    assert payload["clarification"]["clarification_history"] == [
        {
            "clarification_question": (
                "Should the top 5 students be determined by highest CGPA or by "
                "highest average examination marks?"
            ),
            "clarification_answer": "highest CGPA",
        },
        {
            "clarification_question": second_question,
            "clarification_answer": "Include ties",
        },
    ]
    assert payload["results"]["status"] == "success"


def test_cli_retries_failed_clarification_without_losing_context(
    monkeypatch,
    tmp_path,
    capsys,
):
    provider = _MockProvider(
        _clarification(),
        GeminiProviderError("Gemini request failed. Diagnostic: HTTP 503: UNAVAILABLE"),
        _clear(),
        _sql_response(),
    )
    _run_cli(
        monkeypatch,
        tmp_path,
        provider,
        ["highest CGPA", "highest CGPA"],
    )

    cli.main()

    captured = capsys.readouterr()
    payload = _json_output(captured.out)
    assert "Your clarification is retained." in captured.err
    assert len(provider.prompts) == 4
    retry_prompt = provider.prompts[2]
    assert "Show 5 best students." in retry_prompt
    assert "highest CGPA" in retry_prompt
    assert payload["clarification"]["original_question"] == "Show 5 best students."
    answers = [
        event["content"]
        for event in payload["conversation"]["history"]
        if event["kind"] == "clarification_answer"
    ]
    assert answers == ["highest CGPA"]
    assert payload["results"]["status"] == "success"


def test_cli_initial_quota_failure_requires_explicit_retry_and_retains_question(
    monkeypatch,
    tmp_path,
    capsys,
):
    provider = _MockProvider(
        GeminiQuotaError(
            "Gemini quota is exhausted. "
            "The provider suggests retrying in approximately 15 hours. "
            "This is an estimate and does not guarantee the quota will reset. "
            "No automatic retry was made.",
            retry_after_seconds=54000,
            quota_exhausted=True,
        ),
        _clear(
            "Show the 5 students with the highest CGPA.",
            context_action="replace",
        ),
        _sql_response(),
    )
    _run_cli(monkeypatch, tmp_path, provider, ["RETRY"])

    cli.main()

    captured = capsys.readouterr()
    payload = _json_output(captured.out)
    assert "approximately 15 hours" in captured.err
    assert "question analysis input and conversation are retained" in captured.err
    assert len(provider.prompts) == 3
    assert "Show 5 best students." in provider.prompts[1]
    assert payload["clarification"]["original_question"] == (
        "Show 5 best students."
    )
    assert payload["results"]["status"] == "success"


def test_cli_quota_failure_during_clarification_retains_answer_and_context(
    monkeypatch,
    tmp_path,
    capsys,
):
    provider = _MockProvider(
        _clarification(),
        GeminiQuotaError(
            "Gemini quota is exhausted. No automatic retry was made.",
            retry_after_seconds=54000,
            quota_exhausted=True,
        ),
        _clear(),
        _sql_response(),
    )
    _run_cli(
        monkeypatch,
        tmp_path,
        provider,
        ["highest CGPA", "highest CGPA"],
    )

    cli.main()

    captured = capsys.readouterr()
    payload = _json_output(captured.out)
    assert "No automatic retry was made" in captured.err
    assert "Your clarification is retained." in captured.err
    assert len(provider.prompts) == 4
    assert "Show 5 best students." in provider.prompts[2]
    assert "highest CGPA" in provider.prompts[2]
    answers = [
        event["content"]
        for event in payload["conversation"]["history"]
        if event["kind"] == "clarification_answer"
    ]
    assert answers == ["highest CGPA"]
    assert payload["results"]["status"] == "success"


def test_cli_sql_generation_quota_failure_requires_explicit_retry(
    monkeypatch,
    tmp_path,
    capsys,
):
    provider = _MockProvider(
        _clear("Show all CSE students.", context_action="replace"),
        GeminiQuotaError(
            "Gemini quota is exhausted. No automatic retry was made.",
            retry_after_seconds=54000,
            quota_exhausted=True,
        ),
        _sql_response(),
    )
    _run_cli(monkeypatch, tmp_path, provider, ["RETRY"])

    cli.main()

    captured = capsys.readouterr()
    payload = _json_output(captured.out)
    assert "SQL generation input and conversation are retained." in captured.err
    assert len(provider.prompts) == 3
    assert payload["results"]["status"] == "success"


def test_cli_does_not_generate_or_execute_sql_if_clarification_is_unresolved(
    monkeypatch,
    tmp_path,
    capsys,
):
    provider = _MockProvider(_clarification())
    _run_cli(
        monkeypatch,
        tmp_path,
        provider,
        [],
    )
    def end_input(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)

    with pytest.raises(SystemExit):
        cli.main()

    assert len(provider.prompts) == 1
    assert "Input ended before clarification was resolved." in capsys.readouterr().err
