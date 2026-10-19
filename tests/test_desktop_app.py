import json
import queue
import threading
import time
import tkinter
from types import SimpleNamespace

import pytest

from text_to_sql.clarification_engine import ClarificationResult
from text_to_sql.conversation_context import (
    ConversationExecutionResult,
    create_conversation,
)
from text_to_sql.database import initialize_database
from text_to_sql.desktop_app import (
    DesktopTurnResult,
    TextToSQLDesktopApp,
    format_result_table,
    process_desktop_message,
)
from text_to_sql.llm_provider import GeminiProviderError
from text_to_sql.sql_executor import SQLExecutionResult
from text_to_sql.sql_generator import SQLGenerationResult


class _FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_json(self, prompt, response_schema):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)


def _database(tmp_path):
    return initialize_database(tmp_path / "academic.db")


def _clear(question="Show all CSE students."):
    return {
        "status": "clear",
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_reason": None,
        "resolved_question": question,
        "context_action": "replace",
    }


def _needs_clarification():
    return {
        "status": "clarification_needed",
        "clarification_needed": True,
        "clarification_question": "Which department should be shown?",
        "clarification_reason": "A department was not specified.",
        "resolved_question": None,
        "context_action": "replace",
    }


def _sql(query="SELECT student_id, first_name FROM students ORDER BY student_id"):
    return {
        "status": "ready",
        "generated_sql": query,
        "ambiguity_reason": None,
        "clarification_needed": False,
    }


def _ready_result():
    clarification = ClarificationResult(
        status="clear",
        original_question="Show students.",
        clarification_needed=False,
        clarification_question=None,
        clarification_reason=None,
        clarification_answer=None,
        resolved_question="Show students.",
    )
    execution = SQLExecutionResult(
        status="success",
        columns=("student_id",),
        rows=((1001,),),
        row_count=1,
        truncated=False,
        error=None,
    )
    return DesktopTurnResult(
        clarification=clarification,
        execution=ConversationExecutionResult(
            sql=SQLGenerationResult(
                "ready",
                "SELECT student_id FROM students",
                None,
                False,
            ),
            execution=execution,
        ),
    )


def test_desktop_workflow_generates_and_executes_clear_question(tmp_path):
    conversation = create_conversation()
    provider = _FakeProvider(
        _clear("Show all CSE students."),
        _sql(
            "SELECT student_id, first_name FROM students "
            "WHERE department_id = 1 ORDER BY student_id"
        ),
    )

    result = process_desktop_message(
        conversation,
        "Show all CSE students.",
        _database(tmp_path),
        provider=provider,
    )

    assert result.clarification.status == "clear"
    assert result.execution is not None
    assert result.execution.execution.status == "success"
    assert result.execution.execution.columns == ("student_id", "first_name")
    assert result.execution.execution.row_count > 0
    assert len(provider.prompts) == 2


def test_unresolved_clarification_does_not_generate_or_execute_sql(tmp_path):
    conversation = create_conversation()
    provider = _FakeProvider(_needs_clarification())

    result = process_desktop_message(
        conversation,
        "Show students from a department.",
        _database(tmp_path),
        provider=provider,
    )

    assert result.clarification.status == "clarification_needed"
    assert result.execution is None
    assert len(provider.prompts) == 1
    assert conversation.clarification_status == "clarification_needed"


def test_clarification_answer_resolves_then_executes_query(tmp_path):
    conversation = create_conversation()
    provider = _FakeProvider(
        _needs_clarification(),
        _clear("Show students in the CSE department."),
        _sql("SELECT student_id FROM students WHERE department_id = 1"),
    )
    database_path = _database(tmp_path)

    pending = process_desktop_message(
        conversation,
        "Show students from a department.",
        database_path,
        provider=provider,
    )
    completed = process_desktop_message(
        conversation,
        "CSE",
        database_path,
        provider=provider,
    )

    assert pending.execution is None
    assert completed.execution is not None
    assert conversation.history[2].kind == "clarification_answer"
    assert conversation.history[2].content == "CSE"
    assert conversation.history[3].kind == "resolved_request"
    assert conversation.history[3].role == "assistant"
    assert completed.execution.execution.status == "success"


def test_provider_failure_propagates_for_background_error_display(tmp_path):
    with pytest.raises(GeminiProviderError, match="Network"):
        process_desktop_message(
            create_conversation(),
            "Show students.",
            _database(tmp_path),
            provider=_FakeProvider(GeminiProviderError("Network is unavailable.")),
        )


def test_background_worker_returns_errors_through_result_queue(tmp_path):
    def fail_turn(*_args, **_kwargs):
        raise GeminiProviderError("Network is unavailable.")

    app = SimpleNamespace(
        _turn_handler=fail_turn,
        conversation=create_conversation(),
        database_path=tmp_path / "academic.db",
        max_rows=100,
        _worker_results=queue.Queue(),
    )

    TextToSQLDesktopApp._run_turn(app, "Show students.")

    event, error = app._worker_results.get_nowait()
    assert event == "error"
    assert isinstance(error, GeminiProviderError)
    assert str(error) == "Network is unavailable."


def test_new_conversation_has_no_previous_context():
    first = create_conversation()
    second = create_conversation()
    first.append_user_turn("Show CSE students.")

    assert second.history == []
    assert second.active_query_context is None
    assert first.conversation_id != second.conversation_id


def test_result_table_format_includes_headers_and_values():
    rendered = format_result_table(
        ("student_id", "name"),
        ((1001, "Aarav"), (1002, "Diya")),
    )

    assert "student_id | name" in rendered
    assert "1001       | Aarav" in rendered
    assert "1002       | Diya" in rendered


def test_gui_initializes_when_a_graphical_display_is_available():
    try:
        app = TextToSQLDesktopApp(turn_handler=lambda *_args, **_kwargs: _ready_result())
    except tkinter.TclError as error:
        pytest.skip(f"Graphical display is unavailable: {error}")
    try:
        assert app.title() == "Text-to-SQL Clarification Engine"
        assert app.conversation.history == []
        app.new_conversation()
        assert app.conversation.history == []
    finally:
        app.destroy()


def test_gui_message_submission_and_result_display_when_display_is_available():
    completed = threading.Event()

    def handler(*_args, **_kwargs):
        completed.set()
        return _ready_result()

    try:
        app = TextToSQLDesktopApp(turn_handler=handler)
    except tkinter.TclError as error:
        pytest.skip(f"Graphical display is unavailable: {error}")
    try:
        app.message_entry.insert(0, "Show students.")
        app.submit_message()
        assert completed.wait(timeout=3)
        deadline = time.monotonic() + 3
        while app._busy and time.monotonic() < deadline:
            app._poll_worker_results()
            time.sleep(0.02)
        assert app._busy is False
        assert "Query completed" in app.status_label.cget("text")
        assert len(app.result_table.get_children()) == 1
        assert "SELECT student_id" in app.sql_preview.get("1.0", "end")
    finally:
        app.destroy()
