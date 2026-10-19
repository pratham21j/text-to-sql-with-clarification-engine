import csv
import json
import queue
import threading
import time
import tkinter
from types import SimpleNamespace

import pytest

from text_to_sql import desktop_app as desktop_module
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
from text_to_sql.llm_provider import (
    GeminiProviderError,
    GeminiQuotaError,
    GeminiRateLimitError,
)
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

    event, payload = app._worker_results.get_nowait()
    assert event == "error"
    error, retry_message = payload
    assert isinstance(error, GeminiProviderError)
    assert str(error) == "Network is unavailable."
    assert retry_message == "Show students."


def test_desktop_ignores_duplicate_submission_while_request_is_busy():
    app = SimpleNamespace(_busy=True)

    TextToSQLDesktopApp.submit_message(app)


def test_desktop_quota_error_restores_submitted_text_for_manual_retry():
    class _Entry:
        value = ""

        def get(self):
            return self.value

        def insert(self, _index, value):
            self.value = value

        def configure(self, **_kwargs):
            pass

    class _Label:
        def configure(self, **_kwargs):
            pass

    entry = _Entry()
    messages = []
    status_changes = []
    app = SimpleNamespace(
        message_entry=entry,
        execution_status_label=_Label(),
        _append_message=lambda role, text: messages.append((role, text)),
        _set_busy=lambda busy, status: status_changes.append((busy, status)),
    )
    error = GeminiQuotaError(
        "Gemini quota is exhausted. No automatic retry was made.",
        retry_after_seconds=54000,
        quota_exhausted=True,
    )

    TextToSQLDesktopApp._show_error(
        app,
        error,
        retry_message="Show the 5 best students.",
    )

    assert entry.value == "Show the 5 best students."
    assert "No automatic retry was made." in messages[0][1]
    assert status_changes == [(False, "Request failed")]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            GeminiQuotaError(
                "Gemini quota is exhausted. No automatic retry was made.",
                quota_exhausted=True,
            ),
            "Gemini quota is exhausted.",
        ),
        (
            GeminiRateLimitError(
                "Gemini rate limit reached. No further automatic retry was made."
            ),
            "Gemini rate limit reached.",
        ),
        (
            GeminiProviderError(
                "Gemini request failed.",
                diagnostic="HTTP 503: UNAVAILABLE test-secret",
            ),
            "Gemini is temporarily unavailable (HTTP 503).",
        ),
        (
            GeminiProviderError(
                "Gemini request failed.",
                diagnostic="HTTP 400: INVALID_ARGUMENT",
            ),
            "Gemini rejected this request (HTTP 400).",
        ),
        (
            GeminiProviderError("Gemini network request failed."),
            "Gemini could not be reached.",
        ),
    ],
)
def test_desktop_formats_provider_errors_without_exposing_diagnostics(
    error,
    expected,
):
    message = desktop_module.format_desktop_error(error)

    assert expected in message
    assert "test-secret" not in message


def test_desktop_preserves_original_question_and_answer_after_api_failure(tmp_path):
    conversation = create_conversation()
    database_path = _database(tmp_path)
    provider = _FakeProvider(
        _needs_clarification(),
        GeminiProviderError(
            "Gemini request failed.",
            diagnostic="HTTP 503: UNAVAILABLE",
        ),
    )
    pending = process_desktop_message(
        conversation,
        "Show students from a department.",
        database_path,
        provider=provider,
    )

    assert pending.execution is None
    with pytest.raises(GeminiProviderError):
        process_desktop_message(
            conversation,
            "CSE",
            database_path,
            provider=provider,
        )

    assert [event.kind for event in conversation.history] == [
        "question",
        "clarification",
        "clarification_answer",
    ]
    assert conversation.history[0].content == "Show students from a department."
    assert conversation.history[-1].content == "CSE"
    assert conversation.clarification_status == "clarification_needed"


def test_sql_preview_formatting_is_readable_and_preserves_unparseable_text():
    formatted = desktop_module.format_sql_preview(
        "select student_id,cgpa from students where cgpa>8 order by cgpa desc limit 5"
    )

    assert "SELECT\n" in formatted
    assert "FROM students" in formatted
    assert "ORDER BY\n" in formatted
    assert desktop_module.format_sql_preview("SELECT * FROM (") == "SELECT * FROM ("


def test_copy_sql_uses_displayed_query_and_reports_clipboard_failures():
    class _Label:
        def __init__(self):
            self.text = ""

        def configure(self, **kwargs):
            if "text" in kwargs:
                self.text = kwargs["text"]

    label = _Label()
    copied = []
    app = SimpleNamespace(
        _displayed_sql="SELECT\n  student_id\nFROM students",
        sql_action_status_label=label,
        clipboard_clear=lambda: None,
        clipboard_append=copied.append,
    )

    TextToSQLDesktopApp.copy_sql(app)

    assert copied == ["SELECT\n  student_id\nFROM students"]
    assert label.text == "SQL copied to clipboard."

    def fail_clipboard():
        raise tkinter.TclError("clipboard unavailable")

    app.clipboard_clear = fail_clipboard
    TextToSQLDesktopApp.copy_sql(app)
    assert "Could not access the clipboard" in label.text


def test_csv_export_writes_displayed_columns_and_rows_without_rerunning(
    monkeypatch,
    tmp_path,
):
    class _Label:
        def __init__(self):
            self.text = ""

        def configure(self, **kwargs):
            if "text" in kwargs:
                self.text = kwargs["text"]

    destination = tmp_path / "results.csv"
    monkeypatch.setattr(
        desktop_module.filedialog,
        "asksaveasfilename",
        lambda **_kwargs: str(destination),
    )
    label = _Label()
    app = SimpleNamespace(
        _displayed_columns=("student_id", "name"),
        _displayed_rows=((1001, "Aarav"), (1002, "Diya")),
        execution_status_label=label,
    )

    TextToSQLDesktopApp.export_results_csv(app)

    with destination.open(newline="", encoding="utf-8") as csv_file:
        assert list(csv.reader(csv_file)) == [
            ["student_id", "name"],
            ["1001", "Aarav"],
            ["1002", "Diya"],
        ]
    assert label.text == "Exported 2 row(s) to CSV."


def test_csv_export_cancellation_does_not_change_status_or_create_file(
    monkeypatch,
    tmp_path,
):
    class _Label:
        text = "2 row(s) returned"

        def configure(self, *, text):
            self.text = text

    monkeypatch.setattr(
        desktop_module.filedialog,
        "asksaveasfilename",
        lambda **_kwargs: "",
    )
    app = SimpleNamespace(
        _displayed_columns=("student_id",),
        _displayed_rows=((1001,),),
        execution_status_label=_Label(),
    )

    TextToSQLDesktopApp.export_results_csv(app)

    assert app.execution_status_label.text == "2 row(s) returned"
    assert list(tmp_path.iterdir()) == []


def test_csv_export_reports_file_errors_safely(monkeypatch, tmp_path):
    class _Label:
        def __init__(self):
            self.text = ""

        def configure(self, *, text):
            self.text = text

    monkeypatch.setattr(
        desktop_module.filedialog,
        "asksaveasfilename",
        lambda **_kwargs: str(tmp_path),
    )
    label = _Label()
    app = SimpleNamespace(
        _displayed_columns=("student_id",),
        _displayed_rows=((1001,),),
        execution_status_label=label,
    )

    TextToSQLDesktopApp.export_results_csv(app)

    assert "Could not save the CSV file" in label.text


def test_csv_export_reports_save_dialog_failures(monkeypatch):
    class _Label:
        def __init__(self):
            self.text = ""

        def configure(self, *, text):
            self.text = text

    def fail_dialog(**_kwargs):
        raise tkinter.TclError("dialog unavailable")

    monkeypatch.setattr(
        desktop_module.filedialog,
        "asksaveasfilename",
        fail_dialog,
    )
    label = _Label()
    app = SimpleNamespace(
        _displayed_columns=("student_id",),
        _displayed_rows=((1001,),),
        execution_status_label=label,
    )

    TextToSQLDesktopApp.export_results_csv(app)

    assert label.text == "The CSV save dialog could not be opened."


def test_csv_export_allows_empty_result_headers(monkeypatch, tmp_path):
    class _Label:
        def __init__(self):
            self.text = ""

        def configure(self, *, text):
            self.text = text

    destination = tmp_path / "empty.csv"
    monkeypatch.setattr(
        desktop_module.filedialog,
        "asksaveasfilename",
        lambda **_kwargs: str(destination),
    )
    app = SimpleNamespace(
        _displayed_columns=("student_id",),
        _displayed_rows=(),
        execution_status_label=_Label(),
    )

    TextToSQLDesktopApp.export_results_csv(app)

    with destination.open(newline="", encoding="utf-8") as csv_file:
        assert list(csv.reader(csv_file)) == [["student_id"]]
    assert app.execution_status_label.text == "Exported 0 row(s) to CSV."


def test_desktop_displays_returned_sql_and_rows_without_a_display():
    class _Preview:
        def __init__(self):
            self.text = ""

        def configure(self, **_kwargs):
            pass

        def delete(self, *_args):
            self.text = ""

        def insert(self, _index, text):
            self.text = text

    class _Label:
        def __init__(self):
            self.text = ""

        def configure(self, **kwargs):
            if "text" in kwargs:
                self.text = kwargs["text"]

    class _Table:
        def __init__(self):
            self.columns = ()
            self.headings = {}
            self.column_options = {}
            self.rows = []

        def delete(self, *_items):
            self.rows.clear()

        def get_children(self):
            return tuple(range(len(self.rows)))

        def configure(self, *, columns):
            self.columns = columns

        def heading(self, identifier, *, text):
            self.headings[identifier] = text

        def column(self, *_args, **_kwargs):
            self.column_options[_args[0]] = _kwargs

        def insert(self, _parent, _index, *, values, tags):
            self.rows.append(values)

        def tag_configure(self, *_args, **_kwargs):
            pass

    app = SimpleNamespace(
        max_rows=100,
        sql_preview=_Preview(),
        execution_status_label=_Label(),
        sql_action_status_label=_Label(),
        copy_sql_button=_Label(),
        export_csv_button=_Label(),
        result_table=_Table(),
    )
    app._replace_table = lambda columns, rows: TextToSQLDesktopApp._replace_table(
        app,
        columns,
        rows,
    )

    TextToSQLDesktopApp._show_execution(app, _ready_result().execution)

    assert app.sql_preview.text == "SELECT\n  student_id\nFROM students"
    assert app.execution_status_label.text == "1 row(s) returned"
    assert app.result_table.columns == ("column_0",)
    assert app.result_table.headings == {"column_0": "student_id"}
    assert app.result_table.column_options["column_0"]["anchor"] == "w"
    assert app.result_table.column_options["column_0"]["width"] >= 100
    assert app.result_table.column_options["column_0"]["width"] <= 360
    assert app.result_table.rows == [("1001",)]
    assert app._displayed_columns == ("student_id",)
    assert app._displayed_rows == ((1001,),)


def test_empty_and_truncated_results_have_clear_status_messages():
    class _Widget:
        def __init__(self):
            self.text = ""
            self.rows = []
            self.columns = ()

        def configure(self, **_kwargs):
            pass

        def delete(self, *_args):
            self.rows.clear()

        def insert(self, _index, _text):
            pass

        def get_children(self):
            return tuple(range(len(self.rows)))

        def heading(self, *_args, **_kwargs):
            pass

        def column(self, *_args, **_kwargs):
            pass

        def tag_configure(self, *_args, **_kwargs):
            pass

    class _Label(_Widget):
        def configure(self, **kwargs):
            if "text" in kwargs:
                self.text = kwargs["text"]

    class _Table(_Widget):
        def configure(self, *, columns):
            self.columns = columns

        def insert(self, _parent, _index, *, values, tags):
            self.rows.append(values)

    app = SimpleNamespace(
        max_rows=2,
        sql_preview=_Widget(),
        execution_status_label=_Label(),
        sql_action_status_label=_Label(),
        copy_sql_button=_Label(),
        export_csv_button=_Label(),
        result_table=_Table(),
    )
    app._replace_table = lambda columns, rows: TextToSQLDesktopApp._replace_table(
        app,
        columns,
        rows,
    )
    sql_result = _ready_result().execution.sql
    empty = ConversationExecutionResult(
        sql_result,
        SQLExecutionResult("success", ("student_id",), (), 0, False, None),
    )
    TextToSQLDesktopApp._show_execution(app, empty)
    assert app.execution_status_label.text == (
        "No rows matched this query (0 rows returned)."
    )
    assert app._displayed_columns == ("student_id",)
    assert app._displayed_rows == ()

    truncated = ConversationExecutionResult(
        sql_result,
        SQLExecutionResult(
            "success",
            ("student_id",),
            ((1001,), (1002,)),
            2,
            True,
            None,
        ),
    )
    TextToSQLDesktopApp._show_execution(app, truncated)
    assert app.execution_status_label.text == (
        "2 row(s) returned (limited to 2; more rows available)"
    )


def test_chat_messages_receive_distinct_user_and_assistant_styles():
    class _Chat:
        def __init__(self):
            self.entries = []

        def configure(self, **_kwargs):
            pass

        def insert(self, _index, text, tags):
            self.entries.append((text, tags))

        def see(self, _index):
            pass

    app = SimpleNamespace(chat_display=_Chat())

    TextToSQLDesktopApp._append_message(app, "you", "Show students.")
    TextToSQLDesktopApp._append_message(app, "assistant", "Which department?")

    assert app.chat_display.entries == [
        ("You\n", ("user_label",)),
        ("Show students.\n\n", ("user_message",)),
        ("Assistant\n", ("assistant_label",)),
        ("Which department?\n\n", ("assistant_message",)),
    ]


def test_new_conversation_has_no_previous_context():
    first = create_conversation()
    second = create_conversation()
    first.append_user_turn("Show CSE students.")

    assert second.history == []
    assert second.active_query_context is None
    assert first.conversation_id != second.conversation_id


def test_gui_new_conversation_replaces_active_context_without_display():
    class _Widget:
        def configure(self, **_kwargs):
            pass

        def delete(self, *_args):
            pass

    old_conversation = create_conversation()
    old_conversation.append_user_turn("Show CSE students.")
    old_conversation.record_assistant_clarification(
        ClarificationResult(
            status="clear",
            original_question="Show CSE students.",
            clarification_needed=False,
            clarification_question=None,
            clarification_reason=None,
            clarification_answer=None,
            resolved_question="Show CSE students.",
        )
    )
    assert old_conversation.active_query_context == "Show CSE students."
    app = SimpleNamespace(
        _busy=False,
        conversation=old_conversation,
        chat_display=_Widget(),
        message_entry=_Widget(),
        _clear_results=lambda: None,
        _append_message=lambda *_args: None,
        _update_conversation_label=lambda: None,
        _set_busy=lambda *_args: None,
    )

    TextToSQLDesktopApp.new_conversation(app)

    assert app.conversation.conversation_id != old_conversation.conversation_id
    assert app.conversation.history == []
    assert app.conversation.active_query_context is None


def test_database_description_uses_read_only_schema_metadata(tmp_path):
    app = SimpleNamespace(database_path=_database(tmp_path))

    description = TextToSQLDesktopApp._database_description(app)

    assert "Database:" in description
    assert "Tables: courses, departments, enrollments, examinations, students" in description


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
        assert "SELECT\n  student_id" in app.sql_preview.get("1.0", "end")
    finally:
        app.destroy()
