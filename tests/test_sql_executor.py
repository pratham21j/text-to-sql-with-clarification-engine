import hashlib
import json
import sqlite3

import pytest

from text_to_sql.clarification_engine import ClarificationRequiredError
from text_to_sql.conversation_context import (
    create_conversation,
    execute_conversation_query,
    process_user_question,
)
from text_to_sql.sql_executor import (
    DEFAULT_MAX_ROWS,
    execute_sql,
)


class _FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, prompt, response_schema):
        self.calls.append((prompt, response_schema))
        response = self.responses.pop(0)
        if isinstance(response, str):
            return response
        return json.dumps(response)


def _database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE items (item_id INTEGER PRIMARY KEY, label TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO items (item_id, label) VALUES (?, ?)",
            [(index, f"item-{index}") for index in range(1, 6)],
        )
    return path


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clear_response():
    return {
        "status": "clear",
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_reason": None,
        "resolved_question": "Show all items.",
        "context_action": "replace",
    }


def _sql_response(query="SELECT item_id, label FROM items ORDER BY item_id"):
    return {
        "status": "ready",
        "generated_sql": query,
        "ambiguity_reason": None,
        "clarification_needed": False,
    }


def test_valid_select_returns_columns_and_rows(tmp_path):
    database_path = _database(tmp_path / "items.db")

    result = execute_sql(
        "SELECT item_id, label FROM items ORDER BY item_id",
        database_path,
    )

    assert result.status == "success"
    assert result.columns == ("item_id", "label")
    assert result.rows == tuple(
        (index, f"item-{index}") for index in range(1, 6)
    )
    assert result.row_count == 5
    assert result.truncated is False
    assert result.error is None


def test_query_with_no_rows_returns_empty_success_result(tmp_path):
    database_path = _database(tmp_path / "items.db")

    result = execute_sql(
        "SELECT item_id, label FROM items WHERE item_id > 99",
        database_path,
    )

    assert result.status == "success"
    assert result.columns == ("item_id", "label")
    assert result.rows == ()
    assert result.row_count == 0
    assert result.truncated is False


def test_maximum_rows_are_returned_and_truncation_is_reported(tmp_path):
    database_path = _database(tmp_path / "items.db")

    result = execute_sql(
        "SELECT item_id FROM items ORDER BY item_id",
        database_path,
        max_rows=3,
    )

    assert result.rows == ((1,), (2,), (3,))
    assert result.row_count == 3
    assert result.truncated is True
    assert DEFAULT_MAX_ROWS == 100


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "SELECT FROM items",
        "SELECT item_id FROM items; SELECT label FROM items",
        "SELECT missing_column FROM items",
    ],
)
def test_invalid_sql_columns_and_multiple_statements_are_rejected_without_changes(
    tmp_path,
    sql,
):
    database_path = _database(tmp_path / "items.db")
    before = _file_hash(database_path)

    result = execute_sql(sql, database_path)

    assert result.status == "error"
    assert result.rows == ()
    assert result.error
    assert _file_hash(database_path) == before


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO items (item_id, label) VALUES (9, 'new')",
        "UPDATE items SET label = 'changed' WHERE item_id = 1",
        "DELETE FROM items WHERE item_id = 1",
        "DROP TABLE items",
    ],
)
def test_data_modifying_statements_are_rejected_and_database_is_unchanged(
    tmp_path,
    sql,
):
    database_path = _database(tmp_path / "items.db")
    before = _file_hash(database_path)

    result = execute_sql(sql, database_path)

    assert result.status == "error"
    assert result.error == (
        "Query was rejected because it is invalid or unsupported."
    )
    assert _file_hash(database_path) == before
    with sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone() == (5,)


def test_sqlite_connections_used_by_execution_are_read_only(tmp_path, monkeypatch):
    database_path = _database(tmp_path / "items.db")
    real_connect = sqlite3.connect
    connect_calls = []

    def recording_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)

    result = execute_sql("SELECT item_id FROM items", database_path)

    assert result.status == "success"
    assert len(connect_calls) >= 2
    assert all("mode=ro" in args[0] for args, _ in connect_calls)
    assert all(kwargs.get("uri") is True for _, kwargs in connect_calls)


def test_missing_database_returns_safe_database_error(tmp_path):
    result = execute_sql(
        "SELECT item_id FROM items",
        tmp_path / "missing.db",
    )

    assert result.status == "error"
    assert result.error == "The database query could not be completed."


def test_execution_options_must_be_positive_and_valid(tmp_path):
    database_path = _database(tmp_path / "items.db")

    with pytest.raises(ValueError, match="max_rows"):
        execute_sql("SELECT item_id FROM items", database_path, max_rows=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        execute_sql(
            "SELECT item_id FROM items",
            database_path,
            timeout_seconds=float("inf"),
        )


def test_long_running_query_is_interrupted_at_timeout(tmp_path):
    database_path = _database(tmp_path / "items.db")

    result = execute_sql(
        """
        WITH RECURSIVE counter(value) AS (
            VALUES (0)
            UNION ALL
            SELECT value + 1 FROM counter WHERE value < 100000000
        )
        SELECT SUM(value) FROM counter
        """,
        database_path,
        timeout_seconds=0.05,
    )

    assert result.status == "error"
    assert result.error == "Query exceeded the execution time limit."


def test_unresolved_clarification_blocks_sql_generation_and_execution(tmp_path):
    database_path = _database(tmp_path / "items.db")
    provider = _FakeProvider(
        {
            "status": "clarification_needed",
            "clarification_needed": True,
            "clarification_question": "Which items should be shown?",
            "clarification_reason": "The requested item filter is incomplete.",
            "resolved_question": None,
            "context_action": "replace",
        }
    )
    state = create_conversation()
    process_user_question(
        state,
        "Show items from a category.",
        database_path,
        provider=provider,
    )

    with pytest.raises(ClarificationRequiredError, match="before query execution"):
        execute_conversation_query(state, database_path, provider=provider)

    assert len(provider.calls) == 1


def test_conversation_workflow_generates_then_executes_validated_sql(tmp_path):
    database_path = _database(tmp_path / "items.db")
    provider = _FakeProvider(_clear_response(), _sql_response())
    state = create_conversation()
    process_user_question(
        state,
        "Show all items.",
        database_path,
        provider=provider,
    )

    result = execute_conversation_query(
        state,
        database_path,
        provider=provider,
        max_rows=2,
    )

    assert result.sql.status == "ready"
    assert result.execution.status == "success"
    assert result.execution.columns == ("item_id", "label")
    assert result.execution.rows == ((1, "item-1"), (2, "item-2"))
    assert result.execution.row_count == 2
    assert result.execution.truncated is True


def test_sql_generation_rejection_never_returns_execution_results(tmp_path):
    database_path = _database(tmp_path / "items.db")
    provider = _FakeProvider(
        _clear_response(),
        _sql_response("DELETE FROM items"),
    )
    state = create_conversation()
    process_user_question(
        state,
        "Show all items.",
        database_path,
        provider=provider,
    )

    with pytest.raises(ValueError, match="Only read-only SELECT"):
        execute_conversation_query(state, database_path, provider=provider)
