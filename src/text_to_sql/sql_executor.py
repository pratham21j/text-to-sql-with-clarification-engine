"""Execute validated, read-only SQLite queries and return bounded results."""

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.sql_validator import SQLValidationError, validate_read_only_sql

DEFAULT_MAX_ROWS = 100
DEFAULT_QUERY_TIMEOUT_SECONDS = 5.0
_FETCH_BATCH_SIZE = 1000


@dataclass(frozen=True, slots=True)
class SQLExecutionResult:
    """Execution outcome; row_count counts returned rows, not all matches."""

    status: str
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    row_count: int
    truncated: bool
    error: str | None


def _validate_execution_options(max_rows: int, timeout_seconds: float) -> None:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer.")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number.")


def execute_sql(
    sql: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
) -> SQLExecutionResult:
    """Revalidate and execute one read-only query, returning at most max_rows."""
    _validate_execution_options(max_rows, timeout_seconds)
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        return _database_error()

    try:
        validated_sql = validate_read_only_sql(sql, path)
    except SQLValidationError:
        return SQLExecutionResult(
            status="error",
            columns=(),
            rows=(),
            row_count=0,
            truncated=False,
            error="Query was rejected because it is invalid or unsupported.",
        )

    deadline = time.monotonic() + timeout_seconds
    try:
        connection = sqlite3.connect(
            path.as_uri() + "?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        )
    except sqlite3.Error:
        return _database_error()

    try:
        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(
            lambda: int(time.monotonic() >= deadline),
            _FETCH_BATCH_SIZE,
        )
        cursor = connection.execute(validated_sql)
        columns = tuple(
            description[0] for description in (cursor.description or ())
        )
        fetched_rows = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched_rows) > max_rows
        rows = tuple(tuple(row) for row in fetched_rows[:max_rows])
        return SQLExecutionResult(
            status="success",
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            error=None,
        )
    except sqlite3.OperationalError as error:
        if "interrupted" in str(error).lower() and time.monotonic() >= deadline:
            return SQLExecutionResult(
                status="error",
                columns=(),
                rows=(),
                row_count=0,
                truncated=False,
                error="Query exceeded the execution time limit.",
            )
        return _database_error()
    except sqlite3.Error:
        return _database_error()
    finally:
        connection.close()


def _database_error() -> SQLExecutionResult:
    return SQLExecutionResult(
        status="error",
        columns=(),
        rows=(),
        row_count=0,
        truncated=False,
        error="The database query could not be completed.",
    )
