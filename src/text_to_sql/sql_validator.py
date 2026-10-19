"""SQLGlot validation for the read-only SQL query subset."""

import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from text_to_sql.database import DEFAULT_DATABASE_PATH


class SQLValidationError(ValueError):
    """Raised when generated SQL is not one supported read-only query."""


def validate_read_only_sql(
    sql: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> str:
    """Validate one read-only query and prepare it without executing it."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("Generated SQL must be a non-empty string.")

    try:
        statements = [
            statement
            for statement in sqlglot.parse(sql, read="sqlite")
            if statement is not None
        ]
    except ParseError as error:
        raise SQLValidationError("Generated SQL is not valid SQLite syntax.") from error

    if len(statements) != 1:
        raise SQLValidationError("Only one SQL statement is supported.")

    statement = statements[0]
    if not isinstance(
        statement,
        (exp.Select, exp.Union, exp.Intersect, exp.Except),
    ):
        raise SQLValidationError("Only read-only SELECT queries are supported.")

    for node in statement.walk():
        if isinstance(node, (exp.DML, exp.DDL, exp.Command, exp.Into)):
            raise SQLValidationError(
                "Generated SQL contains a data-modifying or unsupported operation."
            )
        if isinstance(node, exp.Placeholder):
            raise SQLValidationError(
                "Generated SQL must not contain unbound parameters."
            )

    validated_sql = sql.strip()
    path = Path(database_path).expanduser().resolve()
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute(f"EXPLAIN {validated_sql}")
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise SQLValidationError(
            "Generated SQL is not valid for the configured SQLite schema."
        ) from error

    return validated_sql
