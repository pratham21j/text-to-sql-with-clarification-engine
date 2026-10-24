"""Database schema retrieval and structured text-to-SQL generation."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.llm_provider import create_provider
from text_to_sql.sql_validator import validate_read_only_sql


class InvalidModelResponseError(ValueError):
    """Raised when the model response does not match the response contract."""


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    name: str
    data_type: str
    not_null: bool
    primary_key_position: int


@dataclass(frozen=True, slots=True)
class ForeignKeySchema:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True, slots=True)
class TableSchema:
    name: str
    columns: tuple[ColumnSchema, ...]


@dataclass(frozen=True, slots=True)
class DatabaseSchema:
    tables: tuple[TableSchema, ...]
    relationships: tuple[ForeignKeySchema, ...]


@dataclass(frozen=True, slots=True)
class SQLGenerationResult:
    status: str
    generated_sql: str | None
    ambiguity_reason: str | None
    clarification_needed: bool


class StructuredResponseProvider(Protocol):
    def generate_json(
        self,
        prompt: str,
        response_schema: dict[str, object],
    ) -> str: ...


_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "status": {
            "type": "STRING",
            "enum": ["ready", "clarification_needed"],
        },
        "generated_sql": {
            "type": "STRING",
            "nullable": True,
            "description": "A single SQLite SELECT query, or null if clarification is needed.",
        },
        "ambiguity_reason": {
            "type": "STRING",
            "nullable": True,
            "description": "Why clarification is needed, or null for a ready query.",
        },
        "clarification_needed": {"type": "BOOLEAN"},
    },
    "required": [
        "status",
        "generated_sql",
        "ambiguity_reason",
        "clarification_needed",
    ],
    "propertyOrdering": [
        "status",
        "generated_sql",
        "ambiguity_reason",
        "clarification_needed",
    ],
}


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def get_database_schema(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> DatabaseSchema:
    """Read table columns and foreign-key relationships from SQLite."""
    path = Path(database_path).expanduser().resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        table_names = [
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]

        tables: list[TableSchema] = []
        relationships: list[ForeignKeySchema] = []
        for table_name in table_names:
            quoted_name = _quote_identifier(table_name)
            columns = tuple(
                ColumnSchema(
                    name=row[1],
                    data_type=row[2],
                    not_null=bool(row[3]),
                    primary_key_position=row[5],
                )
                for row in connection.execute(
                    f"PRAGMA table_info({quoted_name})"
                )
            )
            tables.append(TableSchema(name=table_name, columns=columns))

            relationships.extend(
                ForeignKeySchema(
                    from_table=table_name,
                    from_column=row[3],
                    to_table=row[2],
                    to_column=row[4],
                )
                for row in connection.execute(
                    f"PRAGMA foreign_key_list({quoted_name})"
                )
            )
    finally:
        connection.close()

    return DatabaseSchema(
        tables=tuple(tables),
        relationships=tuple(relationships),
    )


def _build_prompt(question: str, schema: DatabaseSchema) -> str:
    schema_json = json.dumps(asdict(schema), indent=2)
    return f"""Convert the user's question into one SQLite-compatible read-only query.

Use only the tables and columns in the supplied database schema. Never invent
tables or columns. If the question cannot be answered from this schema, or a
material ambiguity prevents a reliable query, request clarification instead.
Only generate a single SELECT query. Never execute SQL, access external systems,
or include data-modifying statements.

Return exactly one JSON object matching the response schema:
- status: "ready" when a query can be generated, otherwise "clarification_needed"
- generated_sql: the SELECT query when ready; null when clarification is needed
- ambiguity_reason: null when ready; otherwise a concise explanation
- clarification_needed: true exactly when status is "clarification_needed"

Database schema:
{schema_json}

User question:
{question}
"""


def _parse_result(
    response_text: str,
    database_path: str | Path,
) -> SQLGenerationResult:
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise InvalidModelResponseError(
            "Model response was not valid JSON."
        ) from error

    required_keys = {
        "status",
        "generated_sql",
        "ambiguity_reason",
        "clarification_needed",
    }
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise InvalidModelResponseError(
            "Model response must contain exactly the required response fields."
        )

    status = payload["status"]
    generated_sql = payload["generated_sql"]
    ambiguity_reason = payload["ambiguity_reason"]
    clarification_needed = payload["clarification_needed"]
    if (
        not isinstance(status, str)
        or not isinstance(clarification_needed, bool)
        or (
            generated_sql is not None
            and not isinstance(generated_sql, str)
        )
        or (
            ambiguity_reason is not None
            and not isinstance(ambiguity_reason, str)
        )
    ):
        raise InvalidModelResponseError(
            "Model response fields have invalid types."
        )

    if status == "clarification_needed" and clarification_needed:
        if generated_sql not in (None, ""):
            raise InvalidModelResponseError(
                "Model must not return SQL when clarification is needed."
            )
        if not ambiguity_reason or not ambiguity_reason.strip():
            raise InvalidModelResponseError(
                "Model must explain why clarification is needed."
            )
        return SQLGenerationResult(
            status=status,
            generated_sql=None,
            ambiguity_reason=ambiguity_reason.strip(),
            clarification_needed=True,
        )

    if status != "ready" or clarification_needed:
        raise InvalidModelResponseError(
            "Model returned an unsupported or inconsistent status."
        )
    if not generated_sql or not generated_sql.strip():
        raise InvalidModelResponseError(
            "Model must return SQL when clarification is not needed."
        )
    if ambiguity_reason not in (None, ""):
        raise InvalidModelResponseError(
            "Model must not return an ambiguity reason for a ready query."
        )

    validated_sql = validate_read_only_sql(
        generated_sql,
        database_path=database_path,
    )
    return SQLGenerationResult(
        status="ready",
        generated_sql=validated_sql,
        ambiguity_reason=None,
        clarification_needed=False,
    )


def generate_sql(
    question: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
) -> SQLGenerationResult:
    """Retrieve live SQLite schema and ask the selected provider for SQL."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    schema = get_database_schema(database_path)
    active_provider = provider if provider is not None else create_provider()
    response_text = active_provider.generate_json(
        _build_prompt(question.strip(), schema),
        _RESPONSE_SCHEMA,
    )
    return _parse_result(response_text, database_path)
