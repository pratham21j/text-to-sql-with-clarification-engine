"""Generate a validated SQL preview for a natural-language question."""

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_to_sql.clarification_engine import (
    ClarificationResponseError,
    analyze_question,
    generate_sql_after_clarification,
    submit_clarification_answer,
)
from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.llm_provider import (
    GeminiConfigurationError,
    GeminiProvider,
    GeminiProviderError,
)
from text_to_sql.sql_generator import InvalidModelResponseError
from text_to_sql.sql_validator import SQLValidationError


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clarify a question when needed, then generate a validated SQL "
            "preview using the configured Gemini model."
        )
    )
    parser.add_argument("question", help="natural-language database question")
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite database file (defaults to data\\academic.db)",
    )
    arguments = parser.parse_args()

    try:
        database_path = arguments.database
        if database_path is None:
            database_path = DEFAULT_DATABASE_PATH
        provider = GeminiProvider()
        clarification = analyze_question(
            arguments.question,
            database_path=database_path,
            provider=provider,
        )
        while clarification.clarification_needed:
            print(clarification.clarification_reason)
            clarification_question = clarification.clarification_question
            if clarification_question is None:
                raise ClarificationResponseError(
                    "Clarification response is missing its follow-up question."
                )
            print(clarification_question)
            answer = input("> ")
            clarification = submit_clarification_answer(
                clarification.original_question,
                clarification_question,
                answer,
                database_path=database_path,
                provider=provider,
                clarification_history=clarification.clarification_history,
            )
        sql_result = generate_sql_after_clarification(
            clarification,
            database_path=database_path,
            provider=provider,
        )
    except (
        GeminiConfigurationError,
        GeminiProviderError,
        InvalidModelResponseError,
        ClarificationResponseError,
        SQLValidationError,
        sqlite3.Error,
        OSError,
        ValueError,
        EOFError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "clarification": asdict(clarification),
                "sql": asdict(sql_result),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
