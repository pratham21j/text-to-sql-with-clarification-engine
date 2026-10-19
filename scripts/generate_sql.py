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
)
from text_to_sql.conversation_context import (
    answer_conversation_clarification,
    create_conversation,
    execute_conversation_query,
    process_user_question,
)
from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.llm_provider import (
    GeminiConfigurationError,
    GeminiProvider,
    GeminiProviderError,
    GeminiRateLimitError,
)
from text_to_sql.sql_generator import InvalidModelResponseError
from text_to_sql.sql_executor import DEFAULT_MAX_ROWS
from text_to_sql.sql_validator import SQLValidationError


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clarify a question when needed, then execute a validated "
            "read-only SQL query using the configured Gemini model."
        )
    )
    parser.add_argument("question", help="natural-language database question")
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite database file (defaults to data\\academic.db)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"maximum number of returned rows (default: {DEFAULT_MAX_ROWS})",
    )
    arguments = parser.parse_args()

    try:
        database_path = arguments.database
        if database_path is None:
            database_path = DEFAULT_DATABASE_PATH
        provider = GeminiProvider()
        conversation = create_conversation()
        while True:
            try:
                clarification = process_user_question(
                    conversation,
                    arguments.question,
                    database_path,
                    provider=provider,
                )
                break
            except GeminiRateLimitError as error:
                if not _confirm_rate_limit_retry(error, "question analysis"):
                    raise
        while conversation.clarification_status in {
            "clarification_needed",
            "unsupported",
        }:
            clarification = conversation.current_result
            if clarification is None or clarification.clarification_question is None:
                raise ClarificationResponseError(
                    "Conversation state is missing its follow-up question."
                )
            print(clarification.clarification_question)
            answer = input("> ")
            while True:
                try:
                    clarification = answer_conversation_clarification(
                        conversation,
                        answer,
                        database_path,
                        provider=provider,
                    )
                    break
                except GeminiProviderError as error:
                    retry_note = (
                        "Your clarification is retained. No automatic retry was "
                        "made; re-enter it only when you are ready to try again."
                        if isinstance(error, GeminiRateLimitError)
                        else "Your clarification is retained. "
                        "Enter it again to retry, or press Ctrl+C to exit."
                    )
                    print(
                        f"{error}\n{retry_note}",
                        file=sys.stderr,
                    )
                    answer = input("> ")
        while True:
            try:
                query_result = execute_conversation_query(
                    conversation,
                    database_path,
                    provider=provider,
                    max_rows=arguments.max_rows,
                )
                break
            except GeminiRateLimitError as error:
                if not _confirm_rate_limit_retry(error, "SQL generation"):
                    raise
    except (
        GeminiConfigurationError,
        GeminiProviderError,
        InvalidModelResponseError,
        ClarificationResponseError,
        SQLValidationError,
        sqlite3.Error,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    except EOFError:
        parser.error("Input ended before clarification was resolved.")

    print(
        json.dumps(
            {
                "clarification": asdict(clarification),
                "sql": asdict(query_result.sql),
                "results": asdict(query_result.execution),
                "conversation": conversation.to_dict(),
            },
            indent=2,
        )
    )


def _confirm_rate_limit_retry(error: GeminiRateLimitError, operation: str) -> bool:
    print(
        f"{error}\nThe {operation} input and conversation are retained. "
        "No automatic retry was made.",
        file=sys.stderr,
    )
    return input(
        "Enter RETRY to make another request when ready, or press Enter to stop: "
    ).strip().casefold() == "retry"


if __name__ == "__main__":
    main()
