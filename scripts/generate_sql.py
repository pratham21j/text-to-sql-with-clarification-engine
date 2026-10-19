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
    generate_conversation_sql,
    process_user_question,
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
        conversation = create_conversation()
        clarification = process_user_question(
            conversation,
            arguments.question,
            database_path,
            provider=provider,
        )
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
            clarification = answer_conversation_clarification(
                conversation,
                answer,
                database_path,
                provider=provider,
            )
        sql_result = generate_conversation_sql(
            conversation,
            database_path,
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
    ) as error:
        parser.error(str(error))
    except EOFError:
        parser.error("Input ended before clarification was resolved.")

    print(
        json.dumps(
            {
                "clarification": asdict(clarification),
                "sql": asdict(sql_result),
                "conversation": conversation.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
