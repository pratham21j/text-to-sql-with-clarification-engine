"""Schema-grounded clarification and explicit conversation-state handling."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.llm_provider import GeminiProvider
from text_to_sql.sql_generator import (
    DatabaseSchema,
    SQLGenerationResult,
    generate_sql,
    get_database_schema,
)


class ClarificationResponseError(ValueError):
    """Raised when the model response violates the clarification contract."""


class ClarificationRequiredError(ValueError):
    """Raised when SQL generation is requested before clarification is resolved."""


class StructuredResponseProvider(Protocol):
    def generate_json(
        self,
        prompt: str,
        response_schema: dict[str, object],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ClarificationTurn:
    clarification_question: str
    clarification_answer: str


@dataclass(frozen=True, slots=True)
class ClarificationResult:
    """Serializable analysis result; valid statuses are clear, clarification_needed, unsupported."""

    status: str
    original_question: str
    clarification_needed: bool
    clarification_question: str | None
    clarification_reason: str | None
    clarification_answer: str | None
    resolved_question: str | None
    clarification_history: tuple[ClarificationTurn, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable conversation state for persistence or follow-ups."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ClarificationResult":
        """Restore a result from its serialized representation."""
        expected_keys = {
            "status",
            "original_question",
            "clarification_needed",
            "clarification_question",
            "clarification_reason",
            "clarification_answer",
            "resolved_question",
            "clarification_history",
        }
        if set(payload) != expected_keys:
            raise ClarificationResponseError(
                "Serialized clarification state has missing or unexpected fields."
            )

        history_value = payload["clarification_history"]
        if not isinstance(history_value, (list, tuple)):
            raise ClarificationResponseError(
                "Serialized clarification history must be a list or tuple."
            )

        history = tuple(_coerce_turn(turn) for turn in history_value)
        result = cls(
            status=payload["status"],
            original_question=payload["original_question"],
            clarification_needed=payload["clarification_needed"],
            clarification_question=payload["clarification_question"],
            clarification_reason=payload["clarification_reason"],
            clarification_answer=payload["clarification_answer"],
            resolved_question=payload["resolved_question"],
            clarification_history=history,
        )
        _validate_result(result)
        return result


_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "status": {
            "type": "STRING",
            "enum": ["clear", "clarification_needed", "unsupported"],
        },
        "clarification_needed": {"type": "BOOLEAN"},
        "clarification_question": {
            "type": "STRING",
            "nullable": True,
        },
        "clarification_reason": {
            "type": "STRING",
            "nullable": True,
        },
        "resolved_question": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "A complete natural-language request preserving the original "
                "intent and all answered clarifications, or null when unresolved."
            ),
        },
    },
    "required": [
        "status",
        "clarification_needed",
        "clarification_question",
        "clarification_reason",
        "resolved_question",
    ],
    "propertyOrdering": [
        "status",
        "clarification_needed",
        "clarification_question",
        "clarification_reason",
        "resolved_question",
    ],
}


def _coerce_turn(value: object) -> ClarificationTurn:
    if isinstance(value, ClarificationTurn):
        question = value.clarification_question
        answer = value.clarification_answer
    elif isinstance(value, Mapping) and set(value) == {
        "clarification_question",
        "clarification_answer",
    }:
        question = value["clarification_question"]
        answer = value["clarification_answer"]
    else:
        raise ClarificationResponseError(
            "Clarification history contains an invalid conversation turn."
        )
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        raise ClarificationResponseError(
            "Clarification history turns require a question and answer."
        )
    return ClarificationTurn(question, answer)


def _validate_result(result: ClarificationResult) -> None:
    if not isinstance(result.status, str) or result.status not in {
        "clear",
        "clarification_needed",
        "unsupported",
    }:
        raise ClarificationResponseError(
            "Clarification status must be clear, clarification_needed, or unsupported."
        )
    if not isinstance(result.original_question, str) or not result.original_question.strip():
        raise ClarificationResponseError(
            "Clarification state must preserve a non-empty original question."
        )
    if not isinstance(result.clarification_needed, bool):
        raise ClarificationResponseError(
            "clarification_needed must be a boolean."
        )
    if result.clarification_answer is not None and not isinstance(
        result.clarification_answer, str
    ):
        raise ClarificationResponseError(
            "clarification_answer must be a string or null."
        )
    if not isinstance(result.clarification_history, tuple):
        raise ClarificationResponseError(
            "Clarification history must be an immutable sequence of turns."
        )
    for turn in result.clarification_history:
        if not isinstance(turn, ClarificationTurn):
            raise ClarificationResponseError(
                "Clarification history must contain valid clarification turns."
            )
        _coerce_turn(turn)
    if result.status == "clear":
        if result.clarification_needed:
            raise ClarificationResponseError(
                "A clear result cannot require clarification."
            )
        if not isinstance(result.resolved_question, str) or not result.resolved_question.strip():
            raise ClarificationResponseError(
                "A clear result requires a resolved question."
            )
        if result.clarification_question is not None or result.clarification_reason is not None:
            raise ClarificationResponseError(
                "A clear result cannot include a pending clarification."
            )
        return

    if not result.clarification_needed:
        raise ClarificationResponseError(
            "Unresolved or unsupported results must require clarification."
        )
    if not isinstance(result.clarification_question, str) or not result.clarification_question.strip():
        raise ClarificationResponseError(
            "A clarification-needed result requires a targeted question."
        )
    if not isinstance(result.clarification_reason, str) or not result.clarification_reason.strip():
        raise ClarificationResponseError(
            "A clarification-needed result requires a reason."
        )
    if result.resolved_question is not None:
        raise ClarificationResponseError(
            "An unresolved result cannot include a resolved question."
        )


def _parse_model_response(response_text: str) -> tuple[
    str, bool, str | None, str | None, str | None
]:
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ClarificationResponseError(
            "Gemini clarification response was not valid JSON."
        ) from error

    required_fields = {
        "status",
        "clarification_needed",
        "clarification_question",
        "clarification_reason",
        "resolved_question",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ClarificationResponseError(
            "Gemini clarification response must contain exactly the required fields."
        )

    status = payload["status"]
    clarification_needed = payload["clarification_needed"]
    clarification_question = payload["clarification_question"]
    clarification_reason = payload["clarification_reason"]
    resolved_question = payload["resolved_question"]
    if (
        not isinstance(status, str)
        or not isinstance(clarification_needed, bool)
        or (
            clarification_question is not None
            and not isinstance(clarification_question, str)
        )
        or (
            clarification_reason is not None
            and not isinstance(clarification_reason, str)
        )
        or (
            resolved_question is not None
            and not isinstance(resolved_question, str)
        )
    ):
        raise ClarificationResponseError(
            "Gemini clarification response contains fields with invalid types."
        )

    return (
        status,
        clarification_needed,
        clarification_question,
        clarification_reason,
        resolved_question,
    )


def _build_prompt(
    original_question: str,
    schema: DatabaseSchema,
    *,
    clarification_history: Sequence[ClarificationTurn] = (),
    latest_clarification_question: str | None = None,
    latest_clarification_answer: str | None = None,
) -> str:
    schema_json = json.dumps(asdict(schema), indent=2)
    history_json = json.dumps(
        [asdict(turn) for turn in clarification_history],
        indent=2,
    )
    follow_up = ""
    if latest_clarification_question is not None:
        follow_up = (
            "\nMost recent clarification question:\n"
            f"{latest_clarification_question}\n"
            "User's answer to that question:\n"
            f"{latest_clarification_answer}\n"
        )

    return f"""Analyze the user's database request using only the supplied SQLite schema.

Do not guess the meaning of vague conditions, missing filters, unspecified
ordering, or ambiguous references. Ask one targeted clarification at a time
when the user must choose an interpretation. If the requested data is absent
from the schema, do not invent a table or column: use status "unsupported",
explain the limitation, and ask whether an available alternative would work.
For a complete, supported request, use status "clear" and produce a
resolved_question that preserves the original intent and all clarification
answers. Do not generate or execute SQL.

Use exactly these statuses:
- "clear": intent and required details are sufficiently specified and supported
- "clarification_needed": a missing or ambiguous detail needs the user's answer
- "unsupported": the requested data cannot be represented by this schema

Return exactly one JSON object with these fields:
- status
- clarification_needed: false only for "clear"; true otherwise
- clarification_question: null for "clear"; otherwise a concise targeted question
- clarification_reason: null for "clear"; otherwise why the request cannot proceed
- resolved_question: a complete request for "clear"; null otherwise

SQLite schema (table names, columns, types, and relationships):
{schema_json}

Clarification history already answered:
{history_json}

Original user question (preserve its intent):
{original_question}
{follow_up}
"""


def _analyze(
    original_question: str,
    schema: DatabaseSchema,
    *,
    provider: StructuredResponseProvider,
    clarification_answer: str | None,
    clarification_history: tuple[ClarificationTurn, ...],
    latest_clarification_question: str | None = None,
) -> ClarificationResult:
    response_text = provider.generate_json(
        _build_prompt(
            original_question,
            schema,
            clarification_history=clarification_history,
            latest_clarification_question=latest_clarification_question,
            latest_clarification_answer=clarification_answer,
        ),
        _RESPONSE_SCHEMA,
    )
    (
        status,
        clarification_needed,
        clarification_question,
        clarification_reason,
        resolved_question,
    ) = _parse_model_response(response_text)
    result = ClarificationResult(
        status=status,
        original_question=original_question,
        clarification_needed=clarification_needed,
        clarification_question=clarification_question,
        clarification_reason=clarification_reason,
        clarification_answer=clarification_answer,
        resolved_question=resolved_question,
        clarification_history=clarification_history,
    )
    _validate_result(result)
    return result


def analyze_question(
    question: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
) -> ClarificationResult:
    """Classify a new question against the actual database schema."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    schema = get_database_schema(database_path)
    active_provider = provider if provider is not None else GeminiProvider()
    return _analyze(
        question,
        schema,
        provider=active_provider,
        clarification_answer=None,
        clarification_history=(),
    )


def submit_clarification_answer(
    original_question: str,
    clarification_question: str,
    clarification_answer: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
    clarification_history: Sequence[ClarificationTurn | Mapping[str, object]] = (),
) -> ClarificationResult:
    """Resolve a clarification turn, retaining prior context for later turns."""
    for label, value in (
        ("Original question", original_question),
        ("Clarification question", clarification_question),
        ("Clarification answer", clarification_answer),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string.")

    prior_turns = tuple(_coerce_turn(turn) for turn in clarification_history)
    turns = (*prior_turns, ClarificationTurn(clarification_question, clarification_answer))
    schema = get_database_schema(database_path)
    active_provider = provider if provider is not None else GeminiProvider()
    return _analyze(
        original_question,
        schema,
        provider=active_provider,
        clarification_answer=clarification_answer,
        clarification_history=turns,
        latest_clarification_question=clarification_question,
    )


def generate_sql_after_clarification(
    result: ClarificationResult,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
) -> SQLGenerationResult:
    """Generate validated SQL only for a clear result, retaining the full context."""
    _validate_result(result)
    if result.status != "clear" or result.resolved_question is None:
        raise ClarificationRequiredError(
            "Resolve the pending clarification before generating SQL."
        )

    request_parts = [f"Original request: {result.original_question}"]
    request_parts.extend(
        f"Clarification: {turn.clarification_question}\n"
        f"User answer: {turn.clarification_answer}"
        for turn in result.clarification_history
    )
    request_parts.append(f"Resolved request: {result.resolved_question}")
    return generate_sql(
        "\n".join(request_parts),
        database_path,
        provider=provider,
    )
