"""Serializable conversation history and follow-up processing."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from text_to_sql.clarification_engine import (
    ClarificationRequiredError,
    ClarificationResult,
    StructuredResponseProvider,
    analyze_question,
    generate_sql_after_clarification,
    submit_clarification_answer,
)
from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.llm_provider import GeminiProvider
from text_to_sql.sql_executor import (
    DEFAULT_MAX_ROWS,
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    SQLExecutionResult,
    execute_sql,
)
from text_to_sql.sql_generator import SQLGenerationResult


class ConversationStateError(ValueError):
    """Raised when conversation state is malformed or inconsistent."""


class PendingClarificationError(ConversationStateError):
    """Raised when a new request is submitted while an answer is pending."""


@dataclass(frozen=True, slots=True)
class ConversationExecutionResult:
    sql: SQLGenerationResult
    execution: SQLExecutionResult


@dataclass(frozen=True, slots=True)
class ConversationEvent:
    role: str
    kind: str
    content: str


@dataclass(slots=True)
class ConversationState:
    """State for one conversation; instances are independent and serializable."""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[ConversationEvent] = field(default_factory=list)
    active_query_context: str | None = None
    clarification_status: str = "idle"
    current_result: ClarificationResult | None = None

    def append_user_turn(self, question: str) -> None:
        """Append a new user question in order."""
        if not isinstance(question, str) or not question.strip():
            raise ConversationStateError("User question must not be empty.")
        self.history.append(
            ConversationEvent("user", "question", question)
        )

    def record_assistant_clarification(
        self,
        result: ClarificationResult,
        *,
        previous_query_context: str | None = None,
    ) -> None:
        """Record an assistant clarification or a resolved request."""
        try:
            result = ClarificationResult.from_dict(result.to_dict())
        except (AttributeError, TypeError, ValueError) as error:
            raise ConversationStateError(
                "Cannot record a malformed clarification result."
            ) from error
        if result.status not in {
            "clear",
            "clarification_needed",
            "unsupported",
        }:
            raise ConversationStateError("Clarification result has an invalid status.")
        if result.original_question not in {
            event.content
            for event in self.history
            if event.role == "user" and event.kind == "question"
        }:
            raise ConversationStateError(
                "Record the original user question before its assistant response."
            )
        if result.clarification_needed:
            if result.clarification_question is None:
                raise ConversationStateError(
                    "Unresolved clarification result has no question."
                )
            self.history.append(
                ConversationEvent(
                    "assistant",
                    "clarification",
                    result.clarification_question,
                )
            )
            self.clarification_status = result.status
            self.current_result = result
            return

        if result.status != "clear" or result.resolved_question is None:
            raise ConversationStateError(
                "Only a clear, resolved result can update query context."
            )
        if result.context_action == "refine":
            if previous_query_context is None:
                raise ConversationStateError(
                    "A refinement requires relevant previous query context."
                )
            resolved_context = (
                "Preserved prior request and constraints:\n"
                f"{previous_query_context}\n"
                "Requested refinement:\n"
                f"{result.resolved_question}"
            )
        else:
            resolved_context = result.resolved_question
        self.history.append(
            ConversationEvent(
                "assistant",
                "resolved_request",
                resolved_context,
            )
        )
        self.active_query_context = resolved_context
        self.clarification_status = "clear"
        self.current_result = result

    def record_clarification_answer(self, answer: str) -> None:
        """Append the user's answer to the current pending clarification."""
        if not isinstance(answer, str) or not answer.strip():
            raise ConversationStateError("Clarification answer must not be empty.")
        if self.clarification_status not in {
            "clarification_needed",
            "unsupported",
        }:
            raise ConversationStateError("There is no pending clarification.")
        self.history.append(
            ConversationEvent("user", "clarification_answer", answer)
        )

    def relevant_history(self) -> tuple[ConversationEvent, ...]:
        """Return the ordered event history for context-aware follow-ups."""
        return tuple(self.history)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of this conversation."""
        return {
            "conversation_id": self.conversation_id,
            "history": [asdict(event) for event in self.history],
            "active_query_context": self.active_query_context,
            "clarification_status": self.clarification_status,
            "current_result": (
                self.current_result.to_dict()
                if self.current_result is not None
                else None
            ),
        }

    def to_json(self) -> str:
        """Serialize the conversation state as JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConversationState":
        """Restore validated state from a dictionary or JSON-decoded object."""
        required_fields = {
            "conversation_id",
            "history",
            "active_query_context",
            "clarification_status",
            "current_result",
        }
        if not isinstance(payload, Mapping) or set(payload) != required_fields:
            raise ConversationStateError(
                "Conversation state has missing or unexpected fields."
            )

        conversation_id = payload["conversation_id"]
        if not isinstance(conversation_id, str):
            raise ConversationStateError("Conversation ID must be a UUID string.")
        try:
            normalized_id = str(uuid.UUID(conversation_id))
        except (ValueError, AttributeError) as error:
            raise ConversationStateError(
                "Conversation ID must be a valid UUID."
            ) from error
        if normalized_id != conversation_id:
            raise ConversationStateError(
                "Conversation ID must use canonical UUID formatting."
            )

        raw_history = payload["history"]
        if not isinstance(raw_history, list):
            raise ConversationStateError("Conversation history must be a list.")
        history = [_restore_event(item) for item in raw_history]

        active_context = payload["active_query_context"]
        if active_context is not None and (
            not isinstance(active_context, str) or not active_context.strip()
        ):
            raise ConversationStateError(
                "Active query context must be a non-empty string or null."
            )

        status = payload["clarification_status"]
        if not isinstance(status, str) or status not in {
            "idle",
            "clear",
            "clarification_needed",
            "unsupported",
        }:
            raise ConversationStateError(
                "Conversation clarification status is invalid."
            )

        raw_result = payload["current_result"]
        if raw_result is not None and not isinstance(raw_result, Mapping):
            raise ConversationStateError(
                "Current clarification result must be an object or null."
            )
        result: ClarificationResult | None = None
        if isinstance(raw_result, Mapping):
            try:
                result = ClarificationResult.from_dict(raw_result)
            except (TypeError, ValueError) as error:
                raise ConversationStateError(
                    "Serialized current clarification result is invalid."
                ) from error

        state = cls(
            conversation_id=conversation_id,
            history=history,
            active_query_context=active_context,
            clarification_status=status,
            current_result=result,
        )
        _validate_state(state)
        return state

    @classmethod
    def from_json(cls, serialized: str) -> "ConversationState":
        """Restore state from serialized JSON and reject malformed input."""
        try:
            payload = json.loads(serialized)
        except (json.JSONDecodeError, TypeError) as error:
            raise ConversationStateError(
                "Conversation state is not valid JSON."
            ) from error
        if not isinstance(payload, Mapping):
            raise ConversationStateError(
                "Conversation state JSON must contain an object."
            )
        return cls.from_dict(payload)


def _restore_event(payload: object) -> ConversationEvent:
    if not isinstance(payload, Mapping) or set(payload) != {
        "role",
        "kind",
        "content",
    }:
        raise ConversationStateError("Conversation history contains an invalid event.")
    role = payload["role"]
    kind = payload["kind"]
    content = payload["content"]
    allowed_events = {
        ("user", "question"),
        ("assistant", "clarification"),
        ("user", "clarification_answer"),
        ("assistant", "resolved_request"),
    }
    if (
        not isinstance(role, str)
        or not isinstance(kind, str)
        or (role, kind) not in allowed_events
        or not isinstance(content, str)
        or not content.strip()
    ):
        raise ConversationStateError("Conversation history contains an invalid event.")
    return ConversationEvent(role, kind, content)


def _validate_state(state: ConversationState) -> None:
    if state.clarification_status == "idle":
        if state.current_result is not None or state.active_query_context is not None:
            raise ConversationStateError(
                "Idle conversation cannot contain resolved or pending query state."
            )
    elif state.current_result is None:
        raise ConversationStateError(
            "Non-idle conversation requires a current clarification result."
        )

    if state.current_result is None:
        return
    if not isinstance(state.current_result, ClarificationResult):
        raise ConversationStateError("Current clarification result is invalid.")
    result = state.current_result
    if result.original_question not in {
        event.content
        for event in state.history
        if event.role == "user" and event.kind == "question"
    }:
        raise ConversationStateError(
            "Conversation history does not contain the current original question."
        )
    if result.status != state.clarification_status:
        raise ConversationStateError(
            "Conversation status does not match its current result."
        )
    if result.status == "clear":
        if (
            result.resolved_question is None
            or state.active_query_context is None
            or (
                result.context_action == "replace"
                and result.resolved_question != state.active_query_context
            )
        ):
            raise ConversationStateError(
                "Active query context does not match the current resolved request."
            )
        if not state.history or state.history[-1] != ConversationEvent(
            "assistant",
            "resolved_request",
            state.active_query_context,
        ):
            raise ConversationStateError(
                "Conversation history is missing its latest resolved request."
            )
    elif not result.clarification_needed:
        raise ConversationStateError(
            "Unresolved conversation status cannot contain a resolved result."
        )
    elif (
        not state.history
        or state.history[-1]
        != ConversationEvent(
            "assistant",
            "clarification",
            result.clarification_question or "",
        )
    ):
        raise ConversationStateError(
            "Conversation history is missing its pending clarification."
        )


def create_conversation() -> ConversationState:
    """Create a clean conversation with a unique identifier."""
    return ConversationState()


def _context_for_model(state: ConversationState) -> str | None:
    if state.active_query_context is None:
        return None
    history = state.relevant_history()
    last_resolved_index = next(
        (
            index
            for index in range(len(history) - 1, -1, -1)
            if history[index].kind == "resolved_request"
        ),
        None,
    )
    active_history = (
        history[last_resolved_index:]
        if last_resolved_index is not None
        else history
    )
    recent = [
        asdict(event)
        for event in active_history
        if event.kind in {"question", "clarification_answer"}
    ]
    return (
        "Previously resolved request:\n"
        f"{state.active_query_context}\n"
        "Relevant ordered user turns:\n"
        f"{json.dumps(recent, ensure_ascii=False)}"
    )


def process_user_question(
    state: ConversationState,
    question: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
) -> ClarificationResult:
    """Analyze a question using this conversation's explicit prior context."""
    _validate_state(state)
    if state.clarification_status in {"clarification_needed", "unsupported"}:
        raise PendingClarificationError(
            "Answer the current clarification before submitting another question."
        )
    if not isinstance(question, str) or not question.strip():
        raise ConversationStateError("User question must not be empty.")

    context = _context_for_model(state)
    active_provider = provider if provider is not None else GeminiProvider()
    result = analyze_question(
        question,
        database_path,
        provider=active_provider,
        conversation_context=context,
    )
    state.append_user_turn(question)
    state.record_assistant_clarification(
        result,
        previous_query_context=state.active_query_context,
    )
    return result


def answer_conversation_clarification(
    state: ConversationState,
    answer: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
) -> ClarificationResult:
    """Answer the pending question while preserving the active query context."""
    _validate_state(state)
    current = state.current_result
    if (
        current is None
        or not current.clarification_needed
        or current.clarification_question is None
    ):
        raise ConversationStateError("There is no pending clarification to answer.")
    if not isinstance(answer, str) or not answer.strip():
        raise ConversationStateError("Clarification answer must not be empty.")

    active_provider = provider if provider is not None else GeminiProvider()
    result = submit_clarification_answer(
        current.original_question,
        current.clarification_question,
        answer,
        database_path,
        provider=active_provider,
        clarification_history=current.clarification_history,
        conversation_context=state.active_query_context,
    )
    state.record_clarification_answer(answer)
    state.record_assistant_clarification(
        result,
        previous_query_context=state.active_query_context,
    )
    return result


def generate_conversation_sql(
    state: ConversationState,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
) -> SQLGenerationResult:
    """Generate validated SQL only when this conversation has a clear request."""
    _validate_state(state)
    if state.clarification_status != "clear" or state.current_result is None:
        raise ClarificationRequiredError(
            "A clear, schema-supported request is required before SQL generation."
        )
    return generate_sql_after_clarification(
        state.current_result,
        database_path,
        provider=provider,
        conversation_context=state.active_query_context,
    )


def execute_conversation_query(
    state: ConversationState,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
) -> ConversationExecutionResult:
    """Generate and execute only after clarification is clear and SQL is ready."""
    if (
        state.clarification_status != "clear"
        or state.current_result is None
        or state.current_result.status != "clear"
        or state.current_result.clarification_needed
    ):
        raise ClarificationRequiredError(
            "A clear, schema-supported request is required before query execution."
        )

    sql_result = generate_conversation_sql(
        state,
        database_path,
        provider=provider,
    )
    if (
        sql_result.status != "ready"
        or sql_result.clarification_needed
        or sql_result.generated_sql is None
    ):
        raise ClarificationRequiredError(
            "SQL generation did not produce a ready query; nothing was executed."
        )

    execution_result = execute_sql(
        sql_result.generated_sql,
        database_path,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
    )
    return ConversationExecutionResult(
        sql=sql_result,
        execution=execution_result,
    )
