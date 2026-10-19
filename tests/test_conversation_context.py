import json
import uuid

import pytest
from google.genai import errors

from text_to_sql.clarification_engine import (
    ClarificationRequiredError,
    ClarificationResult,
    ClarificationTurn,
)
from text_to_sql.conversation_context import (
    ConversationState,
    ConversationStateError,
    PendingClarificationError,
    answer_conversation_clarification,
    create_conversation,
    execute_conversation_query,
    generate_conversation_sql,
    process_user_question,
)
from text_to_sql.database import initialize_database
from text_to_sql.llm_provider import (
    GeminiProvider,
    GeminiProviderError,
    GeminiQuotaError,
)


class _FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_json(self, prompt, response_schema):
        self.prompts.append((prompt, response_schema))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)


def _clear(resolved_question, *, context_action="replace"):
    return {
        "status": "clear",
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_reason": None,
        "resolved_question": resolved_question,
        "context_action": context_action,
    }


def _needs_clarification(
    question="Which department should be used?",
    reason="The department filter is incomplete.",
    status="clarification_needed",
    *,
    context_action="replace",
):
    return {
        "status": status,
        "clarification_needed": True,
        "clarification_question": question,
        "clarification_reason": reason,
        "resolved_question": None,
        "context_action": context_action,
    }


def _sql_response(sql="SELECT student_id FROM students"):
    return json.dumps(
        {
            "status": "ready",
            "generated_sql": sql,
            "ambiguity_reason": None,
            "clarification_needed": False,
        }
    )


def _database(tmp_path):
    return initialize_database(tmp_path / "academic.db")


def test_create_conversation_has_a_unique_id_and_clean_state():
    first = create_conversation()
    second = create_conversation()

    assert str(uuid.UUID(first.conversation_id)) == first.conversation_id
    assert first.conversation_id != second.conversation_id
    assert first.history == []
    assert first.active_query_context is None
    assert first.clarification_status == "idle"
    assert first.current_result is None


def test_appending_multiple_user_turns_preserves_order():
    state = create_conversation()
    state.append_user_turn("Show all CSE students.")
    state.append_user_turn("Sort them by CGPA descending.")

    assert [event.content for event in state.relevant_history()] == [
        "Show all CSE students.",
        "Sort them by CGPA descending.",
    ]
    assert [event.kind for event in state.history] == ["question", "question"]


def test_assistant_clarification_and_answer_are_recorded_in_order(tmp_path):
    state = create_conversation()
    provider = _FakeProvider(
        _needs_clarification(),
        _clear("Show students in the CSE department."),
    )
    database_path = _database(tmp_path)

    process_user_question(
        state,
        "Show students from a department.",
        database_path,
        provider=provider,
    )
    answer_conversation_clarification(
        state,
        "CSE.",
        database_path,
        provider=provider,
    )

    assert [(event.role, event.kind) for event in state.history] == [
        ("user", "question"),
        ("assistant", "clarification"),
        ("user", "clarification_answer"),
        ("assistant", "resolved_request"),
    ]
    assert [event.content for event in state.history] == [
        "Show students from a department.",
        "Which department should be used?",
        "CSE.",
        "Show students in the CSE department.",
    ]
    assert state.clarification_status == "clear"


def test_conversation_state_serializes_and_restores_with_order_and_id(tmp_path):
    state = create_conversation()
    provider = _FakeProvider(
        _needs_clarification(),
        _clear("Show students in the CSE department."),
    )
    database_path = _database(tmp_path)
    process_user_question(
        state,
        "Show students from a department.",
        database_path,
        provider=provider,
    )
    answer_conversation_clarification(
        state,
        "CSE",
        database_path,
        provider=provider,
    )

    restored = ConversationState.from_json(state.to_json())

    assert restored == state
    assert restored.conversation_id == state.conversation_id
    assert restored.active_query_context == (
        "Show students in the CSE department."
    )
    assert restored.current_result == state.current_result
    assert [event.content for event in restored.relevant_history()] == [
        event.content for event in state.history
    ]


@pytest.mark.parametrize(
    "serialized",
    [
        "{not json",
        "[]",
        json.dumps({}),
        json.dumps(
            {
                "conversation_id": "not-a-uuid",
                "history": [],
                "active_query_context": None,
                "clarification_status": "idle",
                "current_result": None,
            }
        ),
        json.dumps(
            {
                "conversation_id": str(uuid.uuid4()),
                "history": [{"role": "alien", "kind": "question", "content": "bad"}],
                "active_query_context": None,
                "clarification_status": "idle",
                "current_result": None,
            }
        ),
        json.dumps(
            {
                "conversation_id": str(uuid.uuid4()),
                "history": [],
                "active_query_context": None,
                "clarification_status": "clear",
                "current_result": None,
            }
        ),
    ],
)
def test_malformed_or_incomplete_serialized_state_is_rejected(serialized):
    with pytest.raises(ConversationStateError):
        ConversationState.from_json(serialized)


def test_restoring_state_missing_latest_assistant_event_is_rejected(tmp_path):
    state = create_conversation()
    process_user_question(
        state,
        "Show students with CGPA above 8.",
        _database(tmp_path),
        provider=_FakeProvider(_clear("Show students with CGPA above 8.")),
    )
    incomplete = state.to_dict()
    incomplete["history"] = incomplete["history"][:-1]

    with pytest.raises(ConversationStateError, match="latest resolved request"):
        ConversationState.from_dict(incomplete)


def test_follow_up_resolves_pronouns_against_explicit_previous_context(tmp_path):
    state = create_conversation()
    database_path = _database(tmp_path)
    provider = _FakeProvider(
        _clear("Show students with CGPA above 8."),
        _clear(
            "Show students with CGPA above 8, sorted by CGPA descending.",
            context_action="refine",
        ),
    )

    process_user_question(
        state,
        "Show students with CGPA above 8.",
        database_path,
        provider=provider,
    )
    result = process_user_question(
        state,
        "Sort them by CGPA descending.",
        database_path,
        provider=provider,
    )

    follow_up_prompt = provider.prompts[1][0]
    assert "Previously resolved request:" in follow_up_prompt
    assert "Show students with CGPA above 8." in follow_up_prompt
    assert "Sort them by CGPA descending." in follow_up_prompt
    assert "preserve every prior filter and constraint" in follow_up_prompt
    assert result.resolved_question == (
        "Show students with CGPA above 8, sorted by CGPA descending."
    )
    assert "Show students with CGPA above 8." in state.active_query_context
    assert result.resolved_question in state.active_query_context


def test_sorting_and_limit_refinements_preserve_previous_filters_and_sql_context(
    tmp_path,
):
    state = create_conversation()
    database_path = _database(tmp_path)
    analysis_provider = _FakeProvider(
        _clear("Show students with CGPA above 8."),
        _clear(
            "Show students with CGPA above 8, sorted by CGPA descending.",
            context_action="refine",
        ),
        _clear(
            "Show the top 3 students with CGPA above 8, sorted by CGPA descending.",
            context_action="refine",
        ),
    )
    process_user_question(
        state,
        "Show students with CGPA above 8.",
        database_path,
        provider=analysis_provider,
    )
    process_user_question(
        state,
        "Sort them by CGPA descending.",
        database_path,
        provider=analysis_provider,
    )
    process_user_question(
        state,
        "Only show the top 3.",
        database_path,
        provider=analysis_provider,
    )
    sql_provider = _FakeProvider(
        json.loads(
            _sql_response(
                "SELECT student_id, cgpa FROM students "
                "WHERE cgpa > 8 ORDER BY cgpa DESC LIMIT 3"
            )
        )
    )

    sql_result = generate_conversation_sql(
        state,
        database_path,
        provider=sql_provider,
    )

    assert sql_result.generated_sql.endswith("LIMIT 3")
    sql_prompt = sql_provider.prompts[0][0]
    assert "CGPA above 8" in sql_prompt
    assert "sorted by CGPA descending" in sql_prompt
    assert "top 3" in sql_prompt
    assert "preserve applicable filters" in sql_prompt


def test_follow_up_without_previous_context_does_not_invent_referent(tmp_path):
    state = create_conversation()
    provider = _FakeProvider(
        _needs_clarification(
            "Which students should 'them' refer to?",
            "There is no previous resolved query in this conversation.",
        )
    )

    result = process_user_question(
        state,
        "Sort them by CGPA descending.",
        _database(tmp_path),
        provider=provider,
    )

    assert result.status == "clarification_needed"
    assert result.resolved_question is None
    assert "Previously resolved request:" not in provider.prompts[0][0]
    assert state.active_query_context is None


def test_ambiguous_follow_up_asks_for_clarification_and_does_not_overwrite_context(
    tmp_path,
):
    state = create_conversation()
    database_path = _database(tmp_path)
    provider = _FakeProvider(
        _clear("Show students with CGPA above 8."),
        _needs_clarification(
            "Which ranking criterion should be used?",
            "The word 'best' does not specify an ordering criterion.",
            context_action="refine",
        ),
    )
    process_user_question(
        state,
        "Show students with CGPA above 8.",
        database_path,
        provider=provider,
    )
    result = process_user_question(
        state,
        "Show the best ones.",
        database_path,
        provider=provider,
    )

    assert result.status == "clarification_needed"
    assert state.clarification_status == "clarification_needed"
    assert state.active_query_context == "Show students with CGPA above 8."
    assert state.history[-1].kind == "clarification"


def test_follow_up_clarification_keeps_prior_query_and_answer_context(tmp_path):
    state = create_conversation()
    database_path = _database(tmp_path)
    provider = _FakeProvider(
        _clear("Show students with CGPA above 8."),
        _needs_clarification(
            "Which ranking criterion should be applied to those students?",
            "The word 'best' does not identify a ranking column.",
            context_action="refine",
        ),
        _clear(
            "Show students with CGPA above 8, ranked by CGPA descending.",
            context_action="refine",
        ),
    )
    process_user_question(
        state,
        "Show students with CGPA above 8.",
        database_path,
        provider=provider,
    )
    process_user_question(
        state,
        "Show the best ones.",
        database_path,
        provider=provider,
    )
    resolved = answer_conversation_clarification(
        state,
        "Rank them by CGPA descending.",
        database_path,
        provider=provider,
    )

    assert resolved.status == "clear"
    assert "Show students with CGPA above 8." in state.active_query_context
    assert resolved.resolved_question in state.active_query_context
    answer_prompt = provider.prompts[2][0]
    assert "Show students with CGPA above 8." in answer_prompt
    assert "Show the best ones." in answer_prompt
    assert "Rank them by CGPA descending." in answer_prompt


def test_separate_conversations_do_not_share_context(tmp_path):
    first = create_conversation()
    second = create_conversation()
    database_path = _database(tmp_path)
    first_provider = _FakeProvider(_clear("Show students with CGPA above 8."))
    process_user_question(
        first,
        "Show students with CGPA above 8.",
        database_path,
        provider=first_provider,
    )
    second_provider = _FakeProvider(
        _needs_clarification(
            "Which students should 'them' refer to?",
            "This conversation has no prior resolved request.",
        )
    )

    result = process_user_question(
        second,
        "Sort them by CGPA descending.",
        database_path,
        provider=second_provider,
    )

    assert result.status == "clarification_needed"
    assert "Show students with CGPA above 8." not in second_provider.prompts[0][0]
    assert second.active_query_context is None
    assert first.active_query_context == "Show students with CGPA above 8."


def test_independent_request_replaces_active_context_without_old_query_leak(
    tmp_path,
):
    state = create_conversation()
    database_path = _database(tmp_path)
    provider = _FakeProvider(
        _clear("Show students with CGPA above 8."),
        _clear("Show all courses."),
        _needs_clarification(
            "Which courses should be sorted?",
            "There is no course query to refine in the active request.",
        ),
    )
    process_user_question(
        state,
        "Show students with CGPA above 8.",
        database_path,
        provider=provider,
    )
    process_user_question(
        state,
        "Show all courses.",
        database_path,
        provider=provider,
    )
    result = process_user_question(
        state,
        "Sort them by credits.",
        database_path,
        provider=provider,
    )

    assert result.status == "clarification_needed"
    third_prompt = provider.prompts[2][0]
    assert "Previously resolved request:\nShow all courses." in third_prompt
    assert "Show students with CGPA above 8." not in third_prompt
    assert state.active_query_context == "Show all courses."


def test_new_question_is_rejected_until_pending_clarification_is_answered(
    tmp_path,
):
    state = create_conversation()
    provider = _FakeProvider(_needs_clarification())
    database_path = _database(tmp_path)
    process_user_question(
        state,
        "Show students from a department.",
        database_path,
        provider=provider,
    )

    with pytest.raises(PendingClarificationError):
        process_user_question(
            state,
            "Show all courses.",
            database_path,
            provider=provider,
        )

    assert len(provider.prompts) == 1


def test_unresolved_or_unsupported_conversation_cannot_generate_sql(tmp_path):
    for status in ("clarification_needed", "unsupported"):
        state = create_conversation()
        provider = _FakeProvider(
            _needs_clarification(status=status)
        )
        process_user_question(
            state,
            "Show the best students.",
            _database(tmp_path / status),
            provider=provider,
        )
        sql_provider = _FakeProvider(
            json.loads(_sql_response())
        )

        with pytest.raises(ClarificationRequiredError):
            generate_conversation_sql(
                state,
                provider=sql_provider,
            )
        assert sql_provider.prompts == []


def test_quota_error_during_initial_analysis_preserves_conversation(tmp_path):
    state = create_conversation()
    initial = state.to_dict()
    provider = _FakeProvider(
        GeminiQuotaError(
            "Gemini quota is exhausted. No automatic retry was made.",
            retry_after_seconds=54000,
            quota_exhausted=True,
        )
    )

    with pytest.raises(GeminiQuotaError):
        process_user_question(
            state,
            "Show the 5 best students.",
            _database(tmp_path),
            provider=provider,
        )

    assert state.to_dict() == initial
    assert state.clarification_status == "idle"
    assert state.active_query_context is None


def test_failed_clarification_follow_up_is_preserved_and_can_be_retried(tmp_path):
    state = create_conversation()
    database_path = _database(tmp_path)
    process_user_question(
        state,
        "Show students from a department.",
        database_path,
        provider=_FakeProvider(_needs_clarification()),
    )

    with pytest.raises(GeminiQuotaError):
        answer_conversation_clarification(
            state,
            "CSE students",
            database_path,
            provider=_FakeProvider(
                GeminiQuotaError(
                    "Gemini quota is exhausted.",
                    retry_after_seconds=54000,
                    quota_exhausted=True,
                )
            ),
        )

    assert state.clarification_status == "clarification_needed"
    assert state.history[-1].kind == "clarification_answer"
    assert state.history[-1].content == "CSE students"
    assert ConversationState.from_json(state.to_json()) == state

    retry_provider = _FakeProvider(
        _clear("Show students in the CSE department.")
    )
    result = answer_conversation_clarification(
        state,
        "Use the CSE department.",
        database_path,
        provider=retry_provider,
    )

    assert result.status == "clear"
    assert state.history[-1].kind == "resolved_request"
    assert [event.content for event in state.history if event.kind == "clarification_answer"] == [
        "CSE students",
        "Use the CSE department.",
    ]
    retry_prompt = retry_provider.prompts[0][0]
    assert "CSE students" in retry_prompt
    assert "Use the CSE department." in retry_prompt


def test_quota_error_during_sql_generation_keeps_resolved_request(tmp_path):
    state = create_conversation()
    database_path = _database(tmp_path)
    process_user_question(
        state,
        "Show all CSE students.",
        database_path,
        provider=_FakeProvider(_clear("Show all CSE students.")),
    )
    before = state.to_dict()

    with pytest.raises(GeminiQuotaError):
        execute_conversation_query(
            state,
            database_path,
            provider=_FakeProvider(
                GeminiQuotaError(
                    "Gemini quota is exhausted.",
                    retry_after_seconds=54000,
                    quota_exhausted=True,
                )
            ),
        )

    assert state.to_dict() == before
    assert state.clarification_status == "clear"
    assert state.current_result is not None
    assert state.current_result.resolved_question == "Show all CSE students."


def test_sdk_api_error_is_reported_without_secret_details(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "GEMINI_API_KEY=test-only-key\nGEMINI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    class _Models:
        def generate_content(self, **kwargs):
            del kwargs
            raise errors.APIError(
                429,
                {
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Quota exceeded for key=test-only-key.",
                    }
                },
            )

    state = create_conversation()
    provider = GeminiProvider(
        client=type("Client", (), {"models": _Models()})(),
        env_file=env_file,
    )

    with pytest.raises(GeminiProviderError) as error:
        process_user_question(
            state,
            "Show CSE students.",
            _database(tmp_path),
            provider=provider,
        )

    assert "HTTP 429" in str(error.value)
    assert "RESOURCE_EXHAUSTED" in str(error.value)
    assert "Quota exceeded" in str(error.value)
    assert "test-only-key" not in str(error.value)
    assert "[REDACTED]" in str(error.value)


def test_pending_clarification_round_trips_and_preserves_state(tmp_path):
    state = create_conversation()
    process_user_question(
        state,
        "Show students from a department.",
        _database(tmp_path),
        provider=_FakeProvider(_needs_clarification()),
    )

    restored = ConversationState.from_dict(state.to_dict())

    assert restored == state
    assert restored.clarification_status == "clarification_needed"
    assert restored.current_result is not None
    assert restored.current_result.original_question == (
        "Show students from a department."
    )
