# Text-to-SQL with Clarification Engine

A Windows desktop project that will turn natural-language questions into safe
SQLite queries. Phases 1–4 establish the project, synthetic academic database,
Gemini provider, validated SQL generation, schema-grounded clarification, and
explicit conversation context. Query execution and the CustomTkinter interface
are planned for later phases.

## Requirements

- Python 3.11 or newer
- Windows (the application is intended to be a desktop app)

## Setup

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `.env` does not already exist, copy `.env.example` to `.env`. Configure
`GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` or as environment variables. The
provider requires both values and does not substitute a default model. Preserve
an existing `.env` when setting up. Keep it private; `.env` is excluded by
`.gitignore`.

## Create the demonstration database

```powershell
python scripts\init_db.py
```

This creates `data\academic.db`. To choose a different path:

```powershell
python scripts\init_db.py --database data\demo.db
```

The initializer refuses to overwrite an existing file. Choose a new path if
you want another copy. The sample data is synthetic and deterministic.

## Clarify a question and generate a SQL preview

With Gemini configuration in place, run the CLI:

```powershell
python scripts\generate_sql.py "Show students with good marks."
```

The CLI asks follow-up questions when the request is ambiguous, incomplete, or
asks for data that the database does not contain. For example, the current
schema includes `students.cgpa` and `examinations.marks`, but it has no
attendance column. The CLI will not invent attendance data; it may explain the
limitation and ask whether CGPA or examination marks would meet the request.
Clarification results use three statuses: `clear` (ready to continue),
`clarification_needed` (intent needs another answer), and `unsupported` (the
schema cannot provide the requested data).

Clear requests, such as `"Show all CSE students."`, can proceed without an
unnecessary follow-up. To use another existing SQLite database:

```powershell
python scripts\generate_sql.py --database data\demo.db "Show all CSE students."
```

Clarification analysis and each answer are sent to the configured Gemini model.
Once resolved, the original question, prior resolved request, and all
clarification turns are included in the SQL-generation request. Output contains
the structured clarification result, SQL-generation result, and serialized
conversation. Generated SQL is parsed and prepared against the database using
SQLite's read-only `EXPLAIN`; it is **not executed**.
Automated clarification and SQL-generation tests use mocked responses and do
not make API calls.

## Conversation context and follow-ups

Create a conversation to keep turns isolated. A follow-up that refers to
previous results is analyzed with the active resolved request; Gemini is asked
to preserve existing filters when sorting or limiting results. A follow-up
without an earlier resolved request receives no invented context and should be
clarified instead. A self-contained unrelated question can become the new
active request.

Example conversation:

```text
User: Show students with CGPA above 8.
User: Sort them by CGPA descending.
User: Only show the top 3.
```

The resolved request used for SQL generation retains the `CGPA above 8` filter,
the descending sort, and the top-three limit. Follow-ups that are unclear
result in another clarification rather than a guessed interpretation.

The Python API can serialize and restore a conversation between application
sessions:

```python
from text_to_sql.conversation_context import (
    ConversationState,
    answer_conversation_clarification,
    create_conversation,
    generate_conversation_sql,
    process_user_question,
)

conversation = create_conversation()
analysis = process_user_question(conversation, "Show students with CGPA above 8.")
analysis = process_user_question(conversation, "Sort them by CGPA descending.")
restored = ConversationState.from_json(conversation.to_json())
analysis = process_user_question(restored, "Only show the top 3.")
```

When clarification is pending, answer it through
`answer_conversation_clarification(conversation, answer)`. SQL generation is
only available after the current result is clear:

```python
if conversation.clarification_status == "clear":
    sql_preview = generate_conversation_sql(conversation)
```

`ConversationState.to_dict()` and `.to_json()` preserve the conversation ID,
ordered user/assistant events, current resolved query, clarification status,
and pending clarification result. Restore with `from_dict()` or `from_json()`;
malformed, inconsistent, or incomplete state is rejected explicitly. Each
conversation is an independent object with a unique ID; no history is held in
module-level global state.

## Run the tests

```powershell
python -m pytest
```

## Project layout

```text
src\text_to_sql\    Application package
scripts\            Database initialization command
tests\              Automated tests
data\               Generated, ignored SQLite database
```

## Current limitations

Follow-up interpretation is Gemini-backed and depends on the quality of the
configured model's response; automated tests use mocks, and live API behavior
has not been verified. Conversation snapshots are serializable, but there is no
built-in persistent store or desktop conversation interface yet. SQL is
validated but not executed; result limits and query execution controls are not
implemented. PyInstaller packaging is optional and has not been configured.
