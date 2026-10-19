# Text-to-SQL with Clarification Engine

A Windows desktop project that turns natural-language questions into safe
SQLite queries. Phases 1–5 establish the project, synthetic academic database,
Gemini provider, validated SQL generation, schema-grounded clarification,
explicit conversation context, and bounded read-only query execution. Phase 6
adds the CustomTkinter desktop interface.

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

## Clarify a question and run a safe query

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
unnecessary follow-up. Once clarification is resolved, the CLI generates SQL,
validates it, executes the read-only query, and prints the SQL and structured
results as JSON. To use another existing SQLite database:

```powershell
python scripts\generate_sql.py --database data\demo.db "Show all CSE students."
```

Set the CLI result cap with `--max-rows` (default `100`):

```powershell
python scripts\generate_sql.py --max-rows 250 "Show all CSE students."
```

## Launch the desktop application

After installing the requirements, initializing `data\academic.db`, and
configuring `GEMINI_API_KEY` and `GEMINI_MODEL` as described above, launch the
CustomTkinter application from the project root:

```powershell
python scripts\run_desktop.py
```

The window provides a conversation view, a message field, a new-conversation
button, a SQL preview, and a results table. While Gemini or SQLite is working,
the request runs in a background thread, the interface displays a loading
status, and duplicate submissions are disabled. When a question needs
clarification, answer in the same input field; the answer and original request
remain in the active conversation context. Starting a new conversation clears
the displayed history and creates an isolated conversation state.

Only the existing conversation workflow generates, validates, and executes SQL.
Unresolved clarification blocks generation and execution. Results show the
returned row count, whether the row cap was reached, and a no-matches message
when appropriate. The GUI does not persist conversation state to disk.

Clarification analysis and each answer are sent to the configured Gemini model.
Once resolved, the original question, prior resolved request, and all
clarification turns are included in the SQL-generation request. The SQL is
validated both during generation and again immediately before execution.
Automated clarification, SQL-generation, and SQL-execution tests use mocked
model responses and do not make API calls.

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
    execute_conversation_query,
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
    query_result = execute_conversation_query(conversation)
```

`ConversationState.to_dict()` and `.to_json()` preserve the conversation ID,
ordered user/assistant events, current resolved query, clarification status,
and pending clarification result. Restore with `from_dict()` or `from_json()`;
malformed, inconsistent, or incomplete state is rejected explicitly. Each
conversation is an independent object with a unique ID; no history is held in
module-level global state.

## Read-only query execution and result limits

The executor opens SQLite in read-only URI mode, enables SQLite's
`query_only` safeguard, and validates the SQL again before running it. Only one
supported read-only SELECT query is accepted. Clarification must be resolved
and SQL generation must return a ready, validated query before the integrated
conversation workflow will execute anything.

`execute_sql(sql, database_path, max_rows=100)` returns a
`SQLExecutionResult` with:

- `status`: `success` or `error`
- `columns`: column names in query order
- `rows`: returned row values, in query order
- `row_count`: number of rows returned (not the total number of matches)
- `truncated`: true only when an additional row was found beyond the cap
- `error`: a safe user-facing message, or null on success

The default maximum is **100 rows**. Set `max_rows` to another positive
integer when calling the Python API; the query itself is not rewritten, and at
most `max_rows + 1` rows are fetched to determine whether results were
truncated. This preserves the query's filter, ordering, and own LIMIT
semantics. Execution is interrupted after a default **5-second** time limit;
both `max_rows` and `timeout_seconds` can be supplied to the API. An error
result has no partial rows or columns.

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
has not been verified. Conversation snapshots are serializable, but the GUI
does not persist conversation state to disk. Returned results are capped and
no total-match count is computed when truncated. PyInstaller packaging is
optional and has not been configured.
