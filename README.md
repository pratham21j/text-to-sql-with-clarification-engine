# Text-to-SQL with Clarification Engine

A Windows desktop project that turns natural-language questions into safe
SQLite queries. Phases 1–5 establish the project, synthetic academic database,
configurable LLM providers, validated SQL generation, schema-grounded clarification,
explicit conversation context, and bounded read-only query execution. Phase 6
adds the CustomTkinter desktop interface.

## Overview and features

The application analyzes natural-language questions against the live SQLite
schema, asks for clarification instead of guessing when intent is materially
ambiguous, and carries resolved context into follow-up questions. It previews
validated SQL, executes supported read-only queries with a row cap, and displays
results in a table that can be exported to CSV.

The project uses Python, CustomTkinter, SQLite, the official OpenAI SDK,
`google-genai`, SQLGlot, `python-dotenv`, pytest, Git, and PyInstaller.
The selected provider is used only for question analysis and SQL generation;
SQL validation, execution, and result display remain local.

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

If `.env` does not already exist, copy `.env.example` to `.env`. The default
provider is OpenAI. Configure these values in `.env` or as environment
variables:

```text
AI_PROVIDER=openai
AI_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your_openai_api_key
```

The [GPT-4.1 mini model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
lists `gpt-4.1-mini` as supporting the [Responses
API](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [structured
outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
The OpenAI provider uses the Responses API with strict JSON-schema output;
model responses still go through the application's existing validation and
read-only SQLite execution path.
Model access and API quota are account-specific and cannot be verified without
an authenticated API request; no request is made during setup or automated
tests. OpenAI API usage may incur charges. A ChatGPT subscription does not
provide API credits.

To explicitly select the retained Gemini provider instead, set
`AI_PROVIDER=gemini` and configure `GEMINI_API_KEY` and `GEMINI_MODEL`. The
application does not fall back to another provider if the selected provider
fails. Preserve an existing `.env` when setting up. Keep it private; `.env` is
excluded by `.gitignore`.

To use OpenRouter, create an API key in your OpenRouter account and set:

```text
AI_PROVIDER=openrouter
AI_MODEL=xingchen-agi/xing4.0-29b-a4b:free
OPENROUTER_API_KEY=your_openrouter_api_key
```

The OpenRouter provider uses the OpenAI-compatible Chat Completions endpoint
with strict JSON Schema output. The model catalog's structured-output filter
listed `xingchen-agi/xing4.0-29b-a4b:free` and
`nvidia/nemotron-3.5-super-vl:free` as free candidates when checked. These are
examples, not a guarantee of continued availability, free access, account
eligibility, or support from every provider endpoint. Check the current
[structured outputs guide](https://openrouter.ai/docs/guides/features/structured-outputs)
and [model catalog](https://openrouter.ai/models) before selecting a model.
The provider requires endpoint support for the requested parameters and
reports an error instead of silently falling back when the model or feature is
unavailable. OpenRouter account limits and model routing can change; no live
request is made during setup or automated tests.

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

With the selected provider configured, run the CLI:

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
configuring the selected provider as described above, launch the CustomTkinter
application from the project root:

```powershell
python scripts\run_desktop.py
```

## Build a Windows desktop distribution

Build on Windows with the same Python architecture and version you intend to
support. The runtime dependencies are listed in
`requirements-runtime.txt`; `requirements.txt` adds pytest, while
`requirements-build.txt` adds PyInstaller without including test dependencies.
From the project root, create a clean build environment and install its
dependencies:

```powershell
python -m venv .venv-build
.\.venv-build\Scripts\Activate.ps1
python -m pip install -r requirements-build.txt
```

The one-folder package includes the synthetic `data\academic.db` as a bundled
starter database. If it is missing from a fresh checkout, initialize it first:

```powershell
python scripts\init_db.py
```

The initializer refuses to replace an existing database. Do not run it over
the existing database; if it already exists, proceed to build. Create the
distribution with the checked-in PyInstaller specification:

```powershell
python -m PyInstaller --clean --noconfirm packaging\text_to_sql.spec
```

The windowed application and its support files are written to
`dist\TextToSQL\`; launch `dist\TextToSQL\TextToSQL.exe`. This is a one-folder
distribution, not a standalone single executable. The packaged synthetic
database is resolved from PyInstaller's bundled resource directory and is
opened read-only by the application. It is not silently initialized or
overwritten at launch. In a source checkout, the default remains
`data\academic.db`.

### Configure the packaged application

The build does not bundle `.env`, `.env.example`, or API credentials. Before
the first question, create a user-owned `.env` file beside
`TextToSQL.exe` (or set the same variables in the process environment):

```text
AI_PROVIDER=openai
AI_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your_openai_api_key
```

For Gemini, use `AI_PROVIDER=gemini`, `GEMINI_API_KEY`, and `GEMINI_MODEL`
instead. For OpenRouter, use `AI_PROVIDER=openrouter`,
`OPENROUTER_API_KEY`, and set `AI_MODEL` to a currently compatible model ID.
Keep this file private. The configured model is read at runtime, the same as
in source runs. Extract the distribution to a folder where you can maintain
this local configuration. Do not copy a real `.env` into `dist` or rebuild
with secrets in the project root.

To create another synthetic database without touching the packaged starter,
choose a new path when invoking the initializer, for example:

```powershell
python scripts\init_db.py --database data\academic-copy.db
```

The packaged executable uses its bundled starter database by default; using
another database requires launching the source application with that path
through `TextToSQLDesktopApp(database_path=...)`.

The window provides a conversation view, a message field, a new-conversation
button, a SQL preview, and a results table. While the configured provider or SQLite is working,
the request runs in a background thread, the interface displays a loading
status, and duplicate submissions are disabled. When a question needs
clarification, answer in the same input field; the answer and original request
remain in the active conversation context. Starting a new conversation clears
the displayed history and creates an isolated conversation state.

Only the existing conversation workflow generates, validates, and executes SQL.
Unresolved clarification blocks generation and execution. Results show the
returned row count, whether the row cap was reached, and a no-matches message
when appropriate. SQL is formatted for readability in a separate preview and
can be copied without executing it. Query results support horizontal and
vertical scrolling, readable headings and column widths, and CSV export of the
currently displayed columns and rows; exporting never reruns the query. The
header shows the configured database path and table names when the database is
available. User and assistant messages have distinct visual styles. The GUI
does not persist conversation state to disk.

Recoverable provider failures restore the submitted message for a manual retry
and preserve pending clarification context. Quota exhaustion is not
automatically retried. HTTP 503, invalid-request, and network failures are
reported with user-facing guidance without displaying provider diagnostics or
credentials.

Clarification analysis and each answer are sent to the configured provider.
Once resolved, the original question, prior resolved request, and all
clarification turns are included in the SQL-generation request. The SQL is
validated both during generation and again immediately before execution.
Automated clarification, SQL-generation, and SQL-execution tests use mocked
model responses and do not make API calls. Phase 7 integration tests exercise
the CLI from initial clarification through validated SQL and bounded SQLite
execution, and check the desktop result-table rendering without requiring a
graphical display.
Phase 8 also tests SQL formatting, clipboard and CSV behavior, row display,
and conversation error recovery with mocked system interactions. Native
clipboard and save-dialog behavior still depends on the user's desktop
environment. Phase 9 adds a reproducible one-folder PyInstaller build
configuration; package build and launch status are tracked in `TASKS.md`.

### Provider authentication, quota, and rate limits

OpenAI authentication failures explain that `OPENAI_API_KEY` and API project
access should be checked. OpenRouter authentication failures identify
`OPENROUTER_API_KEY`; HTTP 402 and quota errors direct users to check account
credits or free-tier limits. Unsupported JSON Schema output and unavailable
models are reported so a compatible model can be selected. Rate limits are
reported with any supplied retry delay, but are not automatically retried.
Temporary service failures are surfaced for a user-initiated retry. Gemini
retains its existing bounded handling of transient failures and its separate
quota/rate-limit messages. No provider silently falls back to another.

The CLI keeps the original question or resolved conversation available while
it is running and asks for explicit confirmation before retrying question
analysis or SQL generation. If a clarification request fails, the clarification
answer remains in the conversation; re-enter it when ready to retry. The
desktop keeps its conversation in memory and restores the failed message in
the input field, so retrying is user initiated. Conversation state is not
persisted across application restarts.

## Demonstration guide

See [docs/demo-guide.md](./docs/demo-guide.md) for a reproducible viva
walkthrough, including a configured-provider desktop flow and separate mocked and
local-only checks for use when the API is unavailable.

## Conversation context and follow-ups

Create a conversation to keep turns isolated. A follow-up that refers to
previous results is analyzed with the active resolved request; the configured
provider is asked
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

The CLI and desktop pass the original request and recorded clarification turns
to the resolver. If a provider request fails while answering a clarification,
the answer remains in the serializable pending conversation. The CLI reports
the failure and lets the user re-enter the answer to retry without restarting
the original question.

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

Provider calls are mocked in the automated tests; a real API key is not
required and no live API requests are made.

```powershell
python -m pytest
```

## Project layout

```text
src\text_to_sql\          Application package and desktop interface
scripts\                  Desktop, CLI, and database initialization entry points
tests\                    Mocked provider and temporary-database tests
data\academic.db          Synthetic starter database (ignored by Git)
packaging\text_to_sql.spec PyInstaller one-folder build specification
docs\demo-guide.md        Viva demonstration plan
requirements*.txt         Runtime, test, and packaging dependencies
```

## Current limitations

Follow-up interpretation depends on the quality and availability of the
configured provider and model; automated tests use mocks, and live API behavior
and account-specific model access have not been verified. API access and billing
are separate from any ChatGPT subscription. Conversation snapshots are serializable, but the CLI
and GUI do not persist conversation state to disk. Returned results are
capped and no total-match count is computed when truncated. PyInstaller output
is a Windows one-folder distribution and must be built on Windows. The
user-owned `.env` must be created beside the executable; the included starter
database is bundled read-only and alternate databases currently require the
source application's `database_path` argument.

This is an educational demonstration, not a production security guarantee.
Question and clarification text plus schema metadata are sent to the selected
provider; database query results are executed and displayed locally.
Do not submit confidential information or connect sensitive databases.
