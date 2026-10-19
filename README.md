# Text-to-SQL with Clarification Engine

A Windows desktop project that will turn natural-language questions into safe
SQLite queries. Phases 1–3 establish the project, synthetic academic database,
Gemini provider, validated SQL generation, and a schema-grounded clarification
workflow. Query execution and the CustomTkinter interface are planned for later
phases.

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
Once resolved, the original question and all clarification turns are included
in the SQL-generation request. Output contains the structured clarification
result and SQL-generation result. Generated SQL is parsed and prepared against
the database using SQLite's read-only `EXPLAIN`; it is **not executed**.
Automated clarification and SQL-generation tests use mocked responses and do
not make API calls.

The Python API can also be used without the interactive CLI:

```python
from text_to_sql.clarification_engine import (
    ClarificationResult,
    analyze_question,
    generate_sql_after_clarification,
    submit_clarification_answer,
)

analysis = analyze_question("Show students with good marks.")
if analysis.clarification_needed:
    clarification_question = analysis.clarification_question
    if clarification_question is None:
        raise ValueError("Missing clarification question")
    answer = input(clarification_question + " ")
    analysis = submit_clarification_answer(
        analysis.original_question,
        clarification_question,
        answer,
        clarification_history=analysis.clarification_history,
    )
if analysis.status == "clear":
    sql_preview = generate_sql_after_clarification(analysis)
```

Persist `analysis.to_dict()` if conversation state needs to survive process
restarts; restore it with `ClarificationResult.from_dict(...)`. Submit each
answer with the original question and returned `clarification_history` to
preserve context across multiple turns.

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

Clarification analysis is Gemini-backed and depends on the quality of the
configured model's response; automated tests use mocks, and live API behavior
has not been verified. The structured clarification state is serializable, but
there is not yet a desktop conversation interface or built-in persistent
conversation store. SQL is validated but not executed, and result limits and
query execution controls are not implemented. PyInstaller packaging is
optional and has not been configured.
