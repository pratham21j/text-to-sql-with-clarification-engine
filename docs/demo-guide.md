# Viva demonstration guide

This guide uses the synthetic academic database. It does not promise exact
result rows because query wording and Gemini's interpretation can vary.

## Prepare the demonstration

For the live desktop flow, follow the source setup and launch instructions in
the [README](../README.md), configure `GEMINI_API_KEY` and `GEMINI_MODEL`
locally, and confirm that Gemini is reachable and has available quota. The
application can open without making an API request, but asking a question
requires the configured provider.

The source database contains `students`, `departments`, `courses`,
`enrollments`, and `examinations`. Student ranking can use `students.cgpa` or
examination marks from `examinations`; attendance is not stored. The data is
synthetic and deterministic.

## Live desktop walkthrough

| Step | User input or action | Expected behavior | What to point out |
| --- | --- | --- | --- |
| 1. Launch | Run `python scripts\run_desktop.py` from the project root. | The CustomTkinter window opens and displays the database path and available tables. | This is a Windows desktop application, not a website. |
| 2. Clear request | `Show all CSE students.` | The request is understood without an unnecessary clarification; validated SQL runs and the results appear. | The generated SQL preview, CSE department filter/join, and result table. |
| 3. Reset context | Click **New conversation**. | A clean conversation starts. | Earlier filters are not carried into a separate conversation. |
| 4. Ambiguous request | `Show the best students.` | The app asks which supported measure should define “best,” rather than selecting a ranking criterion silently. | The schema has CGPA and examination marks, but no single “best” measure. |
| 5. Clarify | `Rank them by CGPA, highest first.` | The original request and answer are combined into a resolved intent; validated SQL and results follow. | The clarification response is retained as part of the conversation. |
| 6. Follow-up | `Only show the top 3.` | The app interprets the follow-up using the active ranking context and generates a bounded top-three query. | The previous CGPA ranking is preserved while the limit is refined. |
| 7. Inspect and export | Review the SQL preview and rows, then click **Export CSV** and choose a destination. | The displayed results are saved as CSV without rerunning the query. | The preview is separate from execution, and export uses the displayed columns and rows. |

Model responses can differ in wording and formatting. If a step receives an
unexpected interpretation, start a new conversation and rephrase it; do not
claim a particular row count or exact result values unless you have observed
them in that run.

## Safe rejection example

The desktop input accepts natural-language questions, not raw SQL. To
demonstrate write-operation rejection reproducibly without putting the bundled
database at risk, run the regression test:

```powershell
python -m pytest tests\test_sql_executor.py -k data_modifying
```

The test supplies `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and `CREATE`
statements to the executor using a temporary database. Each is rejected, and
the test verifies the temporary database remains unchanged. Point out that
generated SQL is parsed and checked before execution, and permitted queries
also run through a read-only SQLite connection.

## Offline preparation when Gemini is unavailable

These checks use mocked model responses or temporary local databases and do
not make live Gemini requests:

```powershell
python -m pytest tests\test_cli_workflow.py tests\test_desktop_app.py
python -m pytest tests\test_sql_executor.py tests\test_sql_generation.py
```

The first command exercises the CLI and desktop workflow with mocked provider
responses. Headless GUI tests cover formatting, results, export behavior, and
error recovery; native window and file-dialog interaction depends on the
desktop environment. The second command exercises schema, SQL validation,
read-only execution, and row limits against test databases.

These automated checks are not a substitute for the live Gemini-backed
walkthrough: they verify deterministic application behavior without depending
on a live model response or API quota.
