# Text-to-SQL with Clarification Engine

## Phase 1: Foundation

- [x] Inspect the development environment.
- [x] Create the project structure.
- [x] Configure Python dependencies.
- [x] Configure environment variables.
- [x] Create the SQLite database initialization script.
- [x] Create the academic schema.
- [x] Generate deterministic sample data.
- [x] Verify database connectivity and relationships with automated tests.
- [x] Document setup and execution instructions in README.md.

## Phase 2: Text-to-SQL

- [x] Integrate the official Gemini SDK through a configurable provider (mocked SDK tests; no live request).
- [x] Retrieve table, column, type, and foreign-key metadata from SQLite.
- [x] Implement and validate structured LLM responses.
- [x] Generate and validate SQL for clear questions.
- [ ] Test filters, sorting, aggregations, and joins.

## Phase 3: Clarification Engine

- [x] Define clear, clarification-needed, and unsupported result statuses.
- [x] Detect vague, incomplete, and unsupported requests using the live schema context.
- [x] Generate targeted clarification questions.
- [x] Incorporate custom clarification responses.
- [x] Preserve the original request and serialized clarification history.
- [x] Resolve the intended query after sufficient clarification.
- [x] Support multiple explicit clarification turns.
- [x] Test conversation-state management with mocked providers.
- [x] Block SQL generation until clarification is resolved; retain context when generation proceeds.

## Phase 4: Conversation Context Management

- [x] Create unique, explicit conversation state with ordered events.
- [x] Append user questions and record assistant clarifications and answers.
- [x] Serialize and restore validated state using dictionaries or JSON.
- [x] Preserve the active resolved request for follow-up understanding.
- [x] Resolve pronoun references and refinements using explicit conversation context.
- [x] Preserve prior filters when follow-up requests sort or limit results.
- [x] Keep separate conversations isolated and block unresolved SQL generation.
- [x] Cover conversation flows and provider failures with mocked tests.

## Phase 5: SQL Safety

- [x] Implement SQL parsing.
- [x] Restrict SQL previews to read-only SELECT queries.
- [x] Validate schema references.
- [x] Apply configurable result limits and report truncation accurately.
- [x] Execute permitted queries through read-only SQLite connections.
- [x] Revalidate SQL immediately before execution and gate execution on resolved clarification.
- [x] Add execution timeout and structured query results.
- [x] Handle provider and validation errors.
- [x] Test unsafe-query rejection, result limits, read-only access, timeouts, and workflow gating.

## Phase 6: Desktop Application

- [x] Build the CustomTkinter window.
- [x] Implement the conversation display.
- [x] Add the question input field and Send control.
- [x] Support clarification responses in the conversation input.
- [x] Add SQL preview.
- [x] Add the query-results table and execution status.
- [x] Add background processing, loading, and error states.
- [x] Integrate clarification, conversation context, SQL generation, validation, and execution.

## Phase 7: Testing and Delivery

- [x] Run final automated tests (126 passed).
- [x] Fix any final test failures.
- [x] Test the desktop send, clarification, SQL-generation, execution, and result-display workflow with mocked provider responses.
- [x] Verify original CLI question and clarification history remain together; verify desktop row rendering headlessly.
- [x] Update README.md with desktop launch instructions.
- [x] Document setup and configuration.
- [x] Report Gemini quota exhaustion and retry guidance without automatic long waits.
- [x] Preserve failed clarification input and require manual retries for quota limits.
- [x] Prepare demonstration questions.
- [x] Verify desktop entry-point startup in the available Windows environment.
- [ ] Test optional PyInstaller packaging.

## Rules

- Work through the phases in order.
- Update this checklist as work progresses.
- Do not mark untested features as complete.
- Document blockers and required user actions.
- Preserve existing work.
- Keep the implementation appropriate for a minor project.
