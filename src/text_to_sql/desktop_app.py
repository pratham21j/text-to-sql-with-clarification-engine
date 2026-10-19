"""CustomTkinter interface for clarification-aware read-only SQL queries."""

import queue
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Callable

import customtkinter as ctk

from text_to_sql.clarification_engine import (
    ClarificationResult,
    ClarificationResponseError,
    StructuredResponseProvider,
)
from text_to_sql.conversation_context import (
    ConversationExecutionResult,
    ConversationState,
    answer_conversation_clarification,
    create_conversation,
    execute_conversation_query,
    process_user_question,
)
from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.llm_provider import (
    GeminiConfigurationError,
    GeminiProvider,
    GeminiProviderError,
)
from text_to_sql.sql_executor import DEFAULT_MAX_ROWS
from text_to_sql.sql_generator import InvalidModelResponseError
from text_to_sql.sql_validator import SQLValidationError


@dataclass(frozen=True, slots=True)
class DesktopTurnResult:
    clarification: ClarificationResult
    execution: ConversationExecutionResult | None


def process_desktop_message(
    conversation: ConversationState,
    message: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    provider: StructuredResponseProvider | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> DesktopTurnResult:
    """Process a user turn and execute only after clarification is resolved."""
    active_provider = provider if provider is not None else GeminiProvider()
    if conversation.clarification_status in {
        "clarification_needed",
        "unsupported",
    }:
        clarification = answer_conversation_clarification(
            conversation,
            message,
            database_path,
            provider=active_provider,
        )
    else:
        clarification = process_user_question(
            conversation,
            message,
            database_path,
            provider=active_provider,
        )

    if clarification.status != "clear" or clarification.clarification_needed:
        return DesktopTurnResult(clarification=clarification, execution=None)

    execution = execute_conversation_query(
        conversation,
        database_path,
        provider=active_provider,
        max_rows=max_rows,
    )
    return DesktopTurnResult(clarification=clarification, execution=execution)


def format_result_table(
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> str:
    """Format result values for a readable text fallback and UI accessibility."""
    if not columns:
        return "No result columns."
    widths = [
        max(
            len(column),
            *(len("" if row[index] is None else str(row[index])) for row in rows),
        )
        for index, column in enumerate(columns)
    ]
    header = " | ".join(
        column.ljust(widths[index]) for index, column in enumerate(columns)
    )
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(
            ("" if value is None else str(value)).ljust(widths[index])
            for index, value in enumerate(row)
        )
        for row in rows
    ]
    return "\n".join((header, separator, *body))


class TextToSQLDesktopApp(ctk.CTk):
    """Desktop chat UI; worker threads never access widgets."""

    _PENDING_STATUSES = {"clarification_needed", "unsupported"}
    _SAFE_ERRORS = (
        GeminiConfigurationError,
        GeminiProviderError,
        InvalidModelResponseError,
        ClarificationResponseError,
        SQLValidationError,
        sqlite3.Error,
        OSError,
        ValueError,
    )

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        turn_handler: Callable[..., DesktopTurnResult] = process_desktop_message,
    ) -> None:
        super().__init__()
        self.title("Text-to-SQL Clarification Engine")
        self.geometry("1180x760")
        self.minsize(900, 600)
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.database_path = Path(database_path)
        self.max_rows = max_rows
        self._turn_handler = turn_handler
        self.conversation = create_conversation()
        self._worker_results: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_chat()
        self._build_results_panel()
        self._append_message(
            "assistant",
            "Welcome. Ask a question about the academic database, or refine a "
            "previous request. I will ask before guessing when clarification "
            "is needed.",
        )
        self._update_conversation_label()
        self.after(100, self._poll_worker_results)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(14, 10))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Text-to-SQL Clarification Engine",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 2))
        self.conversation_label = ctk.CTkLabel(
            header,
            text="",
            text_color=("gray45", "gray70"),
        )
        self.conversation_label.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 12))
        self.new_conversation_button = ctk.CTkButton(
            header,
            text="New conversation",
            command=self.new_conversation,
            width=150,
        )
        self.new_conversation_button.grid(
            row=0, column=1, rowspan=2, padx=16, pady=12
        )

    def _build_chat(self) -> None:
        chat_frame = ctk.CTkFrame(self)
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 12))
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=1)

        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            wrap="word",
            font=ctk.CTkFont(size=14),
            activate_scrollbars=True,
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.chat_display.configure(state="disabled")

        composer = ctk.CTkFrame(chat_frame, fg_color="transparent")
        composer.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        composer.grid_columnconfigure(0, weight=1)
        self.message_entry = ctk.CTkEntry(
            composer,
            placeholder_text="Ask about students, courses, or examinations...",
            height=42,
            font=ctk.CTkFont(size=14),
        )
        self.message_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.message_entry.bind("<Return>", self._on_enter)
        self.send_button = ctk.CTkButton(
            composer,
            text="Send",
            command=self.submit_message,
            width=100,
            height=42,
        )
        self.send_button.grid(row=0, column=1)

        self.status_label = ctk.CTkLabel(
            chat_frame,
            text="Ready",
            anchor="w",
            text_color=("gray45", "gray70"),
        )
        self.status_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))

    def _build_results_panel(self) -> None:
        panel = ctk.CTkFrame(self)
        panel.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 12))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            panel,
            text="SQL preview",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        self.sql_preview = ctk.CTkTextbox(
            panel,
            height=135,
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.sql_preview.grid(row=1, column=0, sticky="ew", padx=12)
        self.sql_preview.configure(state="disabled")

        self.execution_status_label = ctk.CTkLabel(
            panel,
            text="No query executed",
            anchor="w",
            wraplength=420,
        )
        self.execution_status_label.grid(
            row=2, column=0, sticky="ew", padx=12, pady=(10, 4)
        )

        table_frame = ctk.CTkFrame(panel)
        table_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(4, 8))
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
        style = ttk.Style(self)
        style.configure("Result.Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure(
            "Result.Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
        )
        self.result_table = ttk.Treeview(
            table_frame,
            show="headings",
            style="Result.Treeview",
        )
        self.result_table.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.result_table.yview,
        )
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.result_table.xview,
        )
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.result_table.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )

    def _on_enter(self, _event: object) -> str:
        self.submit_message()
        return "break"

    def submit_message(self) -> None:
        if self._busy:
            return
        message = self.message_entry.get().strip()
        if not message:
            return

        pending = self.conversation.clarification_status in self._PENDING_STATUSES
        self._append_message("you", message)
        self.message_entry.delete(0, "end")
        self._set_busy(True, "Resolving your request..." if not pending else "Checking your answer...")
        self._clear_results()
        worker = threading.Thread(
            target=self._run_turn,
            args=(message,),
            daemon=True,
            name="text-to-sql-query",
        )
        worker.start()

    def _run_turn(self, message: str) -> None:
        try:
            result = self._turn_handler(
                self.conversation,
                message,
                self.database_path,
                max_rows=self.max_rows,
            )
        except Exception as error:
            self._worker_results.put(("error", error))
            return
        self._worker_results.put(("result", result))

    def _poll_worker_results(self) -> None:
        try:
            while True:
                event, payload = self._worker_results.get_nowait()
                if event == "error":
                    self._show_error(payload)
                else:
                    self._show_turn_result(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_worker_results)

    def _show_turn_result(self, payload: object) -> None:
        if not isinstance(payload, DesktopTurnResult):
            self._show_error(RuntimeError("Unexpected background response."))
            return
        clarification = payload.clarification
        if clarification.status in self._PENDING_STATUSES:
            question = clarification.clarification_question or (
                "Please clarify the request."
            )
            self._append_message("assistant", question)
            self.message_entry.configure(
                placeholder_text="Type your clarification answer..."
            )
            self._set_busy(False, "Clarification needed")
            self.message_entry.focus_set()
            return

        self.message_entry.configure(
            placeholder_text="Ask a question or refine the previous request..."
        )
        if payload.execution is None:
            self._show_error(RuntimeError("No execution result was returned."))
            return
        self._show_execution(payload.execution)
        execution = payload.execution.execution
        if execution.status == "success":
            self._append_message(
                "assistant",
                f"Query completed. {execution.row_count} row(s) returned."
                + (" Results were limited by the row cap." if execution.truncated else ""),
            )
            self._set_busy(False, "Query completed")
        else:
            self._append_message("assistant", execution.error or "Query failed.")
            self._set_busy(False, "Query failed")

    def _show_execution(self, result: ConversationExecutionResult) -> None:
        sql = result.sql.generated_sql or ""
        self.sql_preview.configure(state="normal")
        self.sql_preview.delete("1.0", "end")
        self.sql_preview.insert("1.0", sql)
        self.sql_preview.configure(state="disabled")

        execution = result.execution
        if execution.status != "success":
            self.execution_status_label.configure(
                text=execution.error or "Query failed."
            )
            self._replace_table((), ())
            return

        status = f"{execution.row_count} row(s) returned"
        if execution.truncated:
            status += f" (limited to {self.max_rows}; more rows available)"
        if execution.row_count == 0:
            status = "No rows matched this query."
        self.execution_status_label.configure(text=status)
        self._replace_table(execution.columns, execution.rows)

    def _replace_table(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[object, ...], ...],
    ) -> None:
        self.result_table.delete(*self.result_table.get_children())
        identifiers = tuple(f"column_{index}" for index in range(len(columns)))
        self.result_table.configure(columns=identifiers)
        for identifier, label in zip(identifiers, columns):
            self.result_table.heading(identifier, text=label)
            self.result_table.column(identifier, width=130, minwidth=80, stretch=True)
        for row in rows:
            self.result_table.insert(
                "",
                "end",
                values=tuple("" if value is None else str(value) for value in row),
            )

    def _show_error(self, error: object) -> None:
        if isinstance(error, self._SAFE_ERRORS):
            message = str(error)
        else:
            message = "An unexpected error occurred. Please try again."
        self._append_message("assistant", f"Unable to complete the request: {message}")
        self.execution_status_label.configure(text=message)
        self._set_busy(False, "Request failed")
        self.message_entry.configure(
            placeholder_text="Ask a question or refine the previous request..."
        )

    def _append_message(self, role: str, message: str) -> None:
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{role.capitalize()}\n{message}\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _clear_results(self) -> None:
        self.sql_preview.configure(state="normal")
        self.sql_preview.delete("1.0", "end")
        self.sql_preview.configure(state="disabled")
        self.execution_status_label.configure(text="Waiting for a resolved query")
        self._replace_table((), ())

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        widget_state = "disabled" if busy else "normal"
        self.send_button.configure(state=widget_state)
        self.message_entry.configure(state=widget_state)
        self.new_conversation_button.configure(state=widget_state)
        self.status_label.configure(text=status)

    def new_conversation(self) -> None:
        if self._busy:
            return
        self.conversation = create_conversation()
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.message_entry.configure(
            placeholder_text="Ask about students, courses, or examinations..."
        )
        self._clear_results()
        self._append_message(
            "assistant",
            "A new conversation has started. Previous context will not be used.",
        )
        self._update_conversation_label()
        self._set_busy(False, "Ready")

    def _update_conversation_label(self) -> None:
        self.conversation_label.configure(
            text=f"Conversation {self.conversation.conversation_id[:8]}"
        )


def main() -> None:
    app = TextToSQLDesktopApp()
    app.mainloop()
