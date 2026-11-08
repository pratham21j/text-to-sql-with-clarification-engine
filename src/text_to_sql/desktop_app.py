"""CustomTkinter interface for clarification-aware read-only SQL queries."""

import csv
import queue
import sqlite3
import threading
import tkinter
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

import customtkinter as ctk
import sqlglot
from sqlglot.errors import ParseError

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
    GeminiProviderError,
    GeminiQuotaError,
    GeminiRateLimitError,
    OpenAIProviderError,
    OpenAIQuotaError,
    OpenAIRateLimitError,
    OpenRouterProviderError,
    OpenRouterRateLimitError,
    ProviderConfigurationError,
    create_provider,
)
from text_to_sql.sql_executor import DEFAULT_MAX_ROWS
from text_to_sql.sql_generator import InvalidModelResponseError, get_database_schema
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
    active_provider = provider if provider is not None else create_provider()
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


def format_sql_preview(sql: str | None) -> str:
    """Format a validated query for display without changing its execution text."""
    if not sql:
        return ""
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(
            pretty=True,
            dialect="sqlite",
        )
    except (ParseError, ValueError):
        return sql


def format_desktop_error(error: object) -> str:
    """Return a useful, credential-safe message for errors shown in the GUI."""
    if isinstance(error, (OpenAIQuotaError, OpenAIRateLimitError)):
        return str(error)
    if isinstance(error, OpenRouterRateLimitError):
        return str(error)
    if isinstance(error, OpenAIProviderError):
        return str(error)
    if isinstance(error, OpenRouterProviderError):
        return str(error)
    if isinstance(error, GeminiQuotaError):
        return str(error)
    if isinstance(error, GeminiRateLimitError):
        return str(error)
    if isinstance(error, GeminiProviderError):
        diagnostic = (error.diagnostic or "").casefold()
        message = str(error).casefold()
        if "http 503" in diagnostic or "unavailable" in diagnostic:
            return (
                "Gemini is temporarily unavailable (HTTP 503). Your message "
                "and conversation are preserved; try again later."
            )
        if "http 400" in diagnostic or "invalid_argument" in diagnostic:
            return (
                "Gemini rejected this request (HTTP 400). Check the question "
                "and configured model, then try again."
            )
        if "network request failed" in message or "network" in message:
            return (
                "Gemini could not be reached. Check your network connection; "
                "your message and conversation are preserved."
            )
        return (
            "Gemini could not complete the request. Your message and "
            "conversation are preserved; try again when ready."
        )
    if isinstance(error, (
        ProviderConfigurationError,
        InvalidModelResponseError,
        ClarificationResponseError,
        SQLValidationError,
        sqlite3.Error,
        OSError,
        ValueError,
    )):
        return str(error)
    return "An unexpected error occurred. Please try again."


class TextToSQLDesktopApp(ctk.CTk):
    """Desktop chat UI; worker threads never access widgets."""

    _PENDING_STATUSES = {"clarification_needed", "unsupported"}

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        turn_handler: Callable[..., DesktopTurnResult] = process_desktop_message,
    ) -> None:
        super().__init__()
        self.title("Text-to-SQL Clarification Engine")
        self.geometry("1240x820")
        self.minsize(900, 640)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.database_path = Path(database_path)
        self.max_rows = max_rows
        self._turn_handler = turn_handler
        self.conversation = create_conversation()
        self._worker_results: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._displayed_sql = ""
        self._displayed_columns: tuple[str, ...] = ()
        self._displayed_rows: tuple[tuple[object, ...], ...] = ()

        self.grid_columnconfigure(0, weight=1, uniform="main")
        self.grid_columnconfigure(1, weight=1, uniform="main")
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
        header = ctk.CTkFrame(self, corner_radius=14)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(18, 12))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="TEXT-TO-SQL WITH CLARIFICATION ENGINE",
            font=ctk.CTkFont(size=21, weight="bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="w", padx=20, pady=(16, 2))
        subtitle = ctk.CTkLabel(
            header,
            text="Natural Language Database Query System",
            text_color=("#64748B", "#A7B3C5"),
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 8))
        self.conversation_label = ctk.CTkLabel(
            header,
            text="",
            text_color=("#64748B", "#A7B3C5"),
            font=ctk.CTkFont(size=11),
        )
        self.conversation_label.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 14))
        self.database_info_label = ctk.CTkLabel(
            header,
            text=self._database_description(),
            text_color=("#64748B", "#8996A8"),
            anchor="w",
            justify="left",
            wraplength=680,
            font=ctk.CTkFont(size=10),
        )
        self.database_info_label.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=20,
            pady=(0, 14),
        )
        self.ready_indicator = ctk.CTkLabel(
            header,
            text="READY",
            text_color="#86EFAC",
            fg_color=("#DCFCE7", "#163627"),
            corner_radius=12,
            font=ctk.CTkFont(size=11, weight="bold"),
            padx=12,
            pady=6,
        )
        self.ready_indicator.grid(row=0, column=1, sticky="e", padx=(8, 20), pady=(16, 6))
        self.new_conversation_button = ctk.CTkButton(
            header,
            text="Clear Conversation",
            command=self.new_conversation,
            width=154,
            height=36,
        )
        self.new_conversation_button.grid(
            row=1, column=1, rowspan=2, padx=(8, 20), pady=(0, 14)
        )

    def _build_chat(self) -> None:
        chat_frame = ctk.CTkFrame(self, corner_radius=14)
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=(18, 8), pady=(0, 18))
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            chat_frame,
            text="Conversation",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))

        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            wrap="word",
            font=ctk.CTkFont(size=13),
            activate_scrollbars=True,
            corner_radius=10,
        )
        self.chat_display.tag_config(
            "user_label",
            foreground="#60A5FA",
            underline=True,
        )
        self.chat_display.tag_config(
            "assistant_label",
            foreground="#AAB8CA",
            underline=True,
        )
        self.chat_display.tag_config(
            "user_message",
            foreground="#DBEAFE",
            background="#172B46",
            lmargin1=12,
            lmargin2=12,
            rmargin=24,
            spacing1=4,
            spacing3=10,
        )
        self.chat_display.tag_config(
            "assistant_message",
            foreground="#E2E8F0",
            background="#202B3A",
            lmargin1=12,
            lmargin2=12,
            rmargin=24,
            spacing1=4,
            spacing3=10,
        )
        self.chat_display.grid(row=1, column=0, sticky="nsew", padx=14, pady=(4, 10))
        self.chat_display.configure(state="disabled")

        self.clarification_panel = ctk.CTkFrame(
            chat_frame,
            corner_radius=10,
            border_width=1,
            border_color=("#93C5FD", "#315A83"),
            fg_color=("#EFF6FF", "#17283B"),
        )
        self.clarification_panel.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.clarification_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.clarification_panel,
            text="CLARIFICATION NEEDED",
            text_color=("#1D4ED8", "#93C5FD"),
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(9, 2))
        self.clarification_question_label = ctk.CTkLabel(
            self.clarification_panel,
            text="",
            anchor="w",
            justify="left",
            wraplength=360,
            font=ctk.CTkFont(size=12),
        )
        self.clarification_question_label.grid(
            row=1, column=0, sticky="ew", padx=12, pady=(0, 9)
        )
        self.clarification_panel.grid_remove()

        composer = ctk.CTkFrame(chat_frame, fg_color="transparent")
        composer.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        composer.grid_columnconfigure(0, weight=1)
        self.message_hint = ctk.CTkLabel(
            composer,
            text="Ask a question about the academic database. Ctrl+Enter to submit.",
            anchor="w",
            text_color=("#64748B", "#A7B3C5"),
            font=ctk.CTkFont(size=10),
        )
        self.message_hint.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.message_entry = ctk.CTkTextbox(
            composer,
            height=96,
            wrap="word",
            font=ctk.CTkFont(size=13),
            corner_radius=10,
            activate_scrollbars=True,
        )
        self.message_entry.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.message_entry.bind("<Control-Return>", self._on_enter)
        self.send_button = ctk.CTkButton(
            composer,
            text="Generate Query",
            command=self.submit_message,
            width=148,
            height=38,
        )
        self.send_button.grid(row=2, column=0, sticky="e")

        self.status_label = ctk.CTkLabel(
            chat_frame,
            text="Ready",
            anchor="w",
            text_color=("#64748B", "#A7B3C5"),
            font=ctk.CTkFont(size=11),
        )
        self.status_label.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 12))

    def _build_results_panel(self) -> None:
        panel = ctk.CTkFrame(self, corner_radius=14)
        panel.grid(row=1, column=1, sticky="nsew", padx=(8, 18), pady=(0, 18))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(
            panel,
            text="Generated SQL",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            panel,
            text="Validated read-only query",
            text_color=("#64748B", "#A7B3C5"),
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.sql_preview = ctk.CTkTextbox(
            panel,
            height=128,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=10,
        )
        self.sql_preview.grid(row=2, column=0, sticky="ew", padx=14)
        self.sql_preview.configure(state="disabled")

        sql_actions = ctk.CTkFrame(panel, fg_color="transparent")
        sql_actions.grid(row=3, column=0, sticky="ew", padx=14, pady=(6, 0))
        sql_actions.grid_columnconfigure(1, weight=1)
        self.copy_sql_button = ctk.CTkButton(
            sql_actions,
            text="Copy SQL",
            command=self.copy_sql,
            width=100,
            height=30,
            state="disabled",
        )
        self.copy_sql_button.grid(row=0, column=0, sticky="w")
        self.sql_action_status_label = ctk.CTkLabel(
            sql_actions,
            text="",
            anchor="e",
            text_color=("#64748B", "#A7B3C5"),
            font=ctk.CTkFont(size=10),
        )
        self.sql_action_status_label.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        results_summary = ctk.CTkFrame(panel, fg_color="transparent")
        results_summary.grid(row=4, column=0, sticky="ew", padx=14, pady=(14, 6))
        results_summary.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            results_summary,
            text="Query Results",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.execution_status_label = ctk.CTkLabel(
            results_summary,
            text="No query executed",
            anchor="w",
            wraplength=340,
            text_color=("#64748B", "#A7B3C5"),
            font=ctk.CTkFont(size=11),
        )
        self.execution_status_label.grid(row=1, column=0, sticky="ew")
        self.export_csv_button = ctk.CTkButton(
            results_summary,
            text="Export CSV",
            command=self.export_results_csv,
            width=100,
            height=30,
            state="disabled",
        )
        self.export_csv_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 0))

        table_frame = ctk.CTkFrame(panel, corner_radius=10)
        table_frame.grid(row=5, column=0, sticky="nsew", padx=14, pady=(0, 14))
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Result.Treeview",
            rowheight=28,
            font=("Segoe UI", 10),
            background="#111827",
            fieldbackground="#111827",
            foreground="#E2E8F0",
            borderwidth=0,
        )
        style.configure(
            "Result.Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            background="#263449",
            foreground="#F8FAFC",
            relief="flat",
        )
        style.map(
            "Result.Treeview",
            background=[("selected", "#1D4ED8")],
            foreground=[("selected", "#FFFFFF")],
        )
        style.map(
            "Result.Treeview.Heading",
            background=[("active", "#34465F")],
            foreground=[("active", "#FFFFFF")],
        )
        style.configure(
            "TScrollbar",
            background="#263449",
            troughcolor="#111827",
            bordercolor="#111827",
            arrowcolor="#CBD5E1",
            relief="flat",
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
        message = self._get_message_text().strip()
        if not message:
            return

        pending = self.conversation.clarification_status in self._PENDING_STATUSES
        self._append_message("you", message)
        self._clear_message_text()
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
            self._worker_results.put(("error", (error, message)))
            return
        self._worker_results.put(("result", result))

    def _poll_worker_results(self) -> None:
        try:
            while True:
                event, payload = self._worker_results.get_nowait()
                if event == "error":
                    if (
                        isinstance(payload, tuple)
                        and len(payload) == 2
                        and isinstance(payload[1], str)
                    ):
                        self._show_error(payload[0], retry_message=payload[1])
                    else:
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
            self.clarification_question_label.configure(text=question)
            self.clarification_panel.grid()
            self._update_input_mode(pending=True)
            self._append_message("assistant", question)
            self._set_busy(False, "Clarification needed")
            self.message_entry.focus_set()
            return

        self.clarification_panel.grid_remove()
        self._update_input_mode(pending=False)
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
        execution = result.execution
        self._displayed_sql = format_sql_preview(sql)
        self.copy_sql_button.configure(
            state="normal" if self._displayed_sql else "disabled"
        )
        self.sql_action_status_label.configure(text="")
        self.sql_preview.configure(state="normal")
        self.sql_preview.delete("1.0", "end")
        self.sql_preview.insert("1.0", self._displayed_sql)
        self.sql_preview.configure(state="disabled")

        if execution.status != "success":
            self._displayed_columns = ()
            self._displayed_rows = ()
            self.export_csv_button.configure(state="disabled")
            self.execution_status_label.configure(
                text=execution.error or "Query failed."
            )
            self._replace_table((), ())
            return

        status = f"{execution.row_count} row(s) returned"
        if execution.truncated:
            status += f" (limited to {self.max_rows}; more rows available)"
        if execution.row_count == 0:
            status = "No rows matched this query (0 rows returned)."
        self._displayed_columns = execution.columns
        self._displayed_rows = execution.rows
        self.export_csv_button.configure(
            state="normal" if execution.columns else "disabled"
        )
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
        for index, (identifier, label) in enumerate(zip(identifiers, columns)):
            self.result_table.heading(identifier, text=label)
            measured_width = max(
                [
                    len(label),
                    *(
                        len("" if row[index] is None else str(row[index]))
                        for row in rows
                    ),
                ]
            )
            width = min(360, max(100, measured_width * 8 + 24))
            self.result_table.column(
                identifier,
                width=width,
                minwidth=80,
                stretch=True,
                anchor="w",
            )
        for index, row in enumerate(rows):
            self.result_table.insert(
                "",
                "end",
                values=tuple("" if value is None else str(value) for value in row),
                tags=("evenrow" if index % 2 == 0 else "oddrow",),
            )
        self.result_table.tag_configure("evenrow", background="#111827")
        self.result_table.tag_configure("oddrow", background="#182338")

    def _show_error(
        self,
        error: object,
        *,
        retry_message: str | None = None,
    ) -> None:
        message = format_desktop_error(error)
        self._append_message("assistant", f"Unable to complete the request: {message}")
        self.execution_status_label.configure(text=message)
        if retry_message and not self._get_message_text().strip():
            self._set_message_text(retry_message)
        self._set_busy(False, "Request failed")
        self._update_input_mode(
            pending=self.conversation.clarification_status in self._PENDING_STATUSES
        )

    def _get_message_text(self) -> str:
        if isinstance(self.message_entry, ctk.CTkTextbox):
            return self.message_entry.get("1.0", "end-1c")
        return self.message_entry.get()

    def _clear_message_text(self) -> None:
        if isinstance(self.message_entry, ctk.CTkTextbox):
            self.message_entry.delete("1.0", "end")
        else:
            self.message_entry.delete(0, "end")

    def _set_message_text(self, message: str) -> None:
        if isinstance(self.message_entry, ctk.CTkTextbox):
            self.message_entry.insert("1.0", message)
        else:
            self.message_entry.insert(0, message)

    def _update_input_mode(self, *, pending: bool) -> None:
        if pending:
            self.message_hint.configure(
                text="Enter your clarification answer below. Ctrl+Enter to submit."
            )
            self.send_button.configure(text="Submit Clarification")
        else:
            self.message_hint.configure(
                text="Ask a question about the academic database. Ctrl+Enter to submit."
            )
            self.send_button.configure(text="Generate Query")

    def _append_message(self, role: str, message: str) -> None:
        is_user = role.casefold() in {"you", "user"}
        label_tag = "user_label" if is_user else "assistant_label"
        message_tag = "user_message" if is_user else "assistant_message"
        self.chat_display.configure(state="normal")
        self.chat_display.insert(
            "end",
            f"{role.capitalize()}\n",
            (label_tag,),
        )
        self.chat_display.insert("end", f"{message}\n\n", (message_tag,))
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _clear_results(self) -> None:
        self._displayed_sql = ""
        self._displayed_columns = ()
        self._displayed_rows = ()
        self.sql_preview.configure(state="normal")
        self.sql_preview.delete("1.0", "end")
        self.sql_preview.configure(state="disabled")
        self.copy_sql_button.configure(state="disabled")
        self.export_csv_button.configure(state="disabled")
        self.sql_action_status_label.configure(text="")
        self.execution_status_label.configure(text="Waiting for a resolved query")
        self._replace_table((), ())

    def copy_sql(self) -> None:
        """Copy only the displayed, already validated SQL preview."""
        if not self._displayed_sql:
            self.sql_action_status_label.configure(text="No SQL query is available to copy.")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(self._displayed_sql)
        except tkinter.TclError:
            self.sql_action_status_label.configure(
                text="Could not access the clipboard. Select and copy SQL from the preview."
            )
            return
        self.sql_action_status_label.configure(text="SQL copied to clipboard.")

    def export_results_csv(self) -> None:
        """Save the currently displayed result set without rerunning its query."""
        if not self._displayed_columns:
            self.execution_status_label.configure(
                text="Run a query with result columns before exporting."
            )
            return
        try:
            destination = filedialog.asksaveasfilename(
                title="Export query results",
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
                initialfile="query_results.csv",
            )
        except tkinter.TclError:
            self.execution_status_label.configure(
                text="The CSV save dialog could not be opened."
            )
            return
        if not destination:
            return
        try:
            with Path(destination).open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self._displayed_columns)
                writer.writerows(self._displayed_rows)
        except (OSError, csv.Error, UnicodeError):
            self.execution_status_label.configure(
                text="Could not save the CSV file. Check the destination and permissions."
            )
            return
        self.execution_status_label.configure(
            text=f"Exported {len(self._displayed_rows)} row(s) to CSV."
        )

    def _database_description(self) -> str:
        database_path = self.database_path.expanduser().resolve()
        try:
            schema = get_database_schema(database_path)
        except (OSError, sqlite3.Error):
            tables = "unavailable"
        else:
            tables = ", ".join(table.name for table in schema.tables) or "none"
        return f"Database: {database_path} | Tables: {tables}"

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        widget_state = "disabled" if busy else "normal"
        self.send_button.configure(state=widget_state)
        self.message_entry.configure(state=widget_state)
        self.new_conversation_button.configure(state=widget_state)
        self.status_label.configure(text=status)
        self.ready_indicator.configure(
            text="PROCESSING" if busy else "READY",
            text_color="#BFDBFE" if busy else "#86EFAC",
            fg_color=("#DBEAFE", "#1E3A5F") if busy else ("#DCFCE7", "#163627"),
        )

    def new_conversation(self) -> None:
        if self._busy:
            return
        self.conversation = create_conversation()
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self._clear_message_text()
        self.clarification_panel.grid_remove()
        self._update_input_mode(pending=False)
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
