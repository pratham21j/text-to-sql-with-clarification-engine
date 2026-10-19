import subprocess
import sqlite3
import sys
from pathlib import Path

import pytest

from text_to_sql.database import initialize_database

PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ROW_COUNTS = {
    "departments": 3,
    "students": 6,
    "courses": 5,
    "enrollments": 10,
    "examinations": 8,
}


def _connect(database_path):
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def test_initialization_creates_schema_and_deterministic_sample_data(tmp_path):
    database_path = initialize_database(tmp_path / "academic.db")

    with _connect(database_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert set(EXPECTED_ROW_COUNTS).issubset(table_names)

        for table_name, expected_count in EXPECTED_ROW_COUNTS.items():
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0] == expected_count

        assert connection.execute(
            "SELECT code FROM departments ORDER BY department_id"
        ).fetchall() == [("CSE",), ("ECE",), ("ME",)]
        assert connection.execute(
            "SELECT email FROM students WHERE student_id = 1002"
        ).fetchone() == ("diya.patel@example.edu",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_foreign_keys_and_relationships_are_enforced(tmp_path):
    database_path = initialize_database(tmp_path / "academic.db")

    with _connect(database_path) as connection:
        linked_enrollments = connection.execute(
            """
            SELECT COUNT(*)
            FROM enrollments AS e
            JOIN students AS s ON s.student_id = e.student_id
            JOIN courses AS c ON c.course_id = e.course_id
            JOIN departments AS d ON d.department_id = s.department_id
            """
        ).fetchone()[0]
        assert linked_enrollments == EXPECTED_ROW_COUNTS["enrollments"]

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO enrollments
                    (student_id, course_id, semester, academic_year)
                VALUES (9999, 201, 'Fall', 2025)
                """
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO examinations
                    (student_id, course_id, exam_type, exam_date, marks, max_marks)
                VALUES (1001, 201, 'Midterm', '2025-10-15', 101, 100)
                """
            )


def test_initialization_refuses_to_overwrite_existing_database(tmp_path):
    database_path = initialize_database(tmp_path / "academic.db")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        initialize_database(database_path)

    with _connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 6


def test_initializer_cli_reports_existing_database_without_traceback(tmp_path):
    database_path = initialize_database(tmp_path / "academic.db")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "init_db.py"),
            "--database",
            str(database_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
    assert "Traceback" not in result.stderr


def test_initialization_is_reproducible(tmp_path):
    first_path = initialize_database(tmp_path / "first.db")
    second_path = initialize_database(tmp_path / "second.db")

    with _connect(first_path) as first, _connect(second_path) as second:
        for table_name in EXPECTED_ROW_COUNTS:
            query = f"SELECT * FROM {table_name} ORDER BY 1"
            assert first.execute(query).fetchall() == second.execute(query).fetchall()
