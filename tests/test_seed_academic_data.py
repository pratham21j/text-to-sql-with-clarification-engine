import sqlite3

import pytest

from scripts.seed_academic_data import seed_academic_database
from text_to_sql.database import initialize_database


_TABLES = ("departments", "students", "courses", "enrollments", "examinations")


def _counts(database_path):
    with sqlite3.connect(database_path) as connection:
        return {
            table: connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            for table in _TABLES
        }


def test_seed_adds_targeted_synthetic_population_and_is_idempotent(tmp_path):
    database_path = initialize_database(tmp_path / "academic.db")
    with sqlite3.connect(database_path) as connection:
        original_rows = {
            table: connection.execute(
                f'SELECT * FROM "{table}" ORDER BY 1'
            ).fetchall()
            for table in _TABLES
        }

    added = seed_academic_database(database_path)

    assert added == {
        "students": 294,
        "enrollments": 1176,
        "examinations": 2352,
    }
    assert _counts(database_path) == {
        "departments": 3,
        "students": 300,
        "courses": 5,
        "enrollments": 1186,
        "examinations": 2360,
    }
    assert seed_academic_database(database_path) == {
        "students": 0,
        "enrollments": 0,
        "examinations": 0,
    }

    with sqlite3.connect(database_path) as connection:
        for table, rows in original_rows.items():
            assert connection.execute(
                f'SELECT * FROM "{table}" ORDER BY 1 LIMIT ?',
                (len(rows),),
            ).fetchall() == rows

        assert connection.execute(
            """
            SELECT COUNT(DISTINCT email), COUNT(*)
            FROM students
            WHERE email LIKE 'synthetic.student.%@example.invalid'
            """
        ).fetchone() == (294, 294)
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM students
            WHERE email LIKE 'synthetic.student.%@example.invalid'
              AND (cgpa NOT BETWEEN 0 AND 10
                   OR enrollment_year NOT BETWEEN 2000 AND 2100)
            """
        ).fetchone()[0] == 0
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM examinations AS e
            JOIN students AS s ON s.student_id = e.student_id
            WHERE s.email LIKE 'synthetic.student.%@example.invalid'
              AND (e.marks < 0 OR e.max_marks <= 0 OR e.marks > e.max_marks)
            """
        ).fetchone()[0] == 0
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM students AS s
            LEFT JOIN enrollments AS e ON e.student_id = s.student_id
            LEFT JOIN examinations AS x ON x.student_id = s.student_id
            WHERE s.email LIKE 'synthetic.student.%@example.invalid'
            GROUP BY s.student_id
            HAVING COUNT(DISTINCT e.enrollment_id) != 4
                OR COUNT(DISTINCT x.examination_id) != 8
            """
        ).fetchall() == []
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_seed_rolls_back_all_new_rows_when_existing_seed_data_conflicts(tmp_path):
    database_path = initialize_database(tmp_path / "academic.db")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO students
                (student_id, first_name, last_name, email, department_id,
                 enrollment_year, cgpa)
            VALUES (9000, 'Conflict', 'Synthetic', 'synthetic.student.0001@example.invalid',
                    1, 2023, 8.0)
            """
        )
        connection.commit()
        before = _counts(database_path)

    with pytest.raises(ValueError, match="conflicts with deterministic seed data"):
        seed_academic_database(database_path)

    assert _counts(database_path) == before
