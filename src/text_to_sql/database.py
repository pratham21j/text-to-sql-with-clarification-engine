"""SQLite database initialization for the synthetic academic dataset."""

import os
import sqlite3
from pathlib import Path

from text_to_sql.app_paths import resource_path

DEFAULT_DATABASE_PATH = resource_path("data", "academic.db")

_SCHEMA = (
    """
    CREATE TABLE departments (
        department_id INTEGER PRIMARY KEY,
        code TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL UNIQUE
    )
    """,
    """
    CREATE TABLE students (
        student_id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        department_id INTEGER NOT NULL,
        enrollment_year INTEGER NOT NULL CHECK (enrollment_year BETWEEN 2000 AND 2100),
        cgpa REAL NOT NULL CHECK (cgpa BETWEEN 0.0 AND 10.0),
        FOREIGN KEY (department_id) REFERENCES departments (department_id)
    )
    """,
    """
    CREATE TABLE courses (
        course_id INTEGER PRIMARY KEY,
        code TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        credits INTEGER NOT NULL CHECK (credits BETWEEN 1 AND 10),
        department_id INTEGER NOT NULL,
        FOREIGN KEY (department_id) REFERENCES departments (department_id)
    )
    """,
    """
    CREATE TABLE enrollments (
        enrollment_id INTEGER PRIMARY KEY,
        student_id INTEGER NOT NULL,
        course_id INTEGER NOT NULL,
        semester TEXT NOT NULL CHECK (semester IN ('Spring', 'Fall')),
        academic_year INTEGER NOT NULL CHECK (academic_year BETWEEN 2000 AND 2100),
        grade TEXT CHECK (grade IS NULL OR grade IN ('A', 'B', 'C', 'D', 'F')),
        FOREIGN KEY (student_id) REFERENCES students (student_id),
        FOREIGN KEY (course_id) REFERENCES courses (course_id),
        UNIQUE (student_id, course_id, semester, academic_year)
    )
    """,
    """
    CREATE TABLE examinations (
        examination_id INTEGER PRIMARY KEY,
        student_id INTEGER NOT NULL,
        course_id INTEGER NOT NULL,
        exam_type TEXT NOT NULL CHECK (exam_type IN ('Midterm', 'Final')),
        exam_date TEXT NOT NULL,
        marks REAL NOT NULL CHECK (marks >= 0),
        max_marks REAL NOT NULL CHECK (max_marks > 0 AND marks <= max_marks),
        FOREIGN KEY (student_id) REFERENCES students (student_id),
        FOREIGN KEY (course_id) REFERENCES courses (course_id),
        UNIQUE (student_id, course_id, exam_type),
        CHECK (date(exam_date) IS NOT NULL)
    )
    """,
)

_SEED_DATA = {
    "departments": (
        (1, "CSE", "Computer Science and Engineering"),
        (2, "ECE", "Electronics and Communication Engineering"),
        (3, "ME", "Mechanical Engineering"),
    ),
    "students": (
        (1001, "Aarav", "Sharma", "aarav.sharma@example.edu", 1, 2023, 8.7),
        (1002, "Diya", "Patel", "diya.patel@example.edu", 1, 2022, 9.1),
        (1003, "Ishaan", "Rao", "ishaan.rao@example.edu", 2, 2023, 7.9),
        (1004, "Meera", "Nair", "meera.nair@example.edu", 2, 2021, 8.4),
        (1005, "Kabir", "Singh", "kabir.singh@example.edu", 3, 2022, 7.6),
        (1006, "Ananya", "Das", "ananya.das@example.edu", 1, 2024, 9.3),
    ),
    "courses": (
        (201, "CS101", "Introduction to Programming", 4, 1),
        (202, "CS205", "Database Systems", 4, 1),
        (203, "EC110", "Digital Logic Design", 3, 2),
        (204, "EC220", "Signals and Systems", 4, 2),
        (205, "ME130", "Engineering Mechanics", 3, 3),
    ),
    "enrollments": (
        (1, 1001, 201, "Fall", 2024, "A"),
        (2, 1001, 202, "Spring", 2025, "B"),
        (3, 1002, 201, "Fall", 2024, "A"),
        (4, 1002, 202, "Spring", 2025, "A"),
        (5, 1003, 203, "Fall", 2024, "B"),
        (6, 1003, 204, "Spring", 2025, "B"),
        (7, 1004, 203, "Fall", 2024, "A"),
        (8, 1005, 205, "Fall", 2024, "C"),
        (9, 1006, 201, "Spring", 2025, "A"),
        (10, 1006, 202, "Fall", 2025, None),
    ),
    "examinations": (
        (1, 1001, 201, "Midterm", "2024-10-15", 84, 100),
        (2, 1001, 202, "Final", "2025-05-20", 88, 100),
        (3, 1002, 201, "Midterm", "2024-10-15", 93, 100),
        (4, 1002, 202, "Final", "2025-05-20", 96, 100),
        (5, 1003, 203, "Midterm", "2024-10-18", 78, 100),
        (6, 1004, 203, "Final", "2025-05-22", 89, 100),
        (7, 1005, 205, "Midterm", "2024-10-20", 74, 100),
        (8, 1006, 201, "Final", "2025-05-21", 97, 100),
    ),
}

_INSERT_STATEMENTS = {
    "departments": (
        "INSERT INTO departments (department_id, code, name) VALUES (?, ?, ?)"
    ),
    "students": (
        "INSERT INTO students "
        "(student_id, first_name, last_name, email, department_id, "
        "enrollment_year, cgpa) VALUES (?, ?, ?, ?, ?, ?, ?)"
    ),
    "courses": (
        "INSERT INTO courses (course_id, code, title, credits, department_id) "
        "VALUES (?, ?, ?, ?, ?)"
    ),
    "enrollments": (
        "INSERT INTO enrollments "
        "(enrollment_id, student_id, course_id, semester, academic_year, grade) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    ),
    "examinations": (
        "INSERT INTO examinations "
        "(examination_id, student_id, course_id, exam_type, exam_date, marks, "
        "max_marks) VALUES (?, ?, ?, ?, ?, ?, ?)"
    ),
}


def initialize_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> Path:
    """Create and seed a new database, refusing to replace an existing file."""
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError(
            f"Database already exists; refusing to overwrite it: {path}"
        ) from error
    else:
        os.close(descriptor)

    try:
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            for statement in _SCHEMA:
                connection.execute(statement)
            for table, rows in _SEED_DATA.items():
                connection.executemany(_INSERT_STATEMENTS[table], rows)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return path
