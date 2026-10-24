"""Add deterministic synthetic students and related academic records."""

import argparse
import random
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_to_sql.database import DEFAULT_DATABASE_PATH

TARGET_STUDENT_COUNT = 300
_EMAIL_PREFIX = "synthetic.student."
_EMAIL_SUFFIX = "@example.invalid"
_EMAIL_PATTERN = re.compile(
    re.escape(_EMAIL_PREFIX) + r"(\d{4})" + re.escape(_EMAIL_SUFFIX)
)
_FIRST_NAMES = (
    "Aarav", "Aditi", "Ayaan", "Diya", "Ira", "Kabir", "Kavya", "Meera",
    "Neel", "Nisha", "Rohan", "Sana", "Tara", "Vihaan", "Zoya", "Arjun",
    "Ishita", "Milan", "Reva", "Samir",
)
_LAST_NAMES = (
    "Anand", "Bose", "Chandra", "Desai", "Iyer", "Kapoor", "Kulkarni",
    "Menon", "Nair", "Pillai", "Rao", "Saxena", "Shah", "Verma", "Mehta",
    "Bhat", "Ghosh", "Joshi", "Patil", "Sen",
)
_DEPARTMENT_CODES = ("CSE", "ECE", "ME")


def _seed_ordinal(email: str) -> int | None:
    match = _EMAIL_PATTERN.fullmatch(email)
    if match:
        ordinal = int(match.group(1))
        if ordinal:
            return ordinal
        raise ValueError(f"Invalid synthetic seed identifier: {email!r}")
    if email.startswith(_EMAIL_PREFIX):
        raise ValueError(f"Unrecognized synthetic seed identifier: {email!r}")
    return None


def _check_schema(connection: sqlite3.Connection) -> None:
    required = {
        "departments": {"department_id", "code"},
        "students": {
            "student_id", "first_name", "last_name", "email", "department_id",
            "enrollment_year", "cgpa",
        },
        "courses": {"course_id", "department_id"},
        "enrollments": {
            "enrollment_id", "student_id", "course_id", "semester",
            "academic_year", "grade",
        },
        "examinations": {
            "examination_id", "student_id", "course_id", "exam_type",
            "exam_date", "marks", "max_marks",
        },
    }
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing_tables = required.keys() - tables
    if missing_tables:
        raise ValueError(f"Database is missing required tables: {sorted(missing_tables)}")
    for table, columns in required.items():
        actual = {
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        if columns - actual:
            raise ValueError(
                f"Table {table!r} is missing columns: {sorted(columns - actual)}"
            )


def _seed_values(ordinal: int) -> tuple[str, str, int, float]:
    generator = random.Random(20261024 + ordinal)
    return (
        _FIRST_NAMES[(ordinal * 7) % len(_FIRST_NAMES)],
        _LAST_NAMES[(ordinal * 13) % len(_LAST_NAMES)],
        generator.randint(2021, 2024),
        round(5.2 + generator.betavariate(3.5, 1.8) * 4.8, 2),
    )


def _grade(marks: float) -> str:
    if marks >= 90:
        return "A"
    if marks >= 80:
        return "B"
    if marks >= 70:
        return "C"
    if marks >= 60:
        return "D"
    return "F"


def _course_list(
    ordinal: int,
    department_id: int,
    by_department: dict[int, list[int]],
) -> list[int]:
    generator = random.Random(20261024 + ordinal)
    major = by_department.get(department_id, [])
    other = [
        course_id
        for dept_courses in by_department.values()
        for course_id in dept_courses
        if course_id not in major
    ]
    if not major or len(major) + len(other) < 4:
        raise ValueError(
            "At least four courses and one course in each seeded student's "
            "department are required."
        )
    if len(major) >= 2:
        selected = generator.sample(major, 2) + generator.sample(other, 2)
    else:
        selected = major + generator.sample(other, 4 - len(major))
    generator.shuffle(selected)
    return selected


def _exam_marks(ordinal: int, course_id: int, exam_type: str, cgpa: float) -> float:
    seed = 20261024 + ordinal * 1009 + course_id * 17
    generator = random.Random(seed + (exam_type == "Final"))
    return round(min(100.0, max(35.0, cgpa * 10 + generator.uniform(-13, 10))), 1)


def _next_id(connection: sqlite3.Connection, table: str, column: str) -> int:
    return connection.execute(
        f'SELECT COALESCE(MAX("{column}"), 0) + 1 FROM "{table}"'
    ).fetchone()[0]


def _records_for_student(
    ordinal: int,
    student_id: int,
    department_id: int,
    enrollment_year: int,
    cgpa: float,
    courses_by_department: dict[int, list[int]],
) -> tuple[list[tuple], list[tuple]]:
    enrollments = []
    examinations = []
    for index, course_id in enumerate(
        _course_list(ordinal, department_id, courses_by_department)
    ):
        semester, academic_year = (
            ("Fall", enrollment_year)
            if index < 2
            else ("Spring", enrollment_year + 1)
        )
        dates = (
            (f"{academic_year}-10-15", f"{academic_year}-12-10")
            if semester == "Fall"
            else (f"{academic_year}-03-15", f"{academic_year}-05-20")
        )
        marks = {
            exam_type: _exam_marks(ordinal, course_id, exam_type, cgpa)
            for exam_type in ("Midterm", "Final")
        }
        enrollments.append(
            (
                student_id, course_id, semester, academic_year,
                _grade(sum(marks.values()) / 2),
            )
        )
        examinations.extend(
            (
                student_id, course_id, exam_type, dates[position],
                marks[exam_type], 100.0,
            )
            for position, exam_type in enumerate(("Midterm", "Final"))
        )
    return enrollments, examinations


def seed_academic_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, int]:
    """Add missing synthetic rows transactionally; preserve all existing rows."""
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database does not exist: {path}")

    connection = sqlite3.connect(path)
    added = {"students": 0, "enrollments": 0, "examinations": 0}
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _check_schema(connection)

        departments = dict(
            connection.execute("SELECT code, department_id FROM departments")
        )
        if not set(_DEPARTMENT_CODES).issubset(departments):
            raise ValueError("Database must contain the CSE, ECE, and ME departments.")

        courses_by_department: dict[int, list[int]] = {}
        for course_id, department_id in connection.execute(
            "SELECT course_id, department_id FROM courses ORDER BY course_id"
        ):
            courses_by_department.setdefault(department_id, []).append(course_id)

        students = {
            email: (student_id, first_name, last_name, department_id, year, cgpa)
            for (
                student_id, first_name, last_name, email, department_id, year, cgpa
            ) in connection.execute(
                """
                SELECT student_id, first_name, last_name, email, department_id,
                       enrollment_year, cgpa
                FROM students
                """
            )
        }
        initial_student_count = len(students)
        synthetic = {}
        for email, row in students.items():
            ordinal = _seed_ordinal(email)
            if ordinal is not None:
                if ordinal in synthetic:
                    raise ValueError(f"Duplicate synthetic seed ordinal: {ordinal}")
                synthetic[ordinal] = row

        count_to_add = max(0, TARGET_STUDENT_COUNT - initial_student_count)
        available_ordinals = (
            ordinal for ordinal in range(1, 10000) if ordinal not in synthetic
        )
        new_ordinals = [next(available_ordinals) for _ in range(count_to_add)]
        next_student_id = _next_id(connection, "students", "student_id")

        for ordinal in new_ordinals:
            email = f"{_EMAIL_PREFIX}{ordinal:04d}{_EMAIL_SUFFIX}"
            department_code = _DEPARTMENT_CODES[(ordinal - 1) % 3]
            department_id = departments[department_code]
            first_name, last_name, year, cgpa = _seed_values(ordinal)
            row = (next_student_id, first_name, last_name, department_id, year, cgpa)
            connection.execute(
                """
                INSERT INTO students
                    (student_id, first_name, last_name, email, department_id,
                     enrollment_year, cgpa)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (next_student_id, first_name, last_name, email, department_id, year, cgpa),
            )
            students[email] = row
            synthetic[ordinal] = row
            next_student_id += 1
            added["students"] += 1

        next_enrollment_id = _next_id(connection, "enrollments", "enrollment_id")
        next_examination_id = _next_id(connection, "examinations", "examination_id")
        for ordinal, row in sorted(synthetic.items()):
            student_id, first_name, last_name, department_id, year, cgpa = row
            if (first_name, last_name, year, cgpa) != _seed_values(ordinal):
                raise ValueError(
                    f"Synthetic student {ordinal:04d} conflicts with deterministic seed data."
                )
            if department_id != departments[_DEPARTMENT_CODES[(ordinal - 1) % 3]]:
                raise ValueError(
                    f"Synthetic student {ordinal:04d} has an unexpected department."
                )

            expected_enrollments, expected_examinations = _records_for_student(
                ordinal, student_id, department_id, year, cgpa, courses_by_department
            )
            for record in expected_enrollments:
                existing = connection.execute(
                    """
                    SELECT student_id, course_id, semester, academic_year, grade
                    FROM enrollments
                    WHERE student_id = ? AND course_id = ?
                      AND semester = ? AND academic_year = ?
                    """,
                    record[:4],
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO enrollments
                            (enrollment_id, student_id, course_id, semester,
                             academic_year, grade)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (next_enrollment_id, *record),
                    )
                    next_enrollment_id += 1
                    added["enrollments"] += 1
                elif tuple(existing) != record:
                    raise ValueError(
                        f"Conflicting enrollment for synthetic student {ordinal:04d}."
                    )

            for record in expected_examinations:
                existing = connection.execute(
                    """
                    SELECT student_id, course_id, exam_type, exam_date, marks, max_marks
                    FROM examinations
                    WHERE student_id = ? AND course_id = ? AND exam_type = ?
                    """,
                    record[:3],
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO examinations
                            (examination_id, student_id, course_id, exam_type,
                             exam_date, marks, max_marks)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (next_examination_id, *record),
                    )
                    next_examination_id += 1
                    added["examinations"] += 1
                elif tuple(existing) != record:
                    raise ValueError(
                        f"Conflicting examination for synthetic student {ordinal:04d}."
                    )

        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Foreign-key validation failed after seeding.")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity validation failed after seeding.")
        if connection.execute("SELECT COUNT(*) FROM students").fetchone()[0] != max(
            TARGET_STUDENT_COUNT, initial_student_count
        ):
            raise ValueError("Final student count does not match the seed target.")

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add deterministic synthetic academic records to an existing database."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"database file path (default: {DEFAULT_DATABASE_PATH})",
    )
    arguments = parser.parse_args()
    added = seed_academic_database(arguments.database)
    print(f"Added records: {added}")
    path = arguments.database.expanduser().resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        for table in (
            "departments", "students", "courses", "enrollments", "examinations"
        ):
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            print(f"{table}: {count}")


if __name__ == "__main__":
    main()
