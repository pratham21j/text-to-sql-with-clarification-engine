"""Command-line entry point for creating the demonstration database."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_to_sql.database import DEFAULT_DATABASE_PATH, initialize_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and seed the synthetic academic SQLite database."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"database file path (default: {DEFAULT_DATABASE_PATH})",
    )
    arguments = parser.parse_args()

    try:
        database_path = initialize_database(arguments.database)
    except FileExistsError as error:
        parser.error(str(error))
    print(f"Created demonstration database: {database_path}")


if __name__ == "__main__":
    main()
