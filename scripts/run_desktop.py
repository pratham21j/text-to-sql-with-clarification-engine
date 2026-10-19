"""Launch the CustomTkinter Text-to-SQL desktop application."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_to_sql.desktop_app import main


if __name__ == "__main__":
    main()
