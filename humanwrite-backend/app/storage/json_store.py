"""Simple JSON file storage utilities."""

import json
from pathlib import Path
from typing import Any


def save_json(path: str | Path, data: Any) -> None:
    """Save data as JSON to the specified path, creating directories if needed."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    """Load and return JSON data from the specified path."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
