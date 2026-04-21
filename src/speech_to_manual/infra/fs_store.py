from __future__ import annotations

import json
from pathlib import Path

from speech_to_manual.domain.ports import FileStore


class LocalFileStore(FileStore):
    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read_text(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"File is empty: {path}")
        return content

    def write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
