from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .enums import StageName
from .models import TranscriptLine


class SttBackend(Protocol):
    def transcribe(self, audio_file: Path, language: str, beam_size: int) -> tuple[str, list[TranscriptLine]]:
        ...


class LlmBackend(Protocol):
    def generate(self, *, stage: StageName, system_prompt: str, user_prompt: str) -> str:
        ...


class FileStore(Protocol):
    def write_text(self, path: Path, text: str) -> None:
        ...

    def read_text(self, path: Path) -> str:
        ...

    def write_json(self, path: Path, data: dict) -> None:
        ...
