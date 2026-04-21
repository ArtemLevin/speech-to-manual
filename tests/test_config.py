from pathlib import Path

import pytest

from speech_to_manual.config import AppConfig
from speech_to_manual.domain.enums import PipelineMode


def test_config_validation_for_missing_input() -> None:
    with pytest.raises(Exception):
        AppConfig(mode=PipelineMode.TEXT_FILE, input_path=Path("/missing/file.txt"))


def test_output_dir_resolution_text_mode(tmp_path: Path) -> None:
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    cfg = AppConfig(mode=PipelineMode.TEXT_FILE, input_path=input_file)
    assert cfg.resolved_output_dir() == input_file.with_suffix("")
