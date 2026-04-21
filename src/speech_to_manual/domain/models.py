from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import PipelineMode


class TranscriptLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @field_validator("end")
    @classmethod
    def end_not_before_start(cls, value: float, info):
        start = info.data.get("start", 0.0)
        if value < start:
            raise ValueError("end must be >= start")
        return value


class PracticeBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needed: bool
    format: str = Field(min_length=1)


class PlanSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    subsections: list[str] = Field(min_length=1)


class ManualPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    sections: list[PlanSection] = Field(min_length=1)
    practice_block: PracticeBlock
    answers_block: PracticeBlock


class PipelineMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    reference_file: str | None = None
    reference_chars: int = Field(ge=0)
    models: dict[str, str]
    files: dict[str, str]
    lengths: dict[str, int]
    audio: dict[str, str | float | int] | None = None


class PipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: PipelineMode
    input_path: Path
    output_dir: Path
    reference_file: Path | None = None
