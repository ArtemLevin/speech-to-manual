from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .domain.enums import PipelineMode


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retries: int = Field(default=3, ge=1, le=10)
    retry_delay: int = Field(default=3, ge=0, le=120)


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clean_chunk_char_limit: int = Field(default=5000, ge=500)
    clean_chunk_overlap: int = Field(default=400, ge=0)
    plan_prep_chunk_char_limit: int = Field(default=7000, ge=500)
    plan_prep_chunk_overlap: int = Field(default=300, ge=0)
    plan_direct_char_limit: int = Field(default=12000, ge=500)

    @field_validator("clean_chunk_overlap")
    @classmethod
    def validate_clean_overlap(cls, value: int, info) -> int:
        limit = info.data.get("clean_chunk_char_limit", 0)
        if limit and value >= limit:
            raise ValueError("clean_chunk_overlap must be smaller than clean_chunk_char_limit")
        return value


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clean: str = "qwen2.5:latest"
    plan: str = "qwen3.5:4b"
    draft: str = "qwen2.5-coder:7b"
    latex: str = "qwen2.5-coder:7b"
    latex_fix: str = "qwen2.5-coder:7b"


class NumPredictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clean: int = Field(default=1200, ge=100)
    plan_prep: int = Field(default=900, ge=100)
    plan: int = Field(default=1400, ge=100)
    draft: int = Field(default=2200, ge=100)
    latex: int = Field(default=2600, ge=100)
    latex_fix: int = Field(default=2600, ge=100)


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    whisper_model: str = "small"
    language: str = "ru"
    beam_size: int = Field(default=5, ge=1, le=20)
    sample_rate: int = Field(default=16000, ge=8000)
    channels: int = Field(default=1, ge=1, le=2)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ollama_base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: PipelineMode
    input_path: Path
    output_dir: Path | None = None
    reference_file: Path | None = None
    student_name: str = "..."
    runtime: RuntimeConfig = RuntimeConfig()
    retries: RetryConfig = RetryConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    models: ModelConfig = ModelConfig()
    num_predict: NumPredictConfig = NumPredictConfig()
    audio: AudioConfig = AudioConfig()

    @field_validator("input_path")
    @classmethod
    def input_must_exist(cls, value: Path) -> Path:
        if not value.exists() or not value.is_file():
            raise ValueError(f"Input file not found: {value}")
        return value

    @field_validator("reference_file")
    @classmethod
    def validate_reference_file(cls, value: Path | None) -> Path | None:
        if value is None:
            return value
        if not value.exists() or not value.is_file():
            raise ValueError(f"Reference file not found: {value}")
        return value

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        if self.mode == PipelineMode.AUDIO:
            stem = self.input_path.with_suffix("")
            return stem.parent / f"{stem.name}_manual"
        return self.input_path.with_suffix("")
