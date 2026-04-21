from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_char_limit: int = Field(ge=1)
    overlap_chars: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_less_than_limit(self) -> "ChunkPolicy":
        if self.overlap_chars >= self.chunk_char_limit:
            raise ValueError("overlap_chars must be smaller than chunk_char_limit")
        return self


class TextChunker:
    @staticmethod
    def split(text: str, policy: ChunkPolicy) -> list[str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        paragraphs = [p.strip() for p in cleaned_text.split("\n") if p.strip()]
        if not paragraphs:
            return [cleaned_text]

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if paragraph_len > policy.chunk_char_limit:
                if current_parts:
                    chunks.append("\n\n".join(current_parts).strip())
                    current_parts = []
                    current_len = 0

                start = 0
                step = max(1, policy.chunk_char_limit - policy.overlap_chars)
                while start < paragraph_len:
                    end = min(start + policy.chunk_char_limit, paragraph_len)
                    piece = paragraph[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end >= paragraph_len:
                        break
                    start += step
                continue

            projected_len = current_len + paragraph_len + (2 if current_parts else 0)
            if projected_len > policy.chunk_char_limit and current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                previous_chunk = chunks[-1]
                tail = previous_chunk[-policy.overlap_chars :].strip() if policy.overlap_chars > 0 else ""
                current_parts = [tail, paragraph] if tail else [paragraph]
                current_len = sum(len(x) for x in current_parts) + (2 if len(current_parts) > 1 else 0)
            else:
                current_parts.append(paragraph)
                current_len = projected_len

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())

        return [chunk.strip() for chunk in chunks if chunk.strip()]
