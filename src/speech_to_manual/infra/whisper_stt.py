from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import whisper

from speech_to_manual.config import AudioConfig
from speech_to_manual.domain.errors import SttError
from speech_to_manual.domain.models import TranscriptLine
from speech_to_manual.infra.ffmpeg_tools import FfmpegTools


class WhisperSttBackend:
    def __init__(self, audio_cfg: AudioConfig) -> None:
        self._cfg = audio_cfg
        self._model = whisper.load_model(audio_cfg.whisper_model)

    def transcribe(self, audio_file: Path, language: str, beam_size: int) -> tuple[str, list[TranscriptLine]]:
        FfmpegTools.ensure_available()
        with TemporaryDirectory(prefix="whisper_prepare_") as tmp_dir:
            prepared_wav = Path(tmp_dir) / f"{audio_file.stem}_prepared.wav"
            FfmpegTools.prepare_wav(
                input_file=audio_file,
                output_wav=prepared_wav,
                sample_rate=self._cfg.sample_rate,
                channels=self._cfg.channels,
            )
            try:
                result = self._model.transcribe(
                    str(prepared_wav),
                    language=language,
                    task="transcribe",
                    fp16=False,
                    temperature=0.0,
                    beam_size=beam_size,
                    best_of=1,
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.5,
                    verbose=False,
                )
            except Exception as exc:
                raise SttError(f"Whisper transcription failed: {exc}") from exc

        full_text = str(result.get("text", "")).strip()
        if not full_text:
            raise SttError("Whisper returned empty transcription")

        lines: list[TranscriptLine] = []
        for seg in result.get("segments", []):
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            lines.append(
                TranscriptLine(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg_text,
                )
            )
        return full_text, lines
