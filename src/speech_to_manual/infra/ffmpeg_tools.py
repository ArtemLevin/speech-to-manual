from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from speech_to_manual.domain.errors import SttError


class FfmpegTools:
    @staticmethod
    def ensure_available() -> None:
        if shutil.which("ffmpeg") is None:
            raise SttError("ffmpeg not found in PATH")
        if shutil.which("ffprobe") is None:
            raise SttError("ffprobe not found in PATH")

    @staticmethod
    def get_duration_seconds(input_file: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(input_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SttError(f"ffprobe failed: {result.stderr}")
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        if duration <= 0:
            raise SttError(f"Invalid audio duration: {duration}")
        return duration

    @staticmethod
    def prepare_wav(input_file: Path, output_wav: Path, sample_rate: int, channels: int) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-af",
            "highpass=f=80,lowpass=f=7600,loudnorm",
            str(output_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SttError(f"ffmpeg prepare failed: {result.stderr}")
