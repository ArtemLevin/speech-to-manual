import argparse  # Разбор аргументов командной строки
import json  # Работа с JSON
import logging  # Логи
import re  # Регулярные выражения для извлечения JSON/LaTeX-блоков
import shutil  # Проверка наличия ffmpeg / ffprobe
import subprocess  # Запуск внешних команд
import time  # Паузы между retry
from dataclasses import dataclass  # Удобные структуры данных
from pathlib import Path  # Работа с путями
from tempfile import TemporaryDirectory  # Временная папка для подготовки WAV

import whisper  # Whisper для транскрибации
from langchain_core.output_parsers import StrOutputParser  # Превращаем ответ модели в обычную строку
from langchain_core.prompts import ChatPromptTemplate  # Шаблон сообщений для LLM
from langchain_ollama import ChatOllama  # LangChain-интеграция с Ollama


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Какие расширения считаем аудио
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

# Параметры Whisper
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_LANGUAGE = "ru"
DEFAULT_BEAM_SIZE = 5

# Модели по стадиям
MODEL_CLEAN = "qwen2.5:latest"
MODEL_PLAN = "qwen3.5:4b"
MODEL_DRAFT = "qwen2.5-coder:7b"
MODEL_LATEX = "qwen2.5-coder:7b"
MODEL_LATEX_FIX = "qwen2.5-coder:7b"

# Параметры Ollama
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.0

# Подготовка аудио
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Chunking
DEFAULT_CLEAN_CHUNK_CHAR_LIMIT = 5000
DEFAULT_CLEAN_CHUNK_OVERLAP = 400

DEFAULT_PLAN_PREP_CHUNK_CHAR_LIMIT = 7000
DEFAULT_PLAN_PREP_CHUNK_OVERLAP = 300
DEFAULT_PLAN_DIRECT_CHAR_LIMIT = 12000

# Ограничения длины ответа модели
DEFAULT_NUM_PREDICT_CLEAN = 1200
DEFAULT_NUM_PREDICT_PLAN_PREP = 900
DEFAULT_NUM_PREDICT_PLAN = 1400
DEFAULT_NUM_PREDICT_DRAFT = 2200
DEFAULT_NUM_PREDICT_LATEX = 2600
DEFAULT_NUM_PREDICT_LATEX_FIX = 2600

# Retry
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 3


# ============================================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================================

@dataclass
class TranscriptLine:
    # Одна строка субтитров
    start: float
    end: float
    text: str


# ============================================================================
# ЛОГИ
# ============================================================================

def build_logger(log_file: Path) -> logging.Logger:
    # Создаём логгер и в файл, и в консоль
    logger_name = f"manual_pipeline_{log_file.parent.name}_{int(time.time())}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Лог в файл
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Лог в консоль
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


# ============================================================================
# ОБЩИЕ УТИЛИТЫ
# ============================================================================

def save_text(text: str, output_file: Path) -> None:
    # Сохраняем текст в UTF-8
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def read_text_file(file_path: Path) -> str:
    # Читаем текстовый файл
    if not file_path.exists() or not file_path.is_file():
        raise RuntimeError(f"Файл не найден: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Файл пустой: {file_path}")

    return text


def read_optional_reference_file(reference_file: Path | None) -> str:
    # Читаем референс, если он передан
    if reference_file is None:
        return ""
    return read_text_file(reference_file)


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    # Запускаем внешнюю команду
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Ошибка при выполнении команды:\n"
            f"{' '.join(cmd)}\n\n"
            f"{result.stderr}"
        )

    return result


def split_text_into_chunks(
        text: str,
        chunk_char_limit: int,
        overlap_chars: int,
) -> list[str]:
    # Делим текст на чанки по символам, стараясь резать по абзацам
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

        # Если абзац слишком длинный — режем его грубо
        if paragraph_len > chunk_char_limit:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_len = 0

            start = 0
            step = max(1, chunk_char_limit - overlap_chars)

            while start < paragraph_len:
                end = min(start + chunk_char_limit, paragraph_len)
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= paragraph_len:
                    break
                start += step

            continue

        projected_len = current_len + paragraph_len + (2 if current_parts else 0)

        # Если не помещается — закрываем текущий чанк
        if projected_len > chunk_char_limit and current_parts:
            chunks.append("\n\n".join(current_parts).strip())

            # Берём хвост прошлого чанка как overlap
            previous_chunk = chunks[-1]
            tail = previous_chunk[-overlap_chars:].strip() if overlap_chars > 0 else ""

            current_parts = [tail, paragraph] if tail else [paragraph]
            current_len = sum(len(x) for x in current_parts) + (2 if len(current_parts) > 1 else 0)
        else:
            current_parts.append(paragraph)
            current_len = projected_len

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def extract_json_block(text: str) -> str:
    # Пытаемся достать JSON из ответа модели
    raw = text.strip()

    # Если модель завернула JSON в ```json ... ```
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    # Иначе берём от первой { до последней }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return raw[first_brace:last_brace + 1].strip()

    raise RuntimeError("Не удалось найти JSON в ответе модели")


def validate_plan_json(plan_data: dict) -> None:
    # Простая проверка структуры JSON-плана
    required_top = {
        "title",
        "topic",
        "audience",
        "goal",
        "sections",
        "practice_block",
        "answers_block",
    }

    missing = required_top - set(plan_data.keys())
    if missing:
        raise RuntimeError(f"В JSON-плане отсутствуют поля: {sorted(missing)}")

    if not isinstance(plan_data["sections"], list) or not plan_data["sections"]:
        raise RuntimeError("Поле sections должно быть непустым списком")

    for idx, section in enumerate(plan_data["sections"], start=1):
        if not isinstance(section, dict):
            raise RuntimeError(f"Раздел #{idx} должен быть объектом")
        for key in ("id", "title", "purpose", "subsections"):
            if key not in section:
                raise RuntimeError(f"В разделе #{idx} отсутствует поле: {key}")
        if not isinstance(section["subsections"], list):
            raise RuntimeError(f"В разделе #{idx} поле subsections должно быть списком")

    for block_name in ("practice_block", "answers_block"):
        block = plan_data[block_name]
        if not isinstance(block, dict):
            raise RuntimeError(f"{block_name} должен быть объектом")
        for key in ("needed", "format"):
            if key not in block:
                raise RuntimeError(f"В {block_name} отсутствует поле: {key}")


def validate_latex(tex_code: str) -> None:
    # Базовая проверка LaTeX-кода
    must_have = [
        r"\documentclass",
        r"\begin{document}",
        r"\end{document}",
    ]

    for token in must_have:
        if token not in tex_code:
            raise RuntimeError(f"LaTeX не содержит обязательный фрагмент: {token}")


def normalize_latex_output(raw_text: str) -> str:
    # Если модель завернула tex в ```latex ... ``` — достаём только код
    text = raw_text.strip()

    fenced_match = re.search(r"```(?:latex|tex)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    return text


# ============================================================================
# FFMPEG / WHISPER
# ============================================================================

def ensure_ffmpeg_tools() -> None:
    # Проверяем наличие ffmpeg и ffprobe
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден в PATH")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe не найден в PATH")


def get_audio_duration_seconds(input_file: Path) -> float:
    # Узнаём длительность файла
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(input_file),
    ]
    result = run_command(cmd)
    data = json.loads(result.stdout)

    duration = float(data["format"]["duration"])
    if duration <= 0:
        raise RuntimeError(f"Не удалось определить длительность файла: {input_file}")

    return duration


def format_srt_timestamp(seconds: float) -> str:
    # Формат времени для SRT
    if seconds < 0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(lines: list[TranscriptLine], output_file: Path) -> None:
    # Сохраняем SRT
    with open(output_file, "w", encoding="utf-8") as f:
        index = 1
        for line in lines:
            text = line.text.strip()
            if not text:
                continue

            f.write(f"{index}\n")
            f.write(
                f"{format_srt_timestamp(line.start)} --> "
                f"{format_srt_timestamp(line.end)}\n"
            )
            f.write(text + "\n\n")
            index += 1


def prepare_audio_for_whisper(input_file: Path, output_wav: Path) -> None:
    # Готовим WAV для Whisper
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-af", "highpass=f=80,lowpass=f=7600,loudnorm",
        str(output_wav),
    ]
    run_command(cmd)


def transcribe_audio_file(
        model,
        prepared_audio_file: Path,
        language: str,
        beam_size: int,
) -> tuple[str, list[TranscriptLine]]:
    # Транскрибируем файл целиком
    result = model.transcribe(
        str(prepared_audio_file),
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

    full_text = result.get("text", "").strip()
    srt_lines: list[TranscriptLine] = []

    for seg in result.get("segments", []):
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue

        srt_lines.append(
            TranscriptLine(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=seg_text,
            )
        )

    return full_text, srt_lines


# ============================================================================
# LANGCHAIN / OLLAMA
# ============================================================================

def build_chain(model_name: str, base_url: str, temperature: float, num_predict: int, user_template: str):
    # Собираем цепочку prompt -> модель -> строка
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", user_template),
        ]
    )

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        reasoning=False,  # Важно: отключаем thinking mode у qwen3.5
        num_predict=num_predict,
    )

    parser = StrOutputParser()
    return prompt | llm | parser


def invoke_chain_with_retry(
        chain,
        payload: dict,
        retries: int,
        retry_delay: int,
        logger: logging.Logger,
        stage_name: str,
) -> str:
    # Вызываем модель с retry
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            result = chain.invoke(payload)
            result = str(result).strip()
            if not result:
                raise RuntimeError("Модель вернула пустой результат")
            return result
        except KeyboardInterrupt:
            # Ручное прерывание не глотаем
            raise
        except Exception as e:
            last_error = e
            logger.warning("%s | попытка %s/%s не удалась: %s", stage_name, attempt, retries, e)

            if attempt < retries:
                logger.info("%s | жду %s сек и пробую снова...", stage_name, retry_delay)
                time.sleep(retry_delay)

    raise RuntimeError(f"{stage_name} | не удалось получить ответ после {retries} попыток: {last_error}")


# ============================================================================
# PROMPT-ШАБЛОНЫ
# ============================================================================

def build_clean_user_prompt(text: str, reference: str) -> str:
    # Prompt для очистки текста
    return (
        "Ниже даны:\n"
        "1. Исходный текст.\n"
        "2. Референс по стилю и формату.\n\n"
        "Задача:\n"
        "- Исправить явные ошибки распознавания речи.\n"
        "- Убрать повторы и слова-паразиты, если это не ломает смысл.\n"
        "- Разбить текст на логические абзацы.\n"
        "- Не выдумывать факты.\n"
        "- Использовать референс только как ориентир по стилю, но не переносить из него факты.\n"
        "- Вернуть только очищенный текст.\n\n"
        "=== ИСХОДНЫЙ ТЕКСТ ===\n"
        f"{text}\n\n"
        "=== РЕФЕРЕНС ===\n"
        f"{reference}\n"
    )


def build_plan_source_user_prompt(text: str, reference: str) -> str:
    # Prompt для извлечения материала под план
    return (
        "Ниже даны:\n"
        "1. Фрагмент очищенного учебного текста.\n"
        "2. Референс.\n\n"
        "Сделай структурированную выжимку для дальнейшего построения плана пособия.\n\n"
        "Нужно выделить:\n"
        "- тему фрагмента\n"
        "- ключевые подтемы\n"
        "- определения\n"
        "- методы/алгоритмы\n"
        "- типичные ошибки\n"
        "- примеры/упражнения, если есть\n\n"
        "Формат ответа: краткий markdown.\n"
        "Не выдумывай.\n"
        "Используй референс только как ориентир по структуре.\n\n"
        "=== ФРАГМЕНТ ===\n"
        f"{text}\n\n"
        "=== РЕФЕРЕНС ===\n"
        f"{reference}\n"
    )


def build_plan_json_user_prompt(text: str, reference: str) -> str:
    # Prompt для финального JSON-плана
    # ВАЖНО: фигурные скобки здесь не должны ломать LangChain, поэтому
    # этот текст мы подставляем уже как готовую строку user_prompt,
    # а не как шаблон с новыми переменными внутри.
    return (
        "Ниже даны:\n"
        "1. Материал для построения плана пособия.\n"
        "2. Референс.\n\n"
        "Нужно вернуть только JSON-объект плана пособия.\n\n"
        "Структура JSON:\n"
        "{\n"
        '  "title": "...",\n'
        '  "topic": "...",\n'
        '  "audience": "...",\n'
        '  "goal": "...",\n'
        '  "sections": [\n'
        "    {\n"
        '      "id": "section_id",\n'
        '      "title": "Название раздела",\n'
        '      "purpose": "Зачем этот раздел",\n'
        '      "subsections": ["...", "..."]\n'
        "    }\n"
        "  ],\n"
        '  "practice_block": {\n'
        '    "needed": true,\n'
        '    "format": "..." \n'
        "  },\n"
        '  "answers_block": {\n'
        '    "needed": true,\n'
        '    "format": "..." \n'
        "  }\n"
        "}\n\n"
        "Правила:\n"
        "- Только валидный JSON.\n"
        "- Без пояснений до и после JSON.\n"
        "- Не выдумывать темы, которых нет в материале.\n"
        "- Использовать референс только как ориентир по формату и структуре.\n\n"
        "=== МАТЕРИАЛ ДЛЯ ПЛАНА ===\n"
        f"{text}\n\n"
        "=== РЕФЕРЕНС ===\n"
        f"{reference}\n"
    )


def build_draft_user_prompt(cleaned_text: str, plan_json: str, reference: str) -> str:
    # Prompt для markdown-черновика пособия
    return (
        "Ниже даны:\n"
        "1. Очищенный учебный текст.\n"
        "2. Готовый JSON-план пособия.\n"
        "3. Референс.\n\n"
        "Нужно написать содержательное текстовое пособие в markdown.\n\n"
        "Требования:\n"
        "- Строго следовать плану.\n"
        "- Сохранять факты только из очищенного текста.\n"
        "- Использовать референс только как ориентир по стилю и подаче.\n"
        "- Сделать понятную учебную структуру.\n"
        "- Включить разделы, подпункты, списки, формулы в текстовом виде там, где это нужно.\n"
        "- Не писать комментарии о своей работе.\n"
        "- Вернуть только markdown-текст пособия.\n\n"
        "=== ОЧИЩЕННЫЙ ТЕКСТ ===\n"
        f"{cleaned_text}\n\n"
        "=== ПЛАН JSON ===\n"
        f"{plan_json}\n\n"
        "=== РЕФЕРЕНС ===\n"
        f"{reference}\n"
    )


def build_latex_user_prompt(draft_text: str, plan_json: str, reference: str) -> str:
    # Prompt для генерации LaTeX
    return (
        "Ниже даны:\n"
        "1. Markdown-черновик пособия.\n"
        "2. JSON-план.\n"
        "3. Референс.\n\n"
        "Нужно вернуть только полный компилируемый LaTeX-код пособия.\n\n"
        "Требования:\n"
        "- Полный минимальный .tex-файл.\n"
        "- pdfLaTeX-совместимость.\n"
        "- Обязательные части: \\documentclass, преамбула, \\begin{document}, \\end{document}.\n"
        "- Академичный, чистый стиль.\n"
        "- Никаких пояснений вне кода.\n"
        "- Не вставлять markdown.\n"
        "- Если есть списки, таблицы, упражнения, оформить их корректно на LaTeX.\n"
        "- Использовать референс только как ориентир по формату и структуре.\n\n"
        "=== MARKDOWN-ЧЕРНОВИК ===\n"
        f"{draft_text}\n\n"
        "=== ПЛАН JSON ===\n"
        f"{plan_json}\n\n"
        "=== РЕФЕРЕНС ===\n"
        f"{reference}\n"
    )


def build_latex_fix_user_prompt(tex_code: str) -> str:
    # Prompt для починки LaTeX
    return (
        "Ниже дан LaTeX-код учебного пособия.\n\n"
        "Нужно:\n"
        "- проверить структуру\n"
        "- исправить явные синтаксические проблемы\n"
        "- сохранить содержание\n"
        "- вернуть только полный исправленный LaTeX-код\n\n"
        "=== LATEX ===\n"
        f"{tex_code}\n"
    )


# ============================================================================
# СТАДИИ PIPELINE
# ============================================================================

def stage_clean_text(
        raw_text: str,
        reference_text: str,
        output_dir: Path,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
        logger: logging.Logger,
        chunk_char_limit: int,
        overlap_chars: int,
) -> str:
    # Стадия 1: очистка текста
    stage_dir = output_dir / "stage_01_clean"
    stage_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_text_into_chunks(
        text=raw_text,
        chunk_char_limit=chunk_char_limit,
        overlap_chars=overlap_chars,
    )

    clean_chain = build_chain(
        model_name=MODEL_CLEAN,
        base_url=base_url,
        temperature=temperature,
        num_predict=DEFAULT_NUM_PREDICT_CLEAN,
        user_template="{user_prompt}",
    )

    cleaned_parts: list[str] = []

    logger.info("Стадия clean | чанков: %s", len(chunks))

    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Стадия clean | чанк %s/%s", idx, len(chunks))

        chunk_in = stage_dir / f"{idx:03d}_input.txt"
        chunk_out = stage_dir / f"{idx:03d}_output.txt"

        save_text(chunk, chunk_in)

        cleaned_chunk = invoke_chain_with_retry(
            chain=clean_chain,
            payload={
                "system_prompt": "Ты аккуратно очищаешь текст после транскрибации.",
                "user_prompt": build_clean_user_prompt(chunk, reference_text),
            },
            retries=retries,
            retry_delay=retry_delay,
            logger=logger,
            stage_name=f"clean_chunk_{idx}",
        )

        save_text(cleaned_chunk, chunk_out)
        cleaned_parts.append(cleaned_chunk)

    cleaned_text = "\n\n".join(cleaned_parts).strip()
    save_text(cleaned_text, output_dir / "02_cleaned.txt")

    if not cleaned_text:
        raise RuntimeError("Стадия clean вернула пустой текст")

    return cleaned_text


def stage_plan_source(
        cleaned_text: str,
        reference_text: str,
        output_dir: Path,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
        logger: logging.Logger,
) -> str:
    # Стадия 2a: готовим материал для плана
    # Если текст короткий — используем его как есть
    if len(cleaned_text) <= DEFAULT_PLAN_DIRECT_CHAR_LIMIT:
        save_text(cleaned_text, output_dir / "03_plan_source.md")
        return cleaned_text

    stage_dir = output_dir / "stage_02_plan_source"
    stage_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_text_into_chunks(
        text=cleaned_text,
        chunk_char_limit=DEFAULT_PLAN_PREP_CHUNK_CHAR_LIMIT,
        overlap_chars=DEFAULT_PLAN_PREP_CHUNK_OVERLAP,
    )

    prep_chain = build_chain(
        model_name=MODEL_PLAN,
        base_url=base_url,
        temperature=temperature,
        num_predict=DEFAULT_NUM_PREDICT_PLAN_PREP,
        user_template="{user_prompt}",
    )

    prep_parts: list[str] = []

    logger.info("Стадия plan_source | чанков: %s", len(chunks))

    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Стадия plan_source | чанк %s/%s", idx, len(chunks))

        chunk_in = stage_dir / f"{idx:03d}_input.txt"
        chunk_out = stage_dir / f"{idx:03d}_output.md"

        save_text(chunk, chunk_in)

        prep_chunk = invoke_chain_with_retry(
            chain=prep_chain,
            payload={
                "system_prompt": "Ты выделяешь структуру учебного материала для будущего плана пособия.",
                "user_prompt": build_plan_source_user_prompt(chunk, reference_text),
            },
            retries=retries,
            retry_delay=retry_delay,
            logger=logger,
            stage_name=f"plan_source_chunk_{idx}",
        )

        save_text(prep_chunk, chunk_out)
        prep_parts.append(f"## Фрагмент {idx}\n\n{prep_chunk}")

    plan_source = "\n\n".join(prep_parts).strip()
    save_text(plan_source, output_dir / "03_plan_source.md")

    if not plan_source:
        raise RuntimeError("Стадия plan_source вернула пустой текст")

    return plan_source


def stage_plan_json(
        plan_source_text: str,
        reference_text: str,
        output_dir: Path,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
        logger: logging.Logger,
) -> dict:
    # Стадия 2b: делаем JSON-план пособия
    plan_chain = build_chain(
        model_name=MODEL_PLAN,
        base_url=base_url,
        temperature=temperature,
        num_predict=DEFAULT_NUM_PREDICT_PLAN,
        user_template="{user_prompt}",
    )

    plan_raw = invoke_chain_with_retry(
        chain=plan_chain,
        payload={
            "system_prompt": "Ты проектируешь структуру учебного пособия и возвращаешь только JSON.",
            "user_prompt": build_plan_json_user_prompt(plan_source_text, reference_text),
        },
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
        stage_name="plan_json",
    )

    save_text(plan_raw, output_dir / "04_plan_raw.txt")

    plan_json_text = extract_json_block(plan_raw)
    plan_data = json.loads(plan_json_text)
    validate_plan_json(plan_data)

    with open(output_dir / "04_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan_data, f, ensure_ascii=False, indent=2)

    return plan_data


def stage_manual_draft(
        cleaned_text: str,
        plan_data: dict,
        reference_text: str,
        output_dir: Path,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
        logger: logging.Logger,
) -> str:
    # Стадия 3: делаем markdown-черновик пособия
    plan_json_pretty = json.dumps(plan_data, ensure_ascii=False, indent=2)

    draft_chain = build_chain(
        model_name=MODEL_DRAFT,
        base_url=base_url,
        temperature=temperature,
        num_predict=DEFAULT_NUM_PREDICT_DRAFT,
        user_template="{user_prompt}",
    )

    draft_text = invoke_chain_with_retry(
        chain=draft_chain,
        payload={
            "system_prompt": "Ты пишешь учебное пособие в markdown по готовому плану.",
            "user_prompt": build_draft_user_prompt(cleaned_text, plan_json_pretty, reference_text),
        },
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
        stage_name="manual_draft",
    )

    save_text(draft_text, output_dir / "05_manual_draft.md")

    if not draft_text:
        raise RuntimeError("Стадия draft вернула пустой текст")

    return draft_text


def stage_latex_generate(
        draft_text: str,
        plan_data: dict,
        reference_text: str,
        output_dir: Path,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
        logger: logging.Logger,
) -> str:
    # Стадия 4: делаем LaTeX
    plan_json_pretty = json.dumps(plan_data, ensure_ascii=False, indent=2)

    latex_chain = build_chain(
        model_name=MODEL_LATEX,
        base_url=base_url,
        temperature=temperature,
        num_predict=DEFAULT_NUM_PREDICT_LATEX,
        user_template="{user_prompt}",
    )

    tex_code_raw = invoke_chain_with_retry(
        chain=latex_chain,
        payload={
            "system_prompt": "Ты пишешь чистый компилируемый LaTeX-код учебного пособия.",
            "user_prompt": build_latex_user_prompt(draft_text, plan_json_pretty, reference_text),
        },
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
        stage_name="latex_generate",
    )

    tex_code = normalize_latex_output(tex_code_raw)
    validate_latex(tex_code)

    save_text(tex_code, output_dir / "06_manual.tex")
    return tex_code


def stage_latex_fix(
        tex_code: str,
        output_dir: Path,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
        logger: logging.Logger,
) -> str:
    # Стадия 5: чиним LaTeX
    fix_chain = build_chain(
        model_name=MODEL_LATEX_FIX,
        base_url=base_url,
        temperature=temperature,
        num_predict=DEFAULT_NUM_PREDICT_LATEX_FIX,
        user_template="{user_prompt}",
    )

    fixed_tex_raw = invoke_chain_with_retry(
        chain=fix_chain,
        payload={
            "system_prompt": "Ты исправляешь LaTeX-код и возвращаешь только полный исправленный .tex-файл.",
            "user_prompt": build_latex_fix_user_prompt(tex_code),
        },
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
        stage_name="latex_fix",
    )

    fixed_tex = normalize_latex_output(fixed_tex_raw)
    validate_latex(fixed_tex)

    save_text(fixed_tex, output_dir / "07_manual_fixed.tex")
    return fixed_tex


# ============================================================================
# ORCHESTRATION
# ============================================================================

def run_pipeline_from_text(
        raw_text: str,
        source_name: str,
        output_dir: Path,
        reference_text: str,
        reference_file: Path | None,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
) -> None:
    # Главная функция pipeline
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "pipeline.log"
    logger = build_logger(log_file)

    logger.info("=" * 80)
    logger.info("Старт пайплайна для: %s", source_name)
    logger.info("=" * 80)

    save_text(raw_text, output_dir / "01_raw.txt")

    if reference_file:
        save_text(reference_text, output_dir / "00_reference.txt")

    # 1. Очистка
    cleaned_text = stage_clean_text(
        raw_text=raw_text,
        reference_text=reference_text,
        output_dir=output_dir,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
        chunk_char_limit=DEFAULT_CLEAN_CHUNK_CHAR_LIMIT,
        overlap_chars=DEFAULT_CLEAN_CHUNK_OVERLAP,
    )

    # 2a. Материал для плана
    plan_source_text = stage_plan_source(
        cleaned_text=cleaned_text,
        reference_text=reference_text,
        output_dir=output_dir,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
    )

    # 2b. План JSON
    plan_data = stage_plan_json(
        plan_source_text=plan_source_text,
        reference_text=reference_text,
        output_dir=output_dir,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
    )

    # 3. Markdown-черновик
    draft_text = stage_manual_draft(
        cleaned_text=cleaned_text,
        plan_data=plan_data,
        reference_text=reference_text,
        output_dir=output_dir,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
    )

    # 4. Генерация LaTeX
    tex_code = stage_latex_generate(
        draft_text=draft_text,
        plan_data=plan_data,
        reference_text=reference_text,
        output_dir=output_dir,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
    )

    # 5. Починка LaTeX
    fixed_tex = stage_latex_fix(
        tex_code=tex_code,
        output_dir=output_dir,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
        logger=logger,
    )

    meta = {
        "source_name": source_name,
        "reference_file": str(reference_file) if reference_file else None,
        "reference_chars": len(reference_text),
        "models": {
            "clean": MODEL_CLEAN,
            "plan": MODEL_PLAN,
            "draft": MODEL_DRAFT,
            "latex": MODEL_LATEX,
            "latex_fix": MODEL_LATEX_FIX,
        },
        "files": {
            "raw": str(output_dir / "01_raw.txt"),
            "cleaned": str(output_dir / "02_cleaned.txt"),
            "plan_source": str(output_dir / "03_plan_source.md"),
            "plan_raw": str(output_dir / "04_plan_raw.txt"),
            "plan_json": str(output_dir / "04_plan.json"),
            "draft_md": str(output_dir / "05_manual_draft.md"),
            "latex": str(output_dir / "06_manual.tex"),
            "latex_fixed": str(output_dir / "07_manual_fixed.tex"),
            "log": str(log_file),
        },
        "lengths": {
            "raw_chars": len(raw_text),
            "cleaned_chars": len(cleaned_text),
            "plan_source_chars": len(plan_source_text),
            "draft_chars": len(draft_text),
            "latex_chars": len(tex_code),
            "latex_fixed_chars": len(fixed_tex),
        },
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Пайплайн завершён. Итоговая папка: %s", output_dir)


def run_from_audio(
        audio_file: Path,
        output_dir: Path,
        reference_text: str,
        reference_file: Path | None,
        whisper_model_name: str,
        language: str,
        beam_size: int,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
) -> None:
    # Режим: аудио -> транскрипт -> pipeline
    ensure_ffmpeg_tools()

    whisper_model = whisper.load_model(whisper_model_name)

    duration = get_audio_duration_seconds(audio_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="whisper_prepare_") as tmp_dir:
        prepared_wav = Path(tmp_dir) / f"{audio_file.stem}_prepared.wav"

        prepare_audio_for_whisper(audio_file, prepared_wav)
        raw_text, srt_lines = transcribe_audio_file(
            model=whisper_model,
            prepared_audio_file=prepared_wav,
            language=language,
            beam_size=beam_size,
        )

    if not raw_text.strip():
        raise RuntimeError("Whisper вернул пустую транскрибацию")

    write_srt(srt_lines, output_dir / "00_transcript.srt")

    run_pipeline_from_text(
        raw_text=raw_text,
        source_name=audio_file.name,
        output_dir=output_dir,
        reference_text=reference_text,
        reference_file=reference_file,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
    )

    # Дописываем информацию об аудио в meta.json
    meta_path = output_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["audio"] = {
        "source_audio": str(audio_file),
        "duration_seconds": duration,
        "whisper_model": whisper_model_name,
        "language": language,
        "beam_size": beam_size,
        "srt_file": str(output_dir / "00_transcript.srt"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def run_from_text_file(
        text_file: Path,
        output_dir: Path,
        reference_text: str,
        reference_file: Path | None,
        base_url: str,
        temperature: float,
        retries: int,
        retry_delay: int,
) -> None:
    # Режим: txt -> pipeline
    raw_text = read_text_file(text_file)

    run_pipeline_from_text(
        raw_text=raw_text,
        source_name=text_file.name,
        output_dir=output_dir,
        reference_text=reference_text,
        reference_file=reference_file,
        base_url=base_url,
        temperature=temperature,
        retries=retries,
        retry_delay=retry_delay,
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    # Аргументы командной строки
    parser = argparse.ArgumentParser(
        description="Многоэтапный pipeline: clean -> plan -> draft -> latex -> latex_fix"
    )

    parser.add_argument(
        "mode",
        choices=["audio", "text-file"],
        help="audio = начать с аудио, text-file = начать с готового txt",
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Путь к аудиофайлу или txt-файлу",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Куда сохранить результат; если не указано, будет создана папка рядом с исходником",
    )

    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="Путь к текстовому референс-файлу",
    )

    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help="Только для mode=audio: модель Whisper",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Только для mode=audio: язык речи",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help="Только для mode=audio: beam search для Whisper",
    )

    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Базовый URL Ollama",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Температура генерации",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Сколько раз повторять неудачный вызов модели",
    )

    parser.add_argument(
        "--retry-delay",
        type=int,
        default=DEFAULT_RETRY_DELAY,
        help="Пауза между retry в секундах",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    reference_file = Path(args.reference_file) if args.reference_file else None
    reference_text = read_optional_reference_file(reference_file)

    # Если output_dir явно передан — используем его
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Иначе строим папку результата рядом с исходником
        if args.mode == "audio":
            output_dir = input_path.with_suffix("")
            output_dir = output_dir.parent / f"{output_dir.name}_manual"
        else:
            output_dir = input_path.with_suffix("")

    if args.mode == "audio":
        if not input_path.exists() or not input_path.is_file():
            print(f"Ошибка: аудиофайл не найден: {input_path}")
            raise SystemExit(1)

        run_from_audio(
            audio_file=input_path,
            output_dir=output_dir,
            reference_text=reference_text,
            reference_file=reference_file,
            whisper_model_name=args.whisper_model,
            language=args.language,
            beam_size=args.beam_size,
            base_url=args.ollama_base_url,
            temperature=args.temperature,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        print("Готово.")
        return

    if not input_path.exists() or not input_path.is_file():
        print(f"Ошибка: txt-файл не найден: {input_path}")
        raise SystemExit(1)

    run_from_text_file(
        text_file=input_path,
        output_dir=output_dir,
        reference_text=reference_text,
        reference_file=reference_file,
        base_url=args.ollama_base_url,
        temperature=args.temperature,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    print("Готово.")


if __name__ == "__main__":
    main()