# CLI Guide: `manual_pipeline_multistage.py`

Этот скрипт запускает многоэтапный pipeline для создания пособия.

Стадии:

1. `clean` — очистка текста через `qwen2.5:latest`
2. `plan` — построение плана пособия через `qwen3.5:4b`
3. `draft` — создание текстового черновика пособия через `qwen2.5-coder:7b`
4. `latex` — генерация LaTeX-кода через `qwen2.5-coder:7b`
5. `latex_fix` — исправление LaTeX-кода через `qwen2.5-coder:7b`

Скрипт умеет работать в двух режимах:

- `audio` — начать с аудиофайла
- `text-file` — начать с уже готового `.txt` файла

---

## 1. Общий синтаксис

```powershell
python manual_pipeline_multistage.py <mode> <input_path> [options]
```

Где:

- `<mode>` — `audio` или `text-file`
- `<input_path>` — путь к аудиофайлу или `.txt` файлу

---

## 2. Режимы

### `audio`

Используется, если у вас есть аудиофайл, который нужно:

1. транскрибировать через Whisper
2. очистить
3. превратить в план пособия
4. превратить в текстовое пособие
5. превратить в LaTeX

Пример:

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --reference-file refs\reference.txt
```

---

### `text-file`

Используется, если у вас уже есть готовый `.txt` файл и транскрибация не нужна.

Пример:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --reference-file refs\reference.txt
```

---

## 3. Основные параметры

### `--output-dir`

Позволяет явно указать папку, куда сохранять результаты.

Пример:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --output-dir results\lesson1
```

Если параметр не указан:

- для `audio` рядом с файлом будет создана папка вида `lesson1_manual`
- для `text-file` рядом с файлом будет создана папка с именем файла без расширения

---

### `--reference-file`

Подключает текстовый референс.

Референс используется как ориентир по:

- стилю
- структуре
- формату
- терминологии

При этом факты должны браться из основного текста, а не из референса.

Пример:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --reference-file refs\reference.txt
```

---

### `--whisper-model`

Только для режима `audio`.

Позволяет выбрать модель Whisper.

Пример:

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --whisper-model medium
```

---

### `--language`

Только для режима `audio`.

Указывает язык речи в аудио.

Пример:

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --language ru
```

---

### `--beam-size`

Только для режима `audio`.

Управляет beam search для Whisper.

Пример:

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --beam-size 5
```

---

### `--ollama-base-url`

Позволяет указать адрес Ollama-сервера.

По умолчанию:

```text
http://localhost:11434
```

Пример:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --ollama-base-url http://localhost:11434
```

---

### `--temperature`

Температура генерации.

Обычно для такого pipeline лучше оставлять `0.0`.

Пример:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --temperature 0.0
```

---

### `--retries`

Сколько раз повторять неудачный вызов модели.

Пример:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --retries 3
```

---

### `--retry-delay`

Пауза между повторными попытками в секундах.

Пример:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --retry-delay 3
```

---

## 4. Полные примеры запуска

### Аудио → полный pipeline

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --reference-file refs\reference.txt
```

### Аудио → полный pipeline с явной папкой результата

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --reference-file refs\reference.txt --output-dir results\lesson1
```

### Готовый `.txt` → полный pipeline

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --reference-file refs\reference.txt
```

### Готовый `.txt` → полный pipeline без референса

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt
```

### Готовый `.txt` → с настройками retry

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --reference-file refs\reference.txt --retries 4 --retry-delay 5
```

---

## 5. Что будет создано в результате

В папке результата появятся файлы:

```text
00_reference.txt        # если был передан reference-file
00_transcript.srt       # только для mode=audio
01_raw.txt              # исходный текст
02_cleaned.txt          # очищенный текст
03_plan_source.md       # материал для плана
04_plan_raw.txt         # сырой ответ модели на этапе plan
04_plan.json            # итоговый JSON-план
05_manual_draft.md      # markdown-черновик пособия
06_manual.tex           # первый LaTeX-вариант
07_manual_fixed.tex     # исправленный LaTeX
meta.json               # метаданные pipeline
pipeline.log            # лог выполнения
```

---

## 6. Что хранится в `meta.json`

Файл `meta.json` содержит:

- имя исходного файла
- путь к референсу
- используемые модели
- пути к результатам
- длины текстов на разных этапах

---

## 7. Что смотреть в случае ошибок

### `pipeline.log`

Это главный файл диагностики.

Там видно:

- на какой стадии произошла ошибка
- какая модель вызывалась
- сколько было попыток
- какая ошибка пришла

---

## 8. Типичные сценарии

### Сценарий 1. У меня аудио лекции
Используйте:

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --reference-file refs\reference.txt
```

### Сценарий 2. У меня уже есть транскрипт в `.txt`
Используйте:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --reference-file refs\reference.txt
```

### Сценарий 3. Хочу просто быстро проверить без референса
Используйте:

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt
```

---

## 9. Рекомендации по использованию

### Для референса
Лучше использовать короткий и структурированный `reference.txt`, где описаны:

- стиль
- формат разделов
- тип подачи
- желаемая терминология

### Для аудио
Если аудио длинное, лучше сначала убедиться, что:

- работает `ffmpeg`
- работает Whisper
- Ollama запущен
- все модели скачаны

---

## 10. Проверка моделей Ollama

Перед запуском pipeline полезно проверить, что все модели доступны:

```powershell
ollama list
```

Ожидаемые модели:

- `qwen2.5:latest`
- `qwen3.5:4b`
- `qwen2.5-coder:7b`

Если какой-то модели нет, скачайте её:

```powershell
ollama pull qwen2.5:latest
ollama pull qwen3.5:4b
ollama pull qwen2.5-coder:7b
```

---

## 11. Проверка зависимостей

Установить Python-библиотеки:

```powershell
pip install openai-whisper langchain langchain-ollama
```

Также нужны:

- `ffmpeg`
- `Ollama`

---

## 12. Быстрый старт

### Из аудио

```powershell
python manual_pipeline_multistage.py audio audio\lesson1.mp3 --reference-file refs\reference.txt
```

### Из готового текста

```powershell
python manual_pipeline_multistage.py text-file texts\lesson1.txt --reference-file refs\reference.txt
```

---

## 13. Шаблон команды

```powershell
python manual_pipeline_multistage.py <audio|text-file> <input_path> --reference-file <reference.txt> --output-dir <result_dir>
```
