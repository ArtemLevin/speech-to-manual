from __future__ import annotations

from speech_to_manual.domain.models import ManualPlan

LATEX_MASTER_PROTOCOL = """
[ФИО ученика: {student_name}]

Ты — Главный редактор, Ведущий методист по математике и Архитектор LaTeX-кода.
Твоя задача — на основе предоставленного материала составить безупречный, готовый к печати LaTeX-документ
в формате структурированного учебного чек-листа.

🛡️ ПРОТОКОЛЫ ПРОВЕРКИ
1) GHOST SOLVER:
- восстанавливай математическое содержание по смыслу;
- исправляй математические/логические ошибки;
- обязательно учитывай ОДЗ;
- проверяй ловушки (деление на ноль, потеря/лишние корни, модуль, знаки, дроби и т.д.);
- делай триангуляцию: тема, определения, примеры, упражнения, ответы — без противоречий.

2) VISUAL PROOF:
- геометрия -> tkz-euclide;
- функции/графики -> pgfplots;
- интервалы/схемы -> TikZ;
- на титульном листе обязательна аккуратная кривая Безье снизу.

3) RUSSIAN TYPESETTING:
- отечественные обозначения: \\tg, \\ctg, \\arctg;
- \\le, \\ge, \\emptyset, \\angle ABC;
- \\sin, \\lim, \\ln прямым шрифтом;
- в интегралах: \\int f(x)\\,dx;
- десятичная запятая через icomma или {{,}}.

4) STYLE OF THE MANUAL:
- сдержанный академичный стиль;
- без визуальной перегрузки и пёстрых блоков;
- структура выделяется типографикой;
- читабельность важнее декора.

5) CLEAN CODE:
- итог обязан компилироваться в pdfLaTeX;
- удалить неиспользуемые/проблемные настройки;
- финальный self-check по математике, методике, графике и компиляции.

⚙️ ТЕХНИЧЕСКИЙ СТАНДАРТ ПРЕАМБУЛЫ (ОБЯЗАТЕЛЬНО):
\\documentclass[a4paper,12pt]{{article}}
\\usepackage[T2A]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[russian]{{babel}}
\\usepackage{{amsmath, amssymb, amsthm, mathtools}}
\\usepackage{{helvet}}
\\renewcommand{{\\familydefault}}{{\\sfdefault}}
\\usepackage{{icomma}}
\\usepackage{{geometry}}
\\usepackage{{microtype}}
\\usepackage{{tikz}}
\\usepackage{{tkz-euclide}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\usepackage{{tabularx, booktabs, longtable}}
\\usepackage{{enumitem}}
\\usepackage{{hyperref}}
\\usepackage{{parskip}}
\\usepackage{{fancyhdr}}
\\usepackage{{setspace}}
\\usepackage{{xcolor}}

Цвета (обязательная база):
\\definecolor{{accent}}{{HTML}}{{008080}}
\\definecolor{{darkaccent}}{{HTML}}{{004D4D}}
\\definecolor{{textgray}}{{HTML}}{{333333}}

СТРУКТУРА ИТОГОВОГО ДОКУМЕНТА (строго в этом порядке):
1. Титульный лист:
   - «Чек-лист» + тема;
   - «Лёвин Артём Александрович эксклюзивно для [ФИО ученика в родительном падеже]»;
   - \\today;
   - кривая Безье снизу;
   - весь текст титула центрирован.
2. Основной чек-лист.
3. Отдельная страница: тренировочный блок.
4. Отдельная страница: таблица ответов к упражнениям.

ТРЕНИРОВОЧНЫЙ БЛОК:
- 10 упражнений (если не указано иное);
- 3 средних, 6 выше средних, 1 сложная;
- строго по теме;
- без решений в блоке упражнений.

ФОРМАТ ВЫВОДА:
- обращение на Вы;
- только LaTeX-код;
- полный минимальный .tex-документ;
- никаких пояснений вне кода.
""".strip()


class PromptFactory:
    @staticmethod
    def clean_user_prompt(text: str, reference: str) -> str:
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

    @staticmethod
    def plan_source_user_prompt(text: str, reference: str) -> str:
        return (
            "Ниже даны:\n"
            "1. Фрагмент очищенного учебного текста.\n"
            "2. Референс.\n\n"
            "Сделай структурированную выжимку для дальнейшего построения плана пособия.\n\n"
            "Нужно выделить:\n"
            "- тему фрагмента\n- ключевые подтемы\n- определения\n- методы/алгоритмы\n"
            "- типичные ошибки\n- примеры/упражнения, если есть\n\n"
            "Формат ответа: краткий markdown.\n"
            "Не выдумывай.\n"
            "Используй референс только как ориентир по структуре.\n\n"
            "=== ФРАГМЕНТ ===\n"
            f"{text}\n\n"
            "=== РЕФЕРЕНС ===\n"
            f"{reference}\n"
        )

    @staticmethod
    def plan_json_user_prompt(text: str, reference: str) -> str:
        # JSON-структура собрана как строка, чтобы избежать конфликтов с {} в ChatPromptTemplate.
        json_schema_text = (
            "{\n"
            '  "title": "...",\n'
            '  "topic": "...",\n'
            '  "audience": "...",\n'
            '  "goal": "...",\n'
            '  "sections": [{"id": "section_id", "title": "...", "purpose": "...", "subsections": ["..."]}],\n'
            '  "practice_block": {"needed": true, "format": "..."},\n'
            '  "answers_block": {"needed": true, "format": "..."}\n'
            "}\n"
        )
        return (
            "Ниже даны материал и референс.\n"
            "Верни только валидный JSON-объект плана.\n\n"
            f"Структура JSON:\n{json_schema_text}\n"
            "Правила: только JSON, без пояснений, без выдумок.\n\n"
            "=== МАТЕРИАЛ ДЛЯ ПЛАНА ===\n"
            f"{text}\n\n"
            "=== РЕФЕРЕНС ===\n"
            f"{reference}\n"
        )

    @staticmethod
    def draft_user_prompt(cleaned_text: str, plan_json: str, reference: str) -> str:
        return (
            "Ниже даны очищенный текст, JSON-план и референс.\n"
            "Напиши учебное пособие в markdown строго по плану.\n"
            "Не добавляй факты вне исходного текста и верни только markdown.\n\n"
            f"=== ОЧИЩЕННЫЙ ТЕКСТ ===\n{cleaned_text}\n\n"
            f"=== ПЛАН JSON ===\n{plan_json}\n\n"
            f"=== РЕФЕРЕНС ===\n{reference}\n"
        )

    @staticmethod
    def latex_user_prompt(draft_text: str, plan: ManualPlan, reference: str, student_name: str) -> str:
        protocol = LATEX_MASTER_PROTOCOL.format(student_name=student_name or "...")
        return (
            f"{protocol}\n\n"
            "Ниже даны markdown-черновик, план и референс. "
            "Строго выполни все протоколы выше и верни только полный LaTeX-код.\n\n"
            f"=== MARKDOWN ===\n{draft_text}\n\n"
            f"=== ПЛАН JSON ===\n{plan.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
            f"=== РЕФЕРЕНС ===\n{reference}\n"
        )

    @staticmethod
    def latex_fix_user_prompt(tex_code: str) -> str:
        return (
            "Ниже дан LaTeX-код. Исправь явные синтаксические проблемы, "
            "сохрани содержание и верни только полный исправленный .tex файл.\n\n"
            f"=== LATEX ===\n{tex_code}\n"
        )
