import pytest

from speech_to_manual.domain.errors import JsonRepairError, PlanValidationError
from speech_to_manual.services.validators import JsonPlanParser


VALID_PLAN = '''
```json
{
  "title": "T",
  "topic": "X",
  "audience": "Y",
  "goal": "Z",
  "sections": [
    {"id": "s1", "title": "Sec", "purpose": "P", "subsections": ["a"]}
  ],
  "practice_block": {"needed": true, "format": "quiz"},
  "answers_block": {"needed": true, "format": "short"}
}
```
'''


def test_plan_parser_valid() -> None:
    plan = JsonPlanParser.parse_and_validate(VALID_PLAN)
    assert plan.title == "T"
    assert len(plan.sections) == 1


def test_plan_parser_missing_required() -> None:
    bad = '{"title": "x"}'
    with pytest.raises(PlanValidationError):
        JsonPlanParser.parse_and_validate(bad)


def test_plan_parser_no_json() -> None:
    with pytest.raises(JsonRepairError):
        JsonPlanParser.parse_and_validate("hello")


def test_plan_parser_python_dict_repair() -> None:
    python_style = """{
      'title': 'T',
      'topic': 'X',
      'audience': 'Y',
      'goal': 'Z',
      'sections': [
        {'id': 's1', 'title': 'Sec', 'purpose': 'P', 'subsections': ['a']}
      ],
      'practice_block': {'needed': True, 'format': 'quiz'},
      'answers_block': {'needed': True, 'format': 'short'},
    }"""
    plan = JsonPlanParser.parse_and_validate(python_style)
    assert plan.topic == "X"
