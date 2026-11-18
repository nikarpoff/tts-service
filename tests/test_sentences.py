"""
Sentence splitting (TTS-1): no character is lost, «…», abbreviations and initials.
"""

import pytest

from app.core.pipeline import SpeechPipeline
from app.core.text import split_sentences


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Впереди стол. Справа стул!", ["Впереди стол. ", "Справа стул!"]),
        ("Подожди… Что это?", ["Подожди… ", "Что это?"]),
        ("Как дела?! Хорошо...", ["Как дела?! ", "Хорошо..."]),
        ("Это ул. Ленина. Дом пять.", ["Это ул. Ленина. ", "Дом пять."]),
        ("То есть т.е. так. Дальше.", ["То есть т.е. так. ", "Дальше."]),
        ("Ну и т.д. Потом стол.", ["Ну и т.д. ", "Потом стол."]),
        ("и т.д. и т.п. потом", ["и т.д. и т.п. потом"]),
        ("А. С. Пушкин жил тут. Да.", ["А. С. Пушкин жил тут. ", "Да."]),
        ("Он сказал: «Стоп.» Потом ушёл.", ["Он сказал: «Стоп.» ", "Потом ушёл."]),
        ("Цена 2.5 рубля. Всё.", ["Цена 2.5 рубля. ", "Всё."]),
        ("без точки", ["без точки"]),
        ("  с пробелами.  Ещё.  ", ["  с пробелами.  ", "Ещё.  "]),
        ("Строка.\nНовая строка.", ["Строка.\n", "Новая строка."]),
        ("", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Впереди стол. Справа стул! Слева дверь? Подождите… Хорошо...",
        "  Пробелы   в начале и в конце.  ",
        "Без знаков в конце",
        "?!…. ... Странные знаки!!! И слова.",
        "ул. Ленина, д. 5, кв. 12. т.е. адрес. И т.д. И т.п.",
        "Цифры 1.5, 2,5 и 3. Конец.",
    ],
)
def test_no_characters_lost(text):
    assert "".join(split_sentences(text)) == text


def test_sentences_to_synthesize():
    # Marks without words are not sent to the model, the spaces are dropped
    assert SpeechPipeline.sentences("Впереди стол. ... Справа стул!  ") == ["Впереди стол.", "Справа стул!"]
