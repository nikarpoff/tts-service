"""
Text normalization (TTS-2): numbers in words agreed with the noun, signs, abbreviations, stress.
"""

import pytest

from app.core.text import normalize, number_to_words


@pytest.mark.parametrize(
    "text, expected",
    [
        # Gender of the number follows the noun after it
        ("2 метра", "два метра."),
        ("2 двери", "две двери."),
        ("21 стул", "двадцать один стул."),
        ("1 дверь", "одна дверь."),
        ("1 окно", "одно окно."),
        ("1 скамейка", "одна скамейка."),
        ("22 двери", "двадцать две двери."),
        ("12 стульев", "двенадцать стульев."),
        ("11 дверей", "одиннадцать дверей."),
        ("2 двери и 3 стула", "две двери и три стула."),
        ("1 мужчина", "один мужчина."),
        # Large, grouped and decimal numbers
        ("1000 шагов", "тысяча шагов."),
        ("10 000 шагов", "десять тысяч шагов."),
        ("2001 год", "две тысячи один год."),
        ("2,5 метра", "две целых пять десятых метра."),
        ("1.5 метра", "одна целая пять десятых метра."),
        ("007", "ноль ноль семь."),
        # Signs
        ("Влажность 50%", "Влажность пятьдесят процентов."),
        ("1 %", "один процент."),
        ("3%", "три процента."),
        ("-5°", "минус пять градусов."),
        ("+21 °C", "плюс двадцать один градус Цельсия."),
        ("2°", "два градуса."),
        ("№ 7", "номер семь."),
        ("Автобус №5", "Автобус номер пять."),
        # Abbreviations
        ("ул. Ленина, д. 5", "улица Ленина, дом пять."),
        ("т.е. направо", "то есть направо."),
        ("Т. е. направо", "То есть направо."),
        ("Стулья, столы и т.д. Потом дверь.", "Стулья, столы и так далее. Потом дверь."),
        ("кв. 12", "квартира двенадцать."),
        # Spaces and the final mark
        ("  Впереди   стол  ", "Впереди стол."),
        ("Где дверь?", "Где дверь?"),
        ("Впереди стол!", "Впереди стол!"),
        ("Подождите…", "Подождите…"),
        ("Диапазон 5-6 метров", "Диапазон пять-шесть метров."),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_number_to_words():
    assert number_to_words(0) == "ноль"
    assert number_to_words(115) == "сто пятнадцать"
    assert number_to_words(2_000_000) == "два миллиона"
    assert number_to_words(1_234_567) == "один миллион двести тридцать четыре тысячи пятьсот шестьдесят семь"
    assert number_to_words(21_000) == "двадцать одна тысяча"
    assert number_to_words(2, "f") == "две"
    assert number_to_words(1, "n") == "одно"


def test_stress_marks():
    # A backend that reads the stress gets it, the others get the plain word
    assert normalize("з+амок на дв+ери") == "з+амок на дв+ери."
    assert normalize("з+амок на дв+ери", keep_stress=False) == "замок на двери."


def test_trailing_period_is_one_phrase():
    # The cache key: with and without the final period it is the same text for the model
    assert normalize("Осторожно, впереди ступени") == normalize("Осторожно, впереди ступени.")
    assert normalize("Осторожно, впереди ступени") != normalize("Осторожно, впереди ступени?")


def test_no_digits_left():
    text = normalize("В 3 метрах 2 стула, 15% заряда, 7°, № 12, 1,25 км")
    assert not any(char.isdigit() for char in text)
    assert "%" not in text and "°" not in text and "№" not in text
