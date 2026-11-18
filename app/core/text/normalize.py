"""
Text normalization for the synthesis (TTS-2): what the model reads aloud.

Models read words, not digits and signs: numbers become words agreed with the noun
after them, «%», «°» and «№» become words, common abbreviations are expanded.
The text ends as a sentence: a missing final mark becomes a period. The normalized
text is the cache key, so "Впереди стол" and "Впереди стол." are one phrase - and
one synthesis input, so a cache hit sounds exactly as a new synthesis would.
"""

import re
import unicodedata

from typing import List, Tuple

Gender = str                        # "m", "f" or "n"
Forms = Tuple[str, str, str]        # the noun after 1, after 2-4 and after 5-20: метр, метра, метров

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
SENTENCE_END = ".!?…"

_UNITS = {
    "m": ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"),
    "f": ("ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"),
    "n": ("ноль", "одно", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"),
}
_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто")
_HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот")

# Scale words with their gender, from the largest one
_SCALES: Tuple[Tuple[int, Gender, Forms], ...] = (
    (10 ** 9, "m", ("миллиард", "миллиарда", "миллиардов")),
    (10 ** 6, "m", ("миллион", "миллиона", "миллионов")),
    (10 ** 3, "f", ("тысяча", "тысячи", "тысяч")),
)
_MAX_NUMBER = 10 ** 12 - 1          # larger numbers are read digit by digit

_FRACTIONS: Tuple[Tuple[str, str], ...] = (("десятая", "десятых"), ("сотая", "сотых"), ("тысячная", "тысячных"))

PERCENT: Forms = ("процент", "процента", "процентов")
DEGREE: Forms = ("градус", "градуса", "градусов")

# Feminine nouns ending in «ь»; the other ones are taken for masculine
_FEMININE_SOFT = frozenset({
    "дверь", "ступень", "мебель", "кровать", "тетрадь", "площадь", "лестница", "вещь", "мышь", "ночь",
    "дочь", "печь", "тень", "часть", "область", "скорость", "сеть", "цепь", "соль", "боль", "роль",
    "лошадь", "морковь", "кровь", "речь", "помощь", "шерсть", "пыль", "ткань", "зелень", "осень",
    "постель", "дробь", "грудь", "бровь", "ель", "мель", "щель", "дрель", "цель", "модель",
    "емкость", "ёмкость", "плоскость", "поверхность", "мелочь", "ладонь", "запись", "подпись",
})
# Masculine nouns ending in «а» or «я»
_MASCULINE_A = frozenset({"мужчина", "юноша", "папа", "дедушка", "дядя", "судья", "слуга", "парень"})


def plural_form(number: int, forms: Forms) -> str:
    """The noun form after `number`: 1 метр, 2 метра, 5 метров, 21 метр."""
    if 11 <= number % 100 <= 14:
        return forms[2]
    if number % 10 == 1:
        return forms[0]
    if 2 <= number % 10 <= 4:
        return forms[1]
    return forms[2]


def _below_thousand(number: int, gender: Gender) -> List[str]:
    words = []
    hundreds, rest = divmod(number, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
    else:
        tens, units = divmod(rest, 10)
        if tens:
            words.append(_TENS[tens])
        if units:
            words.append(_UNITS[gender][units])
    return words


def number_to_words(number: int, gender: Gender = "m") -> str:
    """A cardinal number in words, nominative; the gender agrees one and two with the noun."""
    if number == 0:
        return _UNITS[gender][0]
    if number > _MAX_NUMBER:
        return " ".join(_UNITS["m"][int(digit)] for digit in str(number))

    words = []
    for scale, scale_gender, forms in _SCALES:
        count, number = divmod(number, scale)
        if count:
            # «тысяча двести», not «одна тысяча двести»
            if not (count == 1 and scale == 10 ** 3):
                words += _below_thousand(count, scale_gender)
            words.append(plural_form(count, forms))
    words += _below_thousand(number, gender)
    return " ".join(words)


def decimal_to_words(integer: int, fraction: str) -> str:
    """A decimal fraction in words: «две целых пять десятых»."""
    whole = "целая" if plural_form(integer, ("1", "2", "5")) == "1" else "целых"
    numerator = int(fraction)
    one, many = _FRACTIONS[len(fraction) - 1]
    part = one if plural_form(numerator, ("1", "2", "5")) == "1" else many
    return f"{number_to_words(integer, 'f')} {whole} {number_to_words(numerator, 'f')} {part}"


def noun_gender(noun: str, number: int) -> Gender:
    """The gender of the noun that follows `number` in the form agreed with it."""
    noun = noun.lower()
    last_two = number % 100
    if number % 10 == 2 and last_two != 12:
        # After 2 the noun is genitive singular: feminine ones end in «ы» or «и»
        return "f" if noun.endswith(("ы", "и")) else "m"
    if number % 10 == 1 and last_two != 11:
        # After 1 the noun is nominative singular
        if noun in _MASCULINE_A:
            return "m"
        if noun.endswith(("а", "я")):
            return "f"
        if noun.endswith(("о", "е", "ё")) or noun.endswith("мя"):
            return "n"
        if noun.endswith("ь"):
            return "f" if noun in _FEMININE_SOFT else "m"
    return "m"


# =====================================================================
# Abbreviations
# =====================================================================

# (pattern, words, may end a sentence): the period of «и т.д.» at the end stays a period
_ABBREVIATIONS: Tuple[Tuple[re.Pattern, str, bool], ...] = (
    (re.compile(r"\bт\.\s?е\.", re.IGNORECASE), "то есть", False),
    (re.compile(r"\bт\.\s?к\.", re.IGNORECASE), "так как", False),
    (re.compile(r"\bт\.\s?д\.", re.IGNORECASE), "так далее", True),
    (re.compile(r"\bт\.\s?п\.", re.IGNORECASE), "тому подобное", True),
    (re.compile(r"\bдр\.", re.IGNORECASE), "другие", True),
    (re.compile(r"\bул\.", re.IGNORECASE), "улица", False),
    (re.compile(r"\bпросп\.", re.IGNORECASE), "проспект", False),
    (re.compile(r"\bпер\.", re.IGNORECASE), "переулок", False),
    (re.compile(r"\bпл\.", re.IGNORECASE), "площадь", False),
    (re.compile(r"\bим\.", re.IGNORECASE), "имени", False),
    (re.compile(r"\bд\.(?=\s?\d)", re.IGNORECASE), "дом", False),
    (re.compile(r"\bкв\.(?=\s?\d)", re.IGNORECASE), "квартира", False),
    (re.compile(r"\bг\.(?=\s?[А-ЯЁ])"), "город", False),
)
# What follows a sentence end: nothing or a new sentence
_NEXT_SENTENCE = re.compile(r"\s*(?:$|[А-ЯЁA-Z«\"(])")


def _expand(pattern: re.Pattern, words: str, may_end: bool, text: str) -> str:
    def replace(match: re.Match) -> str:
        expanded = words.capitalize() if match.group(0)[0].isupper() else words
        if may_end and _NEXT_SENTENCE.match(text, match.end()):
            expanded += "."
        return expanded

    return pattern.sub(replace, text)


# =====================================================================
# Numbers and signs
# =====================================================================

# Sign, integer part, decimal part, then the sign or the word after the number
_NUMBER = re.compile(
    r"(?P<sign>(?<![\w)])[+\-−](?=\d))?"
    r"(?P<integer>[1-9]\d{0,2}(?: \d{3})+(?!\d)|\d+)"       # «10 000» is one number
    r"(?:[.,](?P<fraction>\d{1,3})(?![.,]?\d))?"
    r"(?P<after>\s?(?:%|°\s?[CС]\b|°))?"
)
_NEXT_WORD = re.compile(r"\s?([а-яёА-ЯЁ]+)")
_NUMERO = re.compile(r"№\s?")


def _number_words(match: re.Match, text: str) -> str:
    integer, fraction = match.group("integer").replace(" ", ""), match.group("fraction")
    after = (match.group("after") or "").strip()
    sign = {"+": "плюс ", "-": "минус ", "−": "минус "}.get(match.group("sign") or "", "")

    # «007» is read digit by digit
    if len(integer) > 1 and integer.startswith("0"):
        return sign + " ".join(_UNITS["m"][int(digit)] for digit in integer)

    number = int(integer)
    if fraction:
        words = decimal_to_words(number, fraction)
    elif after:
        words = number_to_words(number, "m")
    else:
        next_word = _NEXT_WORD.match(text, match.end())
        words = number_to_words(number, noun_gender(next_word.group(1), number) if next_word else "m")

    if after:
        # After a decimal fraction the noun is genitive singular: «две целых пять десятых процента»
        forms = PERCENT if after == "%" else DEGREE
        noun = forms[1] if fraction else plural_form(number, forms)
        words += f" {noun}" + (" Цельсия" if after.endswith(("C", "С")) else "")
    return sign + words


def _numbers(text: str) -> str:
    text = _NUMERO.sub("номер ", text)
    parts, position = [], 0
    for match in _NUMBER.finditer(text):
        parts.append(text[position:match.start()])
        parts.append(_number_words(match, text))
        position = match.end()
    parts.append(text[position:])
    return "".join(parts)


# =====================================================================
# Normalization
# =====================================================================

_STRESS = re.compile(rf"\+(?=[{VOWELS}])")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+(?=[,.!?…:;])")


def speakable(text: str) -> bool:
    """The text has something to read aloud: a letter or a digit."""
    return any(char.isalnum() for char in text)


def normalize(text: str, keep_stress: bool = True) -> str:
    """The text as the model reads it; `keep_stress=False` drops the `+` stress marks for a backend that cannot read them."""
    text = unicodedata.normalize("NFC", text)
    text = _SPACES.sub(" ", text).strip()

    # 1. Abbreviations go first: «д. 5» needs the number after it
    for pattern, words, may_end in _ABBREVIATIONS:
        text = _expand(pattern, words, may_end, text)

    # 2. Numbers and the signs after them
    text = _numbers(text)

    # 3. Stress marks
    if not keep_stress:
        text = _STRESS.sub("", text)

    # 4. Spaces and the final mark
    text = _SPACE_BEFORE_PUNCTUATION.sub("", _SPACES.sub(" ", text)).strip()
    if text and speakable(text[-1]):
        text += "."
    return text
