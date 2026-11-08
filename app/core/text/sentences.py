"""
Sentence splitting (TTS-1): each sentence is synthesized and sent at once.

A sentence ends with «.», «!», «?» or «…» (and their runs) followed by a space or the
end of the text, as the gateway splits the model stream (PLT-7). A period of an
abbreviation or an initial does not end a sentence. Nothing is lost: the pieces
joined together give the text back character for character.
"""

import re

from typing import List

# The sentence end: marks, closing quotes and brackets, then the spaces before the next one
_END = re.compile(r"[.!?…]+[»\"')\]]*(?=\s|$)\s*")
_WORD_BEFORE = re.compile(r"(?:^|[\s(«\"])([\w.]+)$")

# Abbreviations that never end a sentence; «т.д.» and «т.п.» end it before a capital letter
_ABBREVIATIONS = frozenset({
    "т.е", "т.к", "ул", "д", "кв", "им", "пр", "просп", "пер", "пл", "г", "см", "стр", "рис", "тел", "напр", "проф", "ок",
})
_MAY_END = frozenset({"т.д", "т.п", "др"})


def _is_abbreviation(text: str, end: re.Match) -> bool:
    """The period at `end` belongs to an abbreviation or an initial."""
    # Only a single period: «?», «!», «…» and «...» end a sentence always
    if end.group(0).rstrip() != ".":
        return False

    word = _WORD_BEFORE.search(text, 0, end.start())
    if word is None:
        return False
    word = word.group(1)
    # Initials: «А. С. Пушкин»
    if len(word) == 1 and word.isupper():
        return True
    word = word.lower()
    if word in _MAY_END:
        rest = text[end.end():]
        return bool(rest) and not rest[0].isupper()
    return word in _ABBREVIATIONS


def split_sentences(text: str) -> List[str]:
    """Sentences of the text, each with the spaces after it."""
    sentences, start = [], 0
    for end in _END.finditer(text):
        if _is_abbreviation(text, end):
            continue
        sentences.append(text[start:end.end()])
        start = end.end()
    if start < len(text):
        sentences.append(text[start:])
    return sentences
