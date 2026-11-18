"""
Cache of synthesized phrases (TTS-5).

The key is the backend, voice, sample rate, speed and the normalized text; the value is
the processed PCM of the whole phrase. A hit is sent without synthesis. Phrases from
PREWARM_PHRASES_PATH are synthesized into it at startup, so the warnings of the passive
mode sound without waiting for the model.
"""

import threading

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CacheKey:
    """What the synthesized audio depends on"""
    backend: str
    voice: str
    sample_rate: int
    speed: float
    text: str                           # normalized: the text the model reads


class AudioCache:
    """LRU of synthesized phrases in memory; max_items=0 keeps nothing"""

    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._items: "OrderedDict[CacheKey, bytes]" = OrderedDict()
        # The event loop and the scripts use it; the lock keeps the order consistent anyway
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> Optional[bytes]:
        """The audio of the key, now the most recently used; None on a miss."""
        with self._lock:
            audio = self._items.get(key)
            if audio is not None:
                self._items.move_to_end(key)
            return audio

    def put(self, key: CacheKey, audio: bytes) -> None:
        """Stores the audio, evicting the least recently used phrases over max_items."""
        if self.max_items <= 0:
            return
        with self._lock:
            self._items[key] = audio
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def __contains__(self, key: CacheKey) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def load_phrases(path: Path) -> List[str]:
    """Phrases of the prewarm file, one per line; empty lines are skipped."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
