"""Pick the glossary words a recording needs, from a quick first listen (retrieval for the speech hints).

The speech model takes only about 40 words of hints, while the dictionary holds hundreds. A short,
hint-free pass over a few parts of the recording gives a rough text; every dictionary word is looked
up in it, both as written and as the model tends to mishear it (Hangul compared letter by letter, so
"학과자님" still finds "학과장님"). Words found come first; the packs they belong to fill the rest.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

_LEAD = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWEL = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_TAIL = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

FUZZY_MIN = 0.8  # letter similarity for words of 3+ syllables (one or two letters off)
FUZZY_MIN_SHORT = 0.83  # two-syllable words: one letter off out of six (뒤심 / 뒷심), not out of five


def normalize(text: str) -> str:
    """Hangul syllables, lowercase Latin letters and digits only (spacing and punctuation vary)."""
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def _jamo(syllables: str) -> str:
    out = []
    for ch in syllables:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_LEAD[code // 588] + _VOWEL[code % 588 // 28] + _TAIL[code % 28].strip())
        else:
            out.append(ch)
    return "".join(out)


def _lead(ch: str) -> str:
    code = ord(ch) - 0xAC00
    return _LEAD[code // 588] if 0 <= code < 11172 else ch


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _spoken(sample_text: str) -> tuple[str, list[int]]:
    """The normalized sample and where each of its words starts. Dictionary words are looked up only from the start
    of a spoken word, so a pack word that is also an ending ("에서", Esau, is the particle ~에서) is not "heard"."""
    sample, starts = "", []
    for token in sample_text.split():
        part = normalize(token)
        if part:
            starts.append(len(sample))
            sample += part
    return sample, starts


def _near(term: str, sample: str, starts: Iterable[int] | None = None) -> float:
    """Best letter-level similarity of ``term`` to any stretch of ``sample`` (both normalized) beginning at one of
    ``starts`` (every position when not given)."""
    n = len(term)
    want = _jamo(term)
    best = 0.0
    first, last = _lead(term[0]), term[-1]
    for i in (range(len(sample)) if starts is None else starts):
        # Only stretches that start like the word or end like it: fast, and mishearings keep one of them.
        if _lead(sample[i]) != first and sample[i + n - 1:i + n] != last:
            continue
        for size in (n - 1, n, n + 1):
            piece = sample[i:i + size]
            if len(piece) < max(2, n - 1):
                continue
            got = _jamo(piece)
            score = 1 - _distance(want, got) / max(len(want), len(got))
            if score > best:
                best = score
    return best


def pick_hints(sample_text: str, candidates: Iterable[Sequence[str]], fill: bool = True) -> list[str]:
    """Dictionary words ordered by how surely the recording uses them.

    ``candidates`` are (word, pack) pairs. Returns the words heard in the sample (exactly or as a
    likely mishearing), most certain first, then (``fill``) the other words of the packs those came from,
    best-matching pack first. Words of packs with no match are left out.
    """
    sample, starts = _spoken(sample_text)
    pairs = [(str(w).strip(), str(p or "")) for w, p in candidates if str(w).strip()]
    scores: dict[str, float] = {}
    for word, _ in pairs:
        if word in scores:
            continue
        key = normalize(word)
        if len(key) < 2 or not sample:
            scores[word] = 0.0
            continue
        hits = sum(1 for i in starts if sample.startswith(key, i))
        if hits:
            scores[word] = 1.0 + min(hits, 5) * 0.1
        elif re.fullmatch(r"[가-힣]+", key):
            near = _near(key, sample, starts)
            # Two-syllable words have few letters, so they must be closer (뒤심 / 뒷심 is one letter off).
            scores[word] = near if near >= (FUZZY_MIN if len(key) >= 3 else FUZZY_MIN_SHORT) else 0.0
        else:
            scores[word] = 0.0
    order = {w: i for i, (w, _) in enumerate(pairs)}
    found = sorted((w for w, s in scores.items() if s > 0), key=lambda w: (-scores[w], order[w]))
    if not fill:
        return found
    affinity: Counter[str] = Counter()
    for word, pack in pairs:
        # A two-syllable near miss is weak evidence: it may pick the word, but does not pull in its pack.
        if scores.get(word, 0) >= 1.0 or (scores.get(word, 0) > 0 and len(normalize(word)) >= 3):
            affinity[pack] += scores[word]
    by_pack: dict[str, list[str]] = defaultdict(list)
    for word, pack in pairs:
        by_pack[pack].append(word)
    picked = list(found)
    seen = set(picked)
    # Then the packs that matched, taking their words in turn (best pack first in each round), so the
    # leading words of every relevant pack get a place before the budget runs out.
    queues = [[w for w in by_pack[pack] if w not in seen] for pack, _ in affinity.most_common()]
    while any(queues):
        for queue in queues:
            while queue and queue[0] in seen:
                queue.pop(0)
            if queue:
                word = queue.pop(0)
                seen.add(word)
                picked.append(word)
    return picked
