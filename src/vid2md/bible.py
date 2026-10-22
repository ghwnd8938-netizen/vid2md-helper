"""성경 대조: a transcript next to the Bible (개역한글, 1961).

Only about 40 hint words fit in front of the speech recogniser, so the Bible is used after it instead, with no limit:

- verses read aloud are looked up in all 31,103 verses (nobody has to say the reference). A line that is a whole verse
  but for a few letters becomes that verse; other lines that follow a verse closely are offered in the editor.
- a book name one syllable off before "N장" is put right ("요안복음 3장" → "요한복음 3장"); other Bible names one
  syllable off are offered.

The text stays in the private repository and on the servers that run this: its copyright is unclear (the Korean Bible
Society still marks it © 1961; Wikisource lists it as expired in 2012), so it is never published (not in the public
helper's repository). Without the file, check() finds nothing.
"""

from __future__ import annotations

import difflib
import gzip
import logging
import re
from pathlib import Path

log = logging.getLogger("vid2md.bible")

DATA = Path(__file__).with_name("data") / "krv.tsv.gz"

# 개역한글 book names and their short forms, in order (book 1 is 창세기).
BOOKS = (
    ("창세기", "창"), ("출애굽기", "출"), ("레위기", "레"), ("민수기", "민"), ("신명기", "신"), ("여호수아", "수"),
    ("사사기", "삿"), ("룻기", "룻"), ("사무엘상", "삼상"), ("사무엘하", "삼하"), ("열왕기상", "왕상"), ("열왕기하", "왕하"),
    ("역대상", "대상"), ("역대하", "대하"), ("에스라", "스"), ("느헤미야", "느"), ("에스더", "에"), ("욥기", "욥"),
    ("시편", "시"), ("잠언", "잠"), ("전도서", "전"), ("아가", "아"), ("이사야", "사"), ("예레미야", "렘"),
    ("예레미야애가", "애"), ("에스겔", "겔"), ("다니엘", "단"), ("호세아", "호"), ("요엘", "욜"), ("아모스", "암"),
    ("오바댜", "옵"), ("요나", "욘"), ("미가", "미"), ("나훔", "나"), ("하박국", "합"), ("스바냐", "습"), ("학개", "학"),
    ("스가랴", "슥"), ("말라기", "말"), ("마태복음", "마"), ("마가복음", "막"), ("누가복음", "눅"), ("요한복음", "요"),
    ("사도행전", "행"), ("로마서", "롬"), ("고린도전서", "고전"), ("고린도후서", "고후"), ("갈라디아서", "갈"),
    ("에베소서", "엡"), ("빌립보서", "빌"), ("골로새서", "골"), ("데살로니가전서", "살전"), ("데살로니가후서", "살후"),
    ("디모데전서", "딤전"), ("디모데후서", "딤후"), ("디도서", "딛"), ("빌레몬서", "몬"), ("히브리서", "히"),
    ("야고보서", "약"), ("베드로전서", "벧전"), ("베드로후서", "벧후"), ("요한일서", "요일"), ("요한이서", "요이"),
    ("요한삼서", "요삼"), ("유다서", "유"), ("요한계시록", "계"),
)

# People, places and words of the Bible and of preaching: hint words for the dictionary's 📖 성경 pack, and the names a
# misheard word is compared with (those of three syllables or more that are written in the Bible).
NAMES = """
아담 하와 가인 아벨 셋 에녹 므두셀라 노아 셈 함 야벳 아브라함 아브람 사라 사래 이삭 리브가 야곱 에서 라헬 레아 요셉
유다 레위 베냐민 르우벤 시므온 단 납달리 갓 아셀 잇사갈 스불론 에브라임 므낫세 모세 아론 미리암 여호수아 갈렙 기드온 삼손
드보라 룻 나오미 보아스 한나 엘리 사무엘 사울 다윗 요나단 밧세바 솔로몬 르호보암 여로보암 엘리야 엘리사 아합 이세벨
히스기야 요시야 이사야 예레미야 에스겔 다니엘 느헤미야 에스라 에스더 모르드개 하만 욥 요나 호세아 아모스 미가 말라기
스가랴 학개 세례 요한 베드로 시몬 안드레 야고보 빌립 바돌로매 도마 마태 다대오 가룟 맛디아 바울 바나바 실라 디모데 디도
누가 마가 스데반 니고데모 나사로 마리아 마르다 막달라 삭개오 빌라도 헤롯 가야바 아굴라 브리스길라 아볼로 고넬료 아나니아
삽비라 루디아 빌레몬 오네시모 멜기세덱 이스마엘 하갈 롯 라반 바로 고라 발람 라합 아간 입다 압살롬 나단 므비보셋 우리아
엘리후 엘리바스 빌닷 소발 그리스도 예수 임마누엘 여호와 이스라엘 유대인 이방인 바리새인 사두개인 사마리아인
에덴 바벨 갈대아 우르 하란 가나안 애굽 시내산 호렙 요단 여리고 헤브론 브엘세바 벧엘 세겜 실로 기브온 베들레헴 예루살렘
시온 사마리아 갈릴리 나사렛 가버나움 벳새다 고라신 가이사랴 빌립보 데살로니가 베뢰아 아덴 고린도 에베소 갈라디아 골로새
라오디게아 버가모 두아디라 사데 서머나 빌라델비아 밧모 로마 안디옥 다메섹 다소 구브로 그레데 바벨론 앗수르 니느웨 바사
메대 블레셋 모압 암몬 에돔 미디안 아말렉 시돈 두로 레바논 갈멜 감람산 겟세마네 골고다 엠마오 베다니 므깃도 아마겟돈 소돔
고모라 길갈 라마 미스바 엔게디
성령 성부 성자 삼위일체 할렐루야 호산나 아멘 언약 성막 법궤 언약궤 지성소 성소 번제 화목제 속죄제 유월절 오순절 초막절
안식일 안식년 희년 할례 침례 성찬 만나 메추라기 율법 선지자 사도 제사장 대제사장 레위인 서기관 산헤드린 회당 성전 십계명
산상수훈 팔복 주기도문 천국 구원 속죄 대속 칭의 성화 부활 승천 재림 휴거 환난 천년왕국 생명나무 선악과 무화과 감람나무
포도원 목자 어린양 복음 말씀 계시 예언 비유 이적 표적 기사 은혜 긍휼 회개 중생 영생 심판 사망 음부 낙원 보좌 면류관
""".split()

# Endings that follow a name in speech (조사 and the like), longest first.
ENDINGS = ("에서는", "에게서", "으로는", "께서는", "께서", "에서", "으로", "에게", "까지", "부터", "이며", "이다", "입니다",
           "이라", "님이", "님은", "님의", "님", "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만", "아", "야")
CHAPTER = re.compile(r"([가-힣]{2,8})(\s*)(\d+)\s*장")
WORD = re.compile(r"[가-힣]+")
MIN_MATCH = 15  # letters a line must share with a verse before it is offered
NEAR = 0.8  # share of a line's letters found in the verse(s), in order
SAME = 0.92  # a line this close to whole verses (both ways) becomes them
WHOLE = 0.85  # a line that holds this much of the verse(s) may be replaced by them in the editor (less: shown only)

_index = None  # (verses, normalized texts, gram keys sorted, verse ids for them, vocabulary)


def normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


def _grams(text: str):
    return [(ord(text[i]) << 32) | (ord(text[i + 1]) << 16) | ord(text[i + 2]) for i in range(len(text) - 2)]


def _load():
    global _index
    if _index is not None:
        return _index
    if not DATA.exists():
        _index = ()
        return _index
    import numpy as np

    verses, norm, keys, ids, vocabulary = [], [], [], [], set()
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for line in f:
            book, chapter, verse, text = line.rstrip("\n").split("\t", 3)
            k = len(verses)
            verses.append((int(book), int(chapter), int(verse), text))
            n = normalize(text)
            norm.append(n)
            g = _grams(n)
            keys.extend(g)
            ids.extend([k] * len(g))
            vocabulary.update(_stem(w) for w in WORD.findall(text))
    keys_a, ids_a = np.array(keys, dtype=np.int64), np.array(ids, dtype=np.int32)
    order = np.argsort(keys_a, kind="stable")
    _index = (verses, norm, keys_a[order], ids_a[order], vocabulary)
    log.info("성경 대조: %d절", len(verses))
    return _index


def _stem(word: str) -> str:
    for ending in ENDINGS:
        if len(word) > len(ending) + 1 and word.endswith(ending):
            return word[: -len(ending)]
    return word


def _one_off(a: str, b: str) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def ref(verse: tuple[int, int, int, str], last: tuple[int, int, int, str] | None = None) -> str:
    short = BOOKS[verse[0] - 1][1]
    if last and last[2] != verse[2]:
        return f"{short} {verse[1]}:{verse[2]}-{last[2]}"
    return f"{short} {verse[1]}:{verse[2]}"


def _candidates(text: str, index, top: int = 4) -> list[int]:
    import numpy as np

    _, _, keys, ids, _ = index
    grams = np.array(sorted(set(_grams(text))), dtype=np.int64)
    if not len(grams):
        return []
    left, right = np.searchsorted(keys, grams, "left"), np.searchsorted(keys, grams, "right")
    hits = [ids[a:b] for a, b in zip(left, right) if 0 < b - a <= 400]  # very common runs say little
    if not hits:
        return []
    counts = np.bincount(np.concatenate(hits))
    best = np.argsort(counts)[::-1][:top]
    return [int(k) for k in best if counts[k] >= max(4, len(grams) // 4)]


def _match(text: str, index):
    """The verse (or run of verses in one chapter) that ``text`` (normalized) follows best: (first, last, matched)."""
    verses, norm, *_ = index
    best = None
    for k in _candidates(text, index):
        for a, b in ((k, k), (k - 1, k), (k, k + 1), (k - 1, k + 1), (k, k + 2)):
            if a < 0 or b >= len(verses) or verses[a][:2] != verses[b][:2]:
                continue
            target = "".join(norm[a : b + 1])
            matched = sum(m.size for m in difflib.SequenceMatcher(None, text, target, autojunk=False).get_matching_blocks())
            score = (matched / len(text), matched / len(target))
            if best is None or (matched, score[1]) > (best[2], best[3][1]):
                best = (a, b, matched, score)
    return best


def check(lines: list[str]) -> tuple[list[str], list[dict]]:
    """The lines with what is certain put right, and what was found (for the editor's 📖 성경 대조)."""
    index = _load()
    if not index:
        return list(lines), []
    verses, norm, _, _, vocabulary = index
    out, found = list(lines), []
    full = {name for name, _ in BOOKS}
    names = [n for n in NAMES + [b for b, _ in BOOKS] if len(n) >= 3]

    # Book names before "N장"
    for i, line in enumerate(out):
        def fix(m: re.Match) -> str:
            word = m.group(1)
            if word in full:
                return m.group(0)
            right = [b for b in full if _one_off(word, b)]
            if len(right) != 1:
                return m.group(0)
            found.append({"kind": "name", "line": i, "wrong": word, "right": right[0], "auto": True})
            return right[0] + m.group(0)[len(word):]
        out[i] = CHAPTER.sub(fix, line)

    # Verses read aloud: a line, or two or three lines together. The best fits are taken first, so a line of the
    # preacher's own words next to a verse does not get counted in with it.
    windows = []
    for i in range(len(out)):
        for span in (1, 2, 3):
            if i + span > len(out):
                break
            text = normalize("".join(out[i : i + span]))
            if len(text) < MIN_MATCH:
                continue
            m = _match(text, index)
            if m and m[2] >= MIN_MATCH and m[3][0] >= NEAR:
                windows.append((m[2] - 2 * (len(text) - m[2]), i, span, m))
    taken: set[int] = set()
    for _, i, span, (a, b, matched, (near, cover)) in sorted(windows, key=lambda w: (-w[0], w[2])):
        if any(j in taken for j in range(i, i + span)):
            continue
        taken.update(range(i, i + span))
        verse_text = " ".join(verses[k][3] for k in range(a, b + 1))
        if normalize("".join(out[i : i + span])) == "".join(norm[a : b + 1]):
            continue  # already word for word
        item = {"kind": "verse", "line": i, "lines": span, "ref": ref(verses[a], verses[b] if b > a else None),
                "text": verse_text, "score": round(min(near, cover), 3), "whole": span == 1 and cover >= WHOLE}
        if span == 1 and near >= SAME and cover >= SAME:
            item.update(auto=True, was=out[i])
            out[i] = verse_text
        found.append(item)
    found.sort(key=lambda f: f["line"])

    # Other names one syllable off (offered only: a word of everyday speech can be one syllable from a name)
    for i, line in enumerate(out):
        for word in WORD.findall(line):
            stem = _stem(word)
            if len(stem) < 3 or stem in vocabulary:
                continue
            right = [n for n in names if _one_off(stem, n)]
            if len(right) == 1:
                found.append({"kind": "name", "line": i, "wrong": stem, "right": right[0]})
    return out, sorted(found, key=lambda f: f["line"])


def glossary_text() -> str:
    """The 📖 성경 pack for the dictionary (paste into 용어 사전 → 여러 개 한 번에 붙여넣기)."""
    words = [b for b, _ in BOOKS] + [n for n in NAMES if len(n) >= 2]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return "## 성경\n" + "\n".join(seen) + "\n"
