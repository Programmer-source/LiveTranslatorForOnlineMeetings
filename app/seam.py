"""Word-level de-duplication at forced-cut seams.

Segment A ends at cut T (+overlap), segment B starts at T (−overlap). Both may contain the
words spoken inside the overlap. Each word belongs to the side its midpoint falls on, and
we compare tail/head words (normalized text + timestamps) so a word is never lost and never
duplicated."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Word:
    text: str  # as printed (may carry punctuation)
    start: float  # absolute seconds
    end: float
    prob: float = 1.0

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    @property
    def norm(self) -> str:
        return normalize(self.text)


_norm_re = re.compile(r"[^\w']+", re.UNICODE)


def normalize(t: str) -> str:
    return _norm_re.sub("", t.lower()).strip("'")


TOL = 0.6  # seconds: timestamp disagreement allowed between two decodes of the same word


def split_at_cut(words: list[Word], cut: float) -> tuple[list[Word], list[Word]]:
    """(words before cut, words after cut) by word midpoint."""
    before = [w for w in words if w.mid < cut]
    after = [w for w in words if w.mid >= cut]
    return before, after


def _matches(w: Word, others: list[Word]) -> bool:
    n = w.norm
    if not n:
        return True  # pure punctuation token — nothing to lose
    return any(o.norm == n and abs(o.start - w.start) <= TOL for o in others)


def merge_seam(a_kept: list[Word], a_carry: list[Word], b_words: list[Word], cut: float) -> list[Word]:
    """Return the words B should emit.

    a_kept:  A's words before the cut (already emitted with A)
    a_carry: A's words after the cut (held back, B is expected to contain them)
    b_words: all words recognised in B
    """
    b_head, b_rest = split_at_cut(b_words, cut)
    a_tail = [w for w in a_kept if w.end >= cut - 3.0]
    # B's words before the cut are duplicates unless A did not have them (rescue).
    rescued_head = [w for w in b_head if not _matches(w, a_tail)]
    # A's words after the cut are kept only if B missed them.
    b_front = [w for w in b_rest if w.start <= cut + 3.0]
    rescued_carry = [w for w in a_carry if not _matches(w, b_front) and not _matches(w, rescued_head)]
    out = sorted(rescued_head + rescued_carry + b_rest, key=lambda w: (w.start, w.end))
    # Final guard: identical adjacent words with overlapping times = same word recognised twice.
    dedup: list[Word] = []
    for w in out:
        if dedup and w.norm and dedup[-1].norm == w.norm and w.start < dedup[-1].end - 0.05:
            continue
        dedup.append(w)
    return dedup


def join_words(words: list[Word]) -> str:
    text = ""
    for w in words:
        t = w.text
        if not text:
            text = t.strip()
        elif t.startswith(" ") or not t:
            text += t
        elif re.match(r"^[,.!?;:%)'\]]", t) or t.startswith("'"):
            text += t
        else:
            text += " " + t
    return re.sub(r"\s+", " ", text).strip()
