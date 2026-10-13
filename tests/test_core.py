import asyncio
import json
import random
import re

import numpy as np
import pytest

from app.hub import Hub
from app.seam import Word, join_words, merge_seam, split_at_cut
from app.segmenter import Segment, Segmenter
from app.session import SessionWriter, build_srt, fmt_srt_time, srt_entry
from app.stt import SeamState, collapse_loops, is_hallucination
from app.translate import Ctx, ProviderError, TranslatorManager, clean_output, validate
from app.zoom import parse_url, split_caption


# ----------------------------------------------------------------------------- seam

def W(t, s, e):
    return Word(" " + t, s, e)


def test_seam_dedup_overlap_words():
    cut = 15.0
    a = [W("we", 13.9, 14.2), W("need", 14.2, 14.6), W("to", 14.6, 14.9), W("plan", 15.0, 15.2)]
    b = [W("need", 14.25, 14.6), W("to", 14.6, 14.85), W("plan", 15.02, 15.3), W("carefully.", 15.3, 15.9)]
    a_kept, a_carry = split_at_cut(a, cut)
    out = merge_seam(a_kept, a_carry, b, cut)
    full = join_words(a_kept + out)
    assert full == "we need to plan carefully."


def test_seam_rescues_words_missing_on_one_side():
    cut = 15.0
    a = [W("one", 14.0, 14.3), W("two", 14.4, 14.8)]  # A missed "three"
    b = [W("three", 15.0, 15.3), W("four", 15.4, 15.8)]
    a_kept, a_carry = split_at_cut(a, cut)
    assert join_words(a_kept + merge_seam(a_kept, a_carry, b, cut)) == "one two three four"
    # B missed a word that A recognised after the cut
    a = [W("one", 14.0, 14.3), W("two", 14.6, 14.95), W("three", 15.0, 15.18)]
    b = [W("four", 15.4, 15.8)]
    a_kept, a_carry = split_at_cut(a, cut)
    assert [w.norm for w in a_carry] == ["three"]
    assert join_words(a_kept + merge_seam(a_kept, a_carry, b, cut)) == "one two three four"
    # B recognised a word before the cut that A missed
    a = [W("one", 14.0, 14.3)]
    b = [W("two", 14.6, 14.9), W("three", 15.1, 15.4)]
    a_kept, a_carry = split_at_cut(a, cut)
    assert join_words(a_kept + merge_seam(a_kept, a_carry, b, cut)) == "one two three"


def test_seam_state_through_segments():
    st = SeamState()
    seg_a = Segment(0, np.zeros(1), 0.0, 15.2, 0, 0, 0, kind="forced", cut_end=15.0)
    seg_b = Segment(1, np.zeros(1), 14.8, 20.0, 0, 0, 0, kind="pause", cut_start=15.0)
    a_words = [W("hello", 14.0, 14.5), W("big", 14.8, 15.1)]
    b_words = [W("big", 14.85, 15.1), W("world.", 15.2, 15.6)]
    out_a = st.apply(seg_a, a_words)
    out_b = st.apply(seg_b, b_words)
    assert join_words(out_a + out_b) == "hello big world."


def test_join_words_punctuation():
    assert join_words([Word(" Hello", 0, 1), Word(",", 1, 1), Word(" world", 1, 2), Word("!", 2, 2)]) == "Hello, world!"


def test_collapse_loops_only_long_repeats():
    words = [W("I", 0, 0), W("think", 0, 0), W("that", 0, 0), W("that", 0, 0)]
    assert len(collapse_loops(words)) == 4  # real speech kept
    loop = [W("thank", 0, 0), W("you", 0, 0)] * 8
    assert join_words(collapse_loops(loop)) == "thank you thank you"


def test_hallucination_guard_is_conservative():
    assert is_hallucination("Thank you for watching!", -0.9, 0.5, 0.2)
    assert not is_hallucination("Thank you for watching!", -0.2, 0.01, 0.9)  # confident real speech stays
    assert not is_hallucination("Today we talk about budgets.", -0.8, 0.4, 0.5)
    assert is_hallucination("Something", -1.5, 0.9, 0.0)


# ----------------------------------------------------------------------------- segmenter

class EnergyVAD:
    def __call__(self, frames):
        return (np.sqrt(np.mean(frames ** 2, axis=1)) > 0.02).astype(np.float32)


def _speech(sec, amp=0.3):
    t = np.arange(int(sec * 16000)) / 16000
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_segmenter_cuts_at_pauses_and_forces_long_speech():
    segs = []
    sg = Segmenter(EnergyVAD(), segs.append)
    audio = np.concatenate([
        np.zeros(8000, np.float32), _speech(5), np.zeros(16000, np.float32),  # 5 s + 1 s pause
        _speech(2), np.zeros(int(0.3 * 16000), np.float32), _speech(2),  # short pause inside short seg: not cut
        np.zeros(16000, np.float32),
        _speech(33),  # long continuous speech -> forced cuts
        np.zeros(32000, np.float32),
    ])
    for i in range(0, len(audio), 1600):
        sg.feed(audio[i:i + 1600], 1000.0 + (i + 1600) / 16000)
    sg.flush()
    seqs = [s.seq for s in segs]
    assert seqs == sorted(seqs) and seqs == list(range(len(segs)))
    assert all(s.duration <= 15.0 + 0.3 for s in segs)
    forced = [s for s in segs if s.kind == "forced"]
    assert len(forced) >= 2
    # forced segments overlap the next one around the cut
    for s in forced:
        nxt = segs[s.seq + 1]
        assert nxt.cut_start == s.cut_end
        assert nxt.abs_start < s.cut_end < s.abs_end
    # every speech sample is covered by some segment
    speech_mask = np.abs(audio) > 0.05
    covered = np.zeros(len(audio), bool)
    for s in segs:
        covered[int(s.abs_start * 16000):int(s.abs_end * 16000)] = True
    assert not np.any(speech_mask & ~covered)
    # the 2 s + 0.3 s + 2 s block stays together (segment < 3 s needs a longer pause)
    assert any(4.0 < s.duration < 5.0 for s in segs)
    # wall clock mapping
    assert abs(segs[0].wall_start - (1000.0 + segs[0].abs_start)) < 0.11


# ----------------------------------------------------------------------------- srt / files

def test_srt_generation(tmp_path):
    assert fmt_srt_time(3723.456) == "01:02:03,456"
    assert srt_entry(1, 1.0, 0.5, "x").startswith("1\n00:00:01,000 --> 00:00:02,000\nx")
    lines = [{"t0": 0.5, "t1": 2.0, "uk": "Привіт", "en": "Hi"}, {"t0": 2.5, "t1": 4.0, "uk": None, "en": "x"},
             {"t0": 4.5, "t1": 6.25, "uk": "Світ", "en": "World"}]
    srt = build_srt(lines)
    assert srt == ("1\n00:00:00,500 --> 00:00:02,000\nПривіт\n\n"
                   "2\n00:00:04,500 --> 00:00:06,250\nСвіт\n\n")
    w = SessionWriter(1_700_000_000, base=tmp_path)
    w.write_en({"t0": 0.5, "en": "Hi"})
    w.write_translated({"t0": 0.5, "t1": 2.0, "en": "Hi", "uk": "Привіт"})
    w.close()
    assert (w.dir / "subtitles_uk.srt").read_text("utf-8").startswith("1\n00:00:00,500 --> 00:00:02,000\nПривіт")
    assert "UK: Привіт" in (w.dir / "bilingual.txt").read_text("utf-8")
    assert "Hi" in (w.dir / "transcript_en.txt").read_text("utf-8")


# ----------------------------------------------------------------------------- zoom

def test_zoom_url_and_split():
    base, seq = parse_url("https://wmcc.zoom.us/closedcaption?id=1&ns=QQ%3D%3D&expire=9&signature=a%2Fb&seq=7&lang=en-US")
    assert seq == 7 and "seq=" not in base and "lang=" not in base and "signature=a%2Fb" in base
    assert parse_url("https://x.zoom.us/cc?id=5")[1] == 0
    text = ("Це перше речення, яке є досить довгим і має кілька частин, розділених комами. "
            "Друге речення коротке. ") * 3
    chunks = split_caption(text, 120)
    assert all(len(c) <= 120 for c in chunks)
    assert re.sub(r"\s+", "", "".join(chunks)) == re.sub(r"\s+", "", text)


# ----------------------------------------------------------------------------- translation

class FlakyProvider:
    label = "flaky"

    def __init__(self, fail_times, out=lambda t: "Переклад: " + t):
        self.fail_times = fail_times
        self.calls = 0
        self.out = out

    def available_in(self):
        return 0.0

    async def translate(self, items, ctx):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("boom", "other")
        return [self.out(t) for t in items]


def test_validation():
    assert validate("Hello there my friends, welcome.", "") is not None
    assert validate("Hello there my friends, welcome to the session.", "Привіт") is not None
    assert validate("Hello there my friends", "Hello there my friends") is not None
    assert validate("Hello there my friends", "Привіт, мої друзі") is None
    assert validate("Zoom", "Zoom") is None
    assert clean_output('UK: «Привіт»') == "Привіт"


def test_manager_retries_until_success_and_falls_back():
    async def run():
        primary = FlakyProvider(10 ** 9)
        backup = FlakyProvider(3)
        m = TranslatorManager([primary, backup], max_backoff=0.01)
        retries = []
        res = await m.translate(["Hello world, how are you?"], Ctx([], {}, [], ""), on_retry=lambda e, n: retries.append(n))
        assert res == ["Переклад: Hello world, how are you?"]
        assert retries == [1, 2, 3]

    asyncio.run(run())


def test_manager_rejects_bad_output_then_retries():
    async def run():
        bad = FlakyProvider(0, out=lambda t: "")
        good = FlakyProvider(0, out=lambda t: "Добрий день усім присутнім тут")
        m = TranslatorManager([bad, good], max_backoff=0.01)
        res = await m.translate(["Good afternoon to everyone present here"], Ctx([], {}, [], ""))
        assert res == ["Добрий день усім присутнім тут"]

    asyncio.run(run())


# ----------------------------------------------------------------------------- hub replay

def test_hub_replay_and_snapshot():
    h = Hub()
    for i in range(5):
        ln = h.add_line(f"en {i}", i, i + 1, [i])
        h.set_translation(ln["id"], f"uk {i}")
    msgs = h.sync_messages(h.session, 3)
    replay = [m for m in msgs if m["type"] == "replay"][0]
    assert [e["ev"] for e in replay["events"]] == list(range(4, 10))
    msgs = h.sync_messages("other-session", 3)
    snap = [m for m in msgs if m["type"] == "snapshot"][0]
    assert [l["uk"] for l in snap["lines"]] == [f"uk {i}" for i in range(5)]
    msgs = h.sync_messages(h.session, 9)
    assert [m for m in msgs if m["type"] == "replay"][0]["events"] == []
    msgs = h.sync_messages(h.session, 50)  # client from the future -> snapshot
    assert any(m["type"] == "snapshot" for m in msgs)
