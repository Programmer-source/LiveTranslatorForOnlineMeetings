"""Translation EN → UK with context, glossary, validation, fallbacks and endless retry.

A line is NEVER skipped: if every provider fails we back off and try again, forever."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger("translate")


class ProviderError(Exception):
    def __init__(self, msg: str, kind: str = "other", retry_after: float = 0.0):
        super().__init__(msg)
        self.kind = kind  # rate | quota | auth | model | invalid | other
        self.retry_after = retry_after


# ----------------------------------------------------------------------------- validation

_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT_WORD = re.compile(r"[A-Za-z]{2,}")


def clean_output(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", t).strip()
    t = re.sub(r"^(?:UK|UA|Ukrainian|Українською|Переклад)\s*[:：]\s*", "", t, flags=re.I).strip()
    if len(t) >= 2 and t[0] in "\"«“" and t[-1] in "\"»”" and t.count(t[0]) <= 2:
        t = t[1:-1].strip()
    return t


def validate(en: str, uk: str, keep_terms: list[str] | None = None) -> Optional[str]:
    """Return a reason string if the translation must be rejected, else None."""
    if not uk or not uk.strip():
        return "порожній переклад"
    en_words = _LAT_WORD.findall(en)
    if len(en.strip()) >= 25 and len(uk.strip()) < 0.35 * len(en.strip()):
        return "переклад підозріло короткий"
    if len(en_words) >= 3 and not _CYR.search(uk):
        rest = en
        for t in keep_terms or []:
            rest = re.sub(re.escape(t), " ", rest, flags=re.I)
        if len(_LAT_WORD.findall(rest)) >= 3:
            return "переклад не українською"
    if len(uk) > 4 * len(en) + 80:
        return "переклад підозріло довгий"
    return None


# ----------------------------------------------------------------------------- providers

@dataclass
class Ctx:
    pairs: list[tuple[str, str]]  # previous (en, uk)
    glossary: dict[str, str]
    keep_terms: list[str]
    glossary_prompt: str


SYSTEM_PROMPT = """You are a professional conference interpreter producing live Ukrainian subtitles for an English-language lecture.

Rules:
1. Translate EVERYTHING in the new segment: every clause, number, example, aside and joke. Never summarize, shorten, skip, merge away or add content.
2. Write natural, fluent, grammatical modern Ukrainian (never Russian, never surzhyk).
3. Keep numbers, dates, amounts, names and titles exact. Use the glossary strictly.
4. The segment is a fragment of continuous speech: it may begin or end mid-sentence. Translate it so it reads naturally right after the previous Ukrainian lines, keep pronouns/terms consistent with them, but do NOT translate or repeat the previous lines again.
5. Speech-recognition text may contain small errors; translate the most plausible meaning, but do not drop words.
6. Output ONLY the Ukrainian translation — no quotes, notes, labels or explanations."""


def build_user_prompt(items: list[str], ctx: Ctx) -> str:
    parts = []
    if ctx.glossary_prompt:
        parts.append("Glossary (English → Ukrainian):\n" + ctx.glossary_prompt)
    if ctx.pairs:
        prev = "\n".join(f"EN: {en}\nUK: {uk}" for en, uk in ctx.pairs)
        parts.append("Previous lines (context only, already translated — do not output them):\n" + prev)
    if len(items) == 1:
        parts.append("NEW SEGMENT TO TRANSLATE:\n" + items[0])
    else:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))
        parts.append(
            f"NEW SEGMENTS TO TRANSLATE ({len(items)} consecutive fragments of the same speech). "
            f"Return a JSON array of exactly {len(items)} strings: the full Ukrainian translation of each "
            f"fragment, in the same order.\n{numbered}")
    return "\n\n".join(parts)


class GeminiProvider:
    label = "Gemini"

    def __init__(self, api_key: str, models: list[str], rpm: int, client: httpx.AsyncClient):
        self.key = api_key
        self.models = [m.strip() for m in models if m and m.strip()]
        self.min_interval = 60.0 / max(1, rpm)
        self.client = client
        self.cooldown: dict[str, float] = {}
        self.disabled: set[str] = set()
        self.no_thinking_cfg: set[str] = set()
        self.last_call = 0.0
        self.last_model = ""

    def available_in(self) -> float:
        """Seconds until some model can be used (inf if none)."""
        now = time.time()
        waits = [max(0.0, self.cooldown.get(m, 0) - now) for m in self.models if m not in self.disabled]
        if not waits:
            return float("inf")
        return max(min(waits), self.last_call + self.min_interval - now)

    async def translate(self, items: list[str], ctx: Ctx) -> list[str]:
        if not self.key:
            raise ProviderError("немає ключа Gemini", "auth")
        last_err: Optional[ProviderError] = None
        for model in self.models:
            if model in self.disabled or self.cooldown.get(model, 0) > time.time():
                continue
            wait = self.last_call + self.min_interval - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call = time.time()
            try:
                out = await self._call(model, items, ctx)
                self.last_model = model
                return out
            except ProviderError as e:
                last_err = e
                if e.kind in ("rate", "quota"):
                    self.cooldown[model] = time.time() + max(e.retry_after, 5.0)
                elif e.kind == "model":
                    self.disabled.add(model)
                elif e.kind == "auth":
                    raise
                continue
        raise last_err or ProviderError("усі моделі Gemini тимчасово недоступні (ліміт)", "rate",
                                        max(1.0, self.available_in()))

    async def _call(self, model: str, items: list[str], ctx: Ctx) -> list[str]:
        gen: dict = {"maxOutputTokens": 8192}
        if model not in self.no_thinking_cfg:
            gen["thinkingConfig"] = {"thinkingLevel": "low"}
        if len(items) > 1:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = {"type": "ARRAY", "items": {"type": "STRING"}}
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": build_user_prompt(items, ctx)}]}],
            "generationConfig": gen,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = await self.client.post(url, json=body, headers={"x-goog-api-key": self.key}, timeout=45)
        except httpx.HTTPError as e:
            raise ProviderError(f"мережа: {e.__class__.__name__}", "other", 2)
        if r.status_code == 400 and "thinking" in r.text.lower() and model not in self.no_thinking_cfg:
            self.no_thinking_cfg.add(model)
            return await self._call(model, items, ctx)
        if r.status_code != 200:
            raise _gemini_error(r, model)
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            fb = data.get("promptFeedback", {})
            raise ProviderError(f"Gemini не повернув відповідь ({fb.get('blockReason', 'невідомо')})", "invalid")
        parts = cands[0].get("content", {}).get("parts", []) or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if cands[0].get("finishReason") not in (None, "STOP"):
            raise ProviderError(f"Gemini: finishReason={cands[0].get('finishReason')}", "invalid")
        if len(items) == 1:
            return [clean_output(text)]
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and len(arr) == len(items):
                return [clean_output(str(a)) for a in arr]
        except Exception:
            pass
        raise ProviderError("Gemini: неправильний формат пакетної відповіді", "invalid")


def _gemini_error(r: httpx.Response, model: str) -> ProviderError:
    try:
        err = r.json().get("error", {})
    except Exception:
        err = {}
    msg = err.get("message", r.text[:200])
    if r.status_code == 429:
        retry = 30.0
        per_day = False
        for d in err.get("details", []) or []:
            if "retryDelay" in d:
                try:
                    retry = float(str(d["retryDelay"]).rstrip("s"))
                except ValueError:
                    pass
            for v in d.get("violations", []) or []:
                if "PerDay" in str(v.get("quotaId", "")):
                    per_day = True
        if per_day:
            return ProviderError(f"{model}: денний ліміт вичерпано", "quota", 3600)
        return ProviderError(f"{model}: ліміт запитів (429)", "rate", retry)
    if r.status_code in (401, 403) or "API key" in msg:
        return ProviderError("невірний ключ Gemini або доступ заборонено", "auth")
    if r.status_code == 404:
        return ProviderError(f"модель {model} не знайдена", "model")
    if r.status_code >= 500:
        return ProviderError(f"Gemini {r.status_code}: тимчасова помилка сервера", "other", 3)
    return ProviderError(f"Gemini {r.status_code}: {msg[:150]}", "other", 3)


def _protect_glossary(text: str, glossary: dict[str, str], keep_english) -> str:
    """Wrap glossary terms in <keep> tags (DeepL leaves them untouched)."""
    for term in sorted(glossary, key=len, reverse=True):
        uk = glossary[term]
        repl = term if keep_english(term, uk) else uk
        repl = repl.replace("&", "&amp;").replace("<", "&lt;")
        text = re.sub(r"(?<![\w>])" + re.escape(term) + r"(?![\w<])", f"<keep>{repl}</keep>", text, flags=re.I)
    return text


class DeepLProvider:
    label = "DeepL"

    def __init__(self, api_key: str, client: httpx.AsyncClient, keep_english):
        self.key = api_key
        self.client = client
        self.keep_english = keep_english
        self.cooldown_until = 0.0

    def available_in(self) -> float:
        return max(0.0, self.cooldown_until - time.time()) if self.key else float("inf")

    async def translate(self, items: list[str], ctx: Ctx) -> list[str]:
        if not self.key:
            raise ProviderError("немає ключа DeepL", "auth")
        host = "api-free.deepl.com" if self.key.endswith(":fx") else "api.deepl.com"
        esc = [t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for t in items]
        texts = [_protect_glossary(t, ctx.glossary, self.keep_english) for t in esc]
        body = {"text": texts, "source_lang": "EN", "target_lang": "UK", "tag_handling": "xml",
                "ignore_tags": ["keep"], "preserve_formatting": True}
        if ctx.pairs:
            body["context"] = " ".join(en for en, _ in ctx.pairs)[-2000:]
        try:
            r = await self.client.post(f"https://{host}/v2/translate", json=body, timeout=30,
                                       headers={"Authorization": "DeepL-Auth-Key " + self.key})
        except httpx.HTTPError as e:
            raise ProviderError(f"DeepL мережа: {e.__class__.__name__}", "other", 2)
        if r.status_code == 429:
            self.cooldown_until = time.time() + 10
            raise ProviderError("DeepL: ліміт запитів", "rate", 10)
        if r.status_code == 456:
            self.cooldown_until = time.time() + 3600
            raise ProviderError("DeepL: місячний ліміт вичерпано", "quota", 3600)
        if r.status_code in (401, 403):
            raise ProviderError("DeepL: невірний ключ", "auth")
        if r.status_code != 200:
            raise ProviderError(f"DeepL {r.status_code}", "other", 3)
        outs = [t["text"] for t in r.json()["translations"]]
        return [clean_output(re.sub(r"</?keep>", "", o).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
                for o in outs]


class FreeGoogleProvider:
    """Keyless public Google Translate web endpoint. Last resort only — no context support."""

    label = "Google (резерв)"

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.cooldown_until = 0.0

    def available_in(self) -> float:
        return max(0.0, self.cooldown_until - time.time())

    async def translate(self, items: list[str], ctx: Ctx) -> list[str]:
        out = []
        for t in items:
            try:
                r = await self.client.post(
                    "https://translate.googleapis.com/translate_a/single",
                    params={"client": "gtx", "sl": "en", "tl": "uk", "dt": "t"}, data={"q": t}, timeout=20)
            except httpx.HTTPError as e:
                raise ProviderError(f"Google мережа: {e.__class__.__name__}", "other", 2)
            if r.status_code == 429:
                self.cooldown_until = time.time() + 30
                raise ProviderError("Google: ліміт", "rate", 30)
            if r.status_code != 200:
                raise ProviderError(f"Google {r.status_code}", "other", 3)
            data = r.json()
            out.append(clean_output("".join(seg[0] for seg in (data[0] or []) if seg and seg[0])))
        return out


# ----------------------------------------------------------------------------- manager

class TranslatorManager:
    """Tries providers in priority order; never gives up on a line."""

    def __init__(self, providers: list, max_backoff: float = 30.0):
        self.providers = providers
        self.max_backoff = max_backoff
        self.last_error = ""
        self.last_provider = ""
        self.failures_in_row = 0
        self.state = "idle"  # idle | ok | retrying | error

    async def translate(self, items: list[str], ctx: Ctx, on_retry=None) -> list[str]:
        attempt = 0
        while True:
            errors = []
            tried = False
            for i, p in enumerate(self.providers):
                wait = p.available_in()
                if wait == float("inf"):
                    continue
                # Prefer a better provider if it frees up soon rather than dropping to a worse one.
                if wait > 0:
                    lower_ready = any(q.available_in() == 0 for q in self.providers[i + 1:])
                    if wait > 8 and lower_ready:
                        errors.append(f"{p.label}: зачекати {wait:.0f} с")
                        continue
                    await asyncio.sleep(min(wait, 8))
                tried = True
                try:
                    res = await p.translate(items, ctx)
                except ProviderError as e:
                    errors.append(f"{p.label}: {e}")
                    continue
                except Exception as e:  # unexpected (bad JSON etc.)
                    log.exception("provider crashed")
                    errors.append(f"{p.label}: {e}")
                    continue
                bad = None
                for en, uk in zip(items, res):
                    bad = validate(en, uk, ctx.keep_terms)
                    if bad:
                        break
                if len(res) != len(items):
                    bad = "кількість рядків не збігається"
                if bad:
                    errors.append(f"{p.label}: {bad}")
                    continue
                self.last_provider = getattr(p, "last_model", "") or p.label
                self.failures_in_row = 0
                self.state = "ok"
                if attempt:
                    self.last_error = ""
                return res
            attempt += 1
            self.failures_in_row += 1
            self.state = "retrying"
            self.last_error = "; ".join(errors) or "немає доступного перекладача (додайте ключ Gemini)"
            if not tried and not any(p.available_in() < float("inf") for p in self.providers):
                self.state = "error"
            log.warning("translation attempt %d failed: %s", attempt, self.last_error)
            if on_retry:
                on_retry(self.last_error, attempt)
            await asyncio.sleep(min(self.max_backoff, 0.5 * (2 ** min(attempt, 8))))
