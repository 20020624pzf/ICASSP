#!/usr/bin/env python3
"""Restored WER rules (masterplan 2.3) and corpus-level scoring.

Differences from the archived `cpu/fixed_wer.py` that the audit flagged:

  * Turkish is normalised by mapping I -> dotless i and dotted capital I -> i BEFORE
    lower-casing, so the four-way Turkish I behaviour is correct rather than a plain
    casefold;
  * apostrophes are unified to the ASCII apostrophe and kept when they sit between two
    letters (Italian elisions), instead of every apostrophe becoming a separator;
  * word-internal ZWNJ (U+200C) is preserved as a joiner rather than converted to a space;
  * Unicode decimal digits are unified to ASCII digits;
  * composition marks and accented letters are preserved (Polish, Italian, Turkish).

The rules are fixed here and are not selected by which language scores better. Existing
archived predictions are re-scored with these rules as errata; the original scores are
kept.
"""
from __future__ import annotations

import re
import unicodedata

# Persian-specific orthographic mapping. Never applied to Arabic.
PERSIAN_MAP = str.maketrans({"\u064a": "\u06cc",   # ARABIC YEH        -> FARSI YEH
                             "\u0649": "\u06cc",   # ALEF MAKSURA      -> FARSI YEH
                             "\u0643": "\u06a9"})  # ARABIC KAF        -> KEHEH

APOSTROPHES = "\u2018\u2019\u02bc\u02b9\u2032\u0060\u00b4"
ZWNJ = "\u200c"
TATWEEL = "\u0640"

_CONTROL = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
_ARABIC_INDIC = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_EXTENDED_ARABIC_INDIC = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}


def normalize(text: str, lang: str) -> str:
    s = unicodedata.normalize("NFC", "" if text is None else str(text))
    s = _CONTROL.sub(" ", s)
    # remove decorative tatweel elongation (not semantic; does not change word identity)
    s = s.replace(TATWEEL, "")
    # digits: unify decimal digits, never expand into number words
    s = "".join(str(unicodedata.decimal(ch)) if unicodedata.category(ch) == "Nd" else ch for ch in s)
    if lang == "fa":
        s = s.translate(PERSIAN_MAP)
    # apostrophe handling: unify, keep between two letters, otherwise a separator
    s = "".join("\u0027" if ch in APOSTROPHES else ch for ch in s)
    s = "".join(ch if ch != "'" or (i > 0 and i+1 < len(s) and s[i-1].isalpha() and s[i+1].isalpha()) else " " for i,ch in enumerate(s))
    if lang == "tr":
        # Turkish casing is not the Unicode default: map before lower-casing
        s = s.replace("I", "\u0131").replace("\u0130", "i")
        s = s.lower()
    else:
        s = s.lower()
    # punctuation becomes a fixed separator; ZWNJ is preserved as a joiner
    out = []
    for ch in s:
        if ch == ZWNJ or ch == "\u0027":
            out.append(ch)
        elif unicodedata.category(ch).startswith("P") or unicodedata.category(ch) == "S":
            out.append(" ")
        else:
            out.append(ch)
    s = "".join(out)
    # a hyphen between letters was already converted by the punctuation rule above, so the
    # hyphen rule is fixed and does not depend on the reference/prediction pairing
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


def words(text: str, lang: str) -> list[str]:
    return normalize(text, lang).split()


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        d[0][j] = j
        bt[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                bt[i][j] = "C"
            else:
                opts = [(d[i - 1][j - 1] + 1, "S"), (d[i - 1][j] + 1, "D"), (d[i][j - 1] + 1, "I")]
                d[i][j], bt[i][j] = min(opts, key=lambda x: (x[0], {"S": 0, "D": 1, "I": 2}[x[1]]))
    s = dd = ins = 0
    i, j = n, m
    while i or j:
        op = bt[i][j]
        if op == "S":
            s += 1; i -= 1; j -= 1
        elif op == "D":
            dd += 1; i -= 1
        elif op == "I":
            ins += 1; j -= 1
        else:
            i -= 1; j -= 1
    return s, dd, ins


def score(reference: str, hypothesis: str, lang: str) -> dict:
    r = words(reference, lang)
    h = words(hypothesis, lang)
    s, dd, ins = edit_counts(r, h)
    n = len(r)
    return {"S": s, "D": dd, "I": ins, "N": n,
            "wer": (None if n == 0 else 100.0 * (s + dd + ins) / n)}


def corpus(pairs: list[tuple[str, str]], lang: str) -> dict:
    """Corpus-level WER = 100 * (sum S + sum D + sum I) / sum N."""
    S = D = I = N = 0
    undefined = 0
    for ref, hyp in pairs:
        rec = score(ref, hyp, lang)
        S += rec["S"]; D += rec["D"]; I += rec["I"]; N += rec["N"]
        if rec["N"] == 0:
            undefined += 1
    return {"S": S, "D": D, "I": I, "N": N,
            "wer": (None if N == 0 else 100.0 * (S + D + I) / N),
            "records": len(pairs), "undefined_records": undefined}


def macro_wers(per_language: dict) -> float | None:
    vals = [v["wer"] for v in per_language.values() if v["wer"] is not None]
    return None if not vals else sum(vals) / len(vals)
