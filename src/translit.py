"""Indic-script handling.

Some Source 2/3 names and addresses are written in Devanagari, Bengali, Telugu,
Kannada, Tamil, Gujarati, Malayalam, Gurmukhi or Oriya script while Source 1 uses
Latin script.  Two complementary mechanisms, both trained only on the provided
training data (no external resources):

1. A learned token dictionary  native_token -> latin_token, mined from matched
   training pairs (positional alignment when token counts agree, co-occurrence
   voting otherwise).
2. A rule-based fallback romaniser (unidecode + cleanup) for unseen tokens; the
   downstream phonetic skeleton key makes the fallback robust
   (e.g. 'kRssnn' -> 'krsn' == skeleton('krishna')).
"""
import re
from collections import Counter, defaultdict

import polars as pl
from unidecode import unidecode

INDIC_RE = re.compile(r"[ऀ-෿]")
INDIC_PAT = r"[ऀ-෿]"
_TOK_SPLIT = re.compile(r"[\s,.;:()\[\]{}|/\\\-]+")
_ZW = re.compile(r"[​-‍﻿]")


def has_indic(s):
    return bool(INDIC_RE.search(s))


def _toks(s):
    return [t for t in _TOK_SPLIT.split(_ZW.sub("", s)) if t]


def _latin_toks(s):
    s = unidecode(s).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def romanize_token(t):
    """Fallback romanisation of a single native-script token."""
    r = unidecode(t).lower()
    r = re.sub(r"[^a-z0-9]", "", r)
    r = re.sub(r"(.)\1+", r"\1", r)   # tt->t, aa->a, nn->n
    return r


class Transliterator:
    def __init__(self, table=None):
        self.table = table or {}

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, pairs_native_latin, min_count=2, min_share=0.5):
        """pairs_native_latin: iterable of (native_text, latin_text) for matched pairs."""
        pos = defaultdict(Counter)
        for nat, lat in pairs_native_latin:
            nt = [t for t in _toks(nat) if has_indic(t)]
            if not nt:
                continue
            all_nt = _toks(nat)
            lt = _latin_toks(lat)
            if len(all_nt) == len(lt):
                for a, b in zip(all_nt, lt):
                    if has_indic(a):
                        pos[a][b] += 1.0
            else:
                w = 1.0 / max(len(lt), 1)
                for a in nt:
                    for b in lt:
                        pos[a][b] += w
        table = {}
        for a, c in pos.items():
            b, n = c.most_common(1)[0]
            tot = sum(c.values())
            if n >= min_count and n / tot >= min_share:
                table[a] = b
        return cls(table)

    # ---------------------------------------------------------------- apply
    def convert(self, s):
        if not s or not INDIC_RE.search(s):
            return s
        s = _ZW.sub("", s)
        comps = []
        for comp in s.split(","):
            c = comp.strip()
            if c and INDIC_RE.search(c):
                t = self.table.get(c)
                if t is None:
                    out = []
                    for piece in re.split(r"(\s+|[;()\[\]])", comp):
                        if piece and INDIC_RE.search(piece):
                            out.append(self.table.get(piece) or " ".join(
                                self.table.get(x) or romanize_token(x) for x in _toks(piece)))
                        else:
                            out.append(piece)
                    t = "".join(out)
                comps.append(" " + t)
            else:
                comps.append(comp)
        return ",".join(comps)

    def convert_series(self, col: pl.Expr) -> pl.Expr:
        return pl.when(col.str.contains(INDIC_PAT)).then(
            col.map_elements(self.convert, return_dtype=pl.String)).otherwise(col)


def fit_from_training(recs, gt, s1_ids_allowed=None):
    """recs: all normalised-raw records (entity_id, business_name, business_address);
    gt: frame (s1_id, tgt_id). Uses names and comma-separated address components."""
    g = gt if s1_ids_allowed is None else gt.filter(pl.col("s1_id").is_in(s1_ids_allowed))
    r = recs.select("entity_id", "business_name", "business_address")
    j = (g.join(r, left_on="tgt_id", right_on="entity_id")
          .join(r, left_on="s1_id", right_on="entity_id", suffix="_s1"))
    jn = j.filter(pl.col("business_name").str.contains(INDIC_PAT))
    name_pairs = list(zip(jn["business_name"].to_list(), jn["business_name_s1"].to_list()))
    # address: native components (state / city names) vs every S1 component
    ja = j.filter(pl.col("business_address").str.contains(INDIC_PAT)).head(400_000)
    comp, occ = defaultdict(Counter), Counter()
    for a, b in zip(ja["business_address"].to_list(), ja["business_address_s1"].to_list()):
        bs = {unidecode(x.strip()).lower() for x in b.split(",") if x.strip()}
        for c in a.split(","):
            c = _ZW.sub("", c.strip())
            if c and has_indic(c):
                occ[c] += 1
                for x in bs:
                    comp[c][x] += 1
    tr = Transliterator.fit(name_pairs)
    # component-level (multi-word, e.g. state names) table for addresses
    for a, c in comp.items():
        b, n = c.most_common(1)[0]
        if n >= 3 and n / occ[a] >= 0.5 and a not in tr.table:
            tr.table[a] = b
    return tr
