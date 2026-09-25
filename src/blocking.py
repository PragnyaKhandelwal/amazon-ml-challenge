"""Candidate generation (blocking) by weighted rare-key joins.

Every record emits a set of hashed *keys*:

  n  : core-name token                 (plus DBA / website alternative names)
  k  : phonetic skeleton of a token    (typos, Indic transliterations)
  p  : unordered pair of name tokens   (robust to shuffling / dropped tokens)
  na : name token x rare address word  (common names disambiguated by place)
  ha : house number x rare address word (rebrands / unreadable names)
  aa : pair of rare address words
  x  : squashed name ("soclaronflex") / domain names

Keys are joined *within the same country label* (an open set of strings, so an
unseen country such as France is handled identically).  Keys shared by more than
`max_df` Source-1 records are ignored.  A (target, S1) pair's blocking score is
the sum of IDF weights of its shared keys; every Source-2/3 record keeps its
top-K Source-1 records (each target matches at most one S1 entity, so the target
side is the natural direction).
"""
import gc
import time

import numpy as np
import polars as pl

ADDR_GENERIC = {"st", "rd", "av", "bd", "ln", "dr", "ct", "pl", "plz", "sq", "ter", "cir", "hwy",
                "pkwy", "expy", "fwy", "trl", "rte", "way", "ste", "apt", "unit", "fl", "bldg",
                "rm", "n", "s", "e", "w", "ne", "nw", "se", "sw", "nr", "opp", "bhd", "sec",
                "blk", "col", "ngr", "extn", "soc", "dist", "vill", "indl", "est", "mkt", "main",
                "crs", "gr", "marg", "rue", "all", "imp", "ch", "cnty", "floor", "road"}


def _tok_list(expr):
    return expr.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique()


KINDS = ["n", "k", "p", "na", "ha", "aa", "x"]


def addr_word_df(recs):
    """Document frequency of address words per country (unsupervised statistics)."""
    parts = []
    for st in range(0, recs.height, 1_000_000):
        parts.append(recs.slice(st, 1_000_000).select("cc", w=pl.col("adw").str.split(" ").list.unique())
                         .explode("w").drop_nulls("w")
                         .filter((pl.col("w").str.len_chars() >= 3) & ~pl.col("w").is_in(list(ADDR_GENERIC)))
                         .group_by("cc", "w").agg(df=pl.len()))
    return pl.concat(parts).group_by("cc", "w").agg(pl.col("df").sum())


def build_keys(recs, aw_df, n_addr=4):
    """recs: normalised frame (rid, cc, core, alt, sk, adw, hn, nsp).
    Returns frame (rid, cc, key:u64, kind:i8)."""
    names = recs.select("rid", pl.concat_str([pl.col("core"), pl.col("alt").str.replace_all(r"\|", " ")],
                                             separator=" ").alias("t"))
    nt = names.with_columns(t=_tok_list(pl.col("t"))).explode("t").drop_nulls("t")
    sk = (recs.select("rid", t=_tok_list(pl.col("sk"))).explode("t").drop_nulls("t")
              .filter(pl.col("t").str.len_chars() >= 3))
    aw = (recs.select("rid", "cc", w=pl.col("adw").str.split(" ").list.unique())
              .explode("w").drop_nulls("w")
              .join(aw_df, on=["cc", "w"])
              .sort(["rid", "df"]).group_by("rid", maintain_order=True).head(n_addr)
              .select("rid", "w"))
    np_ = (nt.join(nt, on="rid", suffix="2").filter(pl.col("t") < pl.col("t2"))
             .select("rid", k=pl.concat_str(["t", "t2"], separator="|")))
    na = nt.join(aw, on="rid").select("rid", k=pl.concat_str(["t", "w"], separator="@"))
    ha = (recs.filter(pl.col("hn") != "").select("rid", "hn").join(aw, on="rid")
              .select("rid", k=pl.concat_str(["hn", "w"], separator="#")))
    aa = (aw.join(aw, on="rid", suffix="2").filter(pl.col("w") < pl.col("w2"))
            .select("rid", k=pl.concat_str(["w", "w2"], separator="&")))
    xx = recs.filter(pl.col("nsp").str.len_chars() >= 4).select("rid", k=pl.col("nsp"))
    xa = (recs.filter(pl.col("alt") != "")
              .select("rid", k=pl.col("alt").str.split(" | ")).explode("k")
              .with_columns(k=pl.col("k").str.replace_all(" ", ""))
              .filter(pl.col("k").str.len_chars() >= 4))
    parts = [nt.select("rid", k=pl.col("t")), sk.select("rid", k=pl.col("t")), np_, na, ha, aa,
             pl.concat([xx, xa])]
    keys = pl.concat([p.select("rid", key=(pl.lit(KINDS[i] + ":") + pl.col("k")).hash(seed=11),
                               kind=pl.lit(i, pl.Int8)) for i, p in enumerate(parts)])
    return keys.unique(["rid", "key"]).join(recs.select("rid", "cc"), on="rid")


def generate_candidates(recs, top_k=10, max_df=300, chunk=250_000, log=print):
    """recs: all normalised records of one split (needs cc = country code).
    Returns frame (t_rid, s1_rid, bscore, brank, nkeys, k_<kind>...)."""
    t0 = time.time()
    cols = ["rid", "cc", "core", "alt", "sk", "adw", "hn", "nsp"]
    aw_df = addr_word_df(recs)
    r1 = recs.filter(pl.col("src") == 1).select(cols)
    k1 = pl.concat([build_keys(r1.slice(st, chunk * 2), aw_df) for st in range(0, r1.height, chunk * 2)])
    n1 = r1.group_by("cc").agg(n=pl.len())
    df1 = (k1.group_by("cc", "key").agg(df=pl.len()).filter(pl.col("df") <= max_df)
             .join(n1, on="cc")
             .with_columns(w=(1.0 + pl.col("n") / pl.col("df")).log().cast(pl.Float32))
             .select("cc", "key", "w"))
    # kind counts packed in one int (4 bits per kind) to keep the aggregation cheap
    k1 = (k1.join(df1, on=["cc", "key"])
            .select("cc", "key", "rid", "w", kbit=pl.lit(16, pl.Int64).pow(pl.col("kind").cast(pl.Int64))))
    del df1, r1
    gc.collect()
    log(f"  S1 keys: {k1.height:,} ({time.time() - t0:.0f}s)")
    rt = recs.filter(pl.col("src") != 1).select(cols)
    out = []
    for st in range(0, rt.height, chunk):
        kt = build_keys(rt.slice(st, chunk), aw_df).select("cc", "key", pl.col("rid").alias("t_rid"))
        j = kt.join(k1, on=["cc", "key"])
        nj = j.height
        agg = (j.group_by("t_rid", "rid")
                .agg(pl.col("w").sum().alias("bscore"), pl.len().cast(pl.Int16).alias("nkeys"),
                     pl.col("kbit").sum().alias("kbits")))
        del j, kt
        agg = (agg.sort(["t_rid", "bscore"], descending=[False, True])
                  .with_columns(brank=pl.int_range(pl.len()).over("t_rid").cast(pl.Int16))
                  .filter(pl.col("brank") < top_k))
        out.append(agg.rename({"rid": "s1_rid"}))
        gc.collect()
        if True:
            log(f"  targets {st:,}+: joined {nj:,} -> {agg.height:,} pairs ({time.time() - t0:.0f}s)")
    res = pl.concat(out)
    return res.with_columns([((pl.col("kbits") // (16 ** i)) % 16).cast(pl.Int8).alias(f"k_{k}")
                             for i, k in enumerate(KINDS)]).drop("kbits")
