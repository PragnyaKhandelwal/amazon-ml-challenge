"""Pairwise + context features for (target, Source-1) candidate pairs.

All features are country-agnostic similarities (no country one-hot), so the model
transfers to country labels never seen in training (France in test).
"""
import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

REC_COLS = ["rid", "cc", "nm", "core", "alt", "legal", "is_dom", "sk", "nsp", "ntok", "first",
            "acro", "is_indic", "ad", "adw", "nums", "postal", "hn", "business_name",
            "business_address", "src", "f_nsp1", "f_nspa", "f_adr1", "f_adra"]


def add_frequencies(recs):
    """How many records (S1 / all sources) in the same country share the exact squashed
    name or the exact normalised address: a unique name is far stronger evidence."""
    adr = pl.col("ad").str.split(" ").list.sort().list.join(" ")
    recs = recs.with_columns(_adr=adr, _s1=(pl.col("src") == 1).cast(pl.Int32))
    out = recs.with_columns(
        f_nsp1=pl.col("_s1").sum().over("cc", "nsp").cast(pl.Int32),
        f_nspa=pl.len().over("cc", "nsp").cast(pl.Int32),
        f_adr1=pl.col("_s1").sum().over("cc", "_adr").cast(pl.Int32),
        f_adra=pl.len().over("cc", "_adr").cast(pl.Int32))
    empty_n, empty_a = pl.col("nsp") == "", pl.col("ad") == ""
    return out.with_columns(
        [pl.when(empty_n).then(-1).otherwise(pl.col(c)).alias(c) for c in ("f_nsp1", "f_nspa")] +
        [pl.when(empty_a).then(-1).otherwise(pl.col(c)).alias(c) for c in ("f_adr1", "f_adra")]
    ).drop("_adr", "_s1")


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _tri(a, b):
    """1 both present & equal, -1 both present & different, 0 one side missing."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    both = (a != "") & (b != "")
    return np.where(both, np.where(a == b, 1, -1), 0).astype(np.int8)


def token_idf(recs, col, name):
    """IDF per (country, token) over all records of the split (unsupervised)."""
    n = recs.group_by("cc").agg(n=pl.len())
    return (recs.select("cc", t=pl.col(col).str.split(" ").list.unique()).explode("t")
                .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
                .group_by("cc", "t").agg(df=pl.len()).join(n, on="cc")
                .select("cc", "t", (pl.col("n") / pl.col("df")).log().cast(pl.Float32).alias(name)))


def idf_overlap(pairs, recs, col, idf, prefix):
    """IDF-weighted token overlap between the two sides of every pair (vectorised)."""
    p = pairs.select("pid", "t_rid", "s1_rid")
    tok = recs.select("rid", "cc", t=pl.col(col).str.split(" ").list.unique())
    w = idf.columns[-1]

    def side(rid_col):
        return (p.select("pid", rid=pl.col(rid_col)).join(tok, on="rid").explode("t")
                 .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
                 .join(idf, on=["cc", "t"], how="left").with_columns(pl.col(w).fill_null(0.0))
                 .select("pid", "t", w))

    a, b = side("t_rid"), side("s1_rid")
    wa = a.group_by("pid").agg(wa=pl.col(w).sum(), wmax_a=pl.col(w).max())
    wb = b.group_by("pid").agg(wb=pl.col(w).sum(), wmax_b=pl.col(w).max())
    sh = a.join(b.select("pid", "t"), on=["pid", "t"]).group_by("pid").agg(
        ws=pl.col(w).sum(), ns=pl.len(), wsmax=pl.col(w).max())
    r = (p.select("pid").join(wa, on="pid", how="left").join(wb, on="pid", how="left")
          .join(sh, on="pid", how="left").fill_null(0).sort("pid"))
    ws, wa_, wb_ = r["ws"].to_numpy(), r["wa"].to_numpy(), r["wb"].to_numpy()
    eps = 1e-6
    return {
        f"{prefix}_idf_jac": ws / np.maximum(wa_ + wb_ - ws, eps),
        f"{prefix}_idf_min": ws / np.maximum(np.minimum(wa_, wb_), eps),
        f"{prefix}_idf_t": ws / np.maximum(wa_, eps),
        f"{prefix}_idf_s": ws / np.maximum(wb_, eps),
        f"{prefix}_idf_shared": ws,
        f"{prefix}_idf_unsh_t": wa_ - ws,
        f"{prefix}_idf_unsh_s": wb_ - ws,
        f"{prefix}_idf_maxtok_shared": r["wsmax"].to_numpy(),
        f"{prefix}_idf_maxtok_miss": np.maximum(r["wmax_a"].to_numpy(), r["wmax_b"].to_numpy()) - r["wsmax"].to_numpy(),
        f"{prefix}_nshared": r["ns"].to_numpy().astype(np.int16),
    }


def _num_feats(na, nb, ha, hb):
    """Numeric-token agreement (house numbers are the main precision signal)."""
    n = len(na)
    jac = np.zeros(n, np.float32)
    inter = np.zeros(n, np.int8)
    conf = np.zeros(n, np.int8)
    hpre = np.zeros(n, np.int8)
    for k in range(n):
        x, y = na[k], nb[k]
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            i = len(sx & sy)
            inter[k] = i
            jac[k] = i / len(sx | sy)
            conf[k] = len(sx ^ sy)
        a, b = ha[k], hb[k]
        if a and b and a != b and (a.startswith(b) or b.startswith(a)):
            hpre[k] = 1
    return jac, inter, conf, hpre


def _domain_cover(dom_flag, nsp_t, core_s, nsp_s, core_t):
    """For domain-style names ('healthatlanticwomens.com'): share of the other side's
    tokens found as substrings of the squashed domain."""
    out = np.zeros(len(dom_flag), np.float32)
    for k in np.nonzero(dom_flag)[0]:
        for d, c in ((nsp_t[k], core_s[k]), (nsp_s[k], core_t[k])):
            toks = [t for t in c.split() if len(t) >= 2]
            if d and toks:
                out[k] = max(out[k], sum(len(t) for t in toks if t in d) / max(len(d), 1))
    return out


def _sorted_tokens(col):
    return pl.col(col).str.split(" ").list.sort().list.join(" ")


def pair_features(pairs, recs, idfs):
    """pairs: frame with t_rid, s1_rid + blocking columns. Returns feature frame (same order)."""
    pairs = pairs.with_row_index("pid")
    need = pl.concat([pairs["t_rid"], pairs["s1_rid"]]).unique()
    R = (recs.filter(pl.col("rid").is_in(need.implode())).select(REC_COLS)
             .with_columns(core_srt=_sorted_tokens("core"), ad_srt=_sorted_tokens("ad"),
                           adw_srt=_sorted_tokens("adw"),
                           name_low=pl.col("business_name").str.to_lowercase(),
                           addr_low=pl.col("business_address").str.to_lowercase()))
    T = pairs.select("pid", rid="t_rid").join(R, on="rid", how="left").sort("pid")
    S = pairs.select("pid", rid="s1_rid").join(R, on="rid", how="left").sort("pid")
    g = lambda D, c: D[c].to_list()  # noqa: E731
    F = {}
    # ------------------------------------------------------------ names
    a, b = g(T, "core"), g(S, "core")
    F["n_ratio"] = _cp(a, b, fuzz.ratio)
    F["n_tset"] = _cp(a, b, fuzz.token_set_ratio)
    F["n_jw"] = _cp(a, b, JaroWinkler.normalized_similarity)
    F["n_lev"] = _cp(a, b, Levenshtein.distance)
    F["n_tsort"] = _cp(g(T, "core_srt"), g(S, "core_srt"), fuzz.ratio)
    F["n_full_ratio"] = _cp(g(T, "nm"), g(S, "nm"), fuzz.ratio)
    F["n_raw_ratio"] = _cp(g(T, "name_low"), g(S, "name_low"), fuzz.ratio)
    F["n_sk_ratio"] = _cp(g(T, "sk"), g(S, "sk"), fuzz.ratio)
    nt, ns = g(T, "nsp"), g(S, "nsp")
    F["n_nsp_ratio"] = _cp(nt, ns, fuzz.ratio)
    F["n_first_jw"] = _cp(g(T, "first"), g(S, "first"), JaroWinkler.normalized_similarity)
    at, as_ = T["alt"].to_numpy(), S["alt"].to_numpy()
    has_alt = (at != "") | (as_ != "")
    alt_x = np.zeros(len(a), np.float32)
    if has_alt.any():
        k = np.nonzero(has_alt)[0]
        aa, bb = np.array(a, object)[k], np.array(b, object)[k]
        alt_t = [x.split(" | ")[0] for x in at[k]]
        alt_s = [x.split(" | ")[0] for x in as_[k]]
        alt_x[k] = np.maximum(_cp(alt_t, list(bb), fuzz.token_set_ratio),
                              _cp(list(aa), alt_s, fuzz.token_set_ratio))
    F["n_alt_x"] = alt_x
    F["n_has_alt"] = ((at != "").astype(np.int8) + (as_ != "").astype(np.int8))
    F["n_best"] = np.maximum(F["n_tset"], F["n_alt_x"])
    F["n_legal_eq"] = _tri(g(T, "legal"), g(S, "legal"))
    acro_t, acro_s = T["acro"].to_numpy(), S["acro"].to_numpy()
    nta, nsa = np.array(nt, object), np.array(ns, object)
    F["n_acro"] = (((acro_t != "") & (acro_t == nsa)) | ((acro_s != "") & (acro_s == nta))).astype(np.int8)
    dom_t, dom_s = T["is_dom"].to_numpy(), S["is_dom"].to_numpy()
    dom = (dom_t + dom_s) > 0
    F["n_dom"] = (dom_t + dom_s).astype(np.int8)
    F["n_dom_cover"] = _domain_cover(dom, nt, b, ns, a)
    nsp_p = F["n_nsp_ratio"].copy()
    if dom.any():
        k = np.nonzero(dom)[0]
        nsp_p[k] = _cp(list(nta[k]), list(nsa[k]), fuzz.partial_ratio)
    F["n_nsp_pratio"] = nsp_p
    F["n_indic"] = T["is_indic"].to_numpy()
    F["n_len_t"], F["n_len_s"] = T["core"].str.len_chars().to_numpy(), S["core"].str.len_chars().to_numpy()
    F["n_ntok_t"], F["n_ntok_s"] = T["ntok"].to_numpy(), S["ntok"].to_numpy()
    # ------------------------------------------------------------ addresses
    a, b = g(T, "ad"), g(S, "ad")
    F["a_ratio"] = _cp(a, b, fuzz.ratio)
    F["a_tset"] = _cp(a, b, fuzz.token_set_ratio)
    F["a_tsort"] = _cp(g(T, "ad_srt"), g(S, "ad_srt"), fuzz.ratio)
    F["a_w_tsort"] = _cp(g(T, "adw_srt"), g(S, "adw_srt"), fuzz.ratio)
    F["a_raw_ratio"] = _cp(g(T, "addr_low"), g(S, "addr_low"), fuzz.ratio)
    F["a_empty"] = ((T["ad"].to_numpy() == "").astype(np.int8) + 2 * (S["ad"].to_numpy() == "").astype(np.int8))
    F["a_len_t"], F["a_len_s"] = T["ad"].str.len_chars().to_numpy(), S["ad"].str.len_chars().to_numpy()
    F["a_postal_eq"] = _tri(g(T, "postal"), g(S, "postal"))
    F["a_hn_eq"] = _tri(g(T, "hn"), g(S, "hn"))
    F["a_hn_ratio"] = _cp(g(T, "hn"), g(S, "hn"), fuzz.ratio)
    (F["a_num_jac"], F["a_num_inter"], F["a_num_conf"], F["a_hn_prefix"]) = _num_feats(
        g(T, "nums"), g(S, "nums"), g(T, "hn"), g(S, "hn"))
    # ------------------------------------------------------------ idf overlaps
    for col, key in (("core", "n"), ("sk", "k"), ("adw", "a")):
        F.update(idf_overlap(pairs, R, col, idfs[key], key))
    F["is_s3"] = (T["src"].to_numpy() == 3).astype(np.int8)
    for c in ("f_nsp1", "f_nspa", "f_adr1", "f_adra"):
        F[f"{c}_t"], F[f"{c}_s"] = T[c].to_numpy(), S[c].to_numpy()
    X = pl.DataFrame(F)
    bcols = [c for c in pairs.columns if c.startswith("k_")] + ["bscore", "brank", "nkeys"]
    return pl.concat([pairs.select(["t_rid", "s1_rid"] + bcols), X], how="horizontal")


def fit_idfs(recs):
    return {"n": token_idf(recs, "core", "w"), "k": token_idf(recs, "sk", "w"),
            "a": token_idf(recs, "adw", "w")}


# ---------------------------------------------------------------- context features
CTX_COLS = ["n_best", "n_ratio", "a_tset", "a_idf_jac", "n_idf_jac", "bscore", "a_num_jac"]


def add_context(X, cols=CTX_COLS, group="t_rid"):
    """How a pair compares to the competing candidates of the same group
    (group = t_rid: other S1 entities for this record; s1_rid: other records for this S1)."""
    tag = "t" if group == "t_rid" else "s"
    exprs = []
    for c in cols:
        v = pl.col(c).cast(pl.Float32)
        mx = v.max().over(group)
        n_at_max = (v >= mx).sum().over(group)
        below = pl.when(v < mx).then(v).otherwise(None).max().over(group).fill_null(-1.0)
        other = pl.when(v < mx).then(mx).when(n_at_max > 1).then(mx).otherwise(below)
        exprs += [(v - other).alias(f"cx{tag}_{c}_gap"), (mx - v).alias(f"cx{tag}_{c}_dmax"),
                  v.rank("min", descending=True).over(group).cast(pl.Int16).alias(f"cx{tag}_{c}_rk")]
    exprs.append(pl.len().over(group).cast(pl.Int16).alias(f"cx{tag}_n"))
    return X.with_columns(exprs)
