"""Stage 3: candidate pruning + pairwise features, streamed in target chunks."""
import gc
import glob
import os
import time

import polars as pl

from features import REC_COLS, add_context, add_frequencies, fit_idfs, pair_features

TOP_K = 10        # blocking candidates kept per target
REL_MIN = 0.25    # ... and only if blocking score >= REL_MIN * best score of that target


def prune(cands, top_k=TOP_K, rel_min=REL_MIN):
    return (cands.filter(pl.col("brank") < top_k)
                 .filter(pl.col("bscore") >= rel_min * pl.col("bscore").max().over("t_rid")))


def build_features(recs, cands, out_dir, tag, chunk=400_000, log=print):
    """recs: normalised frame incl. cc; cands: pruned candidate pairs.
    Writes out_dir/{tag}_feat_XXX.parquet (per-target context included)."""
    t0 = time.time()
    for f in glob.glob(os.path.join(out_dir, f"{tag}_feat_*.parquet")):
        os.remove(f)
    recs = add_frequencies(recs).select(REC_COLS)
    idfs = fit_idfs(recs)
    t_ids = cands["t_rid"].unique().sort()
    n = 0
    for k, st in enumerate(range(0, len(t_ids), chunk)):
        lo, hi = t_ids[st], t_ids[min(st + chunk, len(t_ids)) - 1]
        pc = cands.filter(pl.col("t_rid").is_between(lo, hi)).sort(["t_rid", "brank"])
        X = pair_features(pc, recs, idfs)
        X = add_context(X, group="t_rid")
        X.write_parquet(os.path.join(out_dir, f"{tag}_feat_{k:03d}.parquet"), compression="zstd")
        n += X.height
        del X, pc
        gc.collect()
        log(f"  features chunk {k}: {n:,} pairs ({time.time() - t0:.0f}s)")
    return n


def load_features(out_dir, tag, columns=None, filt=None):
    lf = pl.scan_parquet(os.path.join(out_dir, f"{tag}_feat_*.parquet"))
    if filt is not None:
        lf = lf.filter(filt)
    if columns is not None:
        lf = lf.select(columns)
    return lf.collect()
