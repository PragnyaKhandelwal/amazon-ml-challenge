"""Stage 4 (train split): features -> LightGBM matcher -> decision-rule tuning.

Validation protocol: Source-1 entities are split into 5 folds by a hash of their
id.  Fold 0 is never used for fitting (transliteration table, matcher); the
decision thresholds are tuned on it and its macro F0.5 is reported.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

import config as C
from evaluate import macro_f05, tune_decider
from features import REC_COLS
from model import feature_names, iter_parts, predict, s1_context, train
from pairs import build_features, prune
from run_blocking import gt_pairs, with_country_code

W = C.WORK_DIR


def log(m):
    print(m, flush=True)


def s1_folds(recs):
    return (recs.filter(pl.col("src") == 1)
                .select(pl.col("rid").alias("s1_rid"),
                        fold=(pl.col("entity_id").hash(seed=7) % C.N_FOLDS).cast(pl.Int8)))


def main(stage="all", train_frac=0.25, rounds=1500):
    t0 = time.time()
    recs = with_country_code(pl.read_parquet(os.path.join(W, "train_recs.parquet"),
                                             columns=[c for c in REC_COLS if c != "cc"] + ["entity_id", "country_n"]))
    gt = gt_pairs(recs)
    folds = s1_folds(recs)
    if stage in ("all", "features"):
        cands = prune(pl.read_parquet(os.path.join(W, "train_cands.parquet")))
        log(f"pruned candidates: {cands.height:,}")
        build_features(recs, cands, W, "train", log=log)
        del cands
        s1_context(W, "train")
        log(f"features done ({time.time() - t0:.0f}s)")
    del recs
    # ---------------- training sample (non-validation S1 folds, subsample of targets)
    lab = gt.with_columns(y=pl.lit(1, pl.Int8))
    tr_parts, feats = [], None
    for X in iter_parts(W, "train"):
        X = (X.join(lab, on=["s1_rid", "t_rid"], how="left").with_columns(pl.col("y").fill_null(0))
              .join(folds, on="s1_rid", how="left"))
        feats = feats or feature_names(X)
        keep = (pl.col("fold") != C.VALID_FOLD) & ((pl.col("t_rid").hash(seed=3) % 1000) < train_frac * 1000)
        tr_parts.append(X.filter(keep))
    tr = pl.concat(tr_parts)
    del tr_parts
    va = tr.filter((pl.col("t_rid").hash(seed=5) % 20) == 0)
    tr = tr.filter((pl.col("t_rid").hash(seed=5) % 20) != 0)
    log(f"train rows {tr.height:,} (pos {tr['y'].mean():.3f}), early-stop rows {va.height:,}, "
        f"{len(feats)} features ({time.time() - t0:.0f}s)")
    m = train(tr, feats, rounds=rounds, valid=va)
    del tr, va
    m.save_model(os.path.join(W, "model_stage1.txt"))
    log(f"model trained, best iter {m.best_iteration} ({time.time() - t0:.0f}s)")
    # ---------------- predictions on all train pairs (used for validation tuning)
    preds = []
    for X in iter_parts(W, "train"):
        preds.append(X.select("t_rid", "s1_rid").with_columns(p=pl.Series(predict(m, X, feats))))
    P = pl.concat(preds)
    P.write_parquet(os.path.join(W, "train_pred.parquet"))
    v_s1 = folds.filter(pl.col("fold") == C.VALID_FOLD)["s1_rid"].to_list()
    gv = gt.filter(pl.col("s1_rid").is_in(v_s1))
    best, dec, tab = tune_decider(P.select("t_rid", "s1_rid"), P["p"].to_numpy(), gv, v_s1)
    log(f"VALIDATION macro F0.5 = {best:.5f}  {dec.params()}")
    log(str(tab.sort("f05", descending=True).head(10)))
    json.dump({"decider": dec.params(), "valid_f05": best, "feats": feats},
              open(os.path.join(W, "stage1.json"), "w"), indent=1)
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])
    log(str([(f, int(g)) for f, g in imp[:40]]))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "all", float(a[1]) if len(a) > 1 else 0.25, int(a[2]) if len(a) > 2 else 1500)
