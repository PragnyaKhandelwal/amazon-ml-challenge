"""Matcher: LightGBM binary classifier on pair + context features."""
import glob
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from features import CTX_COLS, add_context

ID_COLS = ("t_rid", "s1_rid", "y", "fold")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
              max_bin=127, num_threads=4, verbose=-1, seed=42)


def s1_context(feat_dir, tag):
    """S1-side context needs every candidate of an S1 entity, which spans chunks:
    compute it once on a slim projection of all pairs."""
    files = sorted(glob.glob(os.path.join(feat_dir, f"{tag}_feat_*.parquet")))
    slim = pl.concat([pl.read_parquet(f, columns=["t_rid", "s1_rid"] + CTX_COLS)
                      .with_columns(part=pl.lit(i, pl.Int16), row=pl.int_range(pl.len(), dtype=pl.Int32))
                      for i, f in enumerate(files)])
    ctx = add_context(slim, group="s1_rid").drop(CTX_COLS)
    ctx.write_parquet(os.path.join(feat_dir, f"{tag}_ctxs.parquet"))
    return files


def iter_parts(feat_dir, tag):
    """Yields full feature frames (pair + target ctx + S1 ctx) chunk by chunk."""
    files = sorted(glob.glob(os.path.join(feat_dir, f"{tag}_feat_*.parquet")))
    ctx = pl.scan_parquet(os.path.join(feat_dir, f"{tag}_ctxs.parquet"))
    for i, f in enumerate(files):
        X = pl.read_parquet(f)
        c = ctx.filter(pl.col("part") == i).collect().sort("row").drop("part", "row", "t_rid", "s1_rid")
        yield pl.concat([X, c], how="horizontal")


def feature_names(X):
    return [c for c in X.columns if c not in ID_COLS]


def to_np(X, feats):
    return X.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def train(X, feats, rounds=1500, valid=None, log_every=100):
    dtr = lgb.Dataset(to_np(X, feats), X["y"].to_numpy(), feature_name=feats, free_raw_data=True)
    sets, cbs = [dtr], [lgb.log_evaluation(log_every)]
    if valid is not None:
        dva = lgb.Dataset(to_np(valid, feats), valid["y"].to_numpy(), reference=dtr)
        sets.append(dva)
        cbs.append(lgb.early_stopping(100, verbose=True))
    return lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=sets, callbacks=cbs)


def predict(m, X, feats):
    return m.predict(to_np(X, feats), num_threads=4).astype(np.float32)
