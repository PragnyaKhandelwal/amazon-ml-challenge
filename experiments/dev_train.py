import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "C:/Users/prag1/amazon_ml_er/code/business_entity_resolution/src")
import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
from evaluate import macro_f05, tune_decider
from features import add_context
from pairs import build_features, load_features, prune

W = C.WORK_DIR
frac = sys.argv[1] if len(sys.argv) > 1 else "0.1"
rebuild = len(sys.argv) > 2 and sys.argv[2] == "1"
t0 = time.time()
recs = pl.read_parquet(f"{W}/dev_recs_{frac}.parquet")
cands = prune(pl.read_parquet(f"{W}/dev_cands_{frac}.parquet"))
gt = pl.read_parquet(f"{W}/dev_gt_{frac}.parquet")
print("pruned pairs", cands.height, flush=True)
if rebuild or not os.path.exists(f"{W}/dev_feat_000.parquet"):
    build_features(recs, cands, W, "dev", log=lambda m: print(m, flush=True))
X = load_features(W, "dev")
X = add_context(X, group="s1_rid")
X = X.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_rid", "t_rid"], how="left").with_columns(
    pl.col("y").fill_null(0))
s1 = recs.filter(pl.col("src") == 1).select("rid", "entity_id")
fold = s1.with_columns(fold=(pl.col("entity_id").hash(seed=7) % 5).cast(pl.Int8)).select(
    pl.col("rid").alias("s1_rid"), "fold")
X = X.join(fold, on="s1_rid", how="left")
feats = [c for c in X.columns if c not in ("t_rid", "s1_rid", "y", "fold")]
print("features", len(feats), "rows", X.height, "pos", X["y"].sum(), f"({time.time() - t0:.0f}s)", flush=True)
tr = X.filter(pl.col("fold") != 0)
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=4, verbose=-1)
dtr = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["y"].to_numpy(), free_raw_data=True)
m = lgb.train(params, dtr, num_boost_round=int(sys.argv[3]) if len(sys.argv) > 3 else 600)
p = m.predict(X.select(feats).to_numpy().astype(np.float32), num_threads=4)
print(f"trained ({time.time() - t0:.0f}s)", flush=True)
v_s1 = fold.filter(pl.col("fold") == 0)["s1_rid"].to_list()
gv = gt.filter(pl.col("s1_rid").is_in(v_s1))
best, dec, tab = tune_decider(X.select("t_rid", "s1_rid"), p, gv, v_s1)
print("VALID macro F0.5:", round(best, 5), dec.params())
print(tab.sort("f05", descending=True).head(8))
imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])
print([(f, int(g)) for f, g in imp[:40]])
X.select("t_rid", "s1_rid", "y", "fold").with_columns(p=pl.Series(p)).write_parquet(f"{W}/dev_oof.parquet")
