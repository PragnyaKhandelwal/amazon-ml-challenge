"""Stage 5 (test split): blocking -> features -> matcher -> decision -> submission files."""
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import polars as pl

import config as C
from evaluate import Decider
from features import PARQUET_COLS
from io_utils import write_outputs
from model import iter_parts, predict, s1_context
from pairs import build_features, prune
from run_blocking import with_country_code

W = C.WORK_DIR


def log(m):
    print(m, flush=True)


def main(stage="all"):
    t0 = time.time()
    recs = with_country_code(pl.read_parquet(os.path.join(W, "test_recs.parquet"),
                                             columns=PARQUET_COLS))
    ids = recs.select("rid", "entity_id", "src")
    if stage in ("all", "features"):
        cands = prune(pl.read_parquet(os.path.join(W, "test_cands.parquet")))
        log(f"pruned candidates: {cands.height:,}")
        build_features(recs, cands, W, "test", log=log)
        del cands
        s1_context(W, "test")
        log(f"features done ({time.time() - t0:.0f}s)")
    del recs
    meta = json.load(open(os.path.join(W, "stage1.json")))
    feats = meta["feats"]
    m = lgb.Booster(model_file=os.path.join(W, "model_stage1.txt"))
    P = pl.concat([X.select("t_rid", "s1_rid").with_columns(p=pl.Series(predict(m, X, feats)))
                   for X in iter_parts(W, "test")])
    P.write_parquet(os.path.join(W, "test_pred.parquet"))
    dec = Decider(**meta["decider"])
    sel = dec.select(P.select("t_rid", "s1_rid"), P["p"].to_numpy())
    log(f"selected {sel.height:,} matches for {sel['s1_rid'].n_unique():,} S1 entities")
    name = ids.select(pl.col("rid"), pl.col("entity_id"))
    to_ids = lambda d: (d.join(name.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_rid")  # noqa: E731
                         .join(name.rename({"rid": "t_rid", "entity_id": "tgt_id"}), on="t_rid"))
    matches = to_ids(sel.sort(["s1_rid", "p"], descending=[False, True])).select("s1_id", "tgt_id")
    cand_pairs = to_ids(P.sort(["s1_rid", "p"], descending=[False, True])).select("s1_id", "tgt_id")
    s1_ids = ids.filter(pl.col("src") == 1).sort("rid")["entity_id"].to_list()
    write_outputs(C.OUT_DIR, s1_ids, matches, cand_pairs)
    log(f"wrote outputs to {C.OUT_DIR} ({time.time() - t0:.0f}s)")
    validator = os.path.join(os.path.dirname(C.DATA_DIR), "utils", "validate_submission.py")
    if os.path.exists(validator):
        subprocess.run([sys.executable, validator,
                        "--matching", os.path.join(C.OUT_DIR, "matching_results.tsv"),
                        "--candidate", os.path.join(C.OUT_DIR, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(C.DATA_DIR, "test")])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
