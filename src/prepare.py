"""Stage 1: load raw TSVs, learn the transliteration table, normalise, cache parquet.

Streams each source file in chunks so peak memory stays small.
"""
import os
import pickle
import sys
import time

import polars as pl

import config as C
from io_utils import COLS, load_ground_truth, load_sources, read_tsv
from normalize import make_pool, normalize_frame
from translit import Transliterator, fit_from_training

CHUNK = 1_000_000


def fold_of(col):
    return (pl.col(col).hash(seed=7) % C.N_FOLDS).cast(pl.Int8)


def fit_translit(tr_path):
    t0 = time.time()
    raw = load_sources(C.DATA_DIR, "train")
    gt = load_ground_truth(C.DATA_DIR, "train")
    fit_ids = (raw.filter(pl.col("src") == 1).select("entity_id")
                  .filter(fold_of("entity_id") != C.VALID_FOLD)["entity_id"])
    tr = fit_from_training(raw, gt, fit_ids)
    with open(tr_path, "wb") as f:
        pickle.dump(tr.table, f)
    print(f"translit table: {len(tr.table)} entries ({time.time() - t0:.0f}s)", flush=True)


def normalize_split(split, tr, pool):
    t = time.time()
    parts, rid0 = [], 0
    for k in (1, 2, 3):
        raw = read_tsv(os.path.join(C.DATA_DIR, split, f"{split}_source{k}.tsv"))
        raw = raw.select([pl.col(c) if c in raw.columns else pl.lit("").alias(c) for c in COLS])
        raw = raw.with_columns(src=pl.lit(k, pl.Int8))
        for st in range(0, raw.height, CHUNK):
            part = normalize_frame(raw.slice(st, CHUNK), tr, pool)
            part = part.with_columns(rid=pl.int_range(rid0, rid0 + part.height, dtype=pl.Int32))
            rid0 += part.height
            fn = os.path.join(C.WORK_DIR, f"_{split}_part{len(parts):03d}.parquet")
            part.write_parquet(fn)
            parts.append(fn)
            del part
            print(f"  {split} S{k} rows {st:,}+ ({time.time() - t:.0f}s)", flush=True)
        del raw
    pl.scan_parquet(parts).sink_parquet(os.path.join(C.WORK_DIR, f"{split}_recs.parquet"))
    for fn in parts:
        os.remove(fn)
    print(f"{split}: {rid0} records normalised ({time.time() - t:.0f}s)", flush=True)


def main(splits=("train", "test")):
    tr_path = os.path.join(C.WORK_DIR, "translit.pkl")
    if not os.path.exists(tr_path):
        fit_translit(tr_path)
    with open(tr_path, "rb") as f:
        tr = Transliterator(pickle.load(f))
    pool = make_pool(C.WORKERS)
    try:
        for split in splits:
            normalize_split(split, tr, pool)
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("train", "test"))
