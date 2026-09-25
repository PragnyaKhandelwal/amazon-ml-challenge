"""Stage 2: candidate generation for a split; reports recall on train."""
import os
import sys
import time

import polars as pl

import config as C
from blocking import generate_candidates
from io_utils import load_ground_truth


BLOCK_COLS = ["rid", "entity_id", "src", "country_n", "core", "alt", "sk", "adw", "hn", "nsp"]


def with_country_code(recs):
    codes = recs.select(pl.col("country_n").unique().sort()).with_row_index("cc")
    return recs.join(codes.with_columns(pl.col("cc").cast(pl.Int16)), on="country_n", how="left")


def gt_pairs(recs, split="train"):
    gt = load_ground_truth(C.DATA_DIR, split)
    ids = recs.select("entity_id", "rid")
    return (gt.join(ids.rename({"entity_id": "s1_id", "rid": "s1_rid"}), on="s1_id")
              .join(ids.rename({"entity_id": "tgt_id", "rid": "t_rid"}), on="tgt_id")
              .select("s1_rid", "t_rid"))


def blocking_report(cands, gt, recs, log=print):
    n1 = recs.filter(pl.col("src") == 1).height
    hit = cands.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_rid", "t_rid"], how="left")
    y = hit["y"].fill_null(0)
    log(f"  candidates: {cands.height:,}  per target {cands.height / max(cands['t_rid'].n_unique(), 1):.2f}"
        f"  per S1 {cands.height / n1:.2f}")
    log(f"  pair recall: {y.sum() / gt.height:.4f}  positives rate {y.mean():.4f}")
    for k in (1, 2, 3, 5, 10, 20):
        r = hit.filter(pl.col("brank") < k)["y"].fill_null(0).sum() / gt.height
        log(f"    recall@{k}: {r:.4f}")


def main(split="train", top_k=10, max_df=300):
    t0 = time.time()
    recs = with_country_code(pl.read_parquet(os.path.join(C.WORK_DIR, f"{split}_recs.parquet"),
                                             columns=BLOCK_COLS))
    # one country at a time: candidates never cross country labels, and it halves peak memory
    parts = []
    for cc in recs["cc"].unique().sort().to_list():
        sub = recs.filter(pl.col("cc") == cc)
        print(f"country {sub['country_n'][0]}: {sub.height:,} records", flush=True)
        parts.append(generate_candidates(sub, top_k=top_k, max_df=max_df,
                                         log=lambda m: print(m, flush=True)))
        del sub
    cands = pl.concat(parts)
    cands.write_parquet(os.path.join(C.WORK_DIR, f"{split}_cands.parquet"))
    print(f"{split}: blocking done in {time.time() - t0:.0f}s", flush=True)
    if split == "train":
        blocking_report(cands, gt_pairs(recs), recs)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "train", int(a[1]) if len(a) > 1 else 10, int(a[2]) if len(a) > 2 else 300)
