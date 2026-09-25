"""Reading challenge TSVs and writing / checking submission files."""
import os

import polars as pl

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    # quote_char=None: names/addresses contain stray quotes that must be kept literally
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    df = df.rename({c: c.strip() for c in df.columns})
    return df.with_columns(pl.col(c).fill_null("").str.strip_chars() for c in df.columns)


def load_sources(data_dir, split):
    """Returns one frame with all records of a split (S1, S2, S3 stacked) + column `src`."""
    d = os.path.join(data_dir, split)
    parts = []
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(d, f"{split}_source{k}.tsv"))
        for c in COLS:
            if c not in df.columns:
                df = df.with_columns(pl.lit("").alias(c))
        parts.append(df.select(COLS).with_columns(src=pl.lit(k, pl.Int8)))
    return pl.concat(parts)


def load_ground_truth(data_dir, split="train"):
    """(s1_id, tgt_id) pairs frame; S1 ids without matches are simply absent."""
    path = os.path.join(data_dir, split, f"{split}_ground_truth.tsv")
    g = read_tsv(path)
    return (g.with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "tgt_id"})
             .with_columns(pl.col("tgt_id").str.strip_chars())
             .filter(pl.col("tgt_id").is_not_null() & (pl.col("tgt_id") != "")))


def _write(path, col2, s1_ids, pairs):
    """pairs: frame (s1_id, tgt_id) already in the desired order."""
    lists = pairs.group_by("s1_id", maintain_order=True).agg(pl.col("tgt_id").unique(maintain_order=True))
    lists = dict(zip(lists["s1_id"].to_list(), lists["tgt_id"].to_list()))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col2}\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{','.join(lists.get(sid, []))}\n")


def write_outputs(out_dir, s1_ids, matches, candidates):
    """matches / candidates: frames with columns s1_id, tgt_id."""
    candidates = pl.concat([candidates.select("s1_id", "tgt_id"),
                            matches.select("s1_id", "tgt_id")]).unique(maintain_order=True)
    _write(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", s1_ids, matches)
    _write(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, candidates)
