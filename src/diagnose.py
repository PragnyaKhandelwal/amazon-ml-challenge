"""Error analysis after train.py / predict.py: one compact report to decide what to fix next.

  python diagnose.py            # needs work/{train,test}_recs.parquet, {train,test}_pred.parquet, stage1.json

Sections
  1 validation (fold 0) macro F0.5 per country, and the best threshold per country
  2 where the fold-0 errors come from (blocking miss / wrong S1 / rejected / false merge)
  3 test vs validation confidence per country (France has no labels: compare its profile)
"""
import json
import os

import numpy as np
import polars as pl

import config as C
from evaluate import Decider, f05
from run_blocking import gt_pairs

W = C.WORK_DIR
COLS = ["rid", "src", "entity_id", "country_n"]


def per_s1(sel, gt, s1):
    """s1: frame (s1_rid, country_n) -> adds tp / npred / ntrue / f."""
    tp = sel.join(gt, on=["s1_rid", "t_rid"]).group_by("s1_rid").agg(tp=pl.len())
    d = (s1.join(tp, on="s1_rid", how="left")
           .join(sel.group_by("s1_rid").agg(npred=pl.len()), on="s1_rid", how="left")
           .join(gt.group_by("s1_rid").agg(ntrue=pl.len()), on="s1_rid", how="left").fill_null(0))
    return d.with_columns(f=pl.Series(f05(d["tp"], d["npred"], d["ntrue"])))


def exclusive_best(P):
    return P.filter(pl.col("p") == pl.col("p").max().over("t_rid")).unique("t_rid", keep="first")


def confidence_profile(P, recs, t):
    """Per target country: share of targets with a candidate, and how confident the best one is."""
    best = exclusive_best(P).join(recs.select(pl.col("rid").alias("t_rid"), "country_n"), on="t_rid")
    tg = recs.filter(pl.col("src") != 1).group_by("country_n").agg(targets=pl.len())
    s = best.group_by("country_n").agg(
        has_cand=pl.len(), med_p=pl.col("p").median(),
        p_ge_t=(pl.col("p") >= t).mean(), p_ge_09=(pl.col("p") >= 0.9).mean(),
        grey=((pl.col("p") >= 0.2) & (pl.col("p") < 0.8)).mean())
    return (tg.join(s, on="country_n", how="left")
              .with_columns(has_cand=pl.col("has_cand") / pl.col("targets"))
              .sort("targets", descending=True).head(6))


def assign_profile(sel, recs):
    """Per S1 country: matches per S1 and share of S1 left empty."""
    s1 = recs.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1_rid"), "country_n")
    n = s1.join(sel.group_by("s1_rid").agg(n=pl.len()), on="s1_rid", how="left").fill_null(0)
    return (n.group_by("country_n").agg(s1=pl.len(), matches_per_s1=pl.col("n").mean(),
                                        empty=(pl.col("n") == 0).mean())
             .sort("s1", descending=True).head(6))


def main():
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(40)
    meta = json.load(open(os.path.join(W, "stage1.json")))
    dec = Decider(**meta["decider"])
    print(f"stage1: valid_f05={meta['valid_f05']:.5f} decider={meta['decider']} n_feats={len(meta['feats'])}")

    # ---------------------------------------------------------------- validation fold
    recs = pl.read_parquet(os.path.join(W, "train_recs.parquet"), columns=COLS)
    gt = gt_pairs(recs)
    s1 = (recs.filter(pl.col("src") == 1)
              .filter((pl.col("entity_id").hash(seed=7) % C.N_FOLDS) == C.VALID_FOLD)
              .select(pl.col("rid").alias("s1_rid"), "country_n"))
    gv = gt.join(s1.select("s1_rid"), on="s1_rid", how="semi")
    P = pl.read_parquet(os.path.join(W, "train_pred.parquet"))
    Pv = P.join(s1.select("s1_rid"), on="s1_rid", how="semi")
    # targets whose true S1 is in the fold; plus every target that has a fold-0 candidate
    tv = pl.concat([gv.select("t_rid"), Pv.select("t_rid")]).unique()
    Pt = P.join(tv, on="t_rid", how="semi")          # all candidates of those targets (any fold)
    sel = dec.select(Pt.select("t_rid", "s1_rid"), Pt["p"].to_numpy())
    sel_v = sel.join(s1.select("s1_rid"), on="s1_rid", how="semi")
    d = per_s1(sel_v, gv, s1)

    print("\n== 1. fold-0 macro F0.5 per S1 country ==")
    print(d.group_by("country_n").agg(s1=pl.len(), f05=pl.col("f").mean(),
                                      singleton=(pl.col("ntrue") == 0).mean(),
                                      perfect=(pl.col("f") >= 0.9999).mean())
           .sort("s1", descending=True).head(6))
    print(f"overall {d['f'].mean():.5f}")

    rows = []
    best_v = exclusive_best(Pt).join(s1.select("s1_rid"), on="s1_rid", how="semi")
    for t in np.round(np.arange(0.3, 0.96, 0.05), 2):
        dd = per_s1(best_v.filter(pl.col("p") >= t), gv, s1)
        g = dd.group_by("country_n").agg(pl.col("f").mean()).sort("country_n")
        rows.append({"t": float(t), **dict(zip(g["country_n"].to_list(), g["f"].to_list()))})
    print("\nthreshold sweep (alpha=0) per country:")
    print(pl.DataFrame(rows))

    print("\n== 2. fold-0 error sources ==")
    fn = gv.join(sel_v, on=["s1_rid", "t_rid"], how="anti")
    top = exclusive_best(Pt).select("t_rid", top_s1="s1_rid", top_p="p")
    fn = (fn.join(Pt.select("s1_rid", "t_rid", pair_p="p"), on=["s1_rid", "t_rid"], how="left")
            .join(top, on="t_rid", how="left")
            .with_columns(kind=pl.when(pl.col("pair_p").is_null()).then(pl.lit("FN blocking miss"))
                          .when(pl.col("top_s1") != pl.col("s1_rid")).then(pl.lit("FN other S1 ranked higher"))
                          .otherwise(pl.lit("FN best but p < t"))))
    fp = (sel_v.join(gv, on=["s1_rid", "t_rid"], how="anti")
               .join(gt.select("t_rid", "s1_rid").rename({"s1_rid": "true_s1"}), on="t_rid", how="left")
               .with_columns(kind=pl.when(pl.col("true_s1").is_null()).then(pl.lit("FP distractor merged"))
                             .otherwise(pl.lit("FP target of another S1"))))
    err = pl.concat([fn.select("s1_rid", "kind"), fp.select("s1_rid", "kind")]).join(s1, on="s1_rid")
    print(f"true pairs {gv.height:,}  predicted {sel_v.height:,}  TP {sel_v.height - fp.height:,}")
    print(err.group_by("kind", "country_n").agg(n=pl.len()).sort("n", descending=True))
    lost = d.with_columns(loss=1 - pl.col("f")).group_by(
        pl.when(pl.col("ntrue") == 0).then(pl.lit("singleton S1"))
          .when(pl.col("tp") == 0).then(pl.lit("S1 with 0 TP"))
          .when(pl.col("npred") > pl.col("tp")).then(pl.lit("has FP"))
          .when(pl.col("tp") < pl.col("ntrue")).then(pl.lit("only FN"))
          .otherwise(pl.lit("perfect")).alias("case")).agg(
        s1=pl.len(), f05_points_lost=(pl.col("loss").sum() / d.height))
    print("\nwhere macro-F0.5 points are lost:")
    print(lost.sort("f05_points_lost", descending=True))
    Pv_t = exclusive_best(Pt)
    del P, Pt, Pv, recs, gt

    # ---------------------------------------------------------------- test profile
    print("\n== 3. confidence profile: validation targets vs test targets ==")
    vr = pl.read_parquet(os.path.join(W, "train_recs.parquet"), columns=COLS).join(
        tv.rename({"t_rid": "rid"}), on="rid", how="semi")
    print("validation:")
    print(confidence_profile(Pv_t, vr, dec.t))
    tr = pl.read_parquet(os.path.join(W, "test_recs.parquet"), columns=COLS)
    Ptest = pl.read_parquet(os.path.join(W, "test_pred.parquet"))
    print("test:")
    print(confidence_profile(Ptest, tr, dec.t))
    sel_t = dec.select(Ptest.select("t_rid", "s1_rid"), Ptest["p"].to_numpy())
    print("\nassignments per S1 country -- validation:")
    print(d.group_by("country_n").agg(s1=pl.len(), matches_per_s1=pl.col("npred").mean(),
                                      empty=(pl.col("npred") == 0).mean(),
                                      true_per_s1=pl.col("ntrue").mean(),
                                      true_empty=(pl.col("ntrue") == 0).mean())
           .sort("s1", descending=True))
    print("test:")
    print(assign_profile(sel_t, tr))


if __name__ == "__main__":
    main()
