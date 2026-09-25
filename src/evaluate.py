"""Macro F0.5 (official definition) and the decision rule that turns pair
probabilities into per-Source-1 match lists."""
import itertools

import numpy as np
import polars as pl

BETA2 = 0.25


def f05(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, np.float64) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.0)
        R = np.where(ntrue > 0, tp / ntrue, 0.0)
        F = np.where(tp > 0, (1 + BETA2) * P * R / (BETA2 * P + R), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(np.float64), F)


def macro_f05(pred, gt, s1_rids):
    """pred, gt: frames (s1_rid, t_rid); s1_rids: every evaluated Source-1 rid."""
    base = pl.DataFrame({"s1_rid": s1_rids}).with_columns(pl.col("s1_rid").cast(pl.Int32))
    tp = pred.join(gt, on=["s1_rid", "t_rid"]).group_by("s1_rid").agg(tp=pl.len())
    npred = pred.group_by("s1_rid").agg(np_=pl.len())
    ntrue = gt.group_by("s1_rid").agg(nt=pl.len())
    d = (base.join(tp, on="s1_rid", how="left").join(npred, on="s1_rid", how="left")
             .join(ntrue, on="s1_rid", how="left").fill_null(0))
    return float(f05(d["tp"], d["np_"], d["nt"]).mean())


class Decider:
    """keep (target, S1) if
         S1 is the target's best-scoring S1        (a record belongs to one entity)
         p >= t
         p >= alpha * best p among this S1's kept targets  (drop weak tail)
    """

    def __init__(self, t=0.5, alpha=0.0):
        self.t, self.alpha = t, alpha

    def params(self):
        return {"t": self.t, "alpha": self.alpha}

    def select(self, pairs, p):
        d = pairs.select("t_rid", "s1_rid").with_columns(p=pl.Series(p, dtype=pl.Float32))
        d = d.filter(pl.col("p") == pl.col("p").max().over("t_rid"))
        d = d.unique("t_rid", keep="first").filter(pl.col("p") >= self.t)
        if self.alpha > 0:
            d = d.filter(pl.col("p") >= self.alpha * pl.col("p").max().over("s1_rid"))
        return d


def tune_decider(pairs, p, gt, s1_rids, ts=None, alphas=(0.0, 0.5, 0.7, 0.8, 0.9)):
    """Grid search on out-of-fold probabilities; returns (best_score, Decider, table)."""
    ts = ts if ts is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    # precompute exclusive best per target once
    d = pairs.select("t_rid", "s1_rid").with_columns(p=pl.Series(p, dtype=pl.Float32))
    d = d.filter(pl.col("p") == pl.col("p").max().over("t_rid")).unique("t_rid", keep="first")
    d = d.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_rid", "t_rid"], how="left").with_columns(
        pl.col("y").fill_null(0))
    base = pl.DataFrame({"s1_rid": s1_rids}).with_columns(pl.col("s1_rid").cast(pl.Int32))
    nt = base.join(gt.group_by("s1_rid").agg(nt=pl.len()), on="s1_rid", how="left").fill_null(0)
    idx = {r: i for i, r in enumerate(nt["s1_rid"].to_list())}
    ntrue = nt["nt"].to_numpy()
    d = d.filter(pl.col("s1_rid").is_in(nt["s1_rid"]))
    pos = np.array([idx[r] for r in d["s1_rid"].to_list()])
    pv, yv = d["p"].to_numpy(), d["y"].to_numpy()
    # per-S1 max prob of the selected (exclusive) pairs, for the alpha rule
    gmax = np.zeros(len(ntrue))
    np.maximum.at(gmax, pos, pv)
    rows, best = [], (-1.0, None)
    for t, a in itertools.product(ts, alphas):
        m = (pv >= t) & (pv >= a * gmax[pos])
        tp = np.bincount(pos, weights=m & (yv == 1), minlength=len(ntrue))
        npred = np.bincount(pos, weights=m, minlength=len(ntrue))
        s = float(f05(tp, npred, ntrue).mean())
        rows.append((t, a, s))
        if s > best[0] + 1e-9:
            best = (s, Decider(float(t), float(a)))
    return best[0], best[1], pl.DataFrame(rows, schema=["t", "alpha", "f05"], orient="row")
