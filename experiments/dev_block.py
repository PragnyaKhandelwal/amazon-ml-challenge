import sys, time, os; sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0,".")
import polars as pl
import config as C
from run_blocking import with_country_code, gt_pairs, blocking_report
from blocking import generate_candidates
frac = float(sys.argv[1]) if len(sys.argv)>1 else 0.1
recs = pl.read_parquet(os.path.join(C.WORK_DIR,"train_recs.parquet"))
gt = gt_pairs(recs)
s1 = recs.filter(pl.col("src")==1).select("rid").filter((pl.col("rid").hash(3)%1000) < frac*1000)
gts = gt.join(s1.rename({"rid":"s1_rid"}), on="s1_rid")
matched_all = gt["t_rid"]
tg = recs.filter(pl.col("src")!=1).select("rid")
un = tg.filter(~pl.col("rid").is_in(matched_all)).filter((pl.col("rid").hash(3)%1000) < frac*1000)
keep = pl.concat([s1["rid"], gts["t_rid"], un["rid"]])
sub = with_country_code(recs.filter(pl.col("rid").is_in(keep)))
del recs
print("subset", sub.height, "S1", s1.height, "gt pairs", gts.height, flush=True)
t=time.time()
md=int(sys.argv[2]); c = generate_candidates(sub, top_k=20, max_df=md)
print("time", time.time()-t)
blocking_report(c, gts, sub)
import polars as pl
h=c.join(gts.with_columns(y=pl.lit(1)),on=["s1_rid","t_rid"],how="left").with_columns(pl.col("y").fill_null(0))
for k in ["k_n","k_k","k_p","k_na","k_ha","k_aa","k_x"]: print(k, "pairs w/ kind", (h[k]>0).mean(), "pos only via kind", h.filter((pl.col("y")==1)&(pl.col("nkeys")==pl.col(k))).height)
c.write_parquet(os.path.join(C.WORK_DIR, f"dev_cands_{frac}.parquet"))
sub.write_parquet(os.path.join(C.WORK_DIR, f"dev_recs_{frac}.parquet"))
gts.write_parquet(os.path.join(C.WORK_DIR, f"dev_gt_{frac}.parquet"))
