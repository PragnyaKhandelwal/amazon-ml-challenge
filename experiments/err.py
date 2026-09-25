import sys; sys.stdout.reconfigure(encoding="utf-8")
import polars as pl
W="C:/Users/prag1/amazon_ml_er/work/"
o=pl.read_parquet(W+"dev_oof.parquet")
g=pl.read_parquet(W+"dev_gt_0.1.parquet")
r=pl.read_parquet(W+"dev_recs_0.1.parquet",columns=["rid","business_name","business_address","country"])
v=o.filter(pl.col("fold")==0)
best=o.filter(pl.col("p")==pl.col("p").max().over("t_rid")).unique("t_rid")
sel=best.filter(pl.col("p")>=0.8)
vs1=set(v["s1_rid"].to_list())
sel=sel.filter(pl.col("s1_rid").is_in(list(vs1)))
gv=g.filter(pl.col("s1_rid").is_in(list(vs1)))
fp=sel.filter(pl.col("y")==0); tp=sel.filter(pl.col("y")==1)
fn=gv.join(sel.select("t_rid","s1_rid"),on=["t_rid","s1_rid"],how="anti")
incand=fn.join(o.select("t_rid","s1_rid","p"),on=["t_rid","s1_rid"],how="left")
print("TP",tp.height,"FP",fp.height,"FN",fn.height,"FN not in cands",incand["p"].null_count())
def show(d,n=12):
    d=d.join(r.rename({"rid":"t_rid"}),on="t_rid").join(r.rename({"rid":"s1_rid"}),on="s1_rid",suffix="_s1")
    for x in d.head(n).iter_rows(named=True):
        print(f"p={x.get('p')}\n  T: {x['business_name']} | {x['business_address']}\n  S: {x['business_name_s1']} | {x['business_address_s1']} [{x['country']}]")
print("=== FALSE POSITIVES"); show(fp.sample(12,seed=1))
print("=== FALSE NEGATIVES (in cands)"); show(incand.filter(pl.col("p").is_not_null()).sample(12,seed=1))
print("=== FN not in cands"); show(incand.filter(pl.col("p").is_null()).sample(8,seed=1))
