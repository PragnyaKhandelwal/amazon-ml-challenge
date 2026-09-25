import polars as pl
W="C:/Users/prag1/amazon_ml_er/work/"
c=pl.read_parquet(W+"dev_cands_0.1.parquet"); g=pl.read_parquet(W+"dev_gt_0.1.parquet")
c=c.join(g.with_columns(y=pl.lit(1,pl.Int8)),on=["s1_rid","t_rid"],how="left").with_columns(pl.col("y").fill_null(0))
c=c.with_columns(rel=pl.col("bscore")/pl.col("bscore").max().over("t_rid"))
n=g.height
for K in (3,5,8,10):
  for r in (0.0,0.2,0.3,0.4,0.5):
    s=c.filter((pl.col("brank")<K)&(pl.col("rel")>=r))
    print(K,r,"pairs/target %.2f"%(s.height/c["t_rid"].n_unique()),"recall %.4f"%(s["y"].sum()/n))
