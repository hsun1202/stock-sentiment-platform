# -*- coding: utf-8 -*-
"""
只用 PTT 個股情緒：2025 年「真正樣本外」檢驗（事先寫定的假說，來自 2015-2024）
  H（主要）：淨情緒（今天收盤前留言）→ 隔天「隔夜」超額報酬的係數 > 0
            2015-2024：控制今天/漲跌停/留言量 +4.6 bp（t=3.8）；再加過去5/20天報酬 +2.7~2.8 bp（t≈2.3）
  另外報告：隔天收盤→收盤、隔天盤中；規則「情緒>0買、<0賣」毛/淨報酬；強正/強負事件後 1/5/20 天累積超額
  兩期用完全相同的程式與設定（2015-2024 的 PTT 來自 final_dataset_v3；2025 來自 ptt_2025_stock_labels.csv）
python stock_ptt_2025_test.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


PCOLS = ["ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"]
base = pd.read_csv("final_dataset_v3_2025/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                   usecols=["stock_code", "Date", "Open", "Close", "abnormal_price"] + PCOLS)
lab = pd.read_csv("ptt_2025_stock_labels.csv", dtype={"stock_code": str}, parse_dates=["Date"])
agg = lab.assign(pos=lab.text_sentiment.eq("positive").astype(int), neg=lab.text_sentiment.eq("negative").astype(int)) \
    .groupby(["stock_code", "Date", "part"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).unstack("part").fillna(0)
agg.columns = [f"ptt_{p}_{k}" for k, p in agg.columns]
agg = agg.reset_index()
is25 = base.Date.dt.year == 2025
base.loc[is25, PCOLS] = 0
base = base.merge(agg, on=["stock_code", "Date"], how="left", suffixes=("", "_new"))
for c in PCOLS:
    if c + "_new" in base:
        base.loc[is25, c] = base.loc[is25, c + "_new"].fillna(0)
        base = base.drop(columns=c + "_new")
df = base.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
df[PCOLS] = df[PCOLS].fillna(0)
pos_c, neg_c = df.ptt_pre_pos + df.ptt_intra_pos, df.ptt_pre_neg + df.ptt_intra_neg
pos_o, neg_o = pos_c + g.ptt_pre_pos.shift(-1).fillna(0), neg_c + g.ptt_pre_neg.shift(-1).fillna(0)
df["n_expr"] = pos_c + neg_c
df["net"] = (pos_c - neg_c) / df.n_expr.replace(0, np.nan)
df["net_o"] = (pos_o - neg_o) / (pos_o + neg_o).replace(0, np.nan)
df["att"] = np.log1p(df.ptt_pre_n + df.ptt_intra_n)
abn = df.abnormal_price.fillna(0) != 0
ok0 = ~abn & ~g.abnormal_price.shift(1).fillna(0).ne(0)
okn = g.abnormal_price.shift(-1).fillna(0).eq(0)
df["r0"] = (df.Close / g.Close.shift(1) - 1).where(ok0)
df["r1"] = (g.Close.shift(-1) / df.Close - 1).where(okn)
df["on"] = (g.Open.shift(-1) / df.Close - 1).where(okn)
df["id"] = (g.Close.shift(-1) / g.Open.shift(-1) - 1).where(okn)
for c in ["r0", "r1", "on", "id"]:
    df["x_" + c] = df[c] - df.groupby("Date")[c].transform("mean")
gx = df.groupby("stock_code").x_r0
df["past5"] = gx.transform(lambda s: s.shift(1).rolling(5, min_periods=4).sum())
df["past20"] = gx.transform(lambda s: s.shift(1).rolling(20, min_periods=15).sum())
for h in (5, 20):
    df[f"car{h}"] = sum(gx.shift(-k) for k in range(1, h + 1))
limit = np.where(df.Date < "2015-06-01", 0.07, 0.10)
df["lim_up"] = (df.r0 >= limit - 0.005).astype(int)
df["lim_dn"] = (df.r0 <= -limit + 0.005).astype(int)
C1 = ["x_r0", "lim_up", "lim_dn", "att"]
C2 = C1 + ["past5", "past20"]

for per, (y0, y1) in {"2015-2024（樣本內）": (2015, 2024), "2025（樣本外）": (2025, 2025)}.items():
    S = df[(df.n_expr >= 1) & df.Date.dt.year.between(y0, y1)].dropna(subset=["x_r0", "x_on", "past5", "past20"])
    say(f"\n=== {per}：有表態留言的股票日 {len(S):,}（情緒>0 {int((S.net > 0).sum()):,}、<0 {int((S.net < 0).sum()):,}）===")
    for y, lab_ in [("x_on", "隔夜（主要假說）"), ("x_r1", "隔天收盤→收盤"), ("x_id", "隔天盤中")]:
        for ctrl, cl in [(C1, "今天/漲跌停/留言量"), (C2, "＋過去5/20天")]:
            d = S.dropna(subset=[y])
            res = sm.OLS(d[y] * 1e4, sm.add_constant(d[["net"] + ctrl])).fit(cov_type="cluster", cov_kwds={"groups": d.Date.dt.strftime("%Y%m%d")})
            say(f"  {lab_:<12s}｜{cl:<10s} 淨情緒係數 {res.params['net']:+6.2f} bp（t={res.tvalues['net']:+.2f}）")
    # 規則
    for name, sig_col, rcol, costs in [("A 開盤當沖", "net_o", "id", (0.0032, 0.0044)), ("B 收盤持有一天", "net", "r1", (0.0047, 0.0059))]:
        d = df[df.Date.dt.year.between(y0, y1) & df[sig_col].notna() & (df[sig_col] != 0) & df[rcol].notna()]
        pr = np.sign(d[sig_col]) * d[rcol]
        xr = np.sign(d[sig_col]) * d["x_" + rcol]
        say(f"  規則 {name}（>0買、<0賣）：{len(d):,} 筆｜猜對 {(pr > 0).mean():.1%}｜每筆 原始 {pr.mean() * 1e4:+.1f} bp、超額 {xr.mean() * 1e4:+.1f} bp"
            f"｜扣{costs[0] * 1e4:.0f}bp {(pr.mean() - costs[0]) * 1e4:+.1f}、扣{costs[1] * 1e4:.0f}bp {(pr.mean() - costs[1]) * 1e4:+.1f}")
    for gname, mk in [("強正（≥3則、淨≥0.5）", (S.n_expr >= 3) & (S.net >= 0.5)), ("強負（≥3則、淨≤−0.5）", (S.n_expr >= 3) & (S.net <= -0.5))]:
        e = S[mk]
        say(f"  事件 {gname:<16s} {len(e):,} 個｜之後 1天 {e.x_r1.mean() * 100:+.2f}%、5天 {e.car5.mean() * 100:+.2f}%、20天 {e.car20.mean() * 100:+.2f}%")
open("stock_ptt_2025_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
