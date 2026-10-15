# -*- coding: utf-8 -*-
"""
深入檢驗：PTT 淨情緒（今天收盤前的留言）→ 隔天報酬的正向關係，是真的資訊，還是「今天漲跌／漲停／關注度」的代理？
樣本：final_dataset_v3（2015-2024），有表態留言（正+負≥1）的股票日
報酬一律用「超額」＝個股 − 當天全體股票平均（去掉大盤漂移）
  1) 迴歸（依交易日群聚標準誤）：隔天超額 ~ 淨情緒 + 今天超額 + |今天超額| + 漲停/跌停 + 量能 + 留言量 + 規模
  2) 拆成 隔夜（收盤→開盤）與 盤中（開盤→收盤）
  3) 之後幾天會不會反轉：第 2~5 天、第 6~20 天累積超額
  4) 排除漲跌停日、分前後五年、依留言量分組
  5) 可交易性：收盤買（情緒>0）、隔天開盤或收盤賣，扣一般交易成本 47/59 bp
python ptt_cc_deepdive.py
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


df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Open", "Close", "Volume", "close_unadj", "abnormal_price",
                          "ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
for c in ["ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"]:
    df[c] = df[c].fillna(0)
pos = df.ptt_pre_pos + df.ptt_intra_pos
neg = df.ptt_pre_neg + df.ptt_intra_neg
df["n_expr"] = pos + neg
df["att"] = np.log1p(df.ptt_pre_n + df.ptt_intra_n)
df["net"] = (pos - neg) / df.n_expr.replace(0, np.nan)
abn = df.abnormal_price.fillna(0) != 0
bad = abn | g.abnormal_price.shift(1).fillna(0).ne(0)
C, O = df.Close, df.Open
df["r_today"] = (C / g.Close.shift(1) - 1).where(~bad)
for k in range(1, 21):  # 未來第 k 天的收盤→收盤報酬，以及隔天的隔夜/盤中
    df[f"r{k}"] = (g.Close.shift(-k) / g.Close.shift(-k + 1) - 1).where(g.abnormal_price.shift(-k).fillna(0).eq(0))
df["r_on"] = (g.Open.shift(-1) / C - 1).where(g.abnormal_price.shift(-1).fillna(0).eq(0))
df["r_id"] = (g.Close.shift(-1) / g.Open.shift(-1) - 1).where(g.abnormal_price.shift(-1).fillna(0).eq(0))
limit = np.where(df.Date < "2015-06-01", 0.07, 0.10)
df["lim_up"] = (df.r_today >= limit - 0.005).astype(int)
df["lim_dn"] = (df.r_today <= -limit + 0.005).astype(int)
vol20 = g.Volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
df["volr"] = np.log((df.Volume + 1) / (vol20 + 1))
dv = (df.close_unadj * df.Volume.where(df.Volume > 0))
df["size"] = dv.groupby(df.stock_code).transform(lambda s: s.rolling(60, min_periods=20).median())
df["size_r"] = df.groupby("Date")["size"].rank(pct=True)
# 超額：減掉當天全體股票平均
for col in ["r_today", "r_on", "r_id"] + [f"r{k}" for k in range(1, 21)]:
    df["x_" + col] = df[col] - df.groupby("Date")[col].transform("mean")
df["x_abs_today"] = df.x_r_today.abs()
df["x_r2_5"] = df[[f"x_r{k}" for k in range(2, 6)]].sum(axis=1, min_count=4)
df["x_r6_20"] = df[[f"x_r{k}" for k in range(6, 21)]].sum(axis=1, min_count=12)

S = df[(df.n_expr >= 1) & df.x_r_today.notna() & df.x_r1.notna()].copy()
say(f"樣本：有表態留言的股票日 {len(S):,}（{S.Date.min().date()}~{S.Date.max().date()}）；淨情緒>0 {int((S.net > 0).sum()):,}、<0 {int((S.net < 0).sum()):,}")
say(f"漲停日占 {S.lim_up.mean():.1%}（全體 {df.lim_up.mean():.1%}）；跌停日占 {S.lim_dn.mean():.1%}；淨情緒與今天超額相關 {S.net.corr(S.x_r_today):+.3f}")
say(f"淨情緒>0 的股票日中 漲停占 {S.loc[S.net > 0, 'lim_up'].mean():.1%}；淨情緒<0 中 {S.loc[S.net < 0, 'lim_up'].mean():.1%}")

CTRL = ["x_r_today", "x_abs_today", "lim_up", "lim_dn", "volr", "att", "size_r"]


def reg(d, ycol, xcols, label):
    d = d.dropna(subset=[ycol] + xcols)
    X = sm.add_constant(d[xcols])
    r = sm.OLS(d[ycol] * 1e4, X).fit(cov_type="cluster", cov_kwds={"groups": d.Date.dt.strftime("%Y%m%d")})
    say(f"  {label:<28s} n={len(d):>6,}｜淨情緒係數 {r.params['net']:+6.2f} bp（t={r.tvalues['net']:+.2f}）"
        + (f"｜漲停 {r.params['lim_up']:+.1f}（t={r.tvalues['lim_up']:+.1f}）" if 'lim_up' in xcols else ""))
    return r


say("\n1) 迴歸：隔天超額報酬（bp）對淨情緒（−1~+1）；『淨情緒係數』＝情緒從全負到全正，隔天超額多幾個 bp 的一半")
reg(S, "x_r1", ["net"], "只放淨情緒")
reg(S, "x_r1", ["net", "x_r_today"], "+今天超額")
reg(S, "x_r1", ["net"] + CTRL, "+全部控制變數")
say("\n2) 拆解隔天報酬（全部控制變數）")
reg(S, "x_r_on", ["net"] + CTRL, "隔夜（收盤→開盤）")
reg(S, "x_r_id", ["net"] + CTRL, "盤中（開盤→收盤）")
say("\n3) 之後會不會反轉（全部控制變數）")
reg(S, "x_r2_5", ["net"] + CTRL, "第 2~5 天累積")
reg(S, "x_r6_20", ["net"] + CTRL, "第 6~20 天累積")
say("\n4) 穩健性（隔天超額，全部控制變數）")
reg(S[(S.lim_up == 0) & (S.lim_dn == 0)], "x_r1", ["net", "x_r_today", "x_abs_today", "volr", "att", "size_r"], "排除漲跌停日")
reg(S[S.Date.dt.year <= 2019], "x_r1", ["net"] + CTRL, "2015-2019")
reg(S[S.Date.dt.year >= 2020], "x_r1", ["net"] + CTRL, "2020-2024")
reg(S[S.n_expr >= 5], "x_r1", ["net"] + CTRL, "表態留言≥5則")
reg(S[S.n_expr < 5], "x_r1", ["net"] + CTRL, "表態留言<5則")
reg(S[S.size_r <= 0.5], "x_r1", ["net"] + CTRL, "規模較小的一半")
reg(S[S.size_r > 0.5], "x_r1", ["net"] + CTRL, "規模較大的一半")

say("\n5) 可交易性：今天收盤買『淨情緒>0』的股票（等權），比較賣出時點；成本＝一般交易來回 47bp（手續費六折）/ 59bp")
days = pd.Series(sorted(df.Date.unique()))
for lab, m in [("全部 淨情緒>0", S.net > 0), ("淨情緒>0 且今天非漲停", (S.net > 0) & (S.lim_up == 0)),
               ("淨情緒>0 且今天漲停", (S.net > 0) & (S.lim_up == 1))]:
    d = S[m]
    for hold, col in [("隔天開盤賣", "r_on"), ("隔天收盤賣", "r1")]:
        rr = d[col].dropna()
        xr = d["x_" + col].dropna()
        line = f"  {lab:<16s}｜{hold}：{len(rr):,} 筆｜原始每筆 {rr.mean() * 1e4:+6.1f} bp、超額 {xr.mean() * 1e4:+6.1f} bp"
        for c in (0.0047, 0.0059):
            line += f"｜扣{c * 1e4:.0f}bp後每筆 {(rr.mean() - c) * 1e4:+6.1f}"
        say(line)
open("ptt_cc_deepdive.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
