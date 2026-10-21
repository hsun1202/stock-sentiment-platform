# -*- coding: utf-8 -*-
"""照組員規格重跑：每季底在成交金額前 250 名中，買近 60 日波動最高 30 檔（與最低 30 檔）、等權、持有一季。
TEJ 還原價（含息）、含下市；比較：母體等權、母體市值加權、0050 含息、加權指數（不含息）。只報數字。
cd D:/專題claude && python quarterly_vol/quarterly_vol_cookman.py"""
import numpy as np
import pandas as pd

OUT = "quarterly_vol/"
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def t_(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def perf(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    return (1 + r).prod() ** (4 / len(r)) - 1, r.std() * 2, (r.mean() * 4) / (r.std() * 2), (eq / eq.cummax() - 1).min()


use = ["stock_code", "Date", "Close", "close_unadj", "Volume", "abnormal_price", "mktcap"]
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
abn = df.abnormal_price.fillna(0) != 0
df["r"] = df.groupby("stock_code").Close.pct_change().where(~(abn | abn.groupby(df.stock_code).shift(1).fillna(False)))
df["dv"] = (df.close_unadj * df.Volume).where(df.Volume > 0)
df["vol60"] = df.groupby("stock_code").r.transform(lambda x: x.rolling(60, min_periods=40).std())
df["dv60"] = df.groupby("stock_code").dv.transform(lambda x: x.rolling(60, min_periods=20).mean())
df["yq"] = df.Date.dt.to_period("Q")
P = df[df.Close.notna()].drop_duplicates(["stock_code", "yq"], keep="last")[["stock_code", "yq", "Date", "Close", "vol60", "dv60", "mktcap"]].reset_index(drop=True)
P["qi"] = P.yq.dt.year * 4 + P.yq.dt.quarter
g = P.groupby("stock_code")
last_qi, data_end = g.qi.transform("max"), P.qi.max()
nxt_C, nxt_qi, nxt_d = g.Close.shift(-1), g.qi.shift(-1), g.Date.shift(-1)
P["ret_next"] = nxt_C / P.Close - 1
q_end = (P.yq + 1).dt.end_time.dt.normalize()
dl = (nxt_qi == last_qi) & (nxt_qi < data_end) & (nxt_d < q_end - pd.Timedelta(days=10))
P.loc[dl, "ret_next"] = (1 + P.ret_next[dl]) * 0.5 - 1
P.loc[(P.qi == last_qi) & (P.qi < data_end), "ret_next"] = -0.5


def bench_q(f):
    s = pd.read_csv(OUT + f, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    s = s.groupby(s.index.to_period("Q")).last()
    return s.shift(-1) / s - 1


b50, btw = bench_q("etf0050_adj.csv"), bench_q("twii_price.csv")
rows = {}
for q, z in P[P.ret_next.notna() & P.vol60.notna() & P.dv60.notna()].groupby("yq"):
    if q < pd.Period("2015Q2", "Q") or q > pd.Period("2025Q3", "Q"):
        continue
    U = z.nlargest(250, "dv60")
    hi, lo = U.nlargest(30, "vol60"), U.nsmallest(30, "vol60")
    rows[q] = {"高波動30": hi.ret_next.mean(), "低波動30": lo.ret_next.mean(), "母體等權": U.ret_next.mean(),
               "母體市值加權": np.average(U.ret_next, weights=U.mktcap.fillna(0) + 1e-9), "0050含息": b50.get(q), "加權指數(不含息)": btw.get(q)}
R = pd.DataFrame(rows).T.astype(float)
say(f"照組員規格（成交金額前 250 名、60 日波動、各 30 檔、等權、季頻）｜TEJ 還原價含息、含下市｜持有 {R.index[0] + 1}–{R.index[-1] + 1}，{len(R)} 季")
say(f"  {'組合':<12}{'年化':>8}{'波動':>8}{'Sharpe':>8}{'最大回落':>9}")
for c in R.columns:
    a, v, sh, dd = perf(R[c])
    say(f"  {c:<12}{a:>+8.1%}{v:>8.1%}{sh:>8.2f}{dd:>9.1%}")
for x, y in [("高波動30", "母體等權"), ("低波動30", "母體等權"), ("高波動30", "加權指數(不含息)"), ("高波動30", "0050含息"), ("高波動30", "低波動30"), ("母體等權", "0050含息")]:
    d = R[x] - R[y]
    say(f"  {x} − {y}：{d.mean():+.2%}/季（年化約 {d.mean() * 4:+.1%}），t={t_(d):+.2f}，贏 {(d > 0).mean():.0%} 的季")
beta = np.polyfit(R["0050含息"].values, R["高波動30"].values, 1)[0]
say(f"  高波動30 對 0050 的 beta ≈ {beta:.2f}（季報酬簡單迴歸）")
d = (R["高波動30"] - R["0050含息"])
say("  分年（高波動30 − 0050 含息，季差加總）：" + " ".join(f"{y}:{v:+.0%}" for y, v in d.groupby([(p + 1).year for p in d.index]).sum().items()))
open(OUT + "quarterly_vol_cookman.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
