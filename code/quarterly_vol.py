# -*- coding: utf-8 -*-
"""
季頻高低波動選股 vs 大盤（不扣手續費）。事先寫定的假說見 PREREG_quarterly_vol.md。
每季底依過去 3 個月日報酬標準差分 5 組、等權、持有一季；和 0050 含息、^TWII、市值加權比。
cd D:/專題claude && python quarterly_vol/quarterly_vol.py
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "quarterly_vol/"
LOG = []
T_CRIT = 2.33
COST = 0.00471


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def perf(r):
    r = pd.Series(r).dropna()
    ann = (1 + r).prod() ** (4 / len(r)) - 1
    vol = r.std() * 2
    eq = (1 + r).cumprod()
    return ann, vol, (r.mean() * 4) / vol, (eq / eq.cummax() - 1).min(), eq.iloc[-1] - 1


# ---------- 股票日資料 ----------
use = ["stock_code", "Date", "Close", "close_unadj", "Volume", "abnormal_price", "mktcap"]
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
abn = df.abnormal_price.fillna(0) != 0
df["r"] = df.groupby("stock_code").Close.pct_change().where(~(abn | abn.groupby(df.stock_code).shift(1).fillna(False)))
df["dv"] = (df.close_unadj * df.Volume).where(df.Volume > 0)
df["vol12"] = df.groupby("stock_code").r.transform(lambda x: x.rolling(250, min_periods=150).std())
df["yq"] = df.Date.dt.to_period("Q")
px = df[df.Close.notna()]
last = px.drop_duplicates(["stock_code", "yq"], keep="last").set_index(["stock_code", "yq"])[["Close", "Date", "mktcap", "vol12"]]
agg = df.groupby(["stock_code", "yq"]).agg(vol3=("r", "std"), n_r=("r", "count"), dv_med=("dv", "median"))
P = last.join(agg).reset_index().sort_values(["stock_code", "yq"]).reset_index(drop=True)
P.loc[P.n_r < 40, "vol3"] = np.nan
P["qi"] = P.yq.dt.year * 4 + P.yq.dt.quarter
C = P.Close
g = P.groupby("stock_code")
last_qi = g.qi.transform("max")
data_end = P.qi.max()
nxt_C, nxt_qi, nxt_date = g.Close.shift(-1), g.qi.shift(-1), g.Date.shift(-1)
P["ret_next"] = nxt_C / C - 1                                   # 下一個有價格的季（中間停牌跨季也用這個）
q_end = (P.yq + 1).dt.end_time.dt.normalize()
# 下市：下一季是最後一季、而且最後交易日離季底超過 10 天 → 再記 −50%
dl_in = (nxt_qi == last_qi) & (nxt_qi < data_end) & (nxt_date < q_end.shift(0) - pd.Timedelta(days=10))
P.loc[dl_in, "ret_next"] = (1 + P.ret_next[dl_in]) * 0.5 - 1
dl_now = (P.qi == last_qi) & (P.qi < data_end)               # 下一季完全沒有價格
P.loc[dl_now, "ret_next"] = -0.5
P["liq_ok"] = P.groupby("yq").dv_med.rank(pct=True) > 0.2
P["big"] = P.groupby("yq").mktcap.rank(pct=True) > 0.5
say(f"[面板] 股票×季 {len(P):,} 列、{P.stock_code.nunique()} 檔、{P.yq.min()}~{P.yq.max()}；下市記 −50%：季內 {int(dl_in.sum())}、隔季無價 {int(dl_now.sum())}")

# ---------- 大盤 ----------
def bench_q(f):
    s = pd.read_csv(OUT + f, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    s = s.groupby(s.index.to_period("Q")).last()
    return (s.shift(-1) / s - 1)                                # 以形成季為索引：下一季報酬


B = {"0050含息": bench_q("etf0050_adj.csv"), "加權指數(不含息)": bench_q("twii_price.csv")}
FORM = pd.period_range("2015Q4", "2025Q3", freq="Q")             # 形成季；持有 2016Q1–2025Q4
# 0050 2025 年 1 拆 4：檢查還原價有沒有跳空
e = pd.read_csv(OUT + "etf0050_adj.csv", index_col=0, parse_dates=True).iloc[:, 0]
say(f"[檢查] 0050 還原價最大單日跌幅 {e.pct_change().min():.1%}（若接近 −75% 代表拆股沒還原）")


def run(U, vcol, label, main=False):
    U = U[U.yq.isin(FORM) & U[vcol].notna() & U.ret_next.notna()].copy()
    U["grp"] = U.groupby("yq")[vcol].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)) + 1
    R = U.groupby(["yq", "grp"]).ret_next.mean().unstack()
    R["市值加權"] = U.groupby("yq").apply(lambda z: np.average(z.ret_next, weights=z.mktcap.fillna(0) + 1e-9))
    R["等權全體"] = U.groupby("yq").ret_next.mean()
    for k, b in B.items():
        R[k] = b.reindex(R.index)
    say(f"\n===== {label}（每季平均 {U.groupby('yq').size().mean():.0f} 檔，{len(R)} 季，持有 {(R.index[0] + 1)}–{(R.index[-1] + 1)}）=====")
    say(f"  {'組合':<14}{'年化':>8}{'波動':>8}{'Sharpe':>8}{'最大回落':>9}{'累積':>9}")
    names = {1: "Q1 最低波動", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 最高波動"}
    for c in [1, 2, 3, 4, 5, "等權全體", "市值加權", "0050含息", "加權指數(不含息)"]:
        a, v, sh, dd, cum = perf(R[c])
        say(f"  {str(names.get(c, c)):<14}{a:>+8.1%}{v:>8.1%}{sh:>8.2f}{dd:>9.1%}{cum:>+9.0%}")
    res = {}
    for c, nm in [(1, "低波動"), (5, "高波動")]:
        for bn in ["0050含息", "加權指數(不含息)", "市值加權"]:
            d = R[c] - R[bn]
            say(f"  {nm} − {bn}：{d.mean():+.2%}/季（t={tstat(d):+.2f}，贏 {(d > 0).mean():.0%} 的季）")
            res[(nm, bn)] = d
    ls = R[1] - R[5]
    say(f"  低波動 − 高波動：{ls.mean():+.2%}/季（t={tstat(ls):+.2f}，贏 {(ls > 0).mean():.0%} 的季）")
    d = R[1] - R["0050含息"]
    yr = d.groupby(d.index.year + (d.index.quarter == 4)).sum()   # 持有年（形成季 +1）
    say("  分年（低波動 − 0050 含息，季差加總）：" + " ".join(f"{y}:{x:+.1%}" for y, x in yr.items()) + f"｜為正 {(yr > 0).sum()}/{len(yr)}")
    hy = np.array([(q + 1).year for q in R.index])
    h1, h2 = d[hy <= 2020], d[hy >= 2021]
    say(f"  前半（持有 2016–2020）{h1.mean():+.2%}/季（t={tstat(h1):+.2f}）｜後半（2021–2025）{h2.mean():+.2%}/季（t={tstat(h2):+.2f}）")
    if main:
        hold = U[U.grp == 1].groupby("yq").stock_code.apply(set)
        turn = pd.Series([len(hold.iloc[i] - hold.iloc[i - 1]) / len(hold.iloc[i]) for i in range(1, len(hold))])
        net = R[1].iloc[1:] - turn.values * COST
        say(f"  [參考] 低波動組每季換股 {turn.mean():.0%}；扣 0.471% 來回成本後 年化 {perf(net)[0]:+.1%}（不扣 {perf(R[1].iloc[1:])[0]:+.1%}）")
        mix = 0.5 * R[1] + 0.5 * R["0050含息"]
        a, v, sh, dd, cum = perf(mix)
        say(f"  [自行分配示範] 低波動 50% ＋ 0050 50%、每季再平衡：年化 {a:+.1%}、波動 {v:.1%}、Sharpe {sh:.2f}、最大回落 {dd:.1%}")
    return R, res, ls


R, res, ls = run(P[P.liq_ok], "vol3", "主要：可買範圍（成交值非最低 20%）、過去 3 個月波動", main=True)
d1 = res[("低波動", "0050含息")]
say(f"\n→ 檢定 Q1（低波動 − 0050 含息 > 0）：{d1.mean():+.2%}/季，t={tstat(d1):+.2f} → {'通過' if d1.mean() > 0 and tstat(d1) > T_CRIT else '未通過'}（門檻 t>{T_CRIT}）")
say(f"→ 檢定 Q2（低波動 − 高波動 > 0）：{ls.mean():+.2%}/季，t={tstat(ls):+.2f} → {'通過' if ls.mean() > 0 and tstat(ls) > T_CRIT else '未通過'}")
run(P, "vol3", "不設流動性限制")
run(P[P.liq_ok & P.big], "vol3", "只買大型股（市值前 50%）")
run(P[P.liq_ok], "vol12", "改用過去 12 個月波動")

# ---------- 累積報酬圖 ----------
fig, ax = plt.subplots(figsize=(10, 5.5))
idx = [(q + 1).end_time for q in R.index]
for c, lab, st in [(1, "低波動組（Q1）", "-"), (5, "高波動組（Q5）", "-"), (3, "中間組（Q3）", ":"),
                   ("0050含息", "0050（含息）", "--"), ("加權指數(不含息)", "加權指數（不含息）", "--")]:
    ax.plot(idx, (1 + R[c]).cumprod(), st, label=lab, lw=2 if st != ":" else 1.2)
ax.set_title("季頻高低波動選股 vs 大盤（等權、不扣手續費，2016–2025）")
ax.set_ylabel("累積淨值（起點 = 1）")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "quarterly_vol_cum.png", dpi=150)
R.to_csv(OUT + "quarterly_vol_returns.csv", encoding="utf-8-sig")
open(OUT + "quarterly_vol.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
