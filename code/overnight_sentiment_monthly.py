# -*- coding: utf-8 -*-
"""
用「隔夜報酬」當散戶情緒指標（不靠文字、每檔每天都有），在月頻、全市場檢驗。
文獻：Aboody, Even-Tov, Lehavy & Trueman (2018, JFQA) 隔夜報酬是個股層級的情緒指標，短期延續、長期反轉；
      Zhang, Tsai, Weng & Tsai (2023, PBFJ) 台股也成立，且散戶成交比重越高越明顯；Lou, Polk & Skouras (2019, JFE) 隔夜/盤中的拉鋸。
資料：final_dataset_v3_tejall25（TEJ 全市場、還原開盤價與收盤價）。隔夜＝開盤/前一日收盤−1；盤中＝收盤/開盤−1；都減掉當天全市場等權平均（超額）。
月特徵：on_m＝當月累積隔夜超額、id_m＝當月累積盤中超額、on_3m＝近 3 個月累積隔夜超額、on_share＝當月隔夜為正的天數比例。
面板、模型、成本、下市處理 同 monthly_fullmarket_strategy.py。

事先寫定的主要檢定（2026-10-12，跑之前寫下；2 個，Bonferroni |t|>2.26）：
  O1：當月累積隔夜超額（on_m）與下個月報酬的月 Rank IC < 0（情緒推高的股票之後較差；2015~2025、全市場）。
  O2：把四個隔夜特徵加進月頻選股模型（技術面＋散戶行為），前 20% 組合扣成本後的月報酬高於原模型（逐月配對差 > 0，2018~2025）。
  次要（方向先寫定）：on_m 的效果在「散戶成交比重高」的股票較強（Zhang et al.）；控制當月總報酬（mom1）後 on_m 仍為負；
        on_m 高的股票下個月「隔夜續高、盤中續低」（拉鋸）；原 265 檔與另外 889 檔分開；依 on_m 分五組的下月報酬。
python overnight_sentiment_monthly.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
LOG = []
C_DISC, XSEEDS = 0.00471, 3


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


use = ["stock_code", "Date", "Open", "Close", "close_unadj", "Volume", "abnormal_price", "MA60", "RSI14", "mktcap",
       "margin_bal", "short_margin_ratio", "inst_share", "credit_share", "in265"]
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
okd = ~(abn | abn.groupby(df.stock_code).shift(1).fillna(False))
df["r"] = g.Close.pct_change().where(okd)
tr_ = df.Volume > 0
df["on"] = (df.Open / g.Close.shift(1) - 1).where(okd & tr_)
df["idr"] = (df.Close / df.Open - 1).where(okd & tr_)
for c in ("on", "idr"):
    df["x" + c] = df[c] - df.groupby("Date")[c].transform("mean")
df["on_pos"] = (df.on > 0).where(df.on.notna())
df["dv"] = (df.close_unadj * df.Volume).where(tr_)
df["vpos"] = df.Volume.where(tr_)
df["rs"] = ((100 - df.inst_share) / 100).where(tr_)
df["cr"] = (df.credit_share / 100).where(tr_)
df["ym"] = df.Date.dt.to_period("M")
last = df[df.Close.notna()].drop_duplicates(["stock_code", "ym"], keep="last").set_index(["stock_code", "ym"])[
    ["Close", "MA60", "RSI14", "mktcap", "margin_bal", "short_margin_ratio", "in265"]]
agg = df.groupby(["stock_code", "ym"]).agg(vol1m=("r", "std"), maxret=("r", "max"), n_r=("r", "count"), dv_med=("dv", "median"), vavg=("vpos", "mean"),
                                           rs_m=("rs", "mean"), cr_m=("cr", "mean"), n_tr=("vpos", "count"),
                                           on_m=("xon", "sum"), id_m=("xidr", "sum"), on_share=("on_pos", "mean"), n_on=("xon", "count"))
P = last.join(agg).reset_index().sort_values(["stock_code", "ym"]).reset_index(drop=True)
P["mi"] = P.ym.dt.year * 12 + P.ym.dt.month


def lag(s, k):
    return s.groupby(P.stock_code).shift(k).where(P.mi.groupby(P.stock_code).shift(k) == P.mi - k)


C = P.Close
P["mom1"], P["mom3"], P["mom6"] = C / lag(C, 1) - 1, C / lag(C, 3) - 1, C / lag(C, 6) - 1
P["mom12_1"] = lag(C, 1) / lag(C, 12) - 1
P["hi52"] = C / C.groupby(P.stock_code).transform(lambda x: x.rolling(12, min_periods=6).max()) - 1
P.loc[P.n_r < 10, ["vol1m", "maxret"]] = np.nan
P.loc[P.n_on < 10, ["on_m", "id_m", "on_share"]] = np.nan
P["on_3m"] = P.on_m + lag(P.on_m, 1) + lag(P.on_m, 2)
P["logdv"] = np.log(P.dv_med)
P["voltrend"] = np.log(P.vavg / (sum(lag(P.vavg, k) for k in range(1, 7)) / 6))
P["ma60r"], P["rsi"] = C / P.MA60 - 1, P.RSI14
mb = lag(P.margin_bal, 1)
P["margin_chg_m"] = (P.margin_bal / mb.where(mb > 0) - 1).clip(-0.5, 0.5)
P["smr"] = P.short_margin_ratio.clip(0, 1)
P.loc[P.n_tr < 10, ["rs_m", "cr_m"]] = np.nan
last_mi, data_end = P.groupby("stock_code").mi.transform("max"), P.mi.max()
nxt_ok = P.mi.groupby(P.stock_code).shift(-1) == P.mi + 1
P["ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1).where(nxt_ok)
P.loc[P.ret_next.isna() & (P.mi == last_mi) & (P.mi < data_end), "ret_next"] = -0.5
gap = P.ret_next.isna() & (P.mi < last_mi)
P.loc[gap, "ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1)[gap]
P["on_next"], P["id_next"] = lag(P.on_m, -1), lag(P.id_m, -1)
P["year"] = P.ym.dt.year
P = P.replace([np.inf, -np.inf], np.nan)
P["liq_ok"] = P.groupby("ym").dv_med.rank(pct=True) > 0.2
P["tgt"] = P.groupby("ym").ret_next.rank(pct=True) - 0.5
valid = P.ret_next.notna() & P.tgt.notna()
tstat = lambda s: pd.Series(s).dropna().mean() / (pd.Series(s).dropna().std(ddof=1) / np.sqrt(pd.Series(s).dropna().size))
V = P[valid & P.on_m.notna()].copy()
say(f"[面板] 股票×月 {len(V):,}（{V.ym.min()}~{V.ym.max()}）；當月累積隔夜超額 中位數 {V.on_m.median():+.2%}、標準差 {V.on_m.std():.2%}；與當月總報酬(mom1)的相關 {V.on_m.corr(V.mom1):+.2f}、與盤中累積的相關 {V.on_m.corr(V.id_m):+.2f}")


def ic_series(d, col, y="ret_next"):
    return pd.Series({ym: spearmanr(z[col], z[y])[0] for ym, z in d.dropna(subset=[col, y]).groupby("ym") if len(z) >= 50})


say("\n===== 1) 單一變數：當月隔夜／盤中累積超額 → 下個月報酬（月 Rank IC）=====")
for col, nm in [("on_m", "當月累積隔夜超額★O1"), ("id_m", "當月累積盤中超額"), ("on_3m", "近3月累積隔夜超額"), ("on_share", "隔夜為正的天數比例"), ("mom1", "當月總報酬（參考）")]:
    ic = ic_series(V, col)
    h1, h2 = ic[ic.index.year <= 2019], ic[ic.index.year >= 2020]
    say(f"  {nm:<16s} IC {ic.mean():+.4f}（t={tstat(ic):+.2f}，{len(ic)} 個月）｜2015-19 {h1.mean():+.4f}（t={tstat(h1):+.1f}）｜2020-25 {h2.mean():+.4f}（t={tstat(h2):+.1f}）")
ic1 = ic_series(V, "on_m")
say(f"  → O1：IC {ic1.mean():+.4f}，t={tstat(ic1):+.2f} → {'通過' if (ic1.mean() < 0 and tstat(ic1) < -2.26) else '未通過'}")
for lab, d in [("原 265 檔", V[V.in265 == 1]), ("另外 889 檔", V[V.in265 == 0])]:
    ic = ic_series(d, "on_m")
    say(f"    {lab}：IC {ic.mean():+.4f}（t={tstat(ic):+.2f}）")
V["q"] = V.groupby("ym").on_m.transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
qm = V.groupby(["ym", "q"]).ret_next.mean().unstack()
say("  依當月隔夜累積分五組，下個月報酬：" + " ".join(f"{['最低', '2', '3', '4', '最高'][k]} {qm[k].mean():+.2%}" for k in range(5)) + f"｜最高−最低 {(qm[4] - qm[0]).mean():+.2%}/月（t={tstat(qm[4] - qm[0]):+.2f}）")

say("\n===== 2) 次要 =====")
fm = []
for ym, z in V.dropna(subset=["on_m", "id_m", "mom1", "vol1m", "logdv"]).groupby("ym"):
    if len(z) < 100:
        continue
    X = z[["on_m", "id_m", "vol1m", "logdv"]].rank(pct=True) - 0.5
    fm.append(sm.OLS(z.ret_next.rank(pct=True) - 0.5, sm.add_constant(X)).fit().params)
fm = pd.DataFrame(fm)
say("  每月橫斷面迴歸（排名對排名，Fama-MacBeth）：" + "｜".join(f"{c} {fm[c].mean():+.4f}(t={tstat(fm[c]):+.2f})" for c in ["on_m", "id_m", "vol1m", "logdv"]))
V["rs_g"] = V.groupby("ym").rs_m.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False) if v.notna().sum() >= 60 else np.nan)
parts = []
for k, lab in [(0, "散戶成交比重低"), (1, "中"), (2, "高")]:
    ic = ic_series(V[V.rs_g == k], "on_m")
    parts.append((lab, ic))
say("  依散戶成交比重分三組的 on_m IC：" + "｜".join(f"{lab} {ic.mean():+.4f}(t={tstat(ic):+.1f})" for lab, ic in parts)
    + f"｜高−低 {(parts[2][1] - parts[0][1]).mean():+.4f}（t={tstat(parts[2][1] - parts[0][1]):+.2f}）")
i1, i2 = ic_series(V, "on_m", "on_next"), ic_series(V, "on_m", "id_next")
say(f"  拉鋸：當月隔夜累積 → 下月隔夜累積 IC {i1.mean():+.4f}（t={tstat(i1):+.1f}）；→ 下月盤中累積 IC {i2.mean():+.4f}（t={tstat(i2):+.1f}）")

say("\n===== 3) 加進月頻選股模型（2018~2025，前 20% 等權、扣 0.471%×換股）=====")
TECH = ["mom1", "mom3", "mom6", "mom12_1", "vol1m", "maxret", "logdv", "voltrend", "ma60r", "rsi", "hi52"]
RETAIL = ["smr", "margin_chg_m", "cr_m", "rs_m"]
ON = ["on_m", "id_m", "on_3m", "on_share"]
ARMS = {"基準（技術面＋散戶行為）": TECH + RETAIL, "＋隔夜情緒★": TECH + RETAIL + ON, "只有技術面": TECH, "技術面＋隔夜情緒": TECH + ON}
pred = {a: pd.Series(np.nan, index=P.index) for a in ARMS}
for Y in range(2018, 2026):
    tr, va, te = valid & (P.year <= Y - 2), valid & (P.year == Y - 1), valid & (P.year == Y)
    for arm, cols in ARMS.items():
        ps = []
        for s in range(XSEEDS):
            m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5, early_stopping_rounds=30, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
            m.fit(P.loc[tr, cols], P.loc[tr, "tgt"], eval_set=[(P.loc[va, cols], P.loc[va, "tgt"])], verbose=False)
            ps.append(m.predict(P.loc[te, cols]))
        pred[arm][te] = np.mean(ps, axis=0)
U = P[valid & (P.year >= 2018) & P.liq_ok].copy()
months = sorted(U.ym.unique())
bench = U.groupby("ym").ret_next.mean()
net, ics = {}, {}
for arm in ARMS:
    U["p"] = pred[arm][U.index]
    top = U[U.groupby("ym").p.rank(pct=True) > 0.8]
    gross = top.groupby("ym").ret_next.mean()
    hold = top.groupby("ym").stock_code.apply(set)
    turn = pd.Series({m_: (1.0 if i == 0 else len(hold[m_] - hold[months[i - 1]]) / len(hold[m_])) for i, m_ in enumerate(months)})
    net[arm] = gross - turn * C_DISC
    ics[arm] = U.groupby("ym").apply(lambda z: spearmanr(z.p, z.ret_next)[0])
    ex = net[arm] - bench
    say(f"  {arm:<14s} 月 IC {ics[arm].mean():+.4f}｜年化 {(1 + net[arm]).prod() ** (12 / len(ex)) - 1:+.1%}｜相對等權 {ex.mean():+.2%}/月（t={tstat(ex):+.2f}）｜2025 年相對等權（年化）{ex[ex.index.year == 2025].mean() * 12:+.1%}")
b0 = "基準（技術面＋散戶行為）"
d, di = net["＋隔夜情緒★"] - net[b0], ics["＋隔夜情緒★"] - ics[b0]
say(f"  ＋隔夜情緒 − 基準：淨報酬 {d.mean():+.3%}/月（t={tstat(d):+.2f}，為正的月數 {(d > 0).mean():.0%}）｜IC {di.mean():+.4f}（t={tstat(di):+.2f}）")
d2 = net["技術面＋隔夜情緒"] - net["只有技術面"]
say(f"  （只有技術面的模型上）＋隔夜情緒：淨報酬 {d2.mean():+.3%}/月（t={tstat(d2):+.2f}）")
say(f"  → O2：淨報酬差 {d.mean():+.3%}/月，t={tstat(d):+.2f} → {'通過' if (d.mean() > 0 and tstat(d) > 2.26) else '未通過'}")
open("overnight_sentiment_monthly.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
