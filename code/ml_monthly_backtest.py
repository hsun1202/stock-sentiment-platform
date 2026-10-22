# -*- coding: utf-8 -*-
"""
AI 選股（機器學習月頻）上線前驗證：照上線時的實際規格回測。事先寫定見 PREREG_ai_strategies.md A、B 節。
上市普通股、只用公開資料算得出的特徵、每月第一個交易日收盤出訊號、第二個交易日開盤成交、扣成本；測試 2018–2025。
另外：情緒過濾（剔除 CMoney 淨情緒最悲觀 20%），2020–2025。
cd D:/專題claude && python ai_strategies/ml_monthly_backtest.py
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
OUT = "ai_strategies/"
COST = 0.00471
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
    return (1 + r).prod() ** (12 / len(r)) - 1, r.std() * np.sqrt(12), r.mean() * 12 / (r.std() * np.sqrt(12)), (eq / eq.cummax() - 1).min()


def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(n).mean(), (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---------- 上市名單（T86 有出現的月份）----------
listed = {}
for f in sorted(glob.glob("chip_raw/t86/*.json")):
    ym = os.path.basename(f)[:6]
    j = json.load(open(f, encoding="utf-8"))
    if j.get("data"):
        listed.setdefault(ym, set()).update(r[0].strip() for r in j["data"])
say(f"[上市名單] {len(listed)} 個月，每月平均 {np.mean([len(v) for v in listed.values()]):.0f} 個代號")

# ---------- 日資料與特徵（全部只用到當天為止）----------
use = ["stock_code", "Date", "Open", "Close", "close_unadj", "Volume", "abnormal_price", "mktcap", "margin_bal", "short_bal", "inst_share"]
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df[df.stock_code.str.match(r"^[1-9]\d{3}$")].sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
df["r"] = g.Close.pct_change().where(~(abn | abn.groupby(df.stock_code).shift(1).fillna(False)))
df["dv"] = (df.close_unadj * df.Volume).where(df.Volume > 0)
df["vpos"] = df.Volume.where(df.Volume > 0)
df["rs"] = ((100 - df.inst_share) / 100).where(df.Volume > 0)
C = df.Close
sh = lambda s, k: s.groupby(df.stock_code).shift(k)  # noqa: E731
rl = lambda col, w, f, mp: g[col].transform(lambda x: getattr(x.rolling(w, min_periods=mp), f)())  # noqa: E731
df["mom1"], df["mom3"], df["mom6"] = C / sh(C, 21) - 1, C / sh(C, 63) - 1, C / sh(C, 126) - 1
df["mom12_1"] = sh(C, 21) / sh(C, 252) - 1
df["vol1m"], df["maxret"] = rl("r", 21, "std", 10), rl("r", 21, "max", 10)
df["logdv"] = np.log(rl("dv", 21, "median", 10))
df["liq60"] = rl("dv", 60, "median", 20)
df["voltrend"] = np.log(rl("vpos", 21, "mean", 10) / rl("vpos", 126, "mean", 60))
df["ma60r"] = C / rl("Close", 60, "mean", 40) - 1
df["rsi"] = g.Close.transform(rsi)
df["hi52"] = C / rl("Close", 250, "max", 120) - 1
df["smr"] = (df.short_bal / df.margin_bal.where(df.margin_bal > 0)).clip(0, 1)
df["margin_chg_m"] = (df.margin_bal / sh(df.margin_bal, 21).where(sh(df.margin_bal, 21) > 0) - 1).clip(-0.5, 0.5)
df["rs_m"] = rl("rs", 21, "mean", 10)
FEAT = ["mom1", "mom3", "mom6", "mom12_1", "vol1m", "maxret", "logdv", "voltrend", "ma60r", "rsi", "hi52", "smr", "margin_chg_m", "rs_m"]

# ---------- 每月第一、第二個交易日 ----------
cal = pd.Series(sorted(df.Date.unique()))
first = cal.groupby(cal.dt.to_period("M")).nth(0).reset_index(drop=True)
second = cal.groupby(cal.dt.to_period("M")).nth(1).reset_index(drop=True)
FD = pd.DataFrame({"ym": first.dt.to_period("M"), "fd": first.values}).merge(pd.DataFrame({"ym": second.dt.to_period("M"), "sd": second.values}), on="ym")
FD["sd_next"] = FD.sd.shift(-1)
FD = FD.dropna()
say(f"[期間] {FD.ym.iloc[0]}～{FD.ym.iloc[-1]}，{len(FD)} 個月")

S = df.set_index(["Date", "stock_code"])
op = df.pivot(index="Date", columns="stock_code", values="Open")
cl = df.pivot(index="Date", columns="stock_code", values="Close").ffill()
last_day = df.groupby("stock_code").Date.max()
data_end = df.Date.max()
rows = []
for _, m in FD.iterrows():
    snap = S.loc[m.fd].copy()
    snap = snap[snap.index.isin(listed.get(m.fd.strftime("%Y%m"), set()))]
    o0 = op.loc[m.sd].reindex(snap.index)
    o1 = op.loc[m.sd_next].reindex(snap.index)
    ret = o1 / o0 - 1
    gone = o1.isna() & (last_day.reindex(snap.index) < m.sd_next) & (last_day.reindex(snap.index) < data_end)
    lastpx = cl.loc[:m.sd_next].iloc[-1].reindex(snap.index)
    ret[gone] = (1 + (lastpx[gone] / o0[gone] - 1)) * 0.5 - 1          # 持有期間下市：最後價格再記 −50%
    snap["ret_next"] = ret.where(o0.notna())
    snap["ym"], snap["fd"] = m.ym, m.fd
    rows.append(snap.reset_index().rename(columns={"index": "stock_code"}))
P = pd.concat(rows, ignore_index=True)
P = P[P.ret_next.notna()].replace([np.inf, -np.inf], np.nan)
P["year"] = P.fd.dt.year
P["liq_ok"] = P.groupby("ym").liq60.rank(pct=True) > 0.2
P["tgt"] = P.groupby("ym").ret_next.rank(pct=True) - 0.5
say(f"[面板] {len(P):,} 股票×月、每月可買約 {P[P.liq_ok].groupby('ym').size().mean():.0f} 檔（上市、成交值非最低 20%）")

# ---------- 模型 ----------
P["score"] = np.nan
for Y in range(2018, 2026):
    tr, va, te = (P.year <= Y - 2), (P.year == Y - 1), (P.year == Y)
    ps = []
    for s in range(3):
        mdl = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=5, early_stopping_rounds=30, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
        mdl.fit(P.loc[tr, FEAT], P.loc[tr, "tgt"], eval_set=[(P.loc[va, FEAT], P.loc[va, "tgt"])], verbose=False)
        ps.append(mdl.predict(P.loc[te, FEAT]))
    P.loc[te, "score"] = np.mean(ps, axis=0)
U = P[P.liq_ok & P.score.notna()].copy()

# ---------- 情緒過濾：CMoney 當月淨情緒（上一個訊號日之後～這個訊號日收盤）----------
cm = pd.read_csv("final_dataset_v3_cm3x/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"])
cm["pos"], cm["neg"] = cm.ptt_pre_pos + cm.ptt_intra_pos, cm.ptt_pre_neg + cm.ptt_intra_neg
edges = list(FD.fd)
cm["fd"] = pd.cut(cm.Date, bins=[pd.Timestamp("1900-01-01")] + edges, labels=edges, right=True)
cmm = cm.dropna(subset=["fd"]).groupby(["stock_code", "fd"], observed=True)[["pos", "neg"]].sum().reset_index()
cmm["fd"] = pd.to_datetime(cmm.fd.astype(str))
U = U.merge(cmm, on=["stock_code", "fd"], how="left")
ok3 = (U.pos + U.neg) >= 3
U["cm_net"] = ((U.pos - U.neg) / (U.pos + U.neg)).where(ok3)
U["cm_q"] = U.groupby("ym").cm_net.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if x.notna().sum() >= 10 else np.nan)


def portfolio(sel_fn):
    out, held = {}, None
    for ym, z in U.groupby("ym"):
        h = sel_fn(z)
        s = set(h.stock_code)
        turn = 1.0 if held is None else len(s - held) / max(len(s), 1)
        out[ym] = (h.ret_next.mean(), turn)
        held = s
    r = pd.DataFrame(out, index=["gross", "turn"]).T
    return r.gross - r.turn * COST, r


top = lambda z: z[z.score.rank(pct=True, ascending=False) <= 0.2]  # noqa: E731
net, info = portfolio(top)
ew = U.groupby("ym").ret_next.mean()
cw = U.groupby("ym").apply(lambda z: np.average(z.ret_next, weights=z.mktcap.fillna(0) + 1e-9))
e = pd.read_csv("quarterly_vol/etf0050_adj.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
b50 = pd.Series({m.ym: e.asof(m.sd_next) / e.asof(m.sd) - 1 for _, m in FD.iterrows()})
idx = net.index
say("\n===== A. 機器學習月頻選股（上市、公開資料特徵、開盤成交、扣成本），2018–2025 =====")
say(f"  每月換股 {info.turn.iloc[1:].mean():.0%}｜每期約 {U.groupby('ym').size().mean() * 0.2:.0f} 檔")
for nm, r in [("AI 選股（扣成本）", net), ("一般股票（等權）", ew.reindex(idx)), ("市值加權", cw.reindex(idx)), ("0050 含息", b50.reindex(idx))]:
    a, v, shp, dd = perf(r)
    say(f"  {nm:<12} 年化 {a:+.1%}｜波動 {v:.1%}｜Sharpe {shp:.2f}｜最大回落 {dd:.1%}")
res = {}
for nm, b in [("一般股票", ew), ("市值加權", cw), ("0050 含息", b50)]:
    d = (net - b.reindex(idx)).dropna()
    res[nm] = d
    yr = d.groupby(pd.PeriodIndex(d.index).year).sum()
    say(f"  相對{nm}：{d.mean():+.2%}／月（t={t_(d):+.2f}，贏 {(d > 0).mean():.0%} 的月）｜分年：" + " ".join(f"{y}:{v:+.0%}" for y, v in yr.items()))
d1, d2 = res["一般股票"], res["0050 含息"]
say(f"\n→ M1（扣成本 − 一般股票 > 0）：{d1.mean():+.2%}／月，t={t_(d1):+.2f} → {'通過' if d1.mean() > 0 and t_(d1) > 1.96 else '未通過'}")
say(f"→ M2（扣成本 − 0050 含息 > 0）：{d2.mean():+.2%}／月，t={t_(d2):+.2f} → {'通過' if d2.mean() > 0 and t_(d2) > 1.96 else '未通過'}")
d25 = d2[pd.PeriodIndex(d2.index).year == 2025]
say(f"  2025 年：相對 0050 {d25.sum():+.1%}（12 個月加總）、相對一般股票 {d1[pd.PeriodIndex(d1.index).year == 2025].sum():+.1%}")

say("\n===== B. 情緒過濾：從前 20% 剔除 CMoney 最悲觀 20%（2020–2025）=====")
net_f, info_f = portfolio(lambda z: top(z)[lambda h: h.cm_q != 1])
mask = pd.PeriodIndex(idx).year >= 2020
removed = U[U.score.rank(pct=True, ascending=False) <= 0.2]
n_rm = top_rm = U.groupby("ym").apply(lambda z: int((top(z).cm_q == 1).sum()))
d3 = (net_f - net)[mask].dropna()
say(f"  每月平均剔除 {n_rm[mask].mean():.1f} 檔（前 20% 約 {U.groupby('ym').size().mean() * 0.2:.0f} 檔中）")
say(f"→ M3（剔除後 − 原本 > 0）：{d3.mean():+.3%}／月，t={t_(d3):+.2f}（{len(d3)} 個月） → {'通過' if d3.mean() > 0 and t_(d3) > 1.96 else '未通過'}")
d3y = d3[pd.PeriodIndex(d3.index).year == 2025]
say(f"  2025 年單獨：{d3y.mean():+.3%}／月（{len(d3y)} 個月，為正 {(d3y > 0).sum()}）")
pd.DataFrame({"ai_net": net, "ai_filter_net": net_f, "ew": ew.reindex(idx), "cw": cw.reindex(idx), "etf0050": b50.reindex(idx), "turn": info.turn}).to_csv(
    OUT + "ml_monthly_returns.csv", encoding="utf-8-sig")
open(OUT + "ml_monthly_backtest.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
