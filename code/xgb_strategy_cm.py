# -*- coding: utf-8 -*-
"""
拿 XGBoost 漲跌模型（技術面 vs +CMoney，cm3）去預測隔天開盤→收盤的漲跌與報酬，並做當沖回測。事先寫定見 PREREG_cmoney_full.md 第 C 節。
模型、特徵、切法同 xgb_direction_cm.py（測試 2022–2024、5 seeds）。
python cmoney_full/xgb_strategy_cm.py
"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.getcwd())
from features_v4 import build  # noqa: E402

CM_TAG = os.environ.get("CM_TAG", "cm3")
SEEDS, TEST_YEARS = 5, [2022, 2023, 2024]
COSTS = {"六折(0.321%)": 2 * 0.001425 * 0.6 + 0.0015, "無折扣(0.435%)": 2 * 0.001425 + 0.0015}
OUT = "cmoney_full/"
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


feat, meta, A = build()
feat_c, _, _ = build(f"final_dataset_v3_{CM_TAG}/all_stocks_v3.csv")
TECH = A["技術面"]
PTT = [c for c in A["+PTT(開盤前可得)"] if c not in TECH]
CM = [c.replace("ptt", "cm") for c in PTT]
feat[CM] = feat_c[PTT].values
ARMS = {"技術面": TECH, "+CMoney": TECH + CM, "+PTT": TECH + PTT}
raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "Open", "Close", "margin_eligible"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
assert (raw.stock_code.values == meta.stock_code.values).all()
g = raw.groupby("stock_code")
meta["r_oc"] = g.Close.shift(-1) / g.Open.shift(-1) - 1          # 隔天開盤→收盤
meta["short_ok"] = raw.margin_eligible.fillna(0).values == 1
keep, y, date, end = meta["valid"] & (meta.date >= "2020-01-01"), meta["y"], meta["date"], meta["end_date"]

rows = []
for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30")
    va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    out = meta.loc[te, ["stock_code", "date", "y", "r_oc", "short_ok"]].copy()
    for arm, cols in ARMS.items():
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=8,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        out["p_" + arm] = np.mean(ps, axis=0)
    rows.append(out)
    print(f"{Y} 完成", flush=True)
D = pd.concat(rows, ignore_index=True)
D.to_pickle(OUT + f"xgb_strategy_preds_{CM_TAG}.pkl")
say(f"資料集 {CM_TAG}｜測試 2022–2024：{len(D):,} 股票日、{D.date.nunique()} 個交易日、{D.stock_code.nunique()} 檔｜每天平均 {D.groupby('date').size().mean():.0f} 檔")
say(f"全部股票日的隔天開盤→收盤：平均 {D.r_oc.mean() * 1e4:+.1f} bp、上漲比例 {D.y.mean():.1%}")

# ---------- 1. 分 10 組 ----------
say("\n【1】依預測上漲機率分 10 組（每天各自分組）：隔天開盤→收盤平均報酬（bp）／實際上漲比例")
spread = {}
for arm in ARMS:
    q = D.groupby("date")["p_" + arm].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)) + 1
    D["q_" + arm] = q
    t = D.groupby(q).agg(r=("r_oc", "mean"), up=("y", "mean"), p=("p_" + arm, "mean"))
    say(f"  {arm:6s} 報酬：" + " ".join(f"{v * 1e4:+5.1f}" for v in t.r) + "｜上漲：" + " ".join(f"{v:.0%}" for v in t.up))
    say(f"  {'':6s} 平均預測機率：最低組 {t.p.iloc[0]:.3f} … 最高組 {t.p.iloc[-1]:.3f}")
    top = D[q == 10].groupby("date").r_oc.mean()
    bot = D[q == 1].groupby("date").r_oc.mean()
    spread[arm] = top - bot
    say(f"  {'':6s} 前10% − 後10%：{(top - bot).mean() * 1e4:+.1f} bp/天（t={tstat(top - bot):+.2f}）")
d = (spread["+CMoney"] - spread["技術面"]).dropna()
say(f"→ R1（+CMoney 的前後差 − 技術面的前後差 > 0）：{d.mean() * 1e4:+.2f} bp/天，t={tstat(d):+.2f} → {'通過' if d.mean() > 0 and tstat(d) > 1.96 else '未通過'}")

# ---------- 2. 當沖回測 ----------
say("\n【2】當沖回測（隔天開盤買、收盤賣；每天換一次；等權）")
mkt = D.groupby("date").r_oc.mean()
say(f"  對照：每天把全部股票都買（等權）毛 {mkt.mean() * 1e4:+.1f} bp/天")
res = {}
for arm in ARMS:
    q = D["q_" + arm]
    lo = D[q == 10].groupby("date").r_oc.mean()
    sh = D[(q == 1) & D.short_ok].groupby("date").r_oc.mean()
    ls = (lo - sh.reindex(lo.index).fillna(0)) / 2                     # 多空各半資金
    for name, r in [("只做多 前10%", lo), ("多空 前10%/後10%", ls)]:
        line = f"  {arm:6s} {name:14s} 毛 {r.mean() * 1e4:+6.1f} bp/天（t={tstat(r):+.2f}，賺錢天數 {(r > 0).mean():.0%}）"
        for cn, c in COSTS.items():
            net = r - c
            line += f"｜扣{cn} {net.mean() * 1e4:+6.1f} bp、累積 {(1 + net).prod() - 1:+.0%}"
        say(line)
        res[(arm, name)] = r
lo_c = res[("+CMoney", "只做多 前10%")]
net = lo_c - COSTS["六折(0.321%)"]
say(f"→ R2（+CMoney 只做多前10%，扣 0.321% 後 > 0）：{net.mean() * 1e4:+.1f} bp/天，t={tstat(net):+.2f} → {'通過' if net.mean() > 0 and tstat(net) > 1.96 else '未通過'}")
dd = (res[("+CMoney", "只做多 前10%")] - res[("技術面", "只做多 前10%")]).dropna()
say(f"  +CMoney 相對技術面（只做多前10%，毛）：{dd.mean() * 1e4:+.2f} bp/天（t={tstat(dd):+.2f}）")
be = lo_c.mean()
say(f"  損益兩平：+CMoney 只做多前10% 毛報酬 {be * 1e4:+.1f} bp，要成本低於這個才賺；目前最便宜的當沖成本 32.1 bp")

# ---------- 3. 分年 ----------
say("\n【3】分年（+CMoney，毛 bp/天）")
for Y in TEST_YEARS:
    m_ = lo_c.index.year == Y
    s_ = spread["+CMoney"][spread["+CMoney"].index.year == Y]
    say(f"  {Y}：只做多前10% {lo_c[m_].mean() * 1e4:+.1f}｜前後差 {s_.mean() * 1e4:+.1f}｜全體 {mkt[mkt.index.year == Y].mean() * 1e4:+.1f}")
open(OUT + f"xgb_strategy_{CM_TAG}.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
