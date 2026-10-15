# -*- coding: utf-8 -*-
"""
「訓練 5 年、預測 5 年」：2015-2019 訓練一次（2019 下半年當驗證/early stopping），模型固定，直接預測 2020-2024
與逐年滾動（每年重訓，retail_test_v4 / return_regression_test）對照。
  方向：隔天開盤→收盤漲跌，XGBoost（3 seeds）與邏輯迴歸，AUC 與「加某訊號 − 只有技術面」ΔAUC（週 bootstrap）
  報酬：隔天開盤→收盤報酬，XGBoost 迴歸（2 seeds），每日 Rank IC 與前20%−後20%（bp）
python split5_test.py
"""
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features_v4 import build

warnings.filterwarnings("ignore")
feat, meta, A = build()
feat = feat.replace([np.inf, -np.inf], np.nan)
TECH = A["技術面"]
PTT = [c for c in A["+PTT(開盤前可得)"] if c not in TECH]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
ARMS = {"技術面": TECH, "+PTT": TECH + PTT, "+融資融券": TECH + MARGIN, "+法人(對照)": TECH + INST}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "Open", "Close"],
                  parse_dates=["Date"]).sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = raw.groupby("stock_code")
r_oc = g.Close.shift(-1) / g.Open.shift(-1) - 1

tr = keep & (end <= "2019-06-30")
va = keep & (date >= "2019-07-01") & (end <= "2019-12-31")
te = keep & (date >= "2020-01-01") & (date <= "2024-12-31") & (end <= "2025-01-31")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


say(f"訓練 2015-01~2019-06：{int(tr.sum()):,} 筆｜驗證 2019 下半年：{int(va.sum()):,}｜測試 2020-2024：{int(te.sum()):,}（模型固定不重訓）")


def prep(cols):
    imp = SimpleImputer(strategy="median").fit(feat.loc[tr, cols])
    sc = StandardScaler().fit(imp.transform(feat.loc[tr, cols]))
    return lambda m: np.clip(sc.transform(imp.transform(feat.loc[m, cols])), -5, 5)


def clf(model, cols):
    if model == "XGB":
        ps = []
        for s in range(3):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=8,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        return np.mean(ps, axis=0)
    f = prep(cols)
    return LogisticRegression(C=0.1, max_iter=300).fit(f(tr), y[tr]).predict_proba(f(te))[:, 1]


yte = y[te].values
dte = date[te]
yr = dte.dt.year.values
wk = dte.dt.strftime("%G-%V").values
u = np.unique(wk); wi = {w: np.where(wk == w)[0] for w in u}; rs = np.random.default_rng(0)
boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(300)]
out = {}
for model in ["XGB", "LR"]:
    P = {a: clf(model, cols) for a, cols in ARMS.items()}
    base = roc_auc_score(yte, P["技術面"])
    say(f"\n--- 方向（隔天開盤→收盤）模型 {model} ---")
    say(f"  技術面 AUC {base:.4f}｜各年 " + " ".join(f"{Y}:{roc_auc_score(yte[yr == Y], P['技術面'][yr == Y]):.4f}" for Y in range(2020, 2025)))
    out[model] = {}
    for a in list(ARMS)[1:]:
        d = roc_auc_score(yte, P[a]) - base
        bd = np.array([roc_auc_score(yte[b], P[a][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
        per = [roc_auc_score(yte[yr == Y], P[a][yr == Y]) - roc_auc_score(yte[yr == Y], P["技術面"][yr == Y]) for Y in range(2020, 2025)]
        out[model][a] = {"d": d, "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)), "per_year": per}
        say(f"  {a:<10s} ΔAUC {d:+.4f} [{np.percentile(bd, 2.5):+.4f},{np.percentile(bd, 97.5):+.4f}]｜各年 "
            + " ".join(f"{Y}:{v:+.4f}" for Y, v in zip(range(2020, 2025), per)) + f"｜為正 {sum(v > 0 for v in per)}/5")

# 逐年滾動（每年重訓）的同期間對照：retail_test_v4（XGB 5 seeds）
wf = json.load(open("retail_test_v4_results.json", encoding="utf-8"))
mp = {"+PTT": "+散戶說(PTT)", "+融資融券": "+散戶做(融資融券)", "+法人(對照)": "+法人(對照)"}
say("\n--- 對照：逐年滾動（每年重訓，XGB）2020-2024 平均 ΔAUC ---")
for a, k in mp.items():
    v = [wf[str(Y)][k]["d"] for Y in range(2020, 2025)]
    say(f"  {a:<10s} {np.mean(v):+.4f}｜各年 " + " ".join(f"{Y}:{x:+.4f}" for Y, x in zip(range(2020, 2025), v)))

# 報酬（迴歸）
rtr, rva = r_oc[tr], r_oc[va]
lo, hi = rtr.quantile(0.01), rtr.quantile(0.99)
rte = r_oc[te].values
say("\n--- 報酬（隔天開盤→收盤）XGBoost 迴歸 ---")
ics = {}
for a, cols in ARMS.items():
    ps = []
    for s in range(2):
        m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                             colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=8, early_stopping_rounds=30, verbosity=0)
        m.fit(feat.loc[tr, cols], rtr.clip(lo, hi), eval_set=[(feat.loc[va, cols], rva.clip(lo, hi))], verbose=False)
        ps.append(m.predict(feat.loc[te, cols]))
    d = pd.DataFrame({"date": dte.values, "r": rte, "f": np.mean(ps, axis=0)}).dropna()
    ic = d.groupby("date").apply(lambda x: spearmanr(x.f, x.r)[0]).dropna()
    q = d.groupby("date").f.rank(pct=True)
    ls = d[q > 0.8].groupby("date").r.mean() - d[q <= 0.2].groupby("date").r.mean()
    ics[a] = ic
    line = f"  {a:<10s} Rank IC {ic.mean():+.4f}｜前20%−後20% {ls.mean()*1e4:+.1f} bp/天"
    if a != "技術面":
        dd = (ic - ics["技術面"]).dropna()
        line += f"｜ΔIC {dd.mean():+.4f}（t={dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))):+.2f}）"
    say(line)
open("split5_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
