# -*- coding: utf-8 -*-
"""
邏輯迴歸版（最簡單的基準）：與 retail_test_v4.py（XGBoost）完全相同的特徵、目標（隔天開盤→收盤漲跌）、
逐年滾動（測試年 2018~2024）、週 block bootstrap，只把模型換成邏輯迴歸，方便與 XGBoost、LSTM 並排比較。
  前處理：inf→缺值，訓練集中位數補值，標準化後截尾到 ±5；
  超參數：L2 懲罰強度 C 用驗證期（前一年下半年）AUC 在 {0.001, 0.01, 0.1, 1} 中選。
python logreg_direction_v4.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features_v4 import build

BOOT, TEST_YEARS = 300, range(2018, 2025)
CS = [0.001, 0.01, 0.1, 1.0]
feat, meta, A = build()
feat = feat.replace([np.inf, -np.inf], np.nan)
TECH, PTT = A["技術面"], [c for c in A["+PTT(開盤前可得)"] if c not in A["技術面"]]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
ARMS = {"技術面": TECH, "+散戶說(PTT)": TECH + PTT, "+散戶做(融資融券)": TECH + MARGIN,
        "+散戶說+做": TECH + PTT + MARGIN, "+法人(對照)": TECH + INST}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]


def prep(cols, tr):
    imp = SimpleImputer(strategy="median").fit(feat.loc[tr, cols])
    sc = StandardScaler().fit(imp.transform(feat.loc[tr, cols]))
    return lambda m: np.clip(sc.transform(imp.transform(feat.loc[m, cols])), -5, 5)


lines, res = [], {}
for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    yte = y[te].values; wk = date[te].dt.strftime("%G-%V").values; u = np.unique(wk)
    wi = {w: np.where(wk == w)[0] for w in u}; rs = np.random.default_rng(Y)
    boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
    P, bestC = {}, {}
    for arm, cols in ARMS.items():
        f = prep(cols, tr)
        Xtr, Xva, Xte = f(tr), f(va), f(te)
        best = max(CS, key=lambda c: roc_auc_score(y[va], LogisticRegression(C=c, max_iter=300).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]))
        bestC[arm] = best
        P[arm] = LogisticRegression(C=best, max_iter=300).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    base = roc_auc_score(yte, P["技術面"]); res[Y] = {"base": base}
    line = f"{Y}: 技術面 {base:.4f}"
    for arm in list(ARMS)[1:]:
        d = roc_auc_score(yte, P[arm]) - base
        bd = np.array([roc_auc_score(yte[b], P[arm][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
        lo = float(np.percentile(bd, 2.5)); res[Y][arm] = {"d": d, "lo": lo, "hi": float(np.percentile(bd, 97.5))}
        line += f"｜{arm} {d:+.4f}{'✔' if lo > 0 else ''}"
    print(line, f"（C={bestC['技術面']}）", flush=True); lines.append(line)
bases = np.array([res[Y]["base"] for Y in TEST_YEARS])
lines.append(f"--- 7年彙總 --- 技術面 平均AUC {bases.mean():.4f}")
for arm in list(ARMS)[1:]:
    ds = np.array([res[Y][arm]["d"] for Y in TEST_YEARS]); sig = sum(res[Y][arm]["lo"] > 0 for Y in TEST_YEARS)
    lines.append(f"{arm:14s} 平均ΔAUC {ds.mean():+.4f}（t={ds.mean()/(ds.std(ddof=1)/np.sqrt(7)):.2f}）｜顯著 {sig}/7｜為正 {(ds>0).sum()}/7")
print("\n".join(lines[-6:]))
open("logreg_direction_v4.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump({str(k): v for k, v in res.items()}, open("logreg_direction_v4_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
