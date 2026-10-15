# -*- coding: utf-8 -*-
"""
十年資料逐年滾動：「散戶情緒」兩種量法 vs 法人，對個股隔天開盤→收盤漲跌的預測力（同 direction_test_v4 的設定）
  散戶說的：PTT（開盤前可得）；散戶做的：融資融券（融資增減、融券增減、券資比；t 盤後公布）
  對照：法人（外資、投信、自營商買賣超/成交量；t 盤後公布）
python retail_test_v4.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from features_v4 import build

SEEDS, BOOT, TEST_YEARS = 5, 300, range(2018, 2025)
feat, meta, A = build()
TECH, PTT = A["技術面"], [c for c in A["+PTT(開盤前可得)"] if c not in A["技術面"]]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
ARMS = {"技術面": TECH, "+散戶說(PTT)": TECH + PTT, "+散戶做(融資融券)": TECH + MARGIN,
        "+散戶說+做": TECH + PTT + MARGIN, "+法人(對照)": TECH + INST}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
lines, res = [], {}
for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    yte = y[te].values; wk = date[te].dt.strftime("%G-%V").values; u = np.unique(wk)
    wi = {w: np.where(wk == w)[0] for w in u}; rs = np.random.default_rng(Y)
    boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
    P = {}
    for arm, cols in ARMS.items():
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=16,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        P[arm] = np.mean(ps, axis=0)
    base = roc_auc_score(yte, P["技術面"]); res[Y] = {}
    line = f"{Y}: 技術面 {base:.4f}"
    for arm in list(ARMS)[1:]:
        d = roc_auc_score(yte, P[arm]) - base
        bd = np.array([roc_auc_score(yte[b], P[arm][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
        lo = float(np.percentile(bd, 2.5)); res[Y][arm] = {"d": d, "lo": lo, "hi": float(np.percentile(bd, 97.5))}
        line += f"｜{arm} {d:+.4f}{'✔' if lo > 0 else ''}"
    print(line, flush=True); lines.append(line)
lines.append("--- 7年彙總 ---")
for arm in list(ARMS)[1:]:
    ds = np.array([res[Y][arm]["d"] for Y in TEST_YEARS]); sig = sum(res[Y][arm]["lo"] > 0 for Y in TEST_YEARS)
    lines.append(f"{arm:14s} 平均ΔAUC {ds.mean():+.4f}（t={ds.mean()/(ds.std(ddof=1)/np.sqrt(7)):.2f}）｜顯著 {sig}/7｜為正 {(ds>0).sum()}/7")
print("\n".join(lines[-6:]))
open("retail_test_v4.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump({str(k): v for k, v in res.items()}, open("retail_test_v4_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
