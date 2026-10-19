# -*- coding: utf-8 -*-
"""
XGBoost 隔天開盤→收盤漲跌：CMoney 貼文情緒 vs PTT，同一批 265 檔、同一段 2020-2024。
設定同 retail_test_v4.py（features_v4.build 的技術面特徵與目標 B、XGB 超參數、依週 bootstrap），只改：
  只用 2020 以後的資料（CMoney 只有這段）、測試年 2022-2024、多一組 +CMoney。事先寫定的假說見 cmoney_full/PREREG_cmoney_full.md。
python cmoney_full/xgb_direction_cm.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())
from features_v4 import build  # noqa: E402

SEEDS, BOOT, TEST_YEARS = 5, 300, [2022, 2023, 2024]
OUT = "cmoney_full/"
feat, meta, A = build()
feat_c, meta_c, _ = build("final_dataset_v3_cm2/all_stocks_v3.csv")
assert (meta.stock_code.values == meta_c.stock_code.values).all() and (meta.date.values == meta_c.date.values).all()
TECH = A["技術面"]
PTT = [c for c in A["+PTT(開盤前可得)"] if c not in TECH]
CM = [c.replace("ptt", "cm") for c in PTT]
feat[CM] = feat_c[PTT].values
ARMS = {"技術面": TECH, "+PTT": TECH + PTT, "+CMoney": TECH + CM, "+PTT+CMoney": TECH + PTT + CM}
keep, y, date, end = meta["valid"] & (meta.date >= "2020-01-01"), meta["y"], meta["date"], meta["end_date"]
lines, res, allp = [], {}, []
lines.append(f"情緒特徵：PTT {PTT}；CMoney {CM}｜樣本 {int(keep.sum()):,} 股票日、{meta.stock_code[keep].nunique()} 檔")
for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30")
    va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    yte = y[te].values
    wk = date[te].dt.strftime("%G-%V").values
    u = np.unique(wk)
    wi = {w: np.where(wk == w)[0] for w in u}
    rs = np.random.default_rng(Y)
    boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
    P = {}
    for arm, cols in ARMS.items():
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=8,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        P[arm] = np.mean(ps, axis=0)
    allp.append(pd.DataFrame({"Y": Y, "wk": [f"{Y}_{w}" for w in wk], "y": yte, **P}))
    base = roc_auc_score(yte, P["技術面"])
    res[Y] = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "auc_tech": base}
    line = f"{Y}（訓練 {int(tr.sum()):,}／測試 {int(te.sum()):,}）：技術面 AUC {base:.4f}"
    for arm in list(ARMS)[1:]:
        d = roc_auc_score(yte, P[arm]) - base
        bd = np.array([roc_auc_score(yte[b], P[arm][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
        lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
        res[Y][arm] = {"d": d, "lo": lo, "hi": hi}
        line += f"｜{arm} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    print(line, flush=True)
    lines.append(line)

# 三年合併：把三年的測試樣本接起來算 AUC 差，依週 bootstrap
D = pd.concat(allp, ignore_index=True)
u = D.wk.unique()
wi = {w: np.where(D.wk.values == w)[0] for w in u}
rs = np.random.default_rng(0)
boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
yy = D.y.values
lines.append("--- 2022-2024 合併 ---")
lines.append(f"技術面 AUC {roc_auc_score(yy, D['技術面']):.4f}")
for arm in list(ARMS)[1:]:
    d = roc_auc_score(yy, D[arm]) - roc_auc_score(yy, D["技術面"])
    bd = np.array([roc_auc_score(yy[b], D[arm].values[b]) - roc_auc_score(yy[b], D["技術面"].values[b]) for b in boots])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    res["pooled_" + arm] = {"d": d, "lo": lo, "hi": hi}
    lines.append(f"{arm:12s} ΔAUC {d:+.4f}  95%區間 [{lo:+.4f}, {hi:+.4f}]  三年為正 {sum(res[Y][arm]['d'] > 0 for Y in TEST_YEARS)}/3")
p = res["pooled_+CMoney"]
lines.append(f"→ X1（+CMoney ΔAUC > 0，區間不含 0）：{p['d']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] → {'通過' if p['lo'] > 0 else '未通過'}")
print("\n".join(lines[-7:]))
open(OUT + "xgb_direction_cm.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump({str(k): v for k, v in res.items()}, open(OUT + "xgb_direction_cm_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=float)
