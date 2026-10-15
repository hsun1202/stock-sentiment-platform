# -*- coding: utf-8 -*-
"""
訓練集大小會不會影響結果？（學習曲線）
  目標：隔天開盤→收盤漲跌（同 retail_test_v4）；測試年 2022、2023、2024；驗證期＝前一年下半年。
  訓練期一律到「前一年 6 月底」為止，往前取 1 年、2 年、4 年、全部（2015 起）四種長度。
  比較：技術面 AUC、加 PTT 的 ΔAUC、加融資融券的 ΔAUC；模型 XGBoost（2 seeds）與邏輯迴歸。
  若「資料越多 → PTT 增量越大」，代表訓練資料不夠是問題；若不隨資料量變化，則不是。
python learning_curve_test.py
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
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
ARMS = {"技術面": TECH, "+PTT": TECH + PTT, "+融資融券": TECH + MARGIN}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
WINDOWS = [1, 2, 4, "全部"]
TEST_YEARS = [2022, 2023, 2024]
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def fit_predict(model, cols, tr, va, te):
    if model == "XGB":
        ps = []
        for s in range(2):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=8,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        return np.mean(ps, axis=0)
    imp = SimpleImputer(strategy="median").fit(feat.loc[tr, cols])
    sc = StandardScaler().fit(imp.transform(feat.loc[tr, cols]))
    f = lambda m: np.clip(sc.transform(imp.transform(feat.loc[m, cols])), -5, 5)
    return LogisticRegression(C=0.1, max_iter=300).fit(f(tr), y[tr]).predict_proba(f(te))[:, 1]


rows = []
for Y in TEST_YEARS:
    va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    for w in WINDOWS:
        start = "2015-01-01" if w == "全部" else f"{Y - 1 - w}-07-01"
        tr = keep & (end <= f"{Y-1}-06-30") & (date >= start)
        for model in ["XGB", "LR"]:
            auc = {a: roc_auc_score(y[te], fit_predict(model, cols, tr, va, te)) for a, cols in ARMS.items()}
            rows.append({"Y": Y, "w": str(w), "model": model, "n_train": int(tr.sum()), "tech": auc["技術面"],
                         "d_ptt": auc["+PTT"] - auc["技術面"], "d_margin": auc["+融資融券"] - auc["技術面"]})
            say(f"  {Y} 訓練{w}{'年' if w != '全部' else ''}（{int(tr.sum()):,} 筆）{model}: 技術面 {auc['技術面']:.4f}"
                f"｜+PTT {auc['+PTT'] - auc['技術面']:+.4f}｜+融資融券 {auc['+融資融券'] - auc['技術面']:+.4f}")
R = pd.DataFrame(rows)
say("\n================ 學習曲線（2022~2024 三個測試年平均）================")
for model in ["XGB", "LR"]:
    say(f"\n--- 模型 {model} ---")
    for w in [str(x) for x in WINDOWS]:
        d = R[(R.model == model) & (R.w == w)]
        say(f"  訓練 {w:<3s}{'年' if w != '全部' else ''}：平均 {d.n_train.mean():>9,.0f} 筆｜技術面 AUC {d.tech.mean():.4f}"
            f"｜+PTT ΔAUC {d.d_ptt.mean():+.4f}（為正 {(d.d_ptt > 0).sum()}/3）｜+融資融券 ΔAUC {d.d_margin.mean():+.4f}（為正 {(d.d_margin > 0).sum()}/3）")
R.to_csv("learning_curve_test.csv", index=False, encoding="utf-8-sig")
open("learning_curve_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
