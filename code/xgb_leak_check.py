# -*- coding: utf-8 -*-
"""
檢查 xgb_direction_cm.py 的 +CMoney 是否來自抽樣洩漏（事後追加的檢查，2026-10-19，看到 X1 通過後才寫——要如實標註）。
疑慮：CMoney 爬蟲每個交易日只抓「收盤前最後 20 篇」。t+1 開盤前的貼文有沒有被抓到，取決於 t+1 盤中貼文多不多（盤中熱 → 開盤前的被擠掉），
     而 t+1 盤中熱不熱鬧跟 t+1 漲跌有關 → 「t+1 開盤前」特徵偷看到了 t+1 的資訊。PTT 是全量抓，沒有這個問題。
做法：
  1. 診斷：同一股票日，「t+1 開盤前被抓到的則數」與「t+1 盤中則數」的相關——CMoney 若為負（被擠掉）、PTT 為正，就是抽樣造成的。
  2. 拆開：+CMoney（只用 t 收盤前）／+CMoney（只用 t+1 開盤前）／原本合併版；PTT 同樣拆。乾淨的是「只用 t 收盤前」。
python cmoney_full/xgb_leak_check.py
"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())
from features_v4 import build  # noqa: E402

SEEDS, BOOT, TEST_YEARS = 5, 300, [2022, 2023, 2024]
CM_TAG = os.environ.get("CM_TAG", "cm2")          # cm2＝原本；cm3＝補抓開盤前後（rebuild_cm3.py）
SFX = "" if CM_TAG == "cm2" else "_" + CM_TAG
feat, meta, A = build()
TECH = A["技術面"]
L = []


def say(s):
    print(s, flush=True)
    L.append(s)


def add(path, tag):
    df = pd.read_csv(path, dtype={"stock_code": str}, usecols=["stock_code", "Date"] + [f"ptt_{a}_{b}" for a in ("pre", "intra") for b in ("n", "pos", "neg")])
    df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
    assert (df.stock_code.values == meta.stock_code.values).all()
    nx = lambda c: df.groupby("stock_code")[c].shift(-1).fillna(0)  # noqa: E731
    roll = lambda s: s.groupby(df.stock_code).transform(lambda x: x.rolling(5, min_periods=1).mean())  # noqa: E731
    parts = {"close": (df.ptt_pre_n + df.ptt_intra_n, df.ptt_pre_pos + df.ptt_intra_pos, df.ptt_pre_neg + df.ptt_intra_neg),
             "nextpre": (nx("ptt_pre_n"), nx("ptt_pre_pos"), nx("ptt_pre_neg"))}
    cols = {}
    for k, (n, p, q) in parts.items():
        t = f"{tag}_{k}"
        feat[f"{t}_fac"] = np.where(p + q > 0, (p - q) / (p + q).replace(0, 1), np.nan)
        feat[f"{t}_att"] = np.log1p(n)
        feat[f"{t}_att5"] = roll(np.log1p(n))
        cols[k] = [f"{t}_fac", f"{t}_att", f"{t}_att5"]
    m = (meta.date >= "2020-01-01") & (meta.date <= "2024-12-30")
    a, b = np.log1p(nx("ptt_pre_n"))[m], np.log1p(nx("ptt_intra_n"))[m]
    hot = df.groupby("stock_code").ptt_intra_n.transform("mean")[m] >= df.ptt_intra_n[m].quantile(0.8)
    say(f"[診斷 {tag}] corr(t+1 開盤前則數, t+1 盤中則數)：全部 {np.corrcoef(a, b)[0, 1]:+.3f}｜熱門股 {np.corrcoef(a[hot], b[hot])[0, 1]:+.3f}｜冷門股 {np.corrcoef(a[~hot], b[~hot])[0, 1]:+.3f}")
    return cols


P_ = add("final_dataset_v3/all_stocks_v3.csv", "ptt")
C_ = add(f"final_dataset_v3_{CM_TAG}/all_stocks_v3.csv", "cm")
ARMS = {"技術面": TECH,
        "+CMoney 只用t收盤前（乾淨）": TECH + C_["close"], "+CMoney 只用t+1開盤前（疑慮）": TECH + C_["nextpre"],
        "+CMoney 合併（原版）": TECH + C_["close"] + C_["nextpre"],
        "+PTT 只用t收盤前": TECH + P_["close"], "+PTT 合併": TECH + P_["close"] + P_["nextpre"]}
keep, y, date, end = meta["valid"] & (meta.date >= "2020-01-01"), meta["y"], meta["date"], meta["end_date"]
allp = []
for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30")
    va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
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
    wk = date[te].dt.strftime("%G-%V").values
    allp.append(pd.DataFrame({"wk": [f"{Y}_{w}" for w in wk], "y": y[te].values, **P}))
    yte = y[te].values
    say(f"{Y}：技術面 {roc_auc_score(yte, P['技術面']):.4f}｜" + "｜".join(f"{a} {roc_auc_score(yte, P[a]) - roc_auc_score(yte, P['技術面']):+.4f}" for a in list(ARMS)[1:]))
D = pd.concat(allp, ignore_index=True)
u = D.wk.unique()
wi = {w: np.where(D.wk.values == w)[0] for w in u}
rs = np.random.default_rng(0)
boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
yy = D.y.values
say("--- 2022-2024 合併 ΔAUC（相對技術面，依週 bootstrap 95% 區間）---")
for arm in list(ARMS)[1:]:
    d = roc_auc_score(yy, D[arm]) - roc_auc_score(yy, D["技術面"])
    bd = [roc_auc_score(yy[b], D[arm].values[b]) - roc_auc_score(yy[b], D["技術面"].values[b]) for b in boots]
    lo, hi = np.percentile(bd, [2.5, 97.5])
    say(f"  {arm:24s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
open(f"cmoney_full/xgb_leak_check{SFX}.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
