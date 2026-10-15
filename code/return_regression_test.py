# -*- coding: utf-8 -*-
"""
改成「預測報酬大小（漲跌幅 %）」而不是只猜漲跌方向，也不預測股價水準（原論文作法）
======================================================================
資料：final_dataset_v3（265檔，2015-2024）；特徵同 features_v4.py。逐年滾動（測試年 2018~2024；訓練≤前一年6月、驗證前一年下半年）。
兩個目標（只用當時已知的資訊）：
  cc ＝ 隔天漲跌幅（收盤→隔天收盤）：技術面(t)、融資融券/法人用前一天(t-1)公布的、PTT 只到 t 收盤前的留言
  oc ＝ 隔天開盤→收盤：技術面(t)、融資融券/法人(t 盤後公布)、PTT 到 t+1 開盤前的留言（同 retail_test_v4）
模型：XGBoost 迴歸（2 seeds，訓練目標截尾 1%/99%）、Ridge（線性對照）
評估（樣本外）：
  R²_oos（相對「預測 0」；金融文獻常用，>0 即優於不預測）、每日橫斷面 Rank IC（預測與實際報酬的排名相關）、
  正負號猜對率、每日預測前20%−後20% 的平均報酬（bp，毛報酬）；並算「加某訊號 − 只有技術面」的增量（逐日配對 t）。
輸出：return_regression_test.log；預測存 return_preds/（可重用）
python return_regression_test.py
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features_v4 import build

warnings.filterwarnings("ignore")
SEEDS, YEARS = 2, list(range(2018, 2025))
os.makedirs("return_preds", exist_ok=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


feat, meta, A = build()
feat = feat.replace([np.inf, -np.inf], np.nan)
TECH = A["技術面"]
PTT_OPEN = [c for c in A["+PTT(開盤前可得)"] if c not in TECH]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
date = meta["date"]

raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                  usecols=["stock_code", "Date", "Open", "Close", "abnormal_price", "ptt_pre_n", "ptt_pre_pos",
                           "ptt_pre_neg", "ptt_intra_n", "ptt_intra_pos", "ptt_intra_neg"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
assert (raw.stock_code.values == meta.stock_code.values).all() and (raw.Date.values == meta.date.values).all()
g = raw.groupby("stock_code")
# PTT 只到 t 收盤前（給 cc 用）
n = raw.ptt_pre_n.fillna(0) + raw.ptt_intra_n.fillna(0)
pos = raw.ptt_pre_pos.fillna(0) + raw.ptt_intra_pos.fillna(0)
neg = raw.ptt_pre_neg.fillna(0) + raw.ptt_intra_neg.fillna(0)
feat["pttc_fac"] = np.where(pos + neg > 0, (pos - neg) / (pos + neg).replace(0, 1), np.nan)
feat["pttc_att"] = np.log1p(n)
feat["pttc_att5"] = feat.pttc_att.groupby(raw.stock_code).transform(lambda x: x.rolling(5, min_periods=1).mean())
PTT_CLOSE = ["pttc_fac", "pttc_att", "pttc_att5"]
for c in MARGIN + INST:  # 前一天公布的（給 cc 用）
    feat[c + "_l1"] = feat[c].groupby(raw.stock_code).shift(1)
MARGIN_L1 = [c + "_l1" for c in MARGIN]
INST_L1 = [c + "_l1" for c in INST]

# 目標
abn = raw.abnormal_price.fillna(0) != 0
abn_next = g.abnormal_price.shift(-1).fillna(0) != 0
nO, nC = g.Open.shift(-1), g.Close.shift(-1)
r_oc = nC / nO - 1
r_cc = nC / raw.Close - 1
tech_ok = feat[TECH].notna().all(axis=1)
valid_cc = r_cc.notna() & ~abn & ~abn_next & tech_ok
TARGETS = {
    "cc": (valid_cc, r_cc, {"技術面": TECH, "+融資融券": TECH + MARGIN_L1, "+PTT": TECH + PTT_CLOSE, "+法人(對照)": TECH + INST_L1}),
    "oc": (meta["valid"], r_oc, {"技術面": TECH, "+融資融券": TECH + MARGIN, "+PTT": TECH + PTT_OPEN, "+法人(對照)": TECH + INST}),
}
end = meta["end_date"]


def fit_predict(model, cols, tr, va, te, ytr, yva):
    if model == "XGB":
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                 colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=8,
                                 early_stopping_rounds=30, verbosity=0)
            m.fit(feat.loc[tr, cols], ytr, eval_set=[(feat.loc[va, cols], yva)], verbose=False)
            ps.append(m.predict(feat.loc[te, cols]))
        return np.mean(ps, axis=0)
    imp = SimpleImputer(strategy="median").fit(feat.loc[tr, cols])
    sc = StandardScaler().fit(imp.transform(feat.loc[tr, cols]))
    f = lambda m: np.clip(sc.transform(imp.transform(feat.loc[m, cols])), -5, 5)
    Xtr, Xva, Xte = f(tr), f(va), f(te)
    best = min([1, 100, 1e4, 1e6], key=lambda a: np.mean((Ridge(alpha=a).fit(Xtr, ytr).predict(Xva) - yva) ** 2))
    return Ridge(alpha=best).fit(Xtr, ytr).predict(Xte)


def metrics(d):
    """d: DataFrame(date, r, f)"""
    r2 = 1 - ((d.r - d.f) ** 2).sum() / (d.r ** 2).sum()
    ic = d.groupby("date").apply(lambda x: spearmanr(x.f, x.r)[0] if len(x) >= 30 else np.nan).dropna()
    nz = d[d.r != 0]
    hit = (np.sign(nz.f) == np.sign(nz.r)).mean()
    q = d.groupby("date").f.rank(pct=True)
    ls = d[q > 0.8].groupby("date").r.mean() - d[q <= 0.2].groupby("date").r.mean()
    return r2, ic, hit, ls


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


res = {}
for tgt, (keep, r, ARMS) in TARGETS.items():
    for model in ["Ridge", "XGB"]:
        store = {a: [] for a in ARMS}
        for Y in YEARS:
            path = f"return_preds/{tgt}_{model}_{Y}.pkl"
            tr = keep & (end <= f"{Y-1}-06-30")
            va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
            te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
            if os.path.exists(path):
                P = pickle.load(open(path, "rb"))
            else:
                lo, hi = r[tr].quantile(0.01), r[tr].quantile(0.99)
                ytr, yva = r[tr].clip(lo, hi), r[va].clip(lo, hi)
                P = {a: fit_predict(model, cols, tr, va, te, ytr, yva) for a, cols in ARMS.items()}
                P["idx"] = np.where(te.values)[0]
                pickle.dump(P, open(path, "wb"))
            for a in ARMS:
                store[a].append(pd.DataFrame({"date": date.values[P["idx"]], "r": r.values[P["idx"]], "f": P[a]}))
            say(f"  [完成] {tgt} {model} {Y}")
        res[(tgt, model)] = {a: metrics(pd.concat(v)) for a, v in store.items()}

say("\n================ 樣本外結果（2018~2024）================")
NAMES = {"cc": "隔天漲跌幅（收盤→收盤）", "oc": "隔天開盤→收盤"}
for tgt in TARGETS:
    for model in ["Ridge", "XGB"]:
        R = res[(tgt, model)]
        say(f"\n--- 目標：{NAMES[tgt]}　模型：{model} ---")
        r2b, icb, hitb, lsb = R["技術面"]
        for a, (r2, ic, hit, ls) in R.items():
            line = (f"  {a:<10s} R²_oos {r2*100:+.3f}%｜Rank IC {ic.mean():+.4f}（t={tstat(ic):+.1f}）｜正負號猜對 {hit:.1%}"
                    f"｜前20%−後20% {ls.mean()*1e4:+.1f} bp/天（t={tstat(ls):+.1f}）")
            if a != "技術面":
                dic = (ic - icb).dropna()
                dls = (ls - lsb).dropna()
                line += f"｜ΔIC {dic.mean():+.4f}（t={tstat(dic):+.2f}）、Δ多空 {dls.mean()*1e4:+.2f} bp（t={tstat(dls):+.2f}）"
            say(line)
say("\n判讀：R²_oos 在日報酬上通常只有 0~1%；Rank IC 0.02~0.05 即算有用；台股當沖來回成本約 32~44 bp。")
open("return_regression_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
