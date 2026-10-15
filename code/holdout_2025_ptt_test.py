# -*- coding: utf-8 -*-
"""
真正的樣本外：用 2015-2024 訓練（2024 下半年當驗證/early stopping），預測 2025 全年（模型從沒看過 2025）
資料：final_dataset_v3_2025/all_stocks_v3.csv（build_dataset_v3_2025.py；2015-2024 與原資料列數相同）
PTT 2025 年沒有資料 → 只比較 技術面 / +融資融券 / +法人(對照)
  1) 方向：隔天開盤→收盤漲跌；XGBoost（3 seeds）與邏輯迴歸；AUC、ΔAUC（週 bootstrap）
  2) 報酬：隔天開盤→收盤報酬；XGBoost 迴歸；每日 Rank IC、前20%−後20%（bp）
  3) 交易規則「預測漲就買、預測跌就賣」（當沖，等權），扣 32 / 44 bp 成本
python holdout_2025_test.py
"""
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
PATH = "final_dataset_v3_2025_ptt/all_stocks_v3.csv"
feat, meta, A = build(path=PATH)
feat = feat.replace([np.inf, -np.inf], np.nan)
TECH = A["技術面"]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
PTT = [c for c in A["+PTT(開盤前可得)"] if c not in TECH]
ARMS = {"技術面": TECH, "+PTT": TECH + PTT, "+融資融券": TECH + MARGIN, "只有PTT": PTT}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
raw = pd.read_csv(PATH, dtype={"stock_code": str}, usecols=["stock_code", "Date", "Open", "Close"], parse_dates=["Date"]
                  ).sort_values(["stock_code", "Date"]).reset_index(drop=True)
assert (raw.stock_code.values == meta.stock_code.values).all()
g = raw.groupby("stock_code")
r_oc = g.Close.shift(-1) / g.Open.shift(-1) - 1

tr = keep & (end <= "2024-06-30")
va = keep & (date >= "2024-07-01") & (end <= "2024-12-31")
te = keep & (date >= "2025-01-01") & (date <= "2025-12-31")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


say(f"訓練 2015~2024/06：{int(tr.sum()):,} 筆｜驗證 2024 下半年：{int(va.sum()):,}｜測試 2025：{int(te.sum()):,} 筆"
    f"（{date[te].min().date()}~{date[te].max().date()}，{date[te].nunique()} 個交易日）")


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
half = np.where(dte.dt.month.values <= 6, "上半年", "下半年")
wk = dte.dt.strftime("%G-%V").values
u = np.unique(wk); wi = {w: np.where(wk == w)[0] for w in u}; rs = np.random.default_rng(0)
boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(500)]
PX = {}
for model in ["XGB", "LR"]:
    P = {a: clf(model, cols) for a, cols in ARMS.items()}
    PX[model] = P
    base = roc_auc_score(yte, P["技術面"])
    say(f"\n--- 1) 方向（隔天開盤→收盤）{model} ---")
    say(f"  技術面 AUC {base:.4f}（上半年 {roc_auc_score(yte[half == '上半年'], P['技術面'][half == '上半年']):.4f}、"
        f"下半年 {roc_auc_score(yte[half == '下半年'], P['技術面'][half == '下半年']):.4f}）｜方向猜對率 {((P['技術面'] > 0.5) == (yte == 1)).mean():.1%}")
    for a in list(ARMS)[1:]:
        d = roc_auc_score(yte, P[a]) - base
        bd = np.array([roc_auc_score(yte[b], P[a][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
        hs = [roc_auc_score(yte[half == h], P[a][half == h]) - roc_auc_score(yte[half == h], P["技術面"][half == h]) for h in ("上半年", "下半年")]
        say(f"  {a:<10s} ΔAUC {d:+.4f} [{np.percentile(bd, 2.5):+.4f},{np.percentile(bd, 97.5):+.4f}]｜上半年 {hs[0]:+.4f}、下半年 {hs[1]:+.4f}")
say("  （對照：逐年滾動 2024 年 XGB：技術面 0.5477、+融資融券 +0.0036、+法人 +0.0023）")

# 2) 報酬
rtr, rva = r_oc[tr], r_oc[va]
lo, hi = rtr.quantile(0.01), rtr.quantile(0.99)
rte = r_oc[te].values
say("\n--- 2) 報酬（隔天開盤→收盤）XGBoost 迴歸 ---")
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
    r2 = 1 - ((d.r - d.f) ** 2).sum() / (d.r ** 2).sum()
    line = f"  {a:<10s} R²_oos {r2*100:+.2f}%｜Rank IC {ic.mean():+.4f}（t={ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))):+.1f}）｜前20%−後20% {ls.mean()*1e4:+.1f} bp/天"
    if a != "技術面":
        dd = (ic - ics["技術面"]).dropna()
        line += f"｜ΔIC {dd.mean():+.4f}（t={dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))):+.2f}）"
    say(line)

# 3) 交易規則
say("\n--- 3) 「預測漲就買、預測跌就賣」（XGB 機率 > 0.5＝預測漲；隔天開盤進場、收盤出場；等權）---")
df = pd.DataFrame({"date": dte.values, "r": rte})
for a in ["技術面", "+融資融券"]:
    p = PX["XGB"][a]
    ok = ~np.isnan(rte)
    pos = np.where(p > 0.5, rte, -rte)
    for c, cn in [(0, "不扣成本"), (0.0032, "扣32bp"), (0.0044, "扣44bp")]:
        daily = pd.Series(pos[ok] - c).groupby(df.date.values[ok]).mean()
        say(f"  {a:<8s} {cn:<5s} 日均 {daily.mean()*1e4:+6.1f} bp｜2025 全年累積 {(1 + daily).prod() - 1:+7.1%}")
ew = pd.read_csv(PATH, dtype={"stock_code": str}, usecols=["stock_code", "Date", "Close"], parse_dates=["Date"])
ew = ew[ew.Date.dt.year == 2025].sort_values(["stock_code", "Date"])
bh = ew.groupby("stock_code").Close.agg(lambda s: s.iloc[-1] / s.iloc[0] - 1).mean()
say(f"  （參考：2025 年 267 檔等權買進持有 {bh:+.1%}；全體隔天開盤→收盤平均 {np.nanmean(rte)*1e4:+.1f} bp/天）")
open("holdout_2025_ptt_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
