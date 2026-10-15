# -*- coding: utf-8 -*-
"""
漲跌方向檢驗的分層版：只看「真的有PTT情緒資訊」的股票/股票日。
  top20 / top50 : 只用訓練期(<=2022-06)留言量最高的K檔股票（訓練與測試都在這些股票內）
  days_eval     : 全部股票訓練，但只在「近3日留言>=20」的股票日上評估
  active_only   : 訓練/驗證/測試都只用「近3日留言>=20」的股票日
python direction_stratified.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SEEDS, BOOT, MIN3 = 8, 300, 20
df = pd.read_csv("final_dataset_267_stitched/all_stocks_final_dataset_267.csv")
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
C = df["Close"]

feat = pd.DataFrame(index=df.index)
feat["ret1"] = np.log(C / g["Close"].shift(1))
for k in (1, 2, 3, 4):
    feat[f"ret1_l{k}"] = feat["ret1"].groupby(df["stock_code"]).shift(k)
feat["ret5"] = np.log(C / g["Close"].shift(5))
feat["ret10"] = np.log(C / g["Close"].shift(10))
feat["gap"] = df["Open"] / g["Close"].shift(1) - 1
feat["range"] = (df["High"] - df["Low"]) / C
vol20 = g["Volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
feat["volr"] = np.log((df["Volume"] + 1) / (vol20 + 1))
for c in ("MA5", "MA30", "MA60", "EMA5", "EMA30", "EMA60"):
    feat[f"{c}_r"] = df[c] / C - 1
for c in ("MACD_6_15_6", "MACD_12_26_9", "MACD_30_60_30"):
    feat[f"{c}_r"] = df[c] / C
feat["MOM14_r"] = df["MOM14"] / C
feat["ADOSC_r"] = df["ADOSC_3_10"] / (vol20 + 1)
for c in ("RSI14", "WILLR14", "CMO14", "ULTOSC_7_14_28"):
    feat[c] = df[c]
BASE = list(feat.columns)

fac, att = df["daily_sentiment_factor"], np.log1p(df["total_comments"])
for k in (0, 1, 2):
    feat[f"fac_l{k}"] = fac.groupby(df["stock_code"]).shift(k) if k else fac
    feat[f"att_l{k}"] = att.groupby(df["stock_code"]).shift(k) if k else att
r3 = g["total_comments"].transform(lambda s: s.rolling(3, min_periods=1).sum())

nxt = g["Close"].shift(-1)
keep = nxt.notna() & (nxt != C)
y = (nxt > C).astype(int)
date = pd.to_datetime(df["Date"])
tr0 = keep & (date <= "2022-06-30")
va0 = keep & (date >= "2022-07-01") & (date <= "2022-12-31")
te0 = keep & (date >= "2023-01-01")

train_vol = df[date <= "2022-06-30"].groupby("stock_code")["total_comments"].sum().sort_values(ascending=False)
ARMS = {
    "A0_無情緒": BASE,
    "A1_lag0(可能洩漏)": BASE + ["fac_l0", "att_l0"],
    "A2_lag1": BASE + ["fac_l1", "att_l1"],
    "A3_lag2": BASE + ["fac_l2", "att_l2"],
}
rng = np.random.default_rng(0)
allres = {}


def run(name, tr, va, te, ev):
    """tr/va/te: 訓練/驗證/測試列；ev: 測試列中實際用來算指標的子集"""
    ev_idx = te & ev
    yev = y[ev_idx].values
    wk = date[ev_idx].dt.strftime("%G-%V").values
    uw = np.unique(wk)
    widx = {w: np.where(wk == w)[0] for w in uw}
    print(f"\n### {name}：train {tr.sum()}  val {va.sum()}  評估列 {ev_idx.sum()}  股票數 {df.loc[ev_idx, 'stock_code'].nunique()}  "
          f"上漲率 {yev.mean():.3f}", flush=True)
    boots = [np.concatenate([widx[w] for w in rng.choice(uw, size=len(uw), replace=True)]) for _ in range(BOOT)]
    seed_auc, probs = {}, {}
    for arm, cols in ARMS.items():
        aucs, ps = [], []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50 if tr.sum() > 30000 else 20,
                                  subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=16,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            p = m.predict_proba(feat.loc[ev_idx, cols])[:, 1]
            aucs.append(roc_auc_score(yev, p)); ps.append(p)
        seed_auc[arm], probs[arm] = np.array(aucs), np.mean(ps, axis=0)
    out = {"n_eval": int(ev_idx.sum()), "n_stocks": int(df.loc[ev_idx, "stock_code"].nunique()), "arms": {}}
    for arm in ARMS:
        e = {"auc": float(seed_auc[arm].mean())}
        if arm != "A0_無情緒":
            bd = np.array([roc_auc_score(yev[s], probs[arm][s]) - roc_auc_score(yev[s], probs["A0_無情緒"][s]) for s in boots])
            e["d_auc_seed_mean"] = float((seed_auc[arm] - seed_auc["A0_無情緒"]).mean())
            e["ci95"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
            print(f"  {arm:20s} AUC {e['auc']:.4f}   ΔAUC {e['d_auc_seed_mean']:+.4f}  95%CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]", flush=True)
        else:
            print(f"  {arm:20s} AUC {e['auc']:.4f}", flush=True)
        out["arms"][arm] = e
    allres[name] = out


for k in (20, 50):
    st = df["stock_code"].isin(set(train_vol.head(k).index))
    run(f"top{k}", tr0 & st, va0 & st, te0 & st, st)
active = r3 >= MIN3
run("days_eval(全部股票訓練，只在高討論日評估)", tr0, va0, te0, active)
run("active_only(只用高討論日訓練與評估)", tr0 & active, va0 & active, te0 & active, active)
json.dump(allres, open("direction_stratified_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n完成")
