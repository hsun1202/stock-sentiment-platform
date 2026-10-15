# -*- coding: utf-8 -*-
"""
異常情緒/討論量偵測：不是每天都看情緒因子，只在「討論量相對該股近20日均值
出現異常尖峰」的日子才納入情緒特徵，其餘日子視為沒有訊號。邏輯：多數日子的
留言是雜訊，只有聲量異常放大時才可能反映真正的重大消息或群眾情緒轉向。

沿用 direction_test.py 同一套嚴謹方法：全體265檔池化、XGBoost、
train<=2022-06/val 2022-07~12/test 2023-2024、20 seeds、按週block bootstrap。

python spike_direction_test.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SEEDS, BOOT, Z_TH = 20, 500, 2.0
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
for c in ("RSI14", "WILLR14", "CMO14", "ULTOSC_7_14_28"):
    feat[c] = df[c]
BASE = list(feat.columns)

# --- 異常討論量偵測（用截至昨天為止的近20日均值/標準差，不看今天，避免洩漏）---
tc = df["total_comments"]
roll_mean = g["total_comments"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
roll_std = g["total_comments"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
z = ((tc - roll_mean) / roll_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
is_spike = (z > Z_TH).astype(float)
fac = df["daily_sentiment_factor"]
att = np.log1p(df["total_comments"])
print(f"z>{Z_TH} 尖峰天數: {int(is_spike.sum())} ({is_spike.mean()*100:.2f}%)  涵蓋股票 {df.loc[is_spike==1,'stock_code'].nunique()}", flush=True)

for k in (1, 2):
    feat[f"spike_l{k}"] = is_spike.groupby(df["stock_code"]).shift(k - 1) if k == 1 else is_spike.shift(k)
    # 尖峰當天的情緒/聲量，落後k天餵給模型；非尖峰天視為0(沒有訊號)
    sp = is_spike * fac
    at = is_spike * att
    feat[f"spike_fac_l{k}"] = sp.groupby(df["stock_code"]).shift(k)
    feat[f"spike_att_l{k}"] = at.groupby(df["stock_code"]).shift(k)
    feat[f"fac_l{k}"] = fac.groupby(df["stock_code"]).shift(k)
    feat[f"att_l{k}"] = att.groupby(df["stock_code"]).shift(k)

nxt = g["Close"].shift(-1)
keep = nxt.notna() & (nxt != C) & feat[BASE].notna().all(axis=1)
y = (nxt > C).astype(int)
date = pd.to_datetime(df["Date"])
tr = keep & (date <= "2022-06-30")
va = keep & (date >= "2022-07-01") & (date <= "2022-12-31")
te = keep & (date >= "2023-01-01")
print(f"樣本：train {tr.sum()}  val {va.sum()}  test {te.sum()}", flush=True)

ARMS = {
    "A0_無情緒": BASE,
    "A1_一般情緒lag1(對照)": BASE + ["fac_l1", "att_l1"],
    "A2_尖峰旗標l1(有沒有異常,不含方向)": BASE + ["spike_l1"],
    "A3_尖峰情緒l1(尖峰日的情緒方向)": BASE + ["spike_l1", "spike_fac_l1", "spike_att_l1"],
    "A4_尖峰情緒l2": BASE + ["spike_l2", "spike_fac_l2", "spike_att_l2"],
}
yte = y[te].values
wk = date[te].dt.strftime("%G-%V").values
uw = np.unique(wk)
widx = {w: np.where(wk == w)[0] for w in uw}
rng = np.random.default_rng(0)
boots = [np.concatenate([widx[w] for w in rng.choice(uw, size=len(uw), replace=True)]) for _ in range(BOOT)]

seed_auc, probs = {}, {}
for arm, cols in ARMS.items():
    aucs, ps = [], []
    for s in range(SEEDS):
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50,
                              subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist",
                              n_jobs=16, early_stopping_rounds=30, eval_metric="logloss")
        m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
        p = m.predict_proba(feat.loc[te, cols])[:, 1]
        aucs.append(roc_auc_score(yte, p)); ps.append(p)
    seed_auc[arm], probs[arm] = np.array(aucs), np.mean(ps, axis=0)
    print(f"{arm:34s} AUC {seed_auc[arm].mean():.4f}±{seed_auc[arm].std(ddof=1):.4f}", flush=True)

out = {"n_test": int(te.sum()), "spike_days": int(is_spike.sum()), "arms": {}}
print("\n=== 相對 A0 的差異(全體test集) ===", flush=True)
for arm in ARMS:
    e = {"auc": float(seed_auc[arm].mean())}
    if arm != "A0_無情緒":
        d_auc = seed_auc[arm] - seed_auc["A0_無情緒"]
        bd = np.array([roc_auc_score(yte[s], probs[arm][s]) - roc_auc_score(yte[s], probs["A0_無情緒"][s]) for s in boots])
        e["d_auc_seed_mean"] = float(d_auc.mean())
        e["ci95"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        print(f"{arm:34s} ΔAUC {e['d_auc_seed_mean']:+.4f}  95%CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]", flush=True)
    out["arms"][arm] = e

# --- 進一步：只在「昨天是尖峰日」的子集上比較 A0 vs A3 ---
sub = te & (feat["spike_l1"] == 1)
print(f"\n=== 只看『昨天是尖峰日』的子集 n={int(sub.sum())} ===", flush=True)
ys = y[sub].values
if sub.sum() >= 30:
    sub_auc = {}
    for arm in ("A0_無情緒", "A3_尖峰情緒l1(尖峰日的情緒方向)"):
        aucs = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=20,
                                  subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist",
                                  n_jobs=16, early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, ARMS[arm]], y[tr], eval_set=[(feat.loc[va, ARMS[arm]], y[va])], verbose=False)
            p = m.predict_proba(feat.loc[sub, ARMS[arm]])[:, 1]
            aucs.append(roc_auc_score(ys, p))
        sub_auc[arm] = np.array(aucs)
        print(f"  {arm:34s} AUC {sub_auc[arm].mean():.4f}±{sub_auc[arm].std(ddof=1):.4f}", flush=True)
    d = sub_auc["A3_尖峰情緒l1(尖峰日的情緒方向)"] - sub_auc["A0_無情緒"]
    out["spike_subset"] = {"n": int(sub.sum()), "d_auc_seed_mean": float(d.mean())}
    print(f"  子集內 ΔAUC(種子平均) {d.mean():+.4f}", flush=True)
else:
    print("  子集樣本太少，略過", flush=True)

json.dump(out, open("spike_direction_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n完成")
