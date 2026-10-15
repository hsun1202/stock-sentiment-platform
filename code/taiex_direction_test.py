# -*- coding: utf-8 -*-
"""
把256檔PTT情緒合併成「全市場」每日情緒(每天中位數708則留言，遠比單一個股密集)，
測試能不能預測大盤(TAIEX/^TWII)隔日漲跌，而不是個股——模仿Chen & Hsu(2025)用
大盤而非個股當預測目標的做法，用跟direction_test.py同一套嚴謹方法(XGBoost、多seed、
按週block bootstrap)驗證換題目後情緒因子是否還是測不出效果。
python taiex_direction_test.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

from stock_indicators import calculate_technical_indicators

SEEDS, BOOT = 20, 500

taiex = pd.read_csv("taiex_2020_2024.csv", header=[0, 1], index_col=0)
taiex.columns = [c[0] for c in taiex.columns]
taiex.index = pd.to_datetime(taiex.index)
taiex = taiex.sort_index()
tech = calculate_technical_indicators(taiex).reset_index().rename(columns={"index": "Date"})
tech["Date"] = pd.to_datetime(tech["Date"]).dt.strftime("%Y-%m-%d")

sent = pd.read_csv("market_daily_sentiment.csv")
df = tech.merge(sent, left_on="Date", right_on="resolved_date", how="left")
for c in ("total_comments", "positive_count", "negative_count", "market_sentiment_factor"):
    df[c] = df[c].fillna(0.0)
df = df.sort_values("Date").reset_index(drop=True)
print(f"交易日 {len(df)}  範圍 {df.Date.min()}~{df.Date.max()}", flush=True)

C = df["Close"]
feat = pd.DataFrame(index=df.index)
feat["ret1"] = np.log(C / C.shift(1))
for k in (1, 2, 3, 4):
    feat[f"ret1_l{k}"] = feat["ret1"].shift(k)
feat["ret5"] = np.log(C / C.shift(5))
feat["ret10"] = np.log(C / C.shift(10))
feat["gap"] = df["Open"] / C.shift(1) - 1
feat["range"] = (df["High"] - df["Low"]) / C
vol20 = df["Volume"].rolling(20, min_periods=10).mean()
feat["volr"] = np.log((df["Volume"] + 1) / (vol20 + 1))
for c in ("MA5", "MA30", "MA60", "EMA5", "EMA30", "EMA60"):
    feat[f"{c}_r"] = df[c] / C - 1
for c in ("MACD_6_15_6", "MACD_12_26_9", "MACD_30_60_30"):
    feat[f"{c}_r"] = df[c] / C
feat["MOM14_r"] = df["MOM14"] / C
for c in ("RSI14", "WILLR14", "CMO14", "ULTOSC_7_14_28"):
    feat[c] = df[c]
BASE = list(feat.columns)

att = np.log1p(df["total_comments"])
for k in (0, 1, 2):
    feat[f"fac_l{k}"] = df["market_sentiment_factor"].shift(k) if k else df["market_sentiment_factor"]
    feat[f"att_l{k}"] = att.shift(k) if k else att

nxt = C.shift(-1)
keep = nxt.notna() & (nxt != C) & feat[BASE].notna().all(axis=1)
y = (nxt > C).astype(int)
date = pd.to_datetime(df["Date"])
tr = keep & (date <= "2022-06-30")
va = keep & (date >= "2022-07-01") & (date <= "2022-12-31")
te = keep & (date >= "2023-01-01")
print(f"樣本：train {tr.sum()}  val {va.sum()}  test {te.sum()}", flush=True)
print(f"上漲基準率：test {y[te].mean():.4f}  -> 全猜多數類別Acc {max(y[te].mean(),1-y[te].mean()):.4f}", flush=True)

ARMS = {
    "A0_無情緒": BASE,
    "A1_lag0(同日)": BASE + ["fac_l0", "att_l0"],
    "A2_lag1": BASE + ["fac_l1", "att_l1"],
    "A3_lag2": BASE + ["fac_l2", "att_l2"],
}
yte = y[te].values
wk = date[te].dt.strftime("%G-%V").values
uw = np.unique(wk)
widx = {w: np.where(wk == w)[0] for w in uw}
rng = np.random.default_rng(0)
boots = [np.concatenate([widx[w] for w in rng.choice(uw, size=len(uw), replace=True)]) for _ in range(BOOT)]

seed_auc, seed_acc, probs = {}, {}, {}
for arm, cols in ARMS.items():
    aucs, accs, ps = [], [], []
    for s in range(SEEDS):
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5,
                              subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist",
                              n_jobs=16, early_stopping_rounds=30, eval_metric="logloss")
        m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
        p = m.predict_proba(feat.loc[te, cols])[:, 1]
        aucs.append(roc_auc_score(yte, p)); accs.append(accuracy_score(yte, p > 0.5)); ps.append(p)
    seed_auc[arm], seed_acc[arm], probs[arm] = np.array(aucs), np.array(accs), np.mean(ps, axis=0)
    print(f"{arm:20s} AUC {seed_auc[arm].mean():.4f}±{seed_auc[arm].std(ddof=1):.4f}  Acc {seed_acc[arm].mean():.4f}", flush=True)

out = {"n_test": int(te.sum()), "arms": {}}
print("\n=== 相對 A0 的差異 ===", flush=True)
for arm in ARMS:
    e = {"auc": float(seed_auc[arm].mean()), "acc": float(seed_acc[arm].mean())}
    if arm != "A0_無情緒":
        d_auc = seed_auc[arm] - seed_auc["A0_無情緒"]
        d_acc = seed_acc[arm] - seed_acc["A0_無情緒"]
        bd = np.array([roc_auc_score(yte[s], probs[arm][s]) - roc_auc_score(yte[s], probs["A0_無情緒"][s]) for s in boots])
        e["d_auc_seed_mean"] = float(d_auc.mean())
        e["d_acc_seed_mean"] = float(d_acc.mean())
        e["ci95_auc"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        print(f"{arm:20s} ΔAUC {e['d_auc_seed_mean']:+.4f}  95%CI [{e['ci95_auc'][0]:+.4f}, {e['ci95_auc'][1]:+.4f}]   ΔAcc {e['d_acc_seed_mean']:+.4f}", flush=True)
    out["arms"][arm] = e
json.dump(out, open("taiex_direction_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("完成")
