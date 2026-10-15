# -*- coding: utf-8 -*-
"""
隔日漲跌方向預測：情緒因子有沒有增量資訊？（比舊版更嚴謹）
  - 特徵皆為無單位比值/報酬（不使用整段樣本min-max縮放，避免洩漏）
  - 所有股票合併訓練、依日期切分（train<=2022-06, val 2022-07~12, test 2023-01~2024-12）
  - 情緒分不同落後天數：lag0(同日,可能含事後留言=洩漏診斷), lag1, lag2；另有「只有討論量」對照
  - 指標：AUC、log-loss、Accuracy(對照多數類別基準)；20組種子；按週block bootstrap估信賴區間
python direction_test.py --input final_dataset_267_stitched/all_stocks_final_dataset_267.csv
"""
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="final_dataset_267_stitched/all_stocks_final_dataset_267.csv")
ap.add_argument("--out", default="direction_test_results.json")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--boot", type=int, default=500)
args = ap.parse_args()

df = pd.read_csv(args.input)
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

fac = df["daily_sentiment_factor"]
att = np.log1p(df["total_comments"])
for k in (0, 1, 2):
    feat[f"fac_l{k}"] = fac.groupby(df["stock_code"]).shift(k) if k else fac
    feat[f"att_l{k}"] = att.groupby(df["stock_code"]).shift(k) if k else att

nxt = g["Close"].shift(-1)
keep = nxt.notna() & (nxt != C)
y = (nxt > C).astype(int)

date = pd.to_datetime(df["Date"])
tr = keep & (date <= "2022-06-30")
va = keep & (date >= "2022-07-01") & (date <= "2022-12-31")
te = keep & (date >= "2023-01-01")
print(f"樣本：train {tr.sum()}  val {va.sum()}  test {te.sum()}   股票數 {df.stock_code.nunique()}")
print(f"上漲基準率：train {y[tr].mean():.4f}  test {y[te].mean():.4f}  -> 全猜多數類別的Accuracy = {max(y[te].mean(), 1 - y[te].mean()):.4f}")

ARMS = {
    "A0_無情緒": BASE,
    "A1_情緒lag0(同日,可能洩漏)": BASE + ["fac_l0", "att_l0"],
    "A2_情緒lag1": BASE + ["fac_l1", "att_l1"],
    "A3_情緒lag2": BASE + ["fac_l2", "att_l2"],
    "A4_只有討論量lag1": BASE + ["att_l1"],
}
yte = y[te].values
weeks = date[te].dt.strftime("%G-%V").values
uw = np.unique(weeks)
week_idx = {w: np.where(weeks == w)[0] for w in uw}

probs, seed_metrics = {}, {}
for name, cols in ARMS.items():
    ps, ms = [], []
    for seed in range(args.seeds):
        m = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist",
            n_jobs=16, early_stopping_rounds=30, eval_metric="logloss",
        )
        m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
        p = m.predict_proba(feat.loc[te, cols])[:, 1]
        ps.append(p)
        ms.append({"auc": roc_auc_score(yte, p), "logloss": log_loss(yte, p), "acc": accuracy_score(yte, p > 0.5)})
    probs[name] = np.mean(ps, axis=0)
    seed_metrics[name] = ms
    a = np.array([r["auc"] for r in ms]); ll = np.array([r["logloss"] for r in ms]); ac = np.array([r["acc"] for r in ms])
    print(f"{name:32s} AUC {a.mean():.4f}±{a.std(ddof=1):.4f}  logloss {ll.mean():.4f}  Acc {ac.mean():.4f}", flush=True)

rng = np.random.default_rng(0)
base_name = "A0_無情緒"
res = {"base_rate_test": float(yte.mean()), "arms": {}}
print("\n=== 相對 A0 的 AUC 差異（>0 代表加了資訊比較好） ===")
boots = []
for _ in range(args.boot):
    sel = np.concatenate([week_idx[w] for w in rng.choice(uw, size=len(uw), replace=True)])
    boots.append(sel)
for name in ARMS:
    a_seed = np.array([r["auc"] for r in seed_metrics[name]])
    a0_seed = np.array([r["auc"] for r in seed_metrics[base_name]])
    d_seed = a_seed - a0_seed
    entry = {"auc_mean": float(a_seed.mean()), "logloss_mean": float(np.mean([r["logloss"] for r in seed_metrics[name]])),
             "acc_mean": float(np.mean([r["acc"] for r in seed_metrics[name]])), "seed_diff_mean": float(d_seed.mean()),
             "seed_diff_se": float(d_seed.std(ddof=1) / np.sqrt(len(d_seed)))}
    if name != base_name:
        bd = np.array([roc_auc_score(yte[s], probs[name][s]) - roc_auc_score(yte[s], probs[base_name][s]) for s in boots])
        entry["boot_diff_ci95"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        entry["boot_diff_mean"] = float(bd.mean())
        print(f"{name:32s} ΔAUC(種子平均) {d_seed.mean():+.4f}   按週bootstrap 95%CI [{entry['boot_diff_ci95'][0]:+.4f}, {entry['boot_diff_ci95'][1]:+.4f}]", flush=True)
    res["arms"][name] = entry
json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("完成")
