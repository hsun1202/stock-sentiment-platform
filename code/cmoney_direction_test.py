# -*- coding: utf-8 -*-
"""
用CMoney已爬到的74檔資料做同日vs落後情緒的漲跌方向檢驗，
驗證CMoney有精確時間戳（createTime）是否能避開PTT那種「同日訊號、落後就消失」的
時間洩漏疑慮。

做法：
  - 只用每檔股票「實際被爬到」的日期範圍(cmoney_267_clean的min~max createDate)，
    範圍內沒有文章的日子視為「爬到但沒人討論」(sentiment=0)，範圍外不算(避免假0)
  - 特徵沿用 direction_test.py 的無單位報酬/比值設計(不用漲跌敏感度低的收盤價迴歸)
  - train<=2022-06 / val 2022-07~12 / test 2023-2024 (跟PTT版一致，才能比較)
python cmoney_direction_test.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SEEDS, BOOT = 10, 300
TECH_DIR = "technical_output_267"
cov = pd.read_csv("cmoney_coverage_windows.csv")

sent_frames = []
for _, row in cov.iterrows():
    from glob import glob
    fs = glob(f"sentiment_output_cmoney/{row['code']}_*_sentiment.csv")
    if fs:
        sent_frames.append(pd.read_csv(fs[0]))
sent = pd.concat(sent_frames)
daily_sent = sent.groupby(["stock_code", "resolved_date"], as_index=False).agg(
    article_count=("article_id", "count"), total_comments=("total_comments", "sum"),
    positive_count=("positive_count", "sum"), negative_count=("negative_count", "sum"),
)
daily_sent["daily_sentiment_factor"] = (
    (daily_sent.positive_count - daily_sent.negative_count) / daily_sent.total_comments
).where(daily_sent.total_comments > 0, 0.0)

panels = []
for _, row in cov.iterrows():
    code = str(row["code"])
    from glob import glob
    fs = glob(f"{TECH_DIR}/{code}_*_technical.csv")
    if not fs:
        continue
    t = pd.read_csv(fs[0])
    t["Date"] = pd.to_datetime(t["Date"]).dt.strftime("%Y-%m-%d")
    t = t[(t.Date >= row["min"]) & (t.Date <= row["max"])].copy()
    if len(t) < 30:
        continue
    t["stock_code"] = int(code)
    ds = daily_sent[daily_sent.stock_code == int(code)]
    t = t.merge(ds, left_on=["stock_code", "Date"], right_on=["stock_code", "resolved_date"], how="left")
    for c in ("article_count", "total_comments", "positive_count", "negative_count", "daily_sentiment_factor"):
        t[c] = t[c].fillna(0.0)
    panels.append(t)
df = pd.concat(panels, ignore_index=True).sort_values(["stock_code", "Date"]).reset_index(drop=True)
print(f"股票數 {df.stock_code.nunique()}  股票日 {len(df)}  範圍 {df.Date.min()}~{df.Date.max()}", flush=True)

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

fac, att = df["daily_sentiment_factor"], np.log1p(df["total_comments"])
for k in (0, 1, 2):
    feat[f"fac_l{k}"] = fac.groupby(df["stock_code"]).shift(k) if k else fac
    feat[f"att_l{k}"] = att.groupby(df["stock_code"]).shift(k) if k else att

nxt = g["Close"].shift(-1)
keep = nxt.notna() & (nxt != C) & feat[BASE].notna().all(axis=1)
y = (nxt > C).astype(int)
date = pd.to_datetime(df["Date"])
tr = keep & (date <= "2022-06-30")
va = keep & (date >= "2022-07-01") & (date <= "2022-12-31")
te = keep & (date >= "2023-01-01")
print(f"樣本：train {tr.sum()}  val {va.sum()}  test {te.sum()}  test涵蓋股票 {df.loc[te,'stock_code'].nunique()}", flush=True)
if te.sum() < 200:
    print("[警告] test樣本過少，結果僅供參考", flush=True)

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

seed_auc, probs = {}, {}
for arm, cols in ARMS.items():
    aucs, ps = [], []
    for s in range(SEEDS):
        m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=10,
                              subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist",
                              n_jobs=16, early_stopping_rounds=30, eval_metric="logloss")
        m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
        p = m.predict_proba(feat.loc[te, cols])[:, 1]
        aucs.append(roc_auc_score(yte, p)); ps.append(p)
    seed_auc[arm], probs[arm] = np.array(aucs), np.mean(ps, axis=0)
    print(f"{arm:20s} AUC {seed_auc[arm].mean():.4f}±{seed_auc[arm].std(ddof=1):.4f}", flush=True)

out = {"n_test": int(te.sum()), "n_stocks_test": int(df.loc[te, "stock_code"].nunique()), "arms": {}}
print("\n=== 相對 A0 的 AUC 差異 ===", flush=True)
for arm in ARMS:
    e = {"auc": float(seed_auc[arm].mean())}
    if arm != "A0_無情緒":
        d_seed = seed_auc[arm] - seed_auc["A0_無情緒"]
        bd = np.array([roc_auc_score(yte[s], probs[arm][s]) - roc_auc_score(yte[s], probs["A0_無情緒"][s]) for s in boots])
        e["d_auc_seed_mean"] = float(d_seed.mean())
        e["ci95"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        print(f"{arm:20s} ΔAUC {e['d_auc_seed_mean']:+.4f}  95%CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]", flush=True)
    out["arms"][arm] = e
json.dump(out, open("cmoney_direction_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("完成")
