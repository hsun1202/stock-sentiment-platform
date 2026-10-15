# -*- coding: utf-8 -*-
"""
用v2資料(還原股價、修補壞資料、週末留言歸到下一交易日)重跑「技術面＋PTT」漲跌預測，
對照修正前 direction_test.py 的結果。
  - 預測目標：隔天(h=1)、一週後(h=5個交易日) 收盤是否高於今天
  - 排除：標籤期間內有 abnormal_price(減資/停牌復牌/資料錯誤) 的列
  - 切分：train 標籤結束日<=2022-06-30、val 2022-07~12(標籤結束日<=2022-12-31)、test 2023-2024 → 各段不重疊
  - XGBoost、20 seeds、按週 block bootstrap 估 ΔAUC 信賴區間
python direction_test_v2.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SEEDS, BOOT = 20, 500
df = pd.read_csv("final_dataset_v2/all_stocks_v2.csv", dtype={"stock_code": str})
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

fac, att = df["sentiment_factor"], np.log1p(df["total_comments"])
for k in (0, 1, 2):
    feat[f"fac_l{k}"] = fac.groupby(df["stock_code"]).shift(k) if k else fac
    feat[f"att_l{k}"] = att.groupby(df["stock_code"]).shift(k) if k else att

ARMS = {
    "技術面": BASE,
    "技術面+PTT同日": BASE + ["fac_l0", "att_l0"],
    "技術面+PTT落後1天": BASE + ["fac_l1", "att_l1"],
    "技術面+PTT落後2天": BASE + ["fac_l2", "att_l2"],
}
date = pd.to_datetime(df["Date"])
abn = df["abnormal_price"]
results = {}
for h in (1, 5):
    fut = g["Close"].shift(-h)
    end_date = pd.to_datetime(g["Date"].shift(-h))
    abn_ahead = sum(abn.groupby(df["stock_code"]).shift(-k).fillna(0) for k in range(1, h + 1))
    keep = fut.notna() & (fut != C) & (abn_ahead == 0) & (abn == 0) & feat[BASE].notna().all(axis=1)
    y = (fut > C).astype(int)
    tr = keep & (end_date <= "2022-06-30")
    va = keep & (date >= "2022-07-01") & (end_date <= "2022-12-31")
    te = keep & (date >= "2023-01-01")
    yte = y[te].values
    print(f"\n######## h={h} ########  train {tr.sum()}  val {va.sum()}  test {te.sum()}  測試期上漲率 {yte.mean():.3f}", flush=True)
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
        print(f"  {arm:14s} AUC {seed_auc[arm].mean():.4f}±{seed_auc[arm].std(ddof=1):.4f}", flush=True)
    res = {"n_test": int(te.sum()), "arms": {}}
    for arm in ARMS:
        e = {"auc": float(seed_auc[arm].mean())}
        if arm != "技術面":
            bd = np.array([roc_auc_score(yte[b], probs[arm][b]) - roc_auc_score(yte[b], probs["技術面"][b]) for b in boots])
            e["d_auc"] = float((seed_auc[arm] - seed_auc["技術面"]).mean())
            e["ci95"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
            print(f"  {arm:14s} ΔAUC {e['d_auc']:+.4f}  95%CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]", flush=True)
        res["arms"][arm] = e
    results[f"h{h}"] = res
json.dump(results, open("direction_test_v2_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n完成")
