# -*- coding: utf-8 -*-
"""
用「留言自己的時間戳」檢驗：先前 PTT 同日(lag0)情緒訊號，是真的預測力，還是時間錯置（事後留言被算到發文當天）？
============================================================

背景：舊資料沒有留言時間，只能把留言算在「文章發文日」。舊結果：同日訊號顯著、落後1天就消失，
      而且討論越熱越大（高討論日 ΔAUC +0.0096~+0.0122），疑似時間錯置但無法確認。
      2015-2024 新資料（同學爬的PTT＋鏡像站）每則留言都有 MM/DD HH:MM → 可以把留言放到正確的時間窗。

設定（照 direction_test_v2/v3b，只改情緒的時間歸屬）：
    股價/技術面：final_dataset_v2/all_stocks_v2_chip.csv（2020-2024，還原股價）
    情緒：sentiment_output_267_2015_2024/*_comment_labels.csv（新分類器、清理後、新比對規則），只用 2020-2024
    切分：train 標籤結束日<=2022-06-30、val 2022H2、test 2023-2024；XGBoost 10 seeds；週/月 block bootstrap
    情緒特徵：fac = (正−負)/(正+負)（沒有表態留言時為空值）、att = log(1+留言數)

每個交易日 t 的時間窗（台股 09:00 開盤、13:30 收盤）：
    pre(t)   = 前一交易日收盤後 ~ t 開盤前（盤前/隔夜/假日）
    intra(t) = t 的 09:00 ~ 13:30（盤中）
比較的情緒歸屬（預測目標 A：t收盤→t+1收盤；目標 B：t+1開盤→t+1收盤（可交易））：
    文章日期（舊做法）         ：留言算在文章發文日（非交易日歸下一交易日）
    收盤前可得 pre(t)+intra(t)  ：t 收盤前已發的留言 → 對目標A、B都沒有偷看
    隔夜 pre(t+1)               ：t收盤後~t+1開盤前 → 對目標B沒有偷看；對目標A屬於「部分未來」
    隔天盤中 intra(t+1)         ：與目標期間同時 → 故意偷看答案的對照組
python lag_timestamp_test.py
"""
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SEEDS, BOOT = 10, 500
OPEN_T, CLOSE_T = pd.Timedelta(hours=9), pd.Timedelta(hours=13, minutes=30)

# ---------------- 股價與技術面（同 v3b） ----------------
df = pd.read_csv("final_dataset_v2/all_stocks_v2_chip.csv", dtype={"stock_code": str},
                 usecols=["stock_code", "Date", "Open", "High", "Low", "Close", "Volume", "MA5", "MA30", "MA60",
                          "EMA5", "EMA30", "EMA60", "MACD_6_15_6", "MACD_12_26_9", "MACD_30_60_30", "RSI14",
                          "WILLR14", "MOM14", "CMO14", "ULTOSC_7_14_28", "ADOSC_3_10", "abnormal_price"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"])
g = df.groupby("stock_code")
C = df["Close"]


def lag(s, k):
    return s.groupby(df["stock_code"]).shift(k) if k else s


feat = pd.DataFrame(index=df.index)
feat["ret1"] = np.log(C / g["Close"].shift(1))
for k in (1, 2, 3, 4):
    feat[f"ret1_l{k}"] = lag(feat["ret1"], k)
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
TECH = list(feat.columns)

# ---------------- 留言 → 時間窗 ----------------
cal = np.array(sorted(df["Date"].unique()), dtype="datetime64[ns]")
close_dt = cal + np.timedelta64(int(CLOSE_T.total_seconds()), "s")
parts = []
for f in glob.glob("sentiment_output_267_2015_2024/*_comment_labels.csv"):
    d = pd.read_csv(f, dtype=str, keep_default_na=False,
                    usecols=["stock_code", "resolved_date", "comment_time", "time_anomaly", "text_sentiment"])
    d = d[(d["resolved_date"] >= "2019-12-01") & (d["resolved_date"] <= "2024-12-31")]
    parts.append(d)
cm = pd.concat(parts, ignore_index=True)
n_all = len(cm)
cm = cm[(cm["comment_time"] != "") & (cm["time_anomaly"] != "1")].copy()
cm["ct"] = pd.to_datetime(cm["comment_time"])
cm["ad"] = pd.to_datetime(cm["resolved_date"])
cm["pos"] = (cm["text_sentiment"] == "positive").astype(int)
cm["neg"] = (cm["text_sentiment"] == "negative").astype(int)
# 留言時間 → 第一個「收盤時間 >= 留言時間」的交易日 t；再分成 t 開盤前(pre) / 盤中(intra)
i = np.searchsorted(close_dt, cm["ct"].values.astype("datetime64[ns]"), side="left")
ok = i < len(cal)
cm = cm[ok].copy()
i = i[ok]
cm["t"] = cal[i]
cm["part"] = np.where(cm["ct"].values < (cal[i] + np.timedelta64(int(OPEN_T.total_seconds()), "s")), "pre", "intra")
# 舊做法：文章日期 → 第一個 >= 文章日期的交易日
j = np.searchsorted(cal, cm["ad"].values.astype("datetime64[ns]"), side="left")
cm = cm[j < len(cal)].copy()
cm["t_art"] = cal[j[j < len(cal)]]

# 描述統計：用文章日期歸屬時，有多少留言其實是「那天收盤之後」才發的
after_close = cm["ct"] > (cm["t_art"] + CLOSE_T)
lines = [f"2020-2024 比對到股票的留言 {n_all:,} 則；有時間且非異常 {len(cm):,} 則",
         f"【關鍵事實】用文章日期歸屬時，{after_close.mean():.1%} 的留言其實是在「歸屬那天的收盤(13:30)之後」才發的",
         f"    其中在歸屬日隔天(含)以後才發的：{(cm['ct'] >= cm['t_art'] + pd.Timedelta(days=1)).mean():.1%}"]
nz = cm[cm["pos"] + cm["neg"] > 0]
lines.append(f"    只看有表態(正/負)的留言：收盤後才發的佔 {(nz['ct'] > nz['t_art'] + CLOSE_T).mean():.1%}")
print("\n".join(lines))


def agg(keycol, mask=None):
    x = cm if mask is None else cm[mask]
    a = x.groupby(["stock_code", keycol]).agg(n=("pos", "size"), p=("pos", "sum"), q=("neg", "sum")).reset_index()
    a = a.rename(columns={keycol: "Date"})
    a["fac"] = np.where(a["p"] + a["q"] > 0, (a["p"] - a["q"]) / (a["p"] + a["q"]).replace(0, 1), np.nan)
    a["att"] = np.log1p(a["n"])
    return df[["stock_code", "Date"]].merge(a[["stock_code", "Date", "fac", "att"]], on=["stock_code", "Date"],
                                           how="left").set_index(df.index)


art = agg("t_art")
pre = agg("t", cm["part"] == "pre")
intra = agg("t", cm["part"] == "intra")
byclose = agg("t")  # pre+intra：t 收盤前可得
for name, a in (("art", art), ("close", byclose)):
    feat[f"{name}_fac"] = a["fac"]
    feat[f"{name}_att"] = a["att"].fillna(0)
feat["over_fac"] = lag(pre["fac"], -1)            # t收盤後~t+1開盤前
feat["over_att"] = lag(pre["att"].fillna(0), -1)
feat["intra1_fac"] = lag(intra["fac"], -1)        # t+1 盤中（對照組：偷看答案）
feat["intra1_att"] = lag(intra["att"].fillna(0), -1)
feat["close_fac_l1"] = lag(feat["close_fac"], 1)
feat["close_att_l1"] = lag(feat["close_att"], 1)

S = {k: [f"{k}_fac", f"{k}_att"] for k in ("art", "close", "over", "intra1")}
S["close_l1"] = ["close_fac_l1", "close_att_l1"]

date = df["Date"]
abn = df["abnormal_price"]
nextO, nextC = g["Open"].shift(-1), g["Close"].shift(-1)
TARGETS = {
    "A 收盤到隔天收盤": ((nextC > C), nextC.notna() & (nextC != C), {
        "文章日期(舊做法)": S["art"],
        "收盤前可得(無偷看)": S["close"],
        "收盤前可得・落後1天": S["close_l1"],
        "收盤前＋隔夜(部分未來)": S["close"] + S["over"],
        "隔天盤中(偷看答案・對照)": S["intra1"],
    }),
    "B 隔天開盤到收盤(可交易)": ((nextC > nextO), nextC.notna() & nextO.notna() & (nextC != nextO), {
        "文章日期(舊做法)": S["art"],
        "收盤前可得(無偷看)": S["close"],
        "收盤前＋隔夜(開盤前可得・無偷看)": S["close"] + S["over"],
        "隔天盤中(偷看答案・對照)": S["intra1"],
    }),
}
end_date = g["Date"].shift(-1)
abn_ahead = lag(abn, -1).fillna(0)
results = {"facts": lines}
for tname, (ybool, valid, arms) in TARGETS.items():
    keep = valid & (abn_ahead == 0) & (abn == 0) & feat[TECH].notna().all(axis=1)
    y = ybool.astype(int)
    tr = keep & (end_date <= "2022-06-30")
    va = keep & (date >= "2022-07-01") & (end_date <= "2022-12-31")
    te = keep & (date >= "2023-01-01")
    yte, dte = y[te].values, date[te]
    hot = np.maximum(feat.loc[te, "close_att"].values, feat.loc[te, "art_att"].values) >= np.log1p(20)  # 高討論股票日(≥20則)
    print(f"\n######## {tname} ########  train {tr.sum():,}  test {te.sum():,}（高討論 {hot.sum():,}）", flush=True)
    rng = np.random.default_rng(0)
    boots = {}
    for bname, key in (("週", dte.dt.strftime("%G-%V").values), ("月", dte.dt.strftime("%Y-%m").values)):
        u = np.unique(key)
        idx = {w: np.where(key == w)[0] for w in u}
        boots[bname] = [np.concatenate([idx[w] for w in rng.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
    probs = {}
    all_arms = {"技術面": TECH, **{k: TECH + v for k, v in arms.items()}}
    for arm, cols in all_arms.items():
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50,
                                  subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist",
                                  n_jobs=16, early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        probs[arm] = np.mean(ps, axis=0)
    base_auc = roc_auc_score(yte, probs["技術面"])
    res = {"n_test": int(te.sum()), "n_hot": int(hot.sum()), "base_auc": base_auc, "arms": {}}
    print(f"  技術面 AUC {base_auc:.4f}")
    for arm in arms:
        d_all = roc_auc_score(yte, probs[arm]) - base_auc
        d_hot = roc_auc_score(yte[hot], probs[arm][hot]) - roc_auc_score(yte[hot], probs["技術面"][hot])
        e = {"d_auc": d_all, "d_auc_hot": d_hot}
        for bname, bl in boots.items():
            bd = np.array([roc_auc_score(yte[b], probs[arm][b]) - roc_auc_score(yte[b], probs["技術面"][b]) for b in bl])
            e[f"ci_{bname}"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        res["arms"][arm] = e
        print(f"  {arm:24s} ΔAUC {d_all:+.4f} 週CI[{e['ci_週'][0]:+.4f},{e['ci_週'][1]:+.4f}] "
              f"月CI[{e['ci_月'][0]:+.4f},{e['ci_月'][1]:+.4f}]｜高討論日 {d_hot:+.4f}", flush=True)
    results[tname] = res
json.dump(results, open("lag_timestamp_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n完成 → lag_timestamp_test_results.json")
