# -*- coding: utf-8 -*-
"""
v4：十年資料（2015-2024，final_dataset_v3/all_stocks_v3.csv）逐年滾動檢驗「技術面＋市場情緒」對個股隔天漲跌的預測力
============================================================
設計（沿用 v2/v3b 的特徵與模型，改成 walk-forward）：
  - 測試年 Y = 2018 ~ 2024：訓練用 2015-01 ~ (Y-1)-06 的標籤、驗證(early stopping)用 (Y-1) 下半年、測試 Y 全年
    → 7 個不重疊的測試年，涵蓋 2018貿易戰、2020疫情、2021多頭、2022空頭、2023-24 AI 行情
  - 情緒只用「當下真的拿得到」的資訊（留言依自己的時間歸屬，lag_timestamp_test.py 已證明文章日期歸屬會偷看答案）
  - 兩種預測目標：
      A 收盤→隔天收盤：情緒/籌碼只能用 t 收盤前可得（PTT：前一收盤後~t收盤；籌碼：t-1 盤後公布的）
      B 隔天開盤→收盤（可交易）：可用 t+1 開盤前的一切（PTT：再加 t 收盤後~t+1 開盤前；籌碼：t 盤後公布的）
  - XGBoost 5 seeds 平均機率；ΔAUC 以週 block bootstrap 估 95%CI（每個測試年各自），並彙總7年
  - 排除 abnormal_price（減資/停牌復牌/資料錯誤，漲跌幅限制 2015-06 前 7%、之後 10%）
python direction_test_v4.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SEEDS, BOOT = 5, 300
TEST_YEARS = range(2018, 2025)
df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
C = df["Close"]


def lag(s, k):
    return s.groupby(df["stock_code"]).shift(k) if k else s


def roll(s, w):
    return s.groupby(df["stock_code"]).transform(lambda x: x.rolling(w, min_periods=1).mean())


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


def ptt_feats(n, pos, neg, tag):
    feat[f"{tag}_fac"] = np.where(pos + neg > 0, (pos - neg) / (pos + neg).replace(0, 1), np.nan)
    feat[f"{tag}_att"] = np.log1p(n)
    feat[f"{tag}_att5"] = roll(np.log1p(n), 5)
    return [f"{tag}_fac", f"{tag}_att", f"{tag}_att5"]


n_c = df.ptt_pre_n + df.ptt_intra_n
p_c = df.ptt_pre_pos + df.ptt_intra_pos
q_c = df.ptt_pre_neg + df.ptt_intra_neg
PTT_CLOSE = ptt_feats(n_c, p_c, q_c, "ptt_close")                           # t 收盤前可得
PTT_OPEN = ptt_feats(n_c + lag(df.ptt_pre_n, -1).fillna(0), p_c + lag(df.ptt_pre_pos, -1).fillna(0),
                     q_c + lag(df.ptt_pre_neg, -1).fillna(0), "ptt_open")   # t+1 開盤前可得

CHIP = []
for c in ("foreign", "trust", "dealer", "inst"):
    r = df[f"{c}_ratio"].clip(-1, 1)
    feat[f"{c}_r1"] = r
    feat[f"{c}_r5"] = roll(r, 5)
    CHIP += [f"{c}_r1", f"{c}_r5"]
feat["margin_chg"] = df["margin_chg"].clip(-0.5, 0.5)
feat["margin_chg5"] = roll(feat["margin_chg"], 5)
feat["short_chg"] = df["short_chg"].clip(-1, 3)
feat["short_margin_ratio"] = df["short_margin_ratio"].clip(0, 1)
feat["margin_eligible"] = df["margin_eligible"]
CHIP += ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
CHIP_L1 = []
for c in CHIP:
    feat[c + "_l1"] = lag(feat[c], 1)
    CHIP_L1.append(c + "_l1")

date = pd.to_datetime(df["Date"])
abn = df["abnormal_price"]
O = df["Open"]
nextO, nextC = g["Open"].shift(-1), g["Close"].shift(-1)
end_date = pd.to_datetime(g["Date"].shift(-1))
abn_ahead = lag(abn, -1).fillna(0)
TARGETS = {
    "A 收盤到隔天收盤": ((nextC > C), nextC.notna() & (nextC != C), {
        "+PTT(收盤前可得)": PTT_CLOSE,
        "+籌碼(前一天，收盤前可得)": CHIP_L1,
        "+PTT+籌碼": PTT_CLOSE + CHIP_L1,
    }),
    "B 隔天開盤到收盤(可交易)": ((nextC > nextO), nextC.notna() & nextO.notna() & (nextC != nextO), {
        "+PTT(開盤前可得)": PTT_OPEN,
        "+籌碼(當天盤後公布)": CHIP,
        "+PTT+籌碼": PTT_OPEN + CHIP,
    }),
}

results = {}
for tname, (ybool, valid, arms) in TARGETS.items():
    keep = valid & (abn_ahead == 0) & (abn == 0) & feat[TECH].notna().all(axis=1)
    y = ybool.astype(int)
    all_arms = {"技術面": TECH, **{k: TECH + v for k, v in arms.items()}}
    res = {}
    print(f"\n######## {tname} ########", flush=True)
    for Y in TEST_YEARS:
        tr = keep & (end_date <= f"{Y-1}-06-30")
        va = keep & (date >= f"{Y-1}-07-01") & (end_date <= f"{Y-1}-12-31")
        te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end_date <= f"{Y+1}-01-31")
        yte, dte = y[te].values, date[te]
        wk = dte.dt.strftime("%G-%V").values
        u = np.unique(wk)
        idx = {w: np.where(wk == w)[0] for w in u}
        rng = np.random.default_rng(Y)
        boots = [np.concatenate([idx[w] for w in rng.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
        probs = {}
        for arm, cols in all_arms.items():
            ps = []
            for s in range(SEEDS):
                m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50,
                                      subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist",
                                      n_jobs=16, early_stopping_rounds=30, eval_metric="logloss")
                m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
                ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
            probs[arm] = np.mean(ps, axis=0)
        base = roc_auc_score(yte, probs["技術面"])
        r = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "base_auc": base, "up_rate": float(yte.mean()), "arms": {}}
        line = f"  {Y}: train {tr.sum():,} test {te.sum():,} 上漲率 {yte.mean():.3f} 技術面AUC {base:.4f}"
        for arm in arms:
            d = roc_auc_score(yte, probs[arm]) - base
            bd = np.array([roc_auc_score(yte[b], probs[arm][b]) - roc_auc_score(yte[b], probs["技術面"][b]) for b in boots])
            ci = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
            r["arms"][arm] = {"d_auc": d, "ci95": ci}
            line += f"｜{arm} {d:+.4f}{'✔' if ci[0] > 0 else ''}"
        print(line, flush=True)
        res[Y] = r
    print("  --- 7年彙總（ΔAUC 平均、幾年CI>0、幾年為正）---")
    summ = {}
    for arm in arms:
        ds = np.array([res[Y]["arms"][arm]["d_auc"] for Y in TEST_YEARS])
        sig = sum(res[Y]["arms"][arm]["ci95"][0] > 0 for Y in TEST_YEARS)
        summ[arm] = {"mean_d_auc": float(ds.mean()), "years_sig": int(sig), "years_pos": int((ds > 0).sum()),
                     "t_stat": float(ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds))))}
        print(f"  {arm:24s} 平均ΔAUC {ds.mean():+.4f}（t={summ[arm]['t_stat']:.2f}）｜顯著年數 {sig}/7｜為正年數 {(ds > 0).sum()}/7")
    results[tname] = {"by_year": {str(k): v for k, v in res.items()}, "summary": summ}
json.dump(results, open("direction_test_v4_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n完成 → direction_test_v4_results.json")
