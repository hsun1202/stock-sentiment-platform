# -*- coding: utf-8 -*-
"""
(v3b) 籌碼訊號的穩健性檢查。原始說明：
v2+籌碼資料：比較各類資訊對個股漲跌(隔天h=1、一週後h=5)的預測貢獻。
組合：技術面 / +PTT / +籌碼 / +籌碼+PCR / 全部，另外拆開 PTT情緒方向 vs PTT討論熱度。
所有非技術面特徵都用「當天收盤後已公布」的資訊（PTT用落後1天，避免同日洩漏疑慮）。
XGBoost、10 seeds；ΔAUC 用按週與按月 block bootstrap；並分2023、2024兩年各算一次。
python direction_test_v3.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SEEDS, BOOT = 10, 500
df = pd.read_csv("final_dataset_v2/all_stocks_v2_chip.csv", dtype={"stock_code": str})
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

feat["ptt_fac_l1"] = lag(df["sentiment_factor"], 1)
feat["ptt_att_l1"] = lag(np.log1p(df["total_comments"]), 1)
feat["ptt_att5_l1"] = lag(roll(np.log1p(df["total_comments"]), 5), 1)
PTT_SENT, PTT_ATT = ["ptt_fac_l1"], ["ptt_att_l1", "ptt_att5_l1"]

for c in ("foreign", "trust", "dealer", "inst"):
    r = df[f"{c}_ratio"].clip(-1, 1)
    feat[f"{c}_r1"] = r
    feat[f"{c}_r5"] = roll(r, 5)
feat["margin_chg"] = df["margin_chg"].clip(-0.5, 0.5)
feat["margin_chg5"] = roll(feat["margin_chg"], 5)
feat["short_chg"] = df["short_chg"].clip(-1, 3)
feat["short_margin_ratio"] = df["short_margin_ratio"].clip(0, 1)
feat["margin_eligible"] = df["margin_eligible"]
CHIP = [c for c in feat.columns if c not in TECH + PTT_SENT + PTT_ATT]
feat["pc_vol"] = df["pc_vol_ratio"]
feat["pc_oi"] = df["pc_oi_ratio"]
feat["pc_oi_chg5"] = df["pc_oi_ratio"] - lag(df["pc_oi_ratio"], 5)
PCR = ["pc_vol", "pc_oi", "pc_oi_chg5"]


CHIP_L1 = []
for c in CHIP:
    feat[c + "_l1"] = lag(feat[c], 1)
    CHIP_L1.append(c + "_l1")
GROUPS = {
    "外資": ["foreign_r1", "foreign_r5"],
    "投信": ["trust_r1", "trust_r5"],
    "自營商": ["dealer_r1", "dealer_r5"],
    "融資融券": ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"],
}
ARMS = {"技術面": TECH, "+籌碼(當天)": TECH + CHIP, "+籌碼(前一天)": TECH + CHIP_L1}
ARMS.update({f"+{k}": TECH + v for k, v in GROUPS.items()})

date = pd.to_datetime(df["Date"])
abn = df["abnormal_price"]
O = df["Open"]
nextO, nextC = g["Open"].shift(-1), g["Close"].shift(-1)
TARGETS = {
    "收盤到隔天收盤": (nextC > C, nextC.notna() & (nextC != C)),
    "隔天開盤到收盤(可交易)": (nextC > nextO, nextC.notna() & nextO.notna() & (nextC != nextO)),
    "隔夜跳空(收盤到隔天開盤)": (nextO > C, nextO.notna() & (nextO != C)),
}
end_date = pd.to_datetime(g["Date"].shift(-1))
abn_ahead = lag(abn, -1).fillna(0)
results = {}
for tname, (ybool, valid) in TARGETS.items():
    keep = valid & (abn_ahead == 0) & (abn == 0) & feat[TECH].notna().all(axis=1)
    y = ybool.astype(int)
    tr = keep & (end_date <= "2022-06-30")
    va = keep & (date >= "2022-07-01") & (end_date <= "2022-12-31")
    te = keep & (date >= "2023-01-01")
    yte, dte = y[te].values, date[te]
    print(f"\n######## {tname} ########  train {tr.sum()}  test {te.sum()}  上漲率 {yte.mean():.3f}", flush=True)
    rng = np.random.default_rng(0)
    boots = {}
    for name, key in (("週", dte.dt.strftime("%G-%V").values), ("月", dte.dt.strftime("%Y-%m").values)):
        u = np.unique(key)
        idx = {w: np.where(key == w)[0] for w in u}
        boots[name] = [np.concatenate([idx[w] for w in rng.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
    years = {yr: (dte.dt.year == yr).values for yr in (2023, 2024)}
    probs, seed_auc = {}, {}
    for arm, cols in ARMS.items():
        ps, aucs = [], []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50,
                                  subsample=0.8, colsample_bytree=0.8, random_state=s, tree_method="hist",
                                  n_jobs=16, early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            p = m.predict_proba(feat.loc[te, cols])[:, 1]
            ps.append(p); aucs.append(roc_auc_score(yte, p))
        probs[arm], seed_auc[arm] = np.mean(ps, axis=0), np.array(aucs)
        line = f"  {arm:12s} AUC {seed_auc[arm].mean():.4f}±{seed_auc[arm].std(ddof=1):.4f}"
        if arm != "技術面":
            d = seed_auc[arm] - seed_auc["技術面"]
            line += f"  ΔAUC {d.mean():+.4f}"
            for bn, bl in boots.items():
                bd = [roc_auc_score(yte[b], probs[arm][b]) - roc_auc_score(yte[b], probs["技術面"][b]) for b in bl]
                line += f"  {bn}CI[{np.percentile(bd, 2.5):+.4f},{np.percentile(bd, 97.5):+.4f}]"
            for yr, mk in years.items():
                line += f"  {yr}:{roc_auc_score(yte[mk], probs[arm][mk]) - roc_auc_score(yte[mk], probs['技術面'][mk]):+.4f}"
        print(line, flush=True)
    results[tname] = {a: float(v.mean()) for a, v in seed_auc.items()}
json.dump(results, open("direction_test_v3b_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n完成")
