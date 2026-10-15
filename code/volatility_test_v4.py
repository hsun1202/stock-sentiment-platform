# -*- coding: utf-8 -*-
"""
PTT 討論熱度/情緒能不能預測「波動」（大動作），而不是漲跌方向？（十年資料，逐年滾動 2018~2024）
============================================================
動機：文獻線索——情緒濃烈/討論爆量不代表方向對，但可能代表「接下來會有大動作」。
      如果成立，PTT 在平台上可當「風險提醒」（這檔討論爆量，注意波動），研究也不只有負面結論。
公平的基準：波動有很強的「群聚」（今天大動作 → 明天也容易大動作），所以技術面基準額外加入
      近期已實現波動（5/20日報酬標準差、平均振幅、今天|報酬|），PTT 必須在這之上還有增量才算數。
預測目標（兩種時間設定，都不偷看）：
  A 隔天收盤漲跌幅 |Close_{t+1}/Close_t − 1| 是否 ≥ 3%（大動作）  → 只用 t 收盤前可得的 PTT
  B 隔天盤中振幅 (High_{t+1} − Low_{t+1}) / Open_{t+1} 是否在當天全體前 10%（相對大振幅）
                                                                  → 可用 t+1 開盤前的 PTT
PTT 特徵：討論量、5日平均、討論量異常（相對過去20日 z 分數）、情緒、|情緒|
比較：技術面(含波動) / +PTT / +籌碼 / +PTT+籌碼；XGBoost 5 seeds、週 block bootstrap
輸出：volatility_test_v4.log、volatility_test_v4_results.json
python volatility_test_v4.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features_v4 import build

SEEDS, BOOT = 5, 300
TEST_YEARS = range(2018, 2025)
feat, meta, arms_dir = build()
TECH = arms_dir["技術面"]
CHIP = [c for c in arms_dir["+籌碼(當天盤後公布)"] if c not in TECH]
df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")


def lag(s, k):
    return s.groupby(df["stock_code"]).shift(k) if k else s


def roll(s, w, f="mean"):
    return s.groupby(df["stock_code"]).transform(lambda x: getattr(x.rolling(w, min_periods=3), f)())


# 波動基準特徵（技術面額外加入）
r = feat["ret1"]
feat["vol5"] = roll(r, 5, "std")
feat["vol20"] = roll(r, 20, "std")
feat["absret"] = r.abs()
feat["range5"] = roll(feat["range"], 5)
feat["range20"] = roll(feat["range"], 20)
VOLB = ["vol5", "vol20", "absret", "range5", "range20"]


def ptt(n, p, q, tag):
    a = np.log1p(n)
    feat[f"{tag}_att"] = a
    feat[f"{tag}_att5"] = roll(a, 5)
    m20, s20 = lag(roll(a, 20), 1), lag(roll(a, 20, "std"), 1)
    feat[f"{tag}_spike"] = (a - m20) / (s20 + 0.1)
    feat[f"{tag}_sent"] = (p - q) / (p + q + 1)
    feat[f"{tag}_abs"] = feat[f"{tag}_sent"].abs()
    return [f"{tag}_{x}" for x in ("att", "att5", "spike", "sent", "abs")]


n_c = df.ptt_pre_n + df.ptt_intra_n
p_c = df.ptt_pre_pos + df.ptt_intra_pos
q_c = df.ptt_pre_neg + df.ptt_intra_neg
PTT_CLOSE = ptt(n_c, p_c, q_c, "pc")
nx = lambda c: lag(df[c], -1).fillna(0)  # noqa: E731
PTT_OPEN = ptt(n_c + nx("ptt_pre_n"), p_c + nx("ptt_pre_pos"), q_c + nx("ptt_pre_neg"), "po")
CHIP_L1 = []
for c in CHIP:
    feat[c + "_l1"] = lag(feat[c], 1)
    CHIP_L1.append(c + "_l1")

C, nextC, nextO = df["Close"], g["Close"].shift(-1), g["Open"].shift(-1)
nextH, nextL = g["High"].shift(-1), g["Low"].shift(-1)
date = pd.to_datetime(df["Date"])
end_date = pd.to_datetime(g["Date"].shift(-1))
abn = df["abnormal_price"]
base_keep = (abn == 0) & (lag(abn, -1).fillna(0) == 0) & feat[TECH].notna().all(axis=1) & (g["Volume"].shift(-1) > 0)
rng_next = (nextH - nextL) / nextO
big_range = rng_next >= rng_next.groupby(g["Date"].shift(-1)).transform(lambda s: s.quantile(0.9))
TARGETS = {
    "A 隔天收盤漲跌幅≥3%（只用收盤前可得）": ((nextC / C - 1).abs() >= 0.03, nextC.notna(), {
        "+PTT": PTT_CLOSE, "+籌碼(前一天)": CHIP_L1, "+PTT+籌碼": PTT_CLOSE + CHIP_L1}),
    "B 隔天盤中振幅前10%（開盤前可得）": (big_range, rng_next.notna(), {
        "+PTT": PTT_OPEN, "+籌碼(當天盤後)": CHIP, "+PTT+籌碼": PTT_OPEN + CHIP}),
}
lines, results = [], {}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


for tname, (ybool, valid, arms) in TARGETS.items():
    keep = base_keep & valid
    y = ybool.astype(int)
    all_arms = {"技術面(含波動)": TECH + VOLB, **{k: TECH + VOLB + v for k, v in arms.items()}}
    log(f"\n######## {tname} ########  全期基準率 {y[keep].mean():.3f}")
    res = {}
    for Y in TEST_YEARS:
        tr = keep & (end_date <= f"{Y-1}-06-30")
        va = keep & (date >= f"{Y-1}-07-01") & (end_date <= f"{Y-1}-12-31")
        te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end_date <= f"{Y+1}-01-31")
        yte, dte = y[te].values, date[te]
        wk = dte.dt.strftime("%G-%V").values
        u = np.unique(wk)
        widx = {w: np.where(wk == w)[0] for w in u}
        rs = np.random.default_rng(Y)
        boots = [np.concatenate([widx[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
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
        base = roc_auc_score(yte, probs["技術面(含波動)"])
        rr = {"base_auc": base, "rate": float(yte.mean()), "arms": {}}
        line = f"  {Y}: 基準率 {yte.mean():.3f} 技術面(含波動)AUC {base:.4f}"
        for arm in arms:
            d = roc_auc_score(yte, probs[arm]) - base
            bd = np.array([roc_auc_score(yte[b], probs[arm][b]) - roc_auc_score(yte[b], probs["技術面(含波動)"][b])
                           for b in boots])
            ci = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
            rr["arms"][arm] = {"d_auc": d, "ci95": ci}
            line += f"｜{arm} {d:+.4f}{'✔' if ci[0] > 0 else ''}"
        log(line)
        res[Y] = rr
    log("  --- 7年彙總 ---")
    summ = {}
    for arm in arms:
        ds = np.array([res[Y]["arms"][arm]["d_auc"] for Y in TEST_YEARS])
        sig = sum(res[Y]["arms"][arm]["ci95"][0] > 0 for Y in TEST_YEARS)
        t = float(ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds))))
        summ[arm] = {"mean_d_auc": float(ds.mean()), "t": t, "years_sig": int(sig), "years_pos": int((ds > 0).sum())}
        log(f"  {arm:16s} 平均ΔAUC {ds.mean():+.4f}（t={t:.2f}）｜顯著 {sig}/7｜為正 {(ds > 0).sum()}/7")
    results[tname] = {"by_year": {str(k): v for k, v in res.items()}, "summary": summ}
json.dump(results, open("volatility_test_v4_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open("volatility_test_v4.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("完成")
