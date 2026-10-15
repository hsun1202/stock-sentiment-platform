# -*- coding: utf-8 -*-
"""
2025 年真正樣本外：散戶「做」的訊號（融資融券＋散戶成交比重）在完全沒看過的一年、全市場還成不成立？
資料：final_dataset_v3_tejall25（TEJ 全市場四碼普通股，2015~2025）。2025 年在所有先前的分析、調參、特徵選擇中都沒有用過。
模型與特徵完全沿用 tej_fullmarket_test.py / tej_retail_daily.py（不做任何修改或調參）；訓練到 2024-06、驗證 2024 下半年、測試 2025 全年。

事先寫定的假說（2026-10-08，跑之前寫下；主要檢定 1 個，不需校正）：
  O1：技術面＋融資融券＋散戶成交比重（合稱「散戶行為」）預測 2025 年隔天開盤→收盤方向的 AUC 高於只有技術面，
      且週區塊拔靴 95% 信賴區間下界 > 0。
  次要（先寫定但不算主要）：單獨 +融資融券、單獨 +散戶成交比重、+法人 的 ΔAUC；
      依模型預測分五組的隔天開→收報酬差（散戶行為模型 vs 技術面模型）；只看不在原 265 檔內的股票；券資比每日排序。
python tej_oos2025_test.py
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import roc_auc_score

from features_v4 import build

PATH = "final_dataset_v3_tejall25/all_stocks_v3.csv"
SEEDS, BOOT, Y = 5, 1000, 2025
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def tp(s):
    s = pd.Series(s).dropna()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    return t, 2 * stats.t.sf(abs(t), len(s) - 1)


feat, meta, A = build(PATH)
raw = pd.read_csv(PATH, dtype={"stock_code": str}, parse_dates=["Date"],
                  usecols=["stock_code", "Date", "Open", "Close", "Volume", "abnormal_price", "margin_eligible", "in265",
                           "inst_share", "short_margin_ratio"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
assert (raw.stock_code.values == meta.stock_code.values).all() and (raw.Date.values == meta.date.values).all()
raw["rs"] = ((100 - raw.inst_share) / 100).where(raw.Volume > 0)
g = raw.groupby("stock_code")


def rmean(col, w, mp, shift=0):
    return g[col].transform(lambda x: x.rolling(w, min_periods=mp).mean().shift(shift))


raw["rs_dev"] = raw.rs - rmean("rs", 20, 10, 1)
raw["rs_5"] = rmean("rs", 5, 3)
raw["rs_dev5"] = raw.rs_5 - rmean("rs", 60, 30)
RS = ["rs", "rs_dev", "rs_5", "rs_dev5"]
for c in RS:
    feat[c] = raw[c].values
TECH = A["技術面"]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
ARMS = {"技術面": TECH, "+散戶行為（融資融券＋散戶成交比重）★O1": TECH + MARGIN + RS, "+融資融券": TECH + MARGIN,
        "+散戶成交比重": TECH + RS, "+法人買賣超": TECH + INST}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31")
say(f"[資料] 全市場 {raw.stock_code.nunique()} 檔；訓練 {int(tr.sum()):,}（~2024-06）／驗證 {int(va.sum()):,}／測試 2025 年 {int(te.sum()):,} 個股票日、"
    f"{raw.loc[te, 'stock_code'].nunique()} 檔、{date[te].nunique()} 個交易日（{date[te].min().date()}~{date[te].max().date()}）")
yte = y[te].values; wk = date[te].dt.strftime("%G-%V").values; u = np.unique(wk)
wi = {w: np.where(wk == w)[0] for w in u}; rg = np.random.default_rng(Y)
boots = [np.concatenate([wi[w] for w in rg.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
P = {}
for arm, cols in ARMS.items():
    ps = []
    for s in range(SEEDS):
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                              colsample_bytree=0.8, random_state=s, tree_method="hist", device="cuda",
                              early_stopping_rounds=30, eval_metric="logloss")
        m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
        m.set_params(device="cpu")
        ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
    P[arm] = np.mean(ps, axis=0)
base = roc_auc_score(yte, P["技術面"])
out_mask = (raw.in265 == 0).values[te.values]
say(f"\n(a) 2025 年隔天開盤→收盤方向 AUC｜技術面 {base:.4f}（不在原 265 檔 {roc_auc_score(yte[out_mask], P['技術面'][out_mask]):.4f}）；2018~2024 平均為 0.5576")
res = {"技術面": float(base)}
for arm in list(ARMS)[1:]:
    d = roc_auc_score(yte, P[arm]) - base
    bd = np.array([roc_auc_score(yte[b], P[arm][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    d_out = roc_auc_score(yte[out_mask], P[arm][out_mask]) - roc_auc_score(yte[out_mask], P["技術面"][out_mask])
    res[arm] = {"d": float(d), "lo": float(lo), "hi": float(hi), "p_boot": float((bd <= 0).mean()), "d_out": float(d_out)}
    say(f"  {arm:<26s} ΔAUC {d:+.4f}（95% 區間 {lo:+.4f}~{hi:+.4f}；拔靴中 ≤0 的比例 {(bd <= 0).mean():.3f}）｜不在原 265 檔 {d_out:+.4f}")
k = "+散戶行為（融資融券＋散戶成交比重）★O1"
say(f"  → O1：ΔAUC {res[k]['d']:+.4f}，95% 區間下界 {res[k]['lo']:+.4f} → {'通過' if res[k]['lo'] > 0 else '未通過'}"
    f"（參考：265 檔 2018~2024 平均 +0.0052）")

# (b) 依模型預測分五組的報酬
nO, nC = g.Open.shift(-1), g.Close.shift(-1)
r_oc = ((nC / nO - 1) * 1e4)[te].values
T = pd.DataFrame({"Date": date[te].values, "r": r_oc, **{a: P[a] for a in ARMS}})
say("\n(b) 2025 年依模型預測每天分五組：隔天開盤→收盤報酬（bp，等權毛報酬；當沖來回成本約 32~44 bp）")
ls = {}
for arm in ARMS:
    q = T.groupby("Date")[arm].rank(pct=True)
    top, bot, allm = T[q > 0.8].groupby("Date").r.mean(), T[q <= 0.2].groupby("Date").r.mean(), T.groupby("Date").r.mean()
    ls[arm] = top - bot
    say(f"  {arm:<26s} 最高組 {top.mean():+.1f}｜最低組 {bot.mean():+.1f}｜高−低 {(top - bot).mean():+.1f}（t={tp(top - bot)[0]:+.2f}）｜最高組−全體 {(top - allm).mean():+.1f}（t={tp(top - allm)[0]:+.2f}）")
dd = (ls[k] - ls["技術面"]).dropna()
say(f"  散戶行為模型比技術面模型多出的 高−低：{dd.mean():+.2f} bp/天（t={tp(dd)[0]:+.2f}，{len(dd)} 天）")
res["ls_extra_bp"] = float(dd.mean()); res["ls_extra_t"] = float(tp(dd)[0])

# (c) 券資比每日排序
S = raw[te.values].assign(r=r_oc)
S = S[(S.margin_eligible == 1) & S.short_margin_ratio.notna() & S.r.notna()]
S = S.assign(q=S.groupby("Date").short_margin_ratio.transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1 if len(v) >= 25 else np.nan))
daily = S.groupby(["Date", "q"]).r.mean().unstack()
sp = daily[5] - daily[1]
say(f"\n(c) 2025 年券資比每日五分位：最高−最低 隔天開→收 {sp.mean():+.1f} bp（t={tp(sp)[0]:+.2f}）；2018~2024 全市場為 −21.7 bp")
res["smr_spread"] = float(sp.mean())
open("tej_oos2025_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
json.dump(res, open("tej_oos2025_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
say("DONE")
