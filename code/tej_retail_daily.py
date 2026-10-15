# -*- coding: utf-8 -*-
"""
TEJ 新檢驗 (1)-日頻：散戶成交比重（＝100−法人成交比重，t 日盤後可得）對隔天報酬有沒有預測力，方法比照融資融券。
資料：final_dataset_v3_tej/all_stocks_v3.csv（265 檔、TEJ 還原股價）＋ tej_raw/tej_chip_daily.pkl；測試期 2018~2024。

事先寫定的假說（2026-10-05，跑之前寫下；主要檢定共 4 個〔日頻 2＋月頻 2，月頻見 tej_retail_monthly.py〕，
Bonferroni 門檻 p<0.0125，其餘一律標為探索性）：
  P1（模型增量）：技術面＋散戶成交比重特徵的 XGBoost，隔天開盤→收盤方向 AUC 比只有技術面高（7 個測試年 ΔAUC 的 t 檢定，自由度 6）。
      對照：+融資融券 +0.0022（t=4.85）、+法人買賣超 +0.0040（t=4.59）。
  P2（單一變數排序）：每天依「異常散戶成交比重」（今天 − 自己前 20 天平均）分五組，最高組隔天開盤→收盤報酬低於最低組
      （預期方向為負：散戶湧入後表現較差；雙尾檢定）。
  探索性：散戶成交比重水準、扣掉規模後的水準、隔夜與收盤→收盤報酬、信用交易比重、資券互抵比例、與融資融券合併的增量。
python tej_retail_daily.py
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import roc_auc_score

from features_v4 import build

PATH = "final_dataset_v3_tej/all_stocks_v3.csv"
SEEDS, BOOT, TEST_YEARS = 5, 300, range(2018, 2025)
ALPHA = 0.05 / 4
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def tp(s, df=None):
    s = pd.Series(s).dropna()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    return t, 2 * stats.t.sf(abs(t), len(s) - 1 if df is None else df)


feat, meta, A = build(PATH)
raw = pd.read_csv(PATH, dtype={"stock_code": str}, parse_dates=["Date"],
                  usecols=["stock_code", "Date", "Open", "Close", "Volume", "abnormal_price", "mktcap", "margin_eligible"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
assert (raw.stock_code.values == meta.stock_code.values).all() and (raw.Date.values == meta.date.values).all()
chip = pd.read_pickle("tej_raw/tej_chip_daily.pkl")[["code", "date", "法人成交比重", "信用交易比重", "資券互抵比例"]]
chip = chip.drop_duplicates(["code", "date"]).rename(columns={"code": "stock_code", "date": "Date"})
n0 = len(raw)
raw = raw.merge(chip, on=["stock_code", "Date"], how="left")
assert len(raw) == n0 and (raw.stock_code.values == meta.stock_code.values).all()

traded = raw.Volume > 0
raw["rs"] = ((100 - raw["法人成交比重"]) / 100).where(traded)
raw["cr"] = (raw["信用交易比重"] / 100).where(traded)
raw["dt"] = (raw["資券互抵比例"] / 100).where(traded)
g = raw.groupby("stock_code")


def rmean(col, w, mp, shift=0):
    return g[col].transform(lambda x: x.rolling(w, min_periods=mp).mean().shift(shift))


raw["rs_dev"] = raw.rs - rmean("rs", 20, 10, 1)
raw["rs_5"] = rmean("rs", 5, 3)
raw["rs_dev5"] = raw.rs_5 - rmean("rs", 60, 30)
raw["cr_dev"] = raw.cr - rmean("cr", 20, 10, 1)
raw["dt_dev"] = raw.dt - rmean("dt", 20, 10, 1)
x = np.log(raw.mktcap.where(raw.mktcap > 0))
okxy = x.notna() & raw.rs.notna()
xd = (x - x.where(okxy).groupby(raw.Date).transform("mean")).where(okxy)
yd = (raw.rs - raw.rs.where(okxy).groupby(raw.Date).transform("mean")).where(okxy)
beta = (xd * yd).groupby(raw.Date).transform("sum") / (xd ** 2).groupby(raw.Date).transform("sum")
raw["rs_adj"] = yd - beta * xd
RS = ["rs", "rs_dev", "rs_5", "rs_dev5"]
for c in RS:
    feat[c] = raw[c].values

say(f"[資料] {len(raw):,} 列、{raw.stock_code.nunique()} 檔；散戶成交比重有值 {raw.rs.notna().mean():.1%}；"
    f"中位數 2015 {raw.rs[raw.Date.dt.year == 2015].median():.2f} → 2024 {raw.rs[raw.Date.dt.year == 2024].median():.2f}；"
    f"與 log 市值的橫斷面相關 {raw[['rs']].assign(x=x).groupby(raw.Date).corr().xs('x', level=1).rs.mean():+.2f}")

# ---------------- (a) 單一變數五分位排序 ----------------
nO, nC = g.Open.shift(-1), g.Close.shift(-1)
abn = raw.abnormal_price.fillna(0)
ok = (abn == 0) & (g.abnormal_price.shift(-1).fillna(0) == 0) & nO.notna() & nC.notna() & (nO > 0)
raw["r_oc"] = ((nC / nO - 1) * 1e4).where(ok)
raw["r_on"] = ((nO / raw.Close - 1) * 1e4).where(ok)
raw["r_cc"] = ((nC / raw.Close - 1) * 1e4).where(ok)
mg = pd.read_csv(PATH, dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "margin_chg", "short_chg", "short_margin_ratio"])
mg = mg.sort_values(["stock_code", "Date"]).reset_index(drop=True)
for c in ("margin_chg", "short_chg", "short_margin_ratio"):
    raw[c] = mg[c].values
S = raw[(raw.Date >= "2018-01-01") & (raw.Date <= "2024-12-31")]
say(f"\n(a) 單一變數排序：每天分五組（1=最低…5=最高），隔天報酬 bp（等權、毛報酬）｜2018~2024｜全體平均 開→收 {S.r_oc.mean():+.1f}、隔夜 {S.r_on.mean():+.1f}")
sort_res = {}
VARS = [("rs_dev", "異常散戶成交比重★P2", False), ("rs", "散戶成交比重(水準)", False), ("rs_adj", "散戶成交比重(扣規模)", False),
        ("cr_dev", "異常信用交易比重", False), ("dt_dev", "異常資券互抵比例", False),
        ("margin_chg", "融資增減(對照)", True), ("short_chg", "融券增減(對照)", True), ("short_margin_ratio", "券資比(對照)", True)]
for col, name, elig in VARS:
    d = S[S[col].notna() & S.r_oc.notna()]
    if elig:
        d = d[d.margin_eligible == 1]
    d = d.assign(q=d.groupby("Date")[col].transform(
        lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1 if len(v) >= 25 else np.nan))
    line = f"  {name:<14s}"
    sort_res[col] = {}
    for r, rn in [("r_oc", "開→收"), ("r_on", "隔夜"), ("r_cc", "收→收")]:
        daily = d.groupby(["Date", "q"])[r].mean().unstack()
        sp = daily[5] - daily[1]
        t, p = tp(sp)
        sort_res[col][r] = {"spread": float(sp.mean()), "t": float(t), "p": float(p)}
        if r == "r_oc":
            line += " 各組 " + " ".join(f"{daily[k].mean():+.1f}" for k in range(1, 6))
        line += f"｜{rn} 高−低 {sp.mean():+.1f}（t={t:+.2f}）"
    say(line)
    if col == "rs_dev":
        for a, b in [("2018", "2020"), ("2021", "2024")]:
            dd = d[(d.Date >= a) & (d.Date <= b + "-12-31")]
            daily = dd.groupby(["Date", "q"]).r_oc.mean().unstack()
            sp = daily[5] - daily[1]
            say(f"      {a}~{b}：開→收 高−低 {sp.mean():+.1f}（t={tp(sp)[0]:+.2f}）")
p2 = sort_res["rs_dev"]["r_oc"]
say(f"  → P2：高−低 {p2['spread']:+.1f} bp/天，t={p2['t']:+.2f}，p={p2['p']:.4f}（門檻 {ALPHA}）→ {'通過' if p2['p'] < ALPHA else '未通過'}"
    f"；當沖來回成本約 32~44 bp")

# ---------------- (b) 模型增量：逐年滾動 XGBoost ----------------
TECH = A["技術面"]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
ARMS = {"技術面": TECH, "+散戶成交比重★P1": TECH + RS, "+融資融券": TECH + MARGIN,
        "+融資融券+散戶成交比重": TECH + MARGIN + RS, "+法人買賣超": TECH + INST}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
say("\n(b) 模型增量：隔天開盤→收盤方向，XGBoost 逐年滾動（設定同 retail_test_v4.py），ΔAUC 相對技術面；✔=週區塊拔靴 95% 區間不含 0")
res = {}
for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    yte = y[te].values; wk = date[te].dt.strftime("%G-%V").values; u = np.unique(wk)
    wi = {w: np.where(wk == w)[0] for w in u}; rs_ = np.random.default_rng(Y)
    boots = [np.concatenate([wi[w] for w in rs_.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
    P = {}
    for arm, cols in ARMS.items():
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=16,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        P[arm] = np.mean(ps, axis=0)
    base = roc_auc_score(yte, P["技術面"]); res[Y] = {}
    line = f"  {Y}: 技術面 {base:.4f}"
    for arm in list(ARMS)[1:]:
        d = roc_auc_score(yte, P[arm]) - base
        bd = np.array([roc_auc_score(yte[b], P[arm][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
        lo = float(np.percentile(bd, 2.5)); res[Y][arm] = {"d": float(d), "lo": lo, "hi": float(np.percentile(bd, 97.5))}
        line += f"｜{arm} {d:+.4f}{'✔' if lo > 0 else ''}"
    mm = roc_auc_score(yte, P["+融資融券+散戶成交比重"]) - roc_auc_score(yte, P["+融資融券"])
    res[Y]["rs_over_margin"] = float(mm)
    say(line + f"｜(散戶比重在融資融券之上 {mm:+.4f})")
say("  --- 7 年彙總 ---")
for arm in list(ARMS)[1:]:
    ds = np.array([res[Y][arm]["d"] for Y in TEST_YEARS]); sig = sum(res[Y][arm]["lo"] > 0 for Y in TEST_YEARS)
    t, p = tp(ds)
    say(f"  {arm:<16s} 平均ΔAUC {ds.mean():+.4f}（t={t:.2f}，p={p:.4f}）｜顯著 {sig}/7｜為正 {(ds > 0).sum()}/7")
ds = np.array([res[Y]["rs_over_margin"] for Y in TEST_YEARS]); t, p = tp(ds)
say(f"  散戶比重在融資融券之上（探索性） 平均ΔAUC {ds.mean():+.4f}（t={t:.2f}，p={p:.4f}）｜為正 {(ds > 0).sum()}/7")
ds = np.array([res[Y]["+散戶成交比重★P1"]["d"] for Y in TEST_YEARS]); t, p = tp(ds)
say(f"  → P1：平均ΔAUC {ds.mean():+.4f}，t={t:.2f}，p={p:.4f}（門檻 {ALPHA}）→ {'通過' if (p < ALPHA and ds.mean() > 0) else '未通過'}")
open("tej_retail_daily.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
json.dump({"sort": sort_res, "model": {str(k): v for k, v in res.items()}},
          open("tej_retail_daily_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
say("DONE")
