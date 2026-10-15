# -*- coding: utf-8 -*-
"""
TEJ 新檢驗 (3)：把「散戶做的」檢驗擴到全市場（四碼普通股 1,144 檔、含已下市；final_dataset_v3_tejall），回應「265 檔事後選定、倖存者偏差」的限制。
全市場沒有 PTT 配對，所以只檢驗技術面、融資融券、法人買賣超、散戶成交比重。

事先寫定的假說（2026-10-05，跑之前寫下；主要檢定 3 個，Bonferroni 門檻 p<0.0167）：
  F1：全市場，技術面＋融資融券的 XGBoost，隔天開盤→收盤方向 AUC 比只有技術面高（7 個測試年 ΔAUC 的 t 檢定）。265 檔為 +0.0022。
  F2：全市場，技術面＋散戶成交比重（特徵同 tej_retail_daily.py）ΔAUC > 0。
  F3：全市場，月頻券資比與下月報酬的 Rank IC < 0（2015~2024 逐月 t）。265 檔為 −0.061。
  探索性：只看「不在原 265 檔內」的股票（模型仍用全市場訓練）、依市值分三組、+法人、每日五分位排序、月頻信用交易比重與散戶成交比重。
注意：XGBoost 改用顯示卡（device=cuda）訓練，其餘設定同 retail_test_v4.py；未處理下市當下的報酬（下市前最後一天之後就沒有資料）。
python tej_fullmarket_test.py
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from features_v4 import build

PATH = "final_dataset_v3_tejall/all_stocks_v3.csv"
SEEDS, BOOT, TEST_YEARS, ALPHA = 5, 300, range(2018, 2025), 0.05 / 3
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
                  usecols=["stock_code", "Date", "Open", "Close", "Volume", "close_unadj", "abnormal_price", "mktcap", "margin_eligible", "in265",
                           "inst_share", "credit_share", "margin_chg", "short_chg", "short_margin_ratio", "margin_bal", "short_bal"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
assert (raw.stock_code.values == meta.stock_code.values).all() and (raw.Date.values == meta.date.values).all()
traded = raw.Volume > 0
raw["rs"] = ((100 - raw.inst_share) / 100).where(traded)
raw["cr"] = (raw.credit_share / 100).where(traded)
g = raw.groupby("stock_code")


def rmean(col, w, mp, shift=0):
    return g[col].transform(lambda x: x.rolling(w, min_periods=mp).mean().shift(shift))


raw["rs_dev"] = raw.rs - rmean("rs", 20, 10, 1)
raw["rs_5"] = rmean("rs", 5, 3)
raw["rs_dev5"] = raw.rs_5 - rmean("rs", 60, 30)
RS = ["rs", "rs_dev", "rs_5", "rs_dev5"]
for c in RS:
    feat[c] = raw[c].values
raw["size_g"] = raw.groupby("Date").mktcap.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False) if v.notna().sum() >= 30 else np.nan)
say(f"[資料] 全市場 {len(raw):,} 列、{raw.stock_code.nunique()} 檔（不在原 265 檔內 {raw[raw.in265 == 0].stock_code.nunique()} 檔）；"
    f"可當預測目標的股票日 {int(meta.valid.sum()):,}")

# ---------------- (a) 每日五分位排序 ----------------
nO, nC = g.Open.shift(-1), g.Close.shift(-1)
abn = raw.abnormal_price.fillna(0)
ok = (abn == 0) & (g.abnormal_price.shift(-1).fillna(0) == 0) & nO.notna() & nC.notna() & (nO > 0) & (nO != nC)
raw["r_oc"] = ((nC / nO - 1) * 1e4).where(ok)
raw["r_on"] = ((nO / raw.Close - 1) * 1e4).where(ok)
S = raw[(raw.Date >= "2018-01-01") & (raw.Date <= "2024-12-31")]
say(f"\n(a) 每日五分位排序（探索性）：隔天報酬 bp，等權毛報酬｜2018~2024｜全體 開→收 {S.r_oc.mean():+.1f}、隔夜 {S.r_on.mean():+.1f}")
for col, name, elig in [("rs_dev", "異常散戶成交比重", False), ("margin_chg", "融資增減", True), ("short_chg", "融券增減", True), ("short_margin_ratio", "券資比", True)]:
    for lab, d in [("全市場", S), ("不在265檔", S[S.in265 == 0])]:
        d = d[d[col].notna() & d.r_oc.notna()]
        if elig:
            d = d[d.margin_eligible == 1]
        d = d.assign(q=d.groupby("Date")[col].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1 if len(v) >= 25 else np.nan))
        line = f"  {name:<10s}［{lab}］"
        for r, rn in [("r_oc", "開→收"), ("r_on", "隔夜")]:
            daily = d.groupby(["Date", "q"])[r].mean().unstack()
            sp = daily[5] - daily[1]
            line += f" {rn} 高−低 {sp.mean():+.1f}（t={tp(sp)[0]:+.2f}）"
        say(line)

# ---------------- (b) 模型增量 ----------------
TECH = A["技術面"]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
ARMS = {"技術面": TECH, "+融資融券★F1": TECH + MARGIN, "+散戶成交比重★F2": TECH + RS, "+法人買賣超": TECH + INST}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
SUB = {"全市場": np.ones(len(raw), bool), "不在265檔": (raw.in265 == 0).values, "原265檔": (raw.in265 == 1).values,
       "市值小": (raw.size_g == 0).values, "市值中": (raw.size_g == 1).values, "市值大": (raw.size_g == 2).values}
say("\n(b) 模型增量：隔天開盤→收盤方向，XGBoost 逐年滾動（全市場訓練），ΔAUC 相對技術面；✔=週區塊拔靴 95% 區間不含 0（只對全市場算）")
res = {s: {} for s in SUB}
for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
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
    line = f"  {Y}（訓練 {int(tr.sum()):,}／測試 {int(te.sum()):,}）:"
    for sname, smask in SUB.items():
        k = smask[te.values]
        base = roc_auc_score(yte[k], P["技術面"][k])
        res[sname][Y] = {"技術面": float(base)}
        for arm in list(ARMS)[1:]:
            res[sname][Y][arm] = {"d": float(roc_auc_score(yte[k], P[arm][k]) - base)}
        if sname == "全市場":
            line += f" 技術面 {base:.4f}"
            for arm in list(ARMS)[1:]:
                bd = np.array([roc_auc_score(yte[b], P[arm][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
                lo = float(np.percentile(bd, 2.5)); res[sname][Y][arm]["lo"] = lo
                line += f"｜{arm} {res[sname][Y][arm]['d']:+.4f}{'✔' if lo > 0 else ''}"
    say(line)
say("  --- 7 年彙總（平均 ΔAUC；t 為 7 個測試年）---")
prim = {}
for sname in SUB:
    line = f"  {sname:<6s} 技術面 {np.mean([res[sname][Y]['技術面'] for Y in TEST_YEARS]):.4f}"
    for arm in list(ARMS)[1:]:
        ds = np.array([res[sname][Y][arm]["d"] for Y in TEST_YEARS]); t, p = tp(ds)
        line += f"｜{arm} {ds.mean():+.4f}（t={t:.2f}，正 {(ds > 0).sum()}/7）"
        if sname == "全市場":
            prim[arm] = (ds.mean(), t, p)
    say(line)
for arm, lab in [("+融資融券★F1", "F1"), ("+散戶成交比重★F2", "F2")]:
    m_, t, p = prim[arm]
    say(f"  → {lab}：平均ΔAUC {m_:+.4f}，t={t:.2f}，p={p:.4f}（門檻 {ALPHA:.4f}）→ {'通過' if (p < ALPHA and m_ > 0) else '未通過'}")

# ---------------- (c) 月頻單一變數 ----------------
raw["ym"] = raw.Date.dt.to_period("M")
okr = raw[(abn == 0) & raw.Close.notna()]
last = okr.drop_duplicates(["stock_code", "ym"], keep="last").set_index(["stock_code", "ym"])[["Close", "short_margin_ratio", "mktcap", "in265", "margin_bal"]]
agg = raw.groupby(["stock_code", "ym"]).agg(rs_m=("rs", "mean"), cr_m=("cr", "mean"), n=("rs", "count"))
Pm = last.join(agg).reset_index().sort_values(["stock_code", "ym"]).reset_index(drop=True)
Pm["mi"] = Pm.ym.dt.year * 12 + Pm.ym.dt.month
gm = Pm.groupby("stock_code")
nxt = gm.mi.shift(-1) == Pm.mi + 1
Pm["ret_next"] = (gm.Close.shift(-1) / Pm.Close - 1).where(nxt)
Pm["smr"] = Pm.short_margin_ratio.clip(0, 1)
mb = gm.margin_bal.shift(1).where(gm.mi.shift(1) == Pm.mi - 1)
Pm["margin_chg_m"] = (Pm.margin_bal / mb.where(mb > 0) - 1).clip(-0.5, 0.5)
Pm.loc[Pm.n < 10, ["rs_m", "cr_m"]] = np.nan
Pm["logmv"] = np.log(Pm.mktcap.where(Pm.mktcap > 0))
say(f"\n(c) 月頻單一變數與下月報酬的 Rank IC｜2015~2024｜股票×月 {int(Pm.ret_next.notna().sum()):,} 列、每月平均 {Pm[Pm.ret_next.notna()].groupby('ym').size().mean():.0f} 檔")
f3 = None
for col, name in [("smr", "券資比★F3"), ("margin_chg_m", "融資月增減"), ("cr_m", "信用交易比重"), ("rs_m", "散戶成交比重"), ("logmv", "log市值")]:
    line = f"  {name:<10s}"
    for lab, d in [("全市場", Pm), ("不在265檔", Pm[Pm.in265 == 0]), ("原265檔", Pm[Pm.in265 == 1])]:
        d = d[d[col].notna() & d.ret_next.notna()]
        ic = pd.Series({ym: spearmanr(z[col], z.ret_next)[0] for ym, z in d.groupby("ym") if len(z) >= 30})
        t, p = tp(ic)
        line += f"｜{lab} IC {ic.mean():+.4f}（t={t:+.2f}）"
        if col == "smr" and lab == "全市場":
            f3 = (ic.mean(), t, p)
    say(line)
say(f"  → F3：IC {f3[0]:+.4f}，t={f3[1]:+.2f}，p={f3[2]:.4f}（門檻 {ALPHA:.4f}）→ {'通過' if (f3[2] < ALPHA and f3[0] < 0) else '未通過'}")
open("tej_fullmarket_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
json.dump({s: {str(k): v for k, v in r.items()} for s, r in res.items()}, open("tej_fullmarket_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
say("DONE")
