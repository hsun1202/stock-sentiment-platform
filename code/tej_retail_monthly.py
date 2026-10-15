# -*- coding: utf-8 -*-
"""
TEJ 新檢驗 (1)-月頻：散戶成交比重（＝100−法人成交比重）對「下個月」個股報酬橫斷面有沒有預測力，方法比照 monthly_ml_test.py。
資料：final_dataset_v3_tej/all_stocks_v3.csv（265 檔、TEJ 還原股價）＋ tej_raw/tej_chip_daily.pkl；股票×月面板 2015~2024。

事先寫定的假說（2026-10-05，跑之前寫下；主要檢定共 4 個〔日頻 P1、P2 見 tej_retail_daily.py〕，Bonferroni 門檻 p<0.0125）：
  P3（單一變數）：當月平均散戶成交比重（扣掉 log 市值後的殘差）與下月報酬的 Rank IC ≠ 0，2015~2024 逐月 IC 的 t 檢定
      （預期方向為負：散戶交易占比高的股票之後表現較差）。
  P4（模型增量）：技術面＋散戶成交比重的 XGBoost，下月報酬 Rank IC 比只有技術面高（2018~2024 逐月配對 t）。
      對照：+融資融券、+法人。
  探索性：未扣規模的水準、月變化、信用交易比重、Ridge、分期、五分位多空報酬。
python tej_retail_monthly.py
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PATH = "final_dataset_v3_tej/all_stocks_v3.csv"
TEST_YEARS, XSEEDS, ALPHA = list(range(2018, 2025)), 3, 0.05 / 4
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def tp(s):
    s = pd.Series(s).dropna()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    return t, 2 * stats.t.sf(abs(t), len(s) - 1)


# ---------------- 日資料 → 股票×月 面板（技術面/融資融券/法人定義同 monthly_ml_test.py）----------------
use = ["stock_code", "Date", "Close", "close_unadj", "Volume", "abnormal_price", "MA60", "RSI14", "mktcap",
       "margin_bal", "short_bal", "short_margin_ratio", "foreign_ratio", "trust_ratio", "dealer_ratio"]
df = pd.read_csv(PATH, dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
chip = pd.read_pickle("tej_raw/tej_chip_daily.pkl")[["code", "date", "法人成交比重", "信用交易比重"]]
chip = chip.drop_duplicates(["code", "date"]).rename(columns={"code": "stock_code", "date": "Date"})
n0 = len(df)
df = df.merge(chip, on=["stock_code", "Date"], how="left")
assert len(df) == n0
df["rs"] = ((100 - df["法人成交比重"]) / 100).where(df.Volume > 0)
df["cr"] = (df["信用交易比重"] / 100).where(df.Volume > 0)
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
df["r"] = g.Close.pct_change().where(~(abn | abn.groupby(df.stock_code).shift(1).fillna(False)))
df["dv"] = (df.close_unadj * df.Volume).where(df.Volume > 0)
df["vpos"] = df.Volume.where(df.Volume > 0)
df["ym"] = df.Date.dt.to_period("M")
for c in ("foreign_ratio", "trust_ratio", "dealer_ratio"):
    df[c] = df[c].clip(-1, 1)

ok = df[(~abn) & df.Close.notna()]
last = ok.drop_duplicates(["stock_code", "ym"], keep="last").set_index(["stock_code", "ym"])[
    ["Close", "MA60", "RSI14", "margin_bal", "short_bal", "short_margin_ratio", "mktcap"]]
agg = df.groupby(["stock_code", "ym"]).agg(
    vol1m=("r", "std"), maxret=("r", "max"), n_r=("r", "count"), dv_med=("dv", "median"), vavg=("vpos", "mean"),
    foreign_m=("foreign_ratio", "mean"), trust_m=("trust_ratio", "mean"), dealer_m=("dealer_ratio", "mean"),
    rs_m=("rs", "mean"), cr_m=("cr", "mean"), n_rs=("rs", "count"))
P = last.join(agg).reset_index().sort_values(["stock_code", "ym"]).reset_index(drop=True)
P["mi"] = P.ym.dt.year * 12 + P.ym.dt.month


def lag(s, k):
    return s.groupby(P.stock_code).shift(k).where(P.mi.groupby(P.stock_code).shift(k) == P.mi - k)


C = P.Close
P["mom1"] = C / lag(C, 1) - 1
P["mom3"] = C / lag(C, 3) - 1
P["mom6"] = C / lag(C, 6) - 1
P["mom12_1"] = lag(C, 1) / lag(C, 12) - 1
P["hi52"] = C / C.groupby(P.stock_code).transform(lambda x: x.rolling(12, min_periods=6).max()) - 1
P.loc[P.n_r < 10, ["vol1m", "maxret"]] = np.nan
P.loc[P.n_rs < 10, ["rs_m", "cr_m"]] = np.nan
P["logdv"] = np.log(P.dv_med)
vprev = sum(lag(P.vavg, k) for k in range(1, 7)) / 6
P["voltrend"] = np.log(P.vavg / vprev)
P["ma60r"] = C / P.MA60 - 1
P["rsi"] = P.RSI14
mb, sb = lag(P.margin_bal, 1), lag(P.short_bal, 1)
P["margin_chg_m"] = (P.margin_bal / mb.where(mb > 0) - 1).clip(-0.5, 0.5)
P["short_chg_m"] = (P.short_bal / sb.where(sb > 0) - 1).clip(-1, 3)
P["smr"] = P.short_margin_ratio.clip(0, 1)
P["rs_chg"] = P.rs_m - sum(lag(P.rs_m, k) for k in range(1, 7)) / 6
P["cr_chg"] = P.cr_m - sum(lag(P.cr_m, k) for k in range(1, 7)) / 6
P["logmv"] = np.log(P.mktcap.where(P.mktcap > 0))
okxy = P.logmv.notna() & P.rs_m.notna()
xd = (P.logmv - P.logmv.where(okxy).groupby(P.ym).transform("mean")).where(okxy)
yd = (P.rs_m - P.rs_m.where(okxy).groupby(P.ym).transform("mean")).where(okxy)
P["rs_adj"] = yd - (xd * yd).groupby(P.ym).transform("sum") / (xd ** 2).groupby(P.ym).transform("sum") * xd
P["ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1).where(P.mi.groupby(P.stock_code).shift(-1) == P.mi + 1)
P["tgt"] = P.groupby("ym").ret_next.rank(pct=True) - 0.5
P = P.replace([np.inf, -np.inf], np.nan)
P["year"] = P.ym.dt.year
say(f"[面板] 股票×月 {len(P):,} 列、{P.stock_code.nunique()} 檔、{P.ym.min()}~{P.ym.max()}；有目標 {int(P.ret_next.notna().sum()):,} 列；"
    f"散戶成交比重有值 {P.rs_m.notna().mean():.1%}；與 log 市值的橫斷面相關（逐月平均）"
    f" {P.groupby('ym').apply(lambda z: z.rs_m.corr(z.logmv)).mean():+.2f}")

# ---------------- (a) 單一變數：逐月 Rank IC 與五分位多空 ----------------
say("\n(a) 單一變數與下月報酬：逐月 Rank IC、五分位（最高−最低）等權毛報酬 %/月｜2015~2024")
V = P[P.ret_next.notna()]
single = {}
for col, name in [("rs_adj", "散戶成交比重(扣規模)★P3"), ("rs_m", "散戶成交比重(水準)"), ("rs_chg", "散戶成交比重月變化"),
                  ("cr_m", "信用交易比重(水準)"), ("cr_chg", "信用交易比重月變化"),
                  ("smr", "券資比(對照)"), ("margin_chg_m", "融資月增減(對照)"), ("logmv", "log市值(參考)")]:
    ic, ls = {}, {}
    for ym, z in V[V[col].notna()].groupby("ym"):
        if len(z) >= 30:
            ic[ym] = spearmanr(z[col], z.ret_next)[0]
            q = z[col].rank(pct=True)
            ls[ym] = z.ret_next[q > 0.8].mean() - z.ret_next[q <= 0.2].mean()
    ic, ls = pd.Series(ic), pd.Series(ls)
    t, p = tp(ic)
    single[col] = (ic.mean(), t, p)
    h1, h2 = ic[ic.index.year <= 2019], ic[ic.index.year >= 2020]
    say(f"  {name:<18s} IC {ic.mean():+.4f}（t={t:+.2f}，p={p:.4f}，{len(ic)} 個月）｜2015-19 {h1.mean():+.3f}（t={tp(h1)[0]:+.2f}）"
        f"、2020-24 {h2.mean():+.3f}（t={tp(h2)[0]:+.2f}）｜高−低 {ls.mean()*100:+.2f}%/月（t={tp(ls)[0]:+.2f}）")
m, t, p = single["rs_adj"]
say(f"  → P3：IC {m:+.4f}，t={t:+.2f}，p={p:.4f}（門檻 {ALPHA}）→ {'通過' if p < ALPHA else '未通過'}")

# ---------------- (b) 模型增量 ----------------
TECH = ["mom1", "mom3", "mom6", "mom12_1", "vol1m", "maxret", "logdv", "voltrend", "ma60r", "rsi", "hi52"]
RS = ["rs_m", "rs_chg"]
MARGIN = ["margin_chg_m", "short_chg_m", "smr"]
INST = ["foreign_m", "trust_m", "dealer_m"]
ARMS = {"技術面": TECH, "+散戶成交比重★P4": TECH + RS, "+融資融券": TECH + MARGIN,
        "+融資融券+散戶成交比重": TECH + MARGIN + RS, "+法人(對照)": TECH + INST}
BASE = "技術面"


def ic_by_month(d, pred):
    out = {}
    for ym, z in d.assign(p=pred).groupby("ym"):
        if z.ret_next.notna().sum() >= 30:
            out[ym] = spearmanr(z.p, z.ret_next, nan_policy="omit")[0]
    return pd.Series(out)


def fit_predict(model, cols, tr, va, te):
    ytr, yva = P.loc[tr, "tgt"], P.loc[va, "tgt"]
    if model == "XGB":
        ps = []
        for s in range(XSEEDS):
            mdl = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8,
                                   colsample_bytree=0.8, reg_lambda=5, early_stopping_rounds=30, tree_method="hist",
                                   n_jobs=4, random_state=s, verbosity=0)
            mdl.fit(P.loc[tr, cols], ytr, eval_set=[(P.loc[va, cols], yva)], verbose=False)
            ps.append(mdl.predict(P.loc[te, cols]))
        return np.mean(ps, axis=0)
    imp = SimpleImputer(strategy="median").fit(P.loc[tr, cols])
    sc = StandardScaler().fit(imp.transform(P.loc[tr, cols]))
    f = lambda msk: np.clip(sc.transform(imp.transform(P.loc[msk, cols])), -5, 5)
    Xtr, Xva, Xte = f(tr), f(va), f(te)
    best = max([1, 10, 100, 1e3, 1e4], key=lambda a: ic_by_month(P[va], Ridge(alpha=a).fit(Xtr, ytr).predict(Xva)).mean())
    return Ridge(alpha=best).fit(Xtr, ytr).predict(Xte)


valid_row = P.ret_next.notna() & P.tgt.notna()
MODELS = ["Ridge", "XGB"]
res = {mo: {a: {"ic": [], "ls": []} for a in ARMS} for mo in MODELS}
for Y in TEST_YEARS:
    tr, va, te = valid_row & (P.year <= Y - 2), valid_row & (P.year == Y - 1), valid_row & (P.year == Y)
    d = P[te]
    for model in MODELS:
        for arm, cols in ARMS.items():
            pred = fit_predict(model, cols, tr, va, te)
            z = d.assign(p=pred)
            q = z.groupby("ym").p.rank(pct=True)
            z["q5"], z["q1"] = q > 0.8, q <= 0.2
            res[model][arm]["ic"].append(ic_by_month(d, pred))
            res[model][arm]["ls"].append(z.groupby("ym").apply(lambda w: w.loc[w.q5, "ret_next"].mean() - w.loc[w.q1, "ret_next"].mean()))
say("\n(b) 模型增量：預測下月報酬橫斷面排名，測試年 2018~2024（切分同 monthly_ml_test.py）；ΔIC 為相對技術面的逐月配對")
p4 = None
for model in MODELS:
    say(f"  --- 模型 {model} ---")
    base_ic = pd.concat(res[model][BASE]["ic"])
    for arm in ARMS:
        ic, ls = pd.concat(res[model][arm]["ic"]), pd.concat(res[model][arm]["ls"])
        line = f"  {arm:<16s} IC {ic.mean():+.4f}（t={tp(ic)[0]:+.2f}）｜多空 {ls.mean()*100:+.2f}%/月（t={tp(ls)[0]:+.2f}）"
        if arm != BASE:
            dlt = (ic - base_ic).dropna()
            t, p = tp(dlt)
            line += f"｜ΔIC {dlt.mean():+.4f}（t={t:+.2f}，p={p:.4f}，月數為正 {(dlt > 0).mean():.0%}）"
            if model == "XGB" and arm == "+散戶成交比重★P4":
                p4 = (dlt.mean(), t, p)
        say(line)
    dlt = (pd.concat(res[model]["+融資融券+散戶成交比重"]["ic"]) - pd.concat(res[model]["+融資融券"]["ic"])).dropna()
    say(f"  散戶比重在融資融券之上（探索性） ΔIC {dlt.mean():+.4f}（t={tp(dlt)[0]:+.2f}）")
say(f"  → P4（XGB）：ΔIC {p4[0]:+.4f}，t={p4[1]:+.2f}，p={p4[2]:.4f}（門檻 {ALPHA}）→ {'通過' if (p4[2] < ALPHA and p4[0] > 0) else '未通過'}")
open("tej_retail_monthly.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
