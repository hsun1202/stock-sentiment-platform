# -*- coding: utf-8 -*-
"""
月頻機器學習檢驗（Gu-Kelly-Xiu 式）：預測「下個月」個股報酬的橫斷面排序，看 PTT / 融資融券 / 法人 有沒有增量
============================================================================
目的：回答「結果不好是不是因為預測任務設計（日頻漲跌、股票混在一起）」——改成文獻中機器學習有效的月頻橫斷面設定重測。
資料：final_dataset_v3/all_stocks_v3.csv（265檔、2015-2024）彙總成「股票×月」面板；特徵只用該月最後一個交易日（含）以前的資訊。
目標：下個月報酬（月底收盤→次月底收盤）在當月橫斷面的百分位排名（-0.5~+0.5）。
特徵組：技術面（動能 1/3/6/12-1 月、波動、最大單日漲幅、流動性/成交值、量能趨勢、均線乖離、RSI、52週高點距離）
        +PTT（留言量、淨情緒、3月平均、變化、文章量）；+融資融券（融資/融券月增減、券資比）；+法人（外資/投信/自營）對照
模型：Ridge（線性）、XGBoost（樹）、MLP（神經網路，5 seeds 平均）
切分：測試年 2018~2024。測試年 Y 的模型用「特徵月 ≤ Y-2 年底」訓練、「Y-1 全年」做驗證（Ridge 選 alpha、XGBoost early stopping），
      Y 年內固定不重訓（保守）。測試特徵月 Y 年 1~12 月（預測次月報酬；2024-12 的次月報酬缺，剔除）。
評估：每月 Rank IC（預測與次月報酬的 Spearman）的平均與 t；預測前20% − 後20% 的等權多空報酬（毛報酬）；前20% − 全體等權；
      與「只有技術面」的 ΔIC（逐月配對 t）。
python monthly_ml_test.py            # 完整
SMOKE=1 python monthly_ml_test.py    # 小測試（只 2024）
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SMOKE = os.environ.get("SMOKE") == "1"
TEST_YEARS = [2024] if SMOKE else list(range(2018, 2025))
XSEEDS, NSEEDS = (1, 1) if SMOKE else (3, 5)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------- 日資料 → 股票×月 面板 ----------------
use = ["stock_code", "Date", "Close", "close_unadj", "Volume", "abnormal_price", "MA60", "RSI14",
       "ptt_pre_n", "ptt_pre_pos", "ptt_pre_neg", "ptt_intra_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_art_n",
       "margin_bal", "short_bal", "short_margin_ratio", "foreign_ratio", "trust_ratio", "dealer_ratio"]
df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
df["r"] = g.Close.pct_change().where(~(abn | abn.groupby(df.stock_code).shift(1).fillna(False)))
df["dv"] = (df.close_unadj * df.Volume).where(df.Volume > 0)
df["vpos"] = df.Volume.where(df.Volume > 0)
df["ym"] = df.Date.dt.to_period("M")
for c in ("pre", "intra"):
    for k in ("n", "pos", "neg"):
        df[f"ptt_{c}_{k}"] = df[f"ptt_{c}_{k}"].fillna(0)
df["ptt_n"] = df.ptt_pre_n + df.ptt_intra_n
df["ptt_pos"] = df.ptt_pre_pos + df.ptt_intra_pos
df["ptt_neg"] = df.ptt_pre_neg + df.ptt_intra_neg
df["ptt_art_n"] = df.ptt_art_n.fillna(0)
for c in ("foreign_ratio", "trust_ratio", "dealer_ratio"):
    df[c] = df[c].clip(-1, 1)

ok = df[(~abn) & df.Close.notna()]
last = ok.drop_duplicates(["stock_code", "ym"], keep="last").set_index(["stock_code", "ym"])[
    ["Close", "MA60", "RSI14", "margin_bal", "short_bal", "short_margin_ratio"]]
agg = df.groupby(["stock_code", "ym"]).agg(
    vol1m=("r", "std"), maxret=("r", "max"), n_r=("r", "count"), dv_med=("dv", "median"), vavg=("vpos", "mean"),
    ptt_n=("ptt_n", "sum"), ptt_pos=("ptt_pos", "sum"), ptt_neg=("ptt_neg", "sum"), ptt_art_n=("ptt_art_n", "sum"),
    foreign_m=("foreign_ratio", "mean"), trust_m=("trust_ratio", "mean"), dealer_m=("dealer_ratio", "mean"))
P = last.join(agg).reset_index().sort_values(["stock_code", "ym"]).reset_index(drop=True)
P["mi"] = P.ym.dt.year * 12 + P.ym.dt.month
gp = P.groupby("stock_code")


def lag(s, k):
    return s.groupby(P.stock_code).shift(k).where(P.mi.groupby(P.stock_code).shift(k) == P.mi - k)


C = P.Close
P["mom1"] = C / lag(C, 1) - 1
P["mom3"] = C / lag(C, 3) - 1
P["mom6"] = C / lag(C, 6) - 1
P["mom12_1"] = lag(C, 1) / lag(C, 12) - 1
P["hi52"] = C / C.groupby(P.stock_code).transform(lambda x: x.rolling(12, min_periods=6).max()) - 1
P.loc[P.n_r < 10, ["vol1m", "maxret"]] = np.nan
P["logdv"] = np.log(P.dv_med)
vprev = sum(lag(P.vavg, k) for k in range(1, 7)) / 6
P["voltrend"] = np.log(P.vavg / vprev)
P["ma60r"] = C / P.MA60 - 1
P["rsi"] = P.RSI14
P["ptt_att"] = np.log1p(P.ptt_n)
P["ptt_net"] = np.where(P.ptt_pos + P.ptt_neg >= 5, (P.ptt_pos - P.ptt_neg) / (P.ptt_pos + P.ptt_neg).replace(0, 1), np.nan)
P["ptt_net3"] = P.groupby("stock_code").ptt_net.transform(lambda x: x.rolling(3, min_periods=1).mean())
P["ptt_dnet"] = P.ptt_net - lag(P.ptt_net, 1)
P["ptt_art_att"] = np.log1p(P.ptt_art_n)
mb, sb = lag(P.margin_bal, 1), lag(P.short_bal, 1)
P["margin_chg_m"] = (P.margin_bal / mb.where(mb > 0) - 1).clip(-0.5, 0.5)
P["short_chg_m"] = (P.short_bal / sb.where(sb > 0) - 1).clip(-1, 3)
P["smr"] = P.short_margin_ratio.clip(0, 1)
P["ret_next"] = lag(C, -1) / C - 1
P["ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1).where(P.mi.groupby(P.stock_code).shift(-1) == P.mi + 1)
P["tgt"] = P.groupby("ym").ret_next.rank(pct=True) - 0.5

TECH = ["mom1", "mom3", "mom6", "mom12_1", "vol1m", "maxret", "logdv", "voltrend", "ma60r", "rsi", "hi52"]
PTT = ["ptt_att", "ptt_net", "ptt_net3", "ptt_dnet", "ptt_art_att"]
MARGIN = ["margin_chg_m", "short_chg_m", "smr"]
INST = ["foreign_m", "trust_m", "dealer_m"]
ARMS = {"技術面": TECH, "+PTT": TECH + PTT, "+融資融券": TECH + MARGIN, "+PTT+融資融券": TECH + PTT + MARGIN, "+法人(對照)": TECH + INST}
BASE = "技術面"
P = P.replace([np.inf, -np.inf], np.nan)
P["year"] = P.ym.dt.year
P.to_pickle("monthly_panel.pkl")
say(f"[面板] 股票×月 {len(P):,} 列、{P.stock_code.nunique()} 檔、{P.ym.min()}~{P.ym.max()}；有目標 {int(P.ret_next.notna().sum()):,} 列；"
    f"每月平均 {P.groupby('ym').ret_next.count().mean():.0f} 檔")
say(f"[設定] SMOKE={SMOKE} 測試年 {TEST_YEARS[0]}~{TEST_YEARS[-1]}｜XGB {XSEEDS} seeds、MLP {NSEEDS} seeds")


def ic_by_month(d, pred):
    out = {}
    for ym, x in d.assign(p=pred).groupby("ym"):
        if x.ret_next.notna().sum() >= 30:
            out[ym] = spearmanr(x.p, x.ret_next, nan_policy="omit")[0]
    return pd.Series(out)


def prep(cols, tr):
    imp = SimpleImputer(strategy="median").fit(P.loc[tr, cols])
    sc = StandardScaler().fit(imp.transform(P.loc[tr, cols]))
    return lambda m: np.clip(sc.transform(imp.transform(P.loc[m, cols])), -5, 5)


def fit_predict(model, cols, tr, va, te):
    ytr, yva = P.loc[tr, "tgt"], P.loc[va, "tgt"]
    if model == "XGB":
        ps = []
        for s in range(XSEEDS):
            m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=5, early_stopping_rounds=30, tree_method="hist",
                                 n_jobs=4, random_state=s, verbosity=0)
            m.fit(P.loc[tr, cols], ytr, eval_set=[(P.loc[va, cols], yva)], verbose=False)
            ps.append(m.predict(P.loc[te, cols]))
        return np.mean(ps, axis=0)
    f = prep(cols, tr)
    Xtr, Xva, Xte = f(tr), f(va), f(te)
    if model == "Ridge":
        best = max([1, 10, 100, 1e3, 1e4], key=lambda a: ic_by_month(P[va], Ridge(alpha=a).fit(Xtr, ytr).predict(Xva)).mean())
        return Ridge(alpha=best).fit(Xtr, ytr).predict(Xte)
    ps = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                       n_iter_no_change=15, random_state=s).fit(Xtr, ytr).predict(Xte) for s in range(NSEEDS)]
    return np.mean(ps, axis=0)


valid_row = P.ret_next.notna() & P.tgt.notna()
MODELS = ["Ridge", "XGB", "MLP"]
res = {m: {a: {"ic": [], "ls": [], "q5ew": []} for a in ARMS} for m in MODELS}
for Y in TEST_YEARS:
    tr = valid_row & (P.year <= Y - 2)
    va = valid_row & (P.year == Y - 1)
    te = valid_row & (P.year == Y)
    say(f"[{Y}] 訓練 {int(tr.sum()):,}／驗證 {int(va.sum()):,}／測試 {int(te.sum()):,}（{P.loc[te, 'ym'].nunique()} 個月）")
    d = P[te]
    for model in MODELS:
        for arm, cols in ARMS.items():
            pred = fit_predict(model, cols, tr, va, te)
            x = d.assign(p=pred)
            ic = ic_by_month(d, pred)
            q = x.groupby("ym").p.rank(pct=True)
            x["q5"] = q > 0.8
            x["q1"] = q <= 0.2
            ls = x.groupby("ym").apply(lambda z: z.loc[z.q5, "ret_next"].mean() - z.loc[z.q1, "ret_next"].mean())
            q5ew = x.groupby("ym").apply(lambda z: z.loc[z.q5, "ret_next"].mean() - z.ret_next.mean())
            res[model][arm]["ic"].append(ic)
            res[model][arm]["ls"].append(ls)
            res[model][arm]["q5ew"].append(q5ew)


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan


say("\n================ 月頻橫斷面：Rank IC、多空組合（預測前20%−後20%，毛報酬，%/月）、前20%−全體（%/月）================")
for model in MODELS:
    say(f"\n--- 模型 {model} ---")
    base_ic = pd.concat(res[model][BASE]["ic"])
    for arm in ARMS:
        ic = pd.concat(res[model][arm]["ic"]); ls = pd.concat(res[model][arm]["ls"]); q5 = pd.concat(res[model][arm]["q5ew"])
        line = f"  {arm:<12s} IC {ic.mean():+.4f}（t={tstat(ic):+.2f}）｜多空 {ls.mean()*100:+.2f}%（t={tstat(ls):+.2f}）｜前20%−全體 {q5.mean()*100:+.2f}%（t={tstat(q5):+.2f}）"
        if arm != BASE:
            dlt = (ic - base_ic).dropna()
            line += f"｜ΔIC {dlt.mean():+.4f}（t={tstat(dlt):+.2f}，月數為正 {(dlt > 0).mean():.0%}）"
        say(line)
    for arm in ["技術面", "+PTT", "+融資融券"]:
        ic = pd.concat(res[model][arm]["ic"])
        yr = ic.groupby(ic.index.year).mean()
        say(f"    各年 IC [{arm}] " + " ".join(f"{y}:{v:+.3f}" for y, v in yr.items()))
say("\n（參考）每月約 %d 檔；單月 IC 標準誤約 0.06；IC 0.02~0.05 在文獻中已算有用。" % P[valid_row].groupby("ym").size().mean())
open(f"monthly_ml_test{'_smoke' if SMOKE else ''}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
