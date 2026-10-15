# -*- coding: utf-8 -*-
"""
月頻、全市場、只做多、扣成本：技術面＋散戶交易行為 每月選股，能不能贏過大盤？
（前面所有日頻做法都輸給成本；月頻是唯一毛報酬幅度夠大的地方，且從未在全市場測過、未扣成本，2025 年完全沒用過。）
資料：final_dataset_v3_tejall25（TEJ 全市場四碼普通股 1,154 檔、2015~2025、還原價、含已下市）。
面板：股票×月；特徵只用該月最後一個交易日（含）以前的資訊。
特徵：技術面 11 個（同 monthly_ml_test.py：動能 1/3/6/12-1 月、月波動、最大單日漲幅、成交值、量能趨勢、60 日均線乖離、RSI、距 52 週高點）
      散戶行為 4 個（月底券資比、融資月增減、當月平均信用交易比重、當月平均散戶成交比重）
模型：XGBoost 迴歸（超參數同 monthly_ml_test.py，3 seeds），目標＝下月報酬的當月橫斷面排名；
      測試年 Y 用「≤Y−2 年」訓練、「Y−1 年」驗證，Y 年內不重訓。測試年 2018~2025。
可交易範圍：當月日成交值中位數排在最低 20% 的股票不買（流動性太差）；其餘都可以買。
投資組合：每月底買進預測分數最高的 20%（等權），持有一個月，下月底再換。
下市處理：下個月有交易就用下個月最後一個收盤價；下個月完全沒有價格（已下市）→ 當月報酬記 −50%（保守假設）。
成本：依換股比例計算。買進 0.0855%（手續費六折）、賣出 0.3855%（含證交稅 0.3%）→ 換一檔來回 0.471%；另報無折扣 0.585%。
基準：同一個可交易範圍的全部股票等權（每月再平衡，不計成本）；另報市值加權。

事先寫定的主要檢定（2026-10-11，跑之前寫下；2 個，Bonferroni |t|>2.26〔p<0.025，約 95 個月〕）：
  S1：「技術面＋散戶行為」模型的前 20% 組合，扣 0.471% 成本後的月報酬 − 等權基準 > 0（2018~2025 逐月 t）。
  S2：同上，但只看 2025 年（完全沒用過的年份）——只報數字與方向，12 個月不足以做顯著性判定。
  次要：只有技術面的模型；散戶行為的增量（兩個模型的淨超額差）；分年；年化報酬、波動、Sharpe、最大回落；換股率；
        前 20% − 後 20% 的毛報酬差；只買大型股（市值前 50%）的版本。
python monthly_fullmarket_strategy.py
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
LOG = []
C_DISC, C_FULL = 0.00471, 0.00585
XSEEDS = 3


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


use = ["stock_code", "Date", "Close", "close_unadj", "Volume", "abnormal_price", "MA60", "RSI14", "mktcap",
       "margin_bal", "short_margin_ratio", "margin_eligible", "inst_share", "credit_share"]
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
df["r"] = g.Close.pct_change().where(~(abn | abn.groupby(df.stock_code).shift(1).fillna(False)))
df["dv"] = (df.close_unadj * df.Volume).where(df.Volume > 0)
df["vpos"] = df.Volume.where(df.Volume > 0)
df["rs"] = ((100 - df.inst_share) / 100).where(df.Volume > 0)
df["cr"] = (df.credit_share / 100).where(df.Volume > 0)
df["ym"] = df.Date.dt.to_period("M")
last = df[df.Close.notna()].drop_duplicates(["stock_code", "ym"], keep="last").set_index(["stock_code", "ym"])[
    ["Close", "MA60", "RSI14", "mktcap", "margin_bal", "short_margin_ratio", "margin_eligible"]]
agg = df.groupby(["stock_code", "ym"]).agg(vol1m=("r", "std"), maxret=("r", "max"), n_r=("r", "count"), dv_med=("dv", "median"),
                                           vavg=("vpos", "mean"), rs_m=("rs", "mean"), cr_m=("cr", "mean"), n_tr=("vpos", "count"))
P = last.join(agg).reset_index().sort_values(["stock_code", "ym"]).reset_index(drop=True)
P["mi"] = P.ym.dt.year * 12 + P.ym.dt.month
gp = P.groupby("stock_code")


def lag(s, k):
    return s.groupby(P.stock_code).shift(k).where(P.mi.groupby(P.stock_code).shift(k) == P.mi - k)


C = P.Close
P["mom1"], P["mom3"], P["mom6"] = C / lag(C, 1) - 1, C / lag(C, 3) - 1, C / lag(C, 6) - 1
P["mom12_1"] = lag(C, 1) / lag(C, 12) - 1
P["hi52"] = C / C.groupby(P.stock_code).transform(lambda x: x.rolling(12, min_periods=6).max()) - 1
P.loc[P.n_r < 10, ["vol1m", "maxret"]] = np.nan
P["logdv"] = np.log(P.dv_med)
P["voltrend"] = np.log(P.vavg / (sum(lag(P.vavg, k) for k in range(1, 7)) / 6))
P["ma60r"], P["rsi"] = C / P.MA60 - 1, P.RSI14
mb = lag(P.margin_bal, 1)
P["margin_chg_m"] = (P.margin_bal / mb.where(mb > 0) - 1).clip(-0.5, 0.5)
P["smr"] = P.short_margin_ratio.clip(0, 1)
P.loc[P.n_tr < 10, ["rs_m", "cr_m"]] = np.nan
# 下月報酬（含下市處理）
last_mi = P.groupby("stock_code").mi.transform("max")
data_end = P.mi.max()
nxt_ok = P.mi.groupby(P.stock_code).shift(-1) == P.mi + 1
P["ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1).where(nxt_ok)
delist = P.ret_next.isna() & (P.mi == last_mi) & (P.mi < data_end)
P.loc[delist, "ret_next"] = -0.5
gap = P.ret_next.isna() & (P.mi < last_mi)                   # 中間缺月（停牌）：用下一個有價格的月份
P.loc[gap, "ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1)[gap]
P["year"] = P.ym.dt.year
P = P.replace([np.inf, -np.inf], np.nan)
P["liq_ok"] = P.groupby("ym").dv_med.rank(pct=True) > 0.2
P["big"] = P.groupby("ym").mktcap.rank(pct=True) > 0.5
P["tgt"] = P.groupby("ym").ret_next.rank(pct=True) - 0.5
TECH = ["mom1", "mom3", "mom6", "mom12_1", "vol1m", "maxret", "logdv", "voltrend", "ma60r", "rsi", "hi52"]
RETAIL = ["smr", "margin_chg_m", "cr_m", "rs_m"]
ARMS = {"技術面": TECH, "技術面＋散戶行為★": TECH + RETAIL}
say(f"[面板] 股票×月 {len(P):,} 列、{P.stock_code.nunique()} 檔、{P.ym.min()}~{P.ym.max()}；每月平均 {P.groupby('ym').size().mean():.0f} 檔，其中可交易（成交值非最低 20%）{P[P.liq_ok].groupby('ym').size().mean():.0f} 檔")
say(f"  下市當月記 −50% 的有 {int(delist.sum())} 個 股票×月；中間停牌跨月的 {int(gap.sum())} 個")

valid = P.ret_next.notna() & P.tgt.notna()
pred = {a: pd.Series(np.nan, index=P.index) for a in ARMS}
for Y in range(2018, 2026):
    tr, va, te = valid & (P.year <= Y - 2), valid & (P.year == Y - 1), valid & (P.year == Y)
    for arm, cols in ARMS.items():
        ps = []
        for s in range(XSEEDS):
            m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5, early_stopping_rounds=30, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
            m.fit(P.loc[tr, cols], P.loc[tr, "tgt"], eval_set=[(P.loc[va, cols], P.loc[va, "tgt"])], verbose=False)
            ps.append(m.predict(P.loc[te, cols]))
        pred[arm][te] = np.mean(ps, axis=0)
T = P[valid & (P.year >= 2018)].copy()
for a in ARMS:
    T["p_" + a] = pred[a][T.index]


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def perf(r):
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    return ann, vol, r.mean() * 12 / vol, (eq / eq.cummax() - 1).min()


def run(U, label):
    say(f"\n===== {label}（每月平均 {U.groupby('ym').size().mean():.0f} 檔可買）=====")
    months = sorted(U.ym.unique())
    bench = U.groupby("ym").ret_next.mean()
    vw = U.groupby("ym").apply(lambda z: np.average(z.ret_next, weights=z.mktcap.fillna(0) + 1e-9))
    a, v, sh, dd = perf(bench)
    say(f"  基準 等權：年化 {a:+.1%}｜波動 {v:.1%}｜Sharpe {sh:.2f}｜最大回落 {dd:.1%}；市值加權：年化 {perf(vw)[0]:+.1%}、Sharpe {perf(vw)[2]:.2f}")
    out = {}
    for arm in ARMS:
        col = "p_" + arm
        q = U.groupby("ym")[col].rank(pct=True)
        top, bot = U[q > 0.8], U[q <= 0.2]
        gross = top.groupby("ym").ret_next.mean()
        ls = gross - bot.groupby("ym").ret_next.mean()
        ic = U.groupby("ym").apply(lambda z: spearmanr(z[col], z.ret_next)[0])
        hold = top.groupby("ym").stock_code.apply(set)
        turn = pd.Series({m: (1.0 if i == 0 else len(hold[m] - hold[months[i - 1]]) / len(hold[m])) for i, m in enumerate(months)})
        net, netf = gross - turn * C_DISC, gross - turn * C_FULL
        ex = net - bench
        out[arm] = ex
        a, v, sh, dd = perf(net)
        say(f"  {arm}")
        say(f"    月 Rank IC {ic.mean():+.4f}（t={tstat(ic):+.1f}）｜前20%−後20% 毛 {ls.mean():+.2%}/月（t={tstat(ls):+.1f}）｜每月換股 {turn.iloc[1:].mean():.0%}")
        say(f"    前20% 組合：毛 {gross.mean():+.2%}/月｜扣 0.471% 後 {net.mean():+.2%}/月 → 年化 {a:+.1%}、波動 {v:.1%}、Sharpe {sh:.2f}、最大回落 {dd:.1%}")
        say(f"    相對等權基準：毛 {(gross - bench).mean():+.2%}/月（t={tstat(gross - bench):+.2f}）｜扣 0.471% {ex.mean():+.2%}/月（t={tstat(ex):+.2f}，贏的月數 {(ex > 0).mean():.0%}）"
            f"｜扣 0.585% {(netf - bench).mean():+.2%}/月（t={tstat(netf - bench):+.2f}）")
        yr = ex.groupby(ex.index.year).mean() * 12
        say("    分年（扣 0.471% 後相對基準，年化）：" + " ".join(f"{y}:{x:+.1%}" for y, x in yr.items()) + f"｜為正 {(yr > 0).sum()}/{len(yr)}")
    d = (out["技術面＋散戶行為★"] - out["技術面"]).dropna()
    say(f"  散戶行為的增量（兩模型扣成本後超額的差）：{d.mean():+.2%}/月（t={tstat(d):+.2f}）")
    return out


o = run(T[T.liq_ok], "全市場（成交值非最低 20%）")
ex = o["技術面＋散戶行為★"]
say(f"\n→ S1：2018~2025 扣 0.471% 後相對等權基準 {ex.mean():+.2%}/月，t={tstat(ex):+.2f}，{len(ex)} 個月 → {'通過' if (ex.mean() > 0 and tstat(ex) > 2.26) else '未通過'}")
e25 = ex[ex.index.year == 2025]
say(f"→ S2：2025 年（完全沒用過）{e25.mean():+.2%}/月、{len(e25)} 個月中 {(e25 > 0).sum()} 個月贏基準、全年累計 {(1 + e25).prod() - 1:+.1%}（只報數字）")
run(T[T.liq_ok & T.big], "只買大型股（市值前 50%，且成交值非最低 20%）")
T[["stock_code", "ym", "ret_next", "liq_ok", "big", "mktcap"] + ["p_" + a for a in ARMS]].to_pickle("monthly_fullmarket_preds.pkl")
open("monthly_fullmarket_strategy.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
