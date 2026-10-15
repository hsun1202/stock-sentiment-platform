# -*- coding: utf-8 -*-
"""
把「媒體新聞」與「公司公告」也拉長成月頻，加進月頻全市場選股模型（monthly_fullmarket_strategy.py），有沒有再多一點？
（使用者 2026-10-12 問：其他訊號如果把時間拉長呢？日頻時新聞與公告方向都對、但開盤就反應完；月頻時成本低很多，值得重測。）
面板、模型、切分、成本、下市處理、可交易範圍 完全同 monthly_fullmarket_strategy.py；只多加特徵。
新增特徵（都只用當月最後一個交易日（含）以前「已經影響股價」的訊息；影響日定義同 mops_bigmove.py / cnyes_bigmove.py）：
  新聞（鉅亨網，cnyes_bigmove_matched*.csv）：當月新聞則數(log)、正面則數、負面則數、淨語氣 (正−負)/(正+負)（至少 1 則有語氣才算）、近 3 個月平均淨語氣、則數相對前 6 個月的變化
  公告（重大訊息，mops_bigmove_matched*.csv，去掉代子公司公告）：當月總則數(log)，以及 10 個類別各自的則數——
        庫藏股、澄清媒體報導、訴訟/裁罰/檢調、災害/停工/事故、法人說明會、股利分派/配股、現金增資/私募/可轉債、減資-彌補虧損、併購/公開收購/股份交換、訂單/合約/新產品
比較的模型：基準＝技術面＋散戶行為；＋新聞；＋公告；＋新聞＋公告。

事先寫定的主要檢定（2026-10-12，跑之前寫下；1 個，|t|>2）：
  A1：「基準＋新聞＋公告」前 20% 組合扣 0.471% 後的月報酬，高於「基準」前 20% 組合（逐月配對差 > 0，2018~2025）。
  次要：各自加新聞、加公告的差；月 Rank IC 的差；單一變數（淨語氣、新聞則數、各類公告則數）對下月報酬的月 Rank IC；2025 年。
python monthly_fullmarket_addinfo.py
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
LOG = []
C_DISC, XSEEDS = 0.00471, 3


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------- 面板（同 monthly_fullmarket_strategy.py）----------------
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
    ["Close", "MA60", "RSI14", "mktcap", "margin_bal", "short_margin_ratio"]]
agg = df.groupby(["stock_code", "ym"]).agg(vol1m=("r", "std"), maxret=("r", "max"), n_r=("r", "count"), dv_med=("dv", "median"),
                                           vavg=("vpos", "mean"), rs_m=("rs", "mean"), cr_m=("cr", "mean"), n_tr=("vpos", "count"))
P = last.join(agg).reset_index().sort_values(["stock_code", "ym"]).reset_index(drop=True)
P["mi"] = P.ym.dt.year * 12 + P.ym.dt.month


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
last_mi, data_end = P.groupby("stock_code").mi.transform("max"), P.mi.max()
nxt_ok = P.mi.groupby(P.stock_code).shift(-1) == P.mi + 1
P["ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1).where(nxt_ok)
P.loc[P.ret_next.isna() & (P.mi == last_mi) & (P.mi < data_end), "ret_next"] = -0.5
gap = P.ret_next.isna() & (P.mi < last_mi)
P.loc[gap, "ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1)[gap]
P["year"] = P.ym.dt.year
P = P.replace([np.inf, -np.inf], np.nan)
P["liq_ok"] = P.groupby("ym").dv_med.rank(pct=True) > 0.2
P["tgt"] = P.groupby("ym").ret_next.rank(pct=True) - 0.5
TECH = ["mom1", "mom3", "mom6", "mom12_1", "vol1m", "maxret", "logdv", "voltrend", "ma60r", "rsi", "hi52"]
RETAIL = ["smr", "margin_chg_m", "cr_m", "rs_m"]

# ---------------- 新聞（月）----------------
nw = pd.concat([pd.read_csv(f, dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "Date", "id", "tone"])
                for f in ("cnyes_bigmove_matched.csv", "cnyes_bigmove_matched_out889.csv")]).drop_duplicates(["stock_code", "id"])
nw["ym"] = nw.Date.dt.to_period("M")
nm = nw.groupby(["stock_code", "ym"]).agg(n_news=("id", "size"), n_pos=("tone", lambda s: (s == "正面").sum()), n_neg=("tone", lambda s: (s == "負面").sum())).reset_index()
P = P.merge(nm, on=["stock_code", "ym"], how="left").fillna({"n_news": 0, "n_pos": 0, "n_neg": 0})
P["news_att"] = np.log1p(P.n_news)
P["news_net"] = ((P.n_pos - P.n_neg) / (P.n_pos + P.n_neg).replace(0, np.nan))
P["news_net3"] = P.groupby("stock_code").news_net.transform(lambda x: x.rolling(3, min_periods=1).mean())
P["news_datt"] = P.news_att - sum(lag(P.news_att, k) for k in range(1, 7)) / 6
NEWS = ["news_att", "n_pos", "n_neg", "news_net", "news_net3", "news_datt"]

# ---------------- 公告（月）----------------
ACATS = {"庫藏股": "a_buyback", "澄清媒體報導": "a_clarify", "訴訟/裁罰/檢調": "a_legal", "災害/停工/事故": "a_disaster", "法人說明會": "a_ir",
         "股利分派/配股": "a_div", "現金增資/私募/可轉債": "a_issue", "減資-彌補虧損": "a_cutloss", "併購/公開收購/股份交換": "a_ma", "訂單/合約/新產品": "a_order"}
mp = pd.concat([pd.read_csv(f, dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "Date", "cat", "sub", "title"])
                for f in ("mops_bigmove_matched.csv", "mops_bigmove_matched_out889.csv")])
mp = mp[~mp["sub"].astype(str).isin(["True", "1"])].drop_duplicates(["stock_code", "Date", "title"])
mp["ym"] = mp.Date.dt.to_period("M")
am = mp.groupby(["stock_code", "ym"]).size().rename("n_ann").reset_index()
ac = mp[mp.cat.isin(ACATS)].groupby(["stock_code", "ym", "cat"]).size().unstack(fill_value=0).rename(columns=ACATS).reset_index()
P = P.merge(am, on=["stock_code", "ym"], how="left").merge(ac, on=["stock_code", "ym"], how="left")
ANN = ["ann_att"] + list(ACATS.values())
for c in ["n_ann"] + list(ACATS.values()):
    P[c] = P[c].fillna(0) if c in P else 0
P["ann_att"] = np.log1p(P.n_ann)
say(f"[面板] 股票×月 {len(P):,}；當月有新聞的 {(P.n_news > 0).mean():.1%}、有語氣的 {P.news_net.notna().mean():.1%}、有公告的 {(P.n_ann > 0).mean():.1%}")

ARMS = {"基準（技術面＋散戶行為）": TECH + RETAIL, "＋新聞": TECH + RETAIL + NEWS, "＋公告": TECH + RETAIL + ANN, "＋新聞＋公告★": TECH + RETAIL + NEWS + ANN}
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
U = P[valid & (P.year >= 2018) & P.liq_ok].copy()
for a in ARMS:
    U["p_" + a] = pred[a][U.index]
tstat = lambda s: pd.Series(s).dropna().mean() / (pd.Series(s).dropna().std(ddof=1) / np.sqrt(pd.Series(s).dropna().size))
months = sorted(U.ym.unique())
bench = U.groupby("ym").ret_next.mean()
net, ics = {}, {}
say("\n===== 月頻全市場選股：加入新聞／公告後的變化（2018~2025，前 20% 等權、扣 0.471%×換股）=====")
for arm in ARMS:
    col = "p_" + arm
    top = U[U.groupby("ym")[col].rank(pct=True) > 0.8]
    gross = top.groupby("ym").ret_next.mean()
    hold = top.groupby("ym").stock_code.apply(set)
    turn = pd.Series({m: (1.0 if i == 0 else len(hold[m] - hold[months[i - 1]]) / len(hold[m])) for i, m in enumerate(months)})
    net[arm] = gross - turn * C_DISC
    ics[arm] = U.groupby("ym").apply(lambda z: spearmanr(z[col], z.ret_next)[0])
    ex = net[arm] - bench
    ann = (1 + net[arm]).prod() ** (12 / len(ex)) - 1
    yr = ex.groupby(ex.index.year).mean() * 12
    say(f"  {arm:<14s} 月 IC {ics[arm].mean():+.4f}｜年化 {ann:+.1%}｜相對等權 {ex.mean():+.2%}/月（t={tstat(ex):+.2f}）｜換股 {turn.iloc[1:].mean():.0%}｜2025 年相對等權 {yr.get(2025, np.nan):+.1%}")
b0 = "基準（技術面＋散戶行為）"
for arm in list(ARMS)[1:]:
    d, di = net[arm] - net[b0], ics[arm] - ics[b0]
    say(f"  {arm} − 基準：淨報酬 {d.mean():+.3%}/月（t={tstat(d):+.2f}，為正的月數 {(d > 0).mean():.0%}）｜IC {di.mean():+.4f}（t={tstat(di):+.2f}）")
d = net["＋新聞＋公告★"] - net[b0]
say(f"→ A1：加入新聞＋公告後 淨報酬差 {d.mean():+.3%}/月，t={tstat(d):+.2f} → {'通過' if (d.mean() > 0 and tstat(d) > 2) else '未通過'}")

say("\n===== 單一變數對「下個月」報酬的月 Rank IC（2015~2025，全部股票；只在該變數有變化的股票×月上算）=====")
V = P[valid]
for col, nm_, cond in [("news_net", "新聞淨語氣", V.news_net.notna()), ("news_att", "新聞則數", V.n_news >= 0), ("news_datt", "新聞則數變化", V.news_datt.notna()),
                       ("ann_att", "公告則數", V.n_ann >= 0)]:
    x = V[cond]
    ic = pd.Series({ym: spearmanr(z[col], z.ret_next)[0] for ym, z in x.groupby("ym") if len(z) >= 50 and z[col].nunique() > 1})
    say(f"  {nm_:<8s} IC {ic.mean():+.4f}（t={tstat(ic):+.2f}，{len(ic)} 個月，每月約 {x.groupby('ym').size().mean():.0f} 檔）")
say("  各類公告：當月有發 vs 沒發，下個月超額報酬（個股−當月全體平均）的差")
V = V.assign(xr=V.ret_next - V.groupby("ym").ret_next.transform("mean"))
for cat, c in ACATS.items():
    has = V[V[c] > 0]
    mm = has.groupby("ym").xr.mean()
    say(f"    {cat:<14s} n={len(has):>6,}｜下個月超額 {has.xr.mean():+.2%}（t={tstat(mm):+.2f}）")
for lab, cond in [("當月新聞淨語氣為正", V.news_net > 0), ("當月新聞淨語氣為負", V.news_net < 0)]:
    has = V[cond]
    say(f"    {lab:<14s} n={len(has):>6,}｜下個月超額 {has.xr.mean():+.2%}（t={tstat(has.groupby('ym').xr.mean()):+.2f}）")
open("monthly_fullmarket_addinfo.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
