# -*- coding: utf-8 -*-
"""
週頻、全市場、只做多、扣成本的選股（使用者 2026-10-15 問「那如果看週的呢」）
作法與 monthly_fullmarket_strategy.py 相同，只是把「每月」換成「每週」（以每週最後一個交易日為準）。
資料：final_dataset_v3_tejall25（TEJ 全市場 1,154 檔、2015~2025）＋ 新聞 cnyes_bigmove_matched*.csv ＋ 公告 mops_bigmove_matched*.csv。
特徵（只用該週最後一個交易日（含）以前的資訊）：
  技術面：近 1/4/12 週報酬、26 週−4 週動能、近 20 日波動、近 20 日最大單日漲幅、近 20 日成交值中位數(log)、量能趨勢(5 日/60 日均量)、60 日均線乖離、RSI14、距 250 日高點
  散戶行為：週末券資比、融資 5 日變化、當週平均信用交易比重、當週平均散戶成交比重、散戶成交比重相對前 20 日
  新聞＋公告（當週）：新聞則數、正面/負面則數、淨語氣；公告總則數與 10 類則數（同 monthly_fullmarket_addinfo.py）
  隔夜情緒（當週）：累積隔夜超額、累積盤中超額、近 4 週累積隔夜超額
模型：XGBoost 迴歸（同月頻超參數、3 seeds），目標＝下週報酬的當週橫斷面排名；測試年 Y 用 ≤Y−2 訓練、Y−1 驗證；測試 2018~2025。
可交易：近 20 日成交值最低 20% 不買。下市：下週沒有價格且之後也沒有 → 記 −50%。
組合：A. 每週買分數前 20%（等權）；B.（降低換股）已持有的只要還在前 40% 就續抱，空位由前 20% 中未持有的依分數補滿，維持與 A 相同檔數。
成本：換股比例 × 0.471%（來回，手續費六折含證交稅）。基準：同範圍等權（不計成本）、市值加權。

事先寫定的主要檢定（2026-10-15，跑之前寫下；2 個，Bonferroni |t|>2.26）：
  W1：「技術面＋散戶行為」組合 B（降低換股版）扣成本後的週報酬 − 等權基準 > 0（2018~2025 逐週 t）。
  W2：加入新聞＋公告後，組合 B 的扣成本週報酬高於不加（逐週配對差 > 0）。
  次要：組合 A；＋隔夜情緒；換股率；年化、Sharpe、回落；分年；相對市值加權；2025 年；與月頻結果對照。
python weekly_fullmarket_strategy.py
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


use = ["stock_code", "Date", "Open", "Close", "close_unadj", "Volume", "abnormal_price", "MA60", "RSI14", "mktcap",
       "margin_bal", "short_margin_ratio", "inst_share", "credit_share", "in265"]
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
okd = ~(abn | abn.groupby(df.stock_code).shift(1).fillna(False))
tr_ = df.Volume > 0
df["r"] = g.Close.pct_change().where(okd)
df["on"] = (df.Open / g.Close.shift(1) - 1).where(okd & tr_)
df["idr"] = (df.Close / df.Open - 1).where(okd & tr_)
for c in ("on", "idr"):
    df["x" + c] = df[c] - df.groupby("Date")[c].transform("mean")
df["dv"] = (df.close_unadj * df.Volume).where(tr_)
df["vpos"] = df.Volume.where(tr_)
df["rs"] = ((100 - df.inst_share) / 100).where(tr_)
df["cr"] = (df.credit_share / 100).where(tr_)
roll = lambda col, w, f, mp: g[col].transform(lambda x: getattr(x.rolling(w, min_periods=mp), f)())
df["vol20"] = roll("r", 20, "std", 10)
df["max20"] = roll("r", 20, "max", 10)
df["dv20"] = roll("dv", 20, "median", 10)
df["v5"], df["v60"] = roll("vpos", 5, "mean", 3), roll("vpos", 60, "mean", 30)
df["hi250"] = roll("Close", 250, "max", 120)
df["rs20"] = g.rs.transform(lambda x: x.rolling(20, min_periods=10).mean().shift(5))
df["mb5"] = g.margin_bal.shift(5)
df["wk"] = df.Date.dt.to_period("W-FRI")
last = df[df.Close.notna()].drop_duplicates(["stock_code", "wk"], keep="last").set_index(["stock_code", "wk"])[
    ["Date", "Close", "MA60", "RSI14", "mktcap", "margin_bal", "mb5", "short_margin_ratio", "vol20", "max20", "dv20", "v5", "v60", "hi250", "rs20", "in265"]]
agg = df.groupby(["stock_code", "wk"]).agg(rs_w=("rs", "mean"), cr_w=("cr", "mean"), on_w=("xon", "sum"), id_w=("xidr", "sum"), n_d=("r", "count"))
P = last.join(agg).reset_index().sort_values(["stock_code", "wk"]).reset_index(drop=True)
P["wi"] = P.wk.apply(lambda p: p.ordinal)


def lag(s, k):
    return s.groupby(P.stock_code).shift(k).where(P.wi.groupby(P.stock_code).shift(k) == P.wi - k)


C = P.Close
P["r1w"], P["r4w"], P["r12w"] = C / lag(C, 1) - 1, C / lag(C, 4) - 1, C / lag(C, 12) - 1
P["mom26_4"] = lag(C, 4) / lag(C, 26) - 1
P["logdv"] = np.log(P.dv20)
P["voltrend"] = np.log(P.v5 / P.v60)
P["ma60r"], P["rsi"], P["hi52"] = C / P.MA60 - 1, P.RSI14, C / P.hi250 - 1
P["smr"] = P.short_margin_ratio.clip(0, 1)
P["margin_chg5"] = (P.margin_bal / P.mb5.where(P.mb5 > 0) - 1).clip(-0.5, 0.5)
P["rs_dev"] = P.rs_w - P.rs20
P["on_4w"] = P.on_w + lag(P.on_w, 1) + lag(P.on_w, 2) + lag(P.on_w, 3)
# 下週報酬（含下市）
last_wi, data_end = P.groupby("stock_code").wi.transform("max"), P.wi.max()
nxt = P.wi.groupby(P.stock_code).shift(-1)
P["ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1).where(nxt == P.wi + 1)
P.loc[P.ret_next.isna() & (P.wi == last_wi) & (P.wi < data_end - 1), "ret_next"] = -0.5
gap = P.ret_next.isna() & (P.wi < last_wi)
P.loc[gap, "ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1)[gap]
P["year"] = P.Date.dt.year
# 新聞、公告（當週）
nw = pd.concat([pd.read_csv(f, dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "Date", "id", "tone"])
                for f in ("cnyes_bigmove_matched.csv", "cnyes_bigmove_matched_out889.csv")]).drop_duplicates(["stock_code", "id"])
nw["wk"] = nw.Date.dt.to_period("W-FRI")
nm = nw.groupby(["stock_code", "wk"]).agg(n_news=("id", "size"), n_pos=("tone", lambda s: (s == "正面").sum()), n_neg=("tone", lambda s: (s == "負面").sum())).reset_index()
ACATS = {"庫藏股": "a_buyback", "澄清媒體報導": "a_clarify", "訴訟/裁罰/檢調": "a_legal", "災害/停工/事故": "a_disaster", "法人說明會": "a_ir",
         "股利分派/配股": "a_div", "現金增資/私募/可轉債": "a_issue", "減資-彌補虧損": "a_cutloss", "併購/公開收購/股份交換": "a_ma", "訂單/合約/新產品": "a_order"}
mp = pd.concat([pd.read_csv(f, dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "Date", "cat", "sub", "title"])
                for f in ("mops_bigmove_matched.csv", "mops_bigmove_matched_out889.csv")])
mp = mp[~mp["sub"].astype(str).isin(["True", "1"])].drop_duplicates(["stock_code", "Date", "title"])
mp["wk"] = mp.Date.dt.to_period("W-FRI")
am = mp.groupby(["stock_code", "wk"]).size().rename("n_ann").reset_index()
ac = mp[mp.cat.isin(ACATS)].groupby(["stock_code", "wk", "cat"]).size().unstack(fill_value=0).rename(columns=ACATS).reset_index()
P = P.merge(nm, on=["stock_code", "wk"], how="left").merge(am, on=["stock_code", "wk"], how="left").merge(ac, on=["stock_code", "wk"], how="left")
for c in ["n_news", "n_pos", "n_neg", "n_ann"] + list(ACATS.values()):
    P[c] = P[c].fillna(0)
P["news_att"], P["ann_att"] = np.log1p(P.n_news), np.log1p(P.n_ann)
P["news_net"] = (P.n_pos - P.n_neg) / (P.n_pos + P.n_neg).replace(0, np.nan)
P = P.replace([np.inf, -np.inf], np.nan)
P["liq_ok"] = P.groupby("wk").logdv.rank(pct=True) > 0.2
P["tgt"] = P.groupby("wk").ret_next.rank(pct=True) - 0.5
TECH = ["r1w", "r4w", "r12w", "mom26_4", "vol20", "max20", "logdv", "voltrend", "ma60r", "rsi", "hi52"]
RETAIL = ["smr", "margin_chg5", "cr_w", "rs_w", "rs_dev"]
INFO = ["news_att", "n_pos", "n_neg", "news_net", "ann_att"] + list(ACATS.values())
ON = ["on_w", "id_w", "on_4w"]
ARMS = {"技術面＋散戶行為★": TECH + RETAIL, "＋新聞＋公告": TECH + RETAIL + INFO, "＋隔夜情緒": TECH + RETAIL + ON, "只有技術面": TECH}
valid = P.ret_next.notna() & P.tgt.notna()
say(f"[面板] 股票×週 {len(P):,}、{P.stock_code.nunique()} 檔、{P.wk.min()}~{P.wk.max()}；每週可交易約 {P[P.liq_ok].groupby('wk').size().mean():.0f} 檔；當週有新聞 {(P.n_news > 0).mean():.1%}、有公告 {(P.n_ann > 0).mean():.1%}")
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
    say(f"  {Y} 年預測完成")
U = P[valid & (P.year >= 2018) & P.liq_ok].copy()
weeks = sorted(U.wk.unique())
bench = U.groupby("wk").ret_next.mean()
vw = U.groupby("wk").apply(lambda z: np.average(z.ret_next, weights=z.mktcap.fillna(0) + 1e-9))
tstat = lambda s: pd.Series(s).dropna().mean() / (pd.Series(s).dropna().std(ddof=1) / np.sqrt(pd.Series(s).dropna().size))


def perf(r):
    ann = (1 + r).prod() ** (52 / len(r)) - 1
    vol = r.std() * np.sqrt(52)
    eq = (1 + r).cumprod()
    return ann, vol, r.mean() * 52 / vol, (eq / eq.cummax() - 1).min()


def build(score, buffer):
    held, gross, turn = set(), {}, {}
    G = {w: z for w, z in U.groupby("wk")}
    for w in weeks:
        z = G[w].sort_values(score, ascending=False)
        n = max(int(len(z) * 0.2), 1)
        top20 = list(z.stock_code.iloc[:n])
        if buffer:
            keep = held & set(z.stock_code.iloc[:int(len(z) * 0.4)])
            new = keep | set([c for c in top20 if c not in keep][:max(n - len(keep), 0)])
        else:
            new = set(top20)
        turn[w] = 1.0 if not held else len(new - held) / len(new)
        gross[w] = z[z.stock_code.isin(new)].ret_next.mean()
        held = new
    gross, turn = pd.Series(gross), pd.Series(turn)
    return gross - turn * C_DISC, gross, turn


say(f"\n===== 週頻全市場選股（2018~2025，{len(weeks)} 週）=====")
a, v, sh, dd = perf(bench)
say(f"  基準 等權：年化 {a:+.1%}｜Sharpe {sh:.2f}｜回落 {dd:.1%}；市值加權：年化 {perf(vw)[0]:+.1%}、Sharpe {perf(vw)[2]:.2f}、回落 {perf(vw)[3]:.1%}")
res = {}
for arm in ARMS:
    U["s"] = pred[arm][U.index]
    ic = U.groupby("wk").apply(lambda z: spearmanr(z.s, z.ret_next)[0])
    say(f"  {arm}｜週 Rank IC {ic.mean():+.4f}（t={tstat(ic):+.1f}）")
    for buf, nm in [(False, "A 每週換前20%"), (True, "B 降低換股")]:
        net, gross, turn = build("s", buf)
        ex, exv = net - bench, net - vw
        a, v, sh, dd = perf(net)
        res[(arm, buf)] = net
        yr = ex.groupby(pd.PeriodIndex(ex.index).year).mean() * 52
        say(f"    {nm}：換股 {turn.iloc[1:].mean():.0%}/週｜毛 {gross.mean():+.2%}/週、扣成本 {net.mean():+.2%}/週 → 年化 {a:+.1%}、Sharpe {sh:.2f}、回落 {dd:.1%}"
            f"｜相對等權 {ex.mean():+.3%}/週（t={tstat(ex):+.2f}）｜相對市值加權 {exv.mean():+.3%}/週（t={tstat(exv):+.2f}）")
        say("      分年（相對等權，年化）：" + " ".join(f"{y}:{x:+.1%}" for y, x in yr.items()) + f"｜為正 {(yr > 0).sum()}/{len(yr)}")
ex = res[("技術面＋散戶行為★", True)] - bench
say(f"\n→ W1：組合 B 扣成本後相對等權 {ex.mean():+.3%}/週，t={tstat(ex):+.2f} → {'通過' if (ex.mean() > 0 and tstat(ex) > 2.26) else '未通過'}")
d = res[("＋新聞＋公告", True)] - res[("技術面＋散戶行為★", True)]
say(f"→ W2：加新聞＋公告後 組合 B 淨報酬差 {d.mean():+.3%}/週，t={tstat(d):+.2f} → {'通過' if (d.mean() > 0 and tstat(d) > 2.26) else '未通過'}")
d2 = res[("＋隔夜情緒", True)] - res[("技術面＋散戶行為★", True)]
d3 = res[("技術面＋散戶行為★", True)] - res[("只有技術面", True)]
say(f"  （次要）加隔夜情緒：{d2.mean():+.3%}/週（t={tstat(d2):+.2f}）｜散戶行為的增量：{d3.mean():+.3%}/週（t={tstat(d3):+.2f}）")
open("weekly_fullmarket_strategy.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
