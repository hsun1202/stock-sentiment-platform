# -*- coding: utf-8 -*-
"""
月頻全市場選股（monthly_fullmarket_strategy.py，S1 通過）的穩健性檢查——2026-10-11，跑之前寫定要檢查的四件事，全部照實報：
  (a) 換基準：相對「市值加權」（接近加權指數，這幾年被台積電拉高）還贏不贏？
  (b) 只看不在原 265 檔內的股票（月頻分析以前只在 265 檔做過，這些股票對月頻模型是沒看過的）。
  (c) 晚一天進場：月底收盤算出名單，隔一個交易日收盤才買（持有到下個月第一個交易日收盤）——排除「用收盤價算訊號又用同一個收盤價成交」的疑慮。
  (d) 只買分數最高的 30 檔（一般人不可能同時持有 160 檔）。
模型預測直接讀 monthly_fullmarket_preds.pkl（不重訓）。成本 0.471%×換股比例。只看「技術面＋散戶行為」模型。
python monthly_fullmarket_robust.py
"""
import numpy as np, pandas as pd
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)
T = pd.read_pickle("monthly_fullmarket_preds.pkl"); COL = "p_技術面＋散戶行為★"; C = 0.00471
px = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "Date", "Close", "in265"])
px = px[px.Close.notna()].sort_values(["stock_code", "Date"]); px["ym"] = px.Date.dt.to_period("M")
first = px.drop_duplicates(["stock_code", "ym"], keep="first")[["stock_code", "ym", "Close"]].rename(columns={"Close": "c_first"})
f1 = first.assign(ym=first.ym - 1).rename(columns={"c_first": "c_in"})      # 下個月第一天收盤＝進場價
f2 = first.assign(ym=first.ym - 2).rename(columns={"c_first": "c_out"})     # 再下個月第一天收盤＝出場價
T = T.merge(f1, on=["stock_code", "ym"], how="left").merge(f2, on=["stock_code", "ym"], how="left")
T["ret_delay"] = T.c_out / T.c_in - 1
T.loc[T.ret_delay.isna() & (T.ret_next <= -0.5), "ret_delay"] = -0.5
T.loc[T.ret_delay.isna(), "ret_delay"] = T.ret_next                          # 其餘缺值（極少）沿用原報酬
T = T.merge(px.drop_duplicates("stock_code")[["stock_code", "in265"]], on="stock_code", how="left")
def tstat(s):
    s = pd.Series(s).dropna(); return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
def perf(r):
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12); eq = (1 + r).cumprod()
    return ann, vol, r.mean() * 12 / vol, (eq / eq.cummax() - 1).min()
def run(U, label, rcol="ret_next", topn=None):
    months = sorted(U.ym.unique())
    ew = U.groupby("ym")[rcol].mean(); vw = U.groupby("ym").apply(lambda z: np.average(z[rcol], weights=z.mktcap.fillna(0) + 1e-9))
    if topn: top = U[U.groupby("ym")[COL].rank(ascending=False, method="first") <= topn]
    else: top = U[U.groupby("ym")[COL].rank(pct=True) > 0.8]
    gross = top.groupby("ym")[rcol].mean(); hold = top.groupby("ym").stock_code.apply(set)
    turn = pd.Series({m: (1.0 if i == 0 else len(hold[m] - hold[months[i - 1]]) / len(hold[m])) for i, m in enumerate(months)})
    net = gross - turn * C; a, v, sh, dd = perf(net)
    e1, e2 = net - ew, net - vw; yr = (e2.groupby(e2.index.year).mean() * 12)
    say(f"\n{label}（每月持有約 {top.groupby('ym').size().mean():.0f} 檔，換股 {turn.iloc[1:].mean():.0%}）")
    say(f"  扣成本後：年化 {a:+.1%}｜波動 {v:.1%}｜Sharpe {sh:.2f}｜最大回落 {dd:.1%}；等權基準 年化 {perf(ew)[0]:+.1%}（Sharpe {perf(ew)[2]:.2f}）；市值加權 年化 {perf(vw)[0]:+.1%}（Sharpe {perf(vw)[2]:.2f}、最大回落 {perf(vw)[3]:.1%}）")
    say(f"  相對等權 {e1.mean():+.2%}/月（t={tstat(e1):+.2f}）｜相對市值加權 {e2.mean():+.2%}/月（t={tstat(e2):+.2f}，贏的月數 {(e2 > 0).mean():.0%}）")
    say("  分年（相對市值加權，年化）：" + " ".join(f"{y}:{x:+.1%}" for y, x in yr.items()) + f"｜為正 {(yr > 0).sum()}/{len(yr)}")
U = T[T.liq_ok]
run(U, "(原設定) 全市場、前 20%、月底收盤進場")
run(U, "(a)(c) 晚一天進場", rcol="ret_delay")
run(U[U.in265 == 0], "(b) 只看不在原 265 檔內的股票")
run(U[U.in265 == 0], "(b)(c) 不在原 265 檔＋晚一天進場", rcol="ret_delay")
run(U, "(d) 只買分數最高的 30 檔", topn=30)
run(U, "(c)(d) 最高 30 檔＋晚一天進場", rcol="ret_delay", topn=30)
run(U[U.big], "(d') 大型股裡分數最高的 30 檔＋晚一天進場", rcol="ret_delay", topn=30)
open("monthly_fullmarket_robust.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
