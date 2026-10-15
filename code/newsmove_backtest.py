# -*- coding: utf-8 -*-
"""
「有消息的大漲之後會續漲」——實際進出場回測（排除漲停買不到的、扣成本）
事件：bigmove_decompose.csv（265 檔）與 bigmove_decompose_out889.csv（889 檔）裡的大漲日（x>0），依歸因分「有消息」（1 公司公告、2 媒體新聞）與「沒消息」（4、5、6）。
      有消息的定義＝影響當天的公告/新聞在當天收盤前就已發布，所以收盤時是知道的。
股價：final_dataset_v3_tejall25（TEJ 還原價、漲跌停旗標）。

事先寫定的規則（2026-10-10，跑之前寫下；只報這一條，不搜尋參數）：
  大漲日當天收盤買進、持有 5 個交易日後收盤賣出。當天收在漲停（limit_flag＝'+'）→ 視為買不到，剔除。
  成本：來回 0.47%（手續費六折）與 0.585%。
  報：每筆原始報酬、超額報酬（減同期全市場等權）、勝率；分「有消息／沒消息」、分 265 檔／889 檔、分年；
      另報「沒有剔除漲停」的數字，看漲停占了多少。
  主要判準：889 檔、有消息、剔除漲停後，扣 0.47% 的超額報酬 > 0（t 依月份群聚，|t|>2）。
python newsmove_backtest.py
"""
import numpy as np
import pandas as pd

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


px = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str, "limit_flag": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Close", "limit_flag", "abnormal_price"])
px = px.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = px.groupby("stock_code")
px["r"] = px.Close / g.Close.shift(1) - 1
px["mkt"] = px.groupby("Date").r.transform("mean")
px["c5"] = g.Close.shift(-5)
lm = np.log1p(px.mkt.fillna(0))
cs = lm.groupby(px.stock_code).cumsum()
px["m5"] = np.expm1(cs.groupby(px.stock_code).shift(-5) - cs)
px["bad5"] = sum(g.abnormal_price.shift(-k).fillna(0).ne(0) for k in range(1, 6)) > 0
px["raw"] = (px.c5 / px.Close - 1).where(~px.bad5)
px["ex"] = px.raw - px.m5
key = ["stock_code", "Date"]


def tm(v, ym):
    z = pd.DataFrame({"v": v, "m": ym}).dropna()
    mm = z.groupby("m").v.mean()
    return z.v.mean(), mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))), len(z)


res = {}
for tag, lab in [("", "原 265 檔"), ("_out889", "另外 889 檔")]:
    B = pd.read_csv(f"bigmove_decompose{tag}.csv", dtype={"stock_code": str}, parse_dates=["Date"])
    B = B[B.x > 0].merge(px[key + ["limit_flag", "raw", "ex"]], on=key, how="left")
    B["ym"], B["year"] = B.Date.dt.to_period("M"), B.Date.dt.year
    B["lim"] = B.limit_flag == "+"
    say(f"\n===== {lab}：大漲日 {len(B):,} 個；其中當天收漲停 {B.lim.mean():.1%}（有消息的 {B[B['info'] == '有消息'].lim.mean():.1%}、沒消息的 {B[B['info'] == '沒消息'].lim.mean():.1%}）=====")
    for info in ["有消息", "沒消息"]:
        s0 = B[B["info"] == info]
        for nm, s in [("全部（含漲停，買不到）", s0), ("剔除漲停（買得到）", s0[~s0.lim])]:
            a, t, n = tm(s.ex, s.ym)
            r_, tr, _ = tm(s.raw, s.ym)
            say(f"  {info}・{nm} n={n:,}｜毛：原始 {r_:+.2%}（t={tr:+.1f}）、超額 {a:+.2%}（t={t:+.1f}）｜勝率 {(s.raw > 0).mean():.1%}"
                f"｜扣0.47%：超額 {a - 0.0047:+.2%}（t={tm(s.ex - 0.0047, s.ym)[1]:+.1f}）｜扣0.585%：超額 {a - 0.00585:+.2%}（t={tm(s.ex - 0.00585, s.ym)[1]:+.1f}）")
        s = s0[~s0.lim]
        yr = s.groupby("year").agg(n=("ex", "count"), ex=("ex", "mean"))
        say(f"    {info}・剔除漲停 分年超額(毛)：" + " ".join(f"{y}:{r.ex:+.1%}({int(r.n)})" for y, r in yr.iterrows())
            + f"｜為正 {(yr.ex > 0).sum()}/{len(yr)}；扣0.47%後為正 {((yr.ex - 0.0047) > 0).sum()}/{len(yr)}")
        if info == "有消息":
            res[lab] = tm(s.ex - 0.0047, s.ym)
            for w in ["1 公司公告", "2 媒體新聞"]:
                x = s[s.why == w]
                a, t, n = tm(x.ex, x.ym)
                say(f"      其中 {w}：n={n:,}｜超額(毛) {a:+.2%}（t={t:+.1f}）")
a, t, n = res["另外 889 檔"]
say(f"\n→ 主要判準（889 檔、有消息、剔除漲停、扣 0.47%）：超額 {a:+.2%}，t={t:+.2f}，n={n:,} → {'通過' if (a > 0 and t > 2) else '未通過'}")
open("newsmove_backtest.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
