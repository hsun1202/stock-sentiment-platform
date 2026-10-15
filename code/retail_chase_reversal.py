# -*- coding: utf-8 -*-
"""
把「事件」和「散戶」接起來：沒消息的大漲之後會回吐（bigmove_decompose / newsmove_backtest 的發現）——散戶追得越兇，回吐越多嗎？
樣本：大漲日（超額>0 的大波動日）、當天沒收漲停。主要看「沒消息」的（歸因 4/5/6），並列「有消息」的當對照。
散戶追價的三個量法（都是大漲當天收盤後就知道的）：
  M 融資增減：當天融資餘額變化率（只取可信用交易的股票）——散戶借錢追
  S 散戶成交比重異常：當天（100−法人成交比重）減自己前 20 天平均——散戶成交占比升高
  P PTT 討論（只有原 265 檔、2015~2024 有）：當天有沒有 PTT 個股留言
結果變數：之後 5 個交易日的累積超額報酬（post5）。每年在樣本內依 M、S 各分三組。
用法：python retail_chase_reversal.py        # 原 265 檔（探索）
      python retail_chase_reversal.py out    # 另外 889 檔（驗證）

事先寫定、要在 889 檔上驗的主要檢定（2026-10-10，看過 265 檔之後、跑 889 檔之前寫下；見檔案最下方 HYP，門檻依個數做 Bonferroni）。
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

LOG = []
OUTS = len(sys.argv) > 1 and sys.argv[1] == "out"
TAG = "_out889" if OUTS else ""


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


key = ["stock_code", "Date"]
px = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str, "limit_flag": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Volume", "limit_flag", "margin_chg", "short_chg", "margin_eligible", "inst_share", "mktcap", "in265"])
px = px[px.in265 == (0 if OUTS else 1)].sort_values(key).reset_index(drop=True)
px["rs"] = ((100 - px.inst_share) / 100).where(px.Volume > 0)
px["rs_dev"] = px.rs - px.groupby("stock_code").rs.transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
px["M"] = px.margin_chg.where(px.margin_eligible == 1).clip(-0.5, 0.5)
B = pd.read_csv(f"bigmove_decompose{TAG}.csv", dtype={"stock_code": str}, parse_dates=["Date"])
B = B[B.x > 0].merge(px[key + ["limit_flag", "M", "short_chg", "rs_dev", "mktcap"]], on=key, how="left")
B = B[(B.limit_flag != "+") & B.post5.notna()].copy()
B["ym"], B["year"] = B.Date.dt.to_period("M"), B.Date.dt.year
if not OUTS:
    pt = pd.read_csv("final_dataset_v3_tej/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                     usecols=["stock_code", "Date", "ptt_pre_n", "ptt_intra_n"]).fillna(0)
    pt["ptt_n"] = pt.ptt_pre_n + pt.ptt_intra_n
    B = B.merge(pt[key + ["ptt_n"]], on=key, how="left")
say(f"[{'驗證：另外 889 檔' if OUTS else '探索：原 265 檔'}] 沒收漲停的大漲日 {len(B):,}：沒消息 {int((B['info'] == '沒消息').sum()):,}、有消息 {int((B['info'] == '有消息').sum()):,}"
    f"；當天融資增減中位數 {B.M.median():+.2%}、散戶成交比重異常中位數 {B.rs_dev.median():+.3f}")


def tm(v, ym):
    z = pd.DataFrame({"v": v, "m": ym}).dropna()
    if len(z) < 30:
        return np.nan, np.nan, len(z)
    mm = z.groupby("m").v.mean()
    return z.v.mean(), mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))), len(z)


res = {}
for info in ["沒消息", "有消息"]:
    S = B[B["info"] == info].copy()
    say(f"\n===== {info}的大漲日（沒收漲停）：依當天散戶追價程度分三組，看之後 5 天累積超額 =====")
    for col, nm in [("M", "融資增減"), ("rs_dev", "散戶成交比重異常")]:
        d = S[S[col].notna()].copy()
        d["g"] = d.groupby("year")[col].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
        parts = []
        for k, lab in [(0, "低"), (1, "中"), (2, "高")]:
            a, t, n = tm(d[d.g == k].post5, d[d.g == k].ym)
            parts.append(f"{lab} {a:+.2%}(t={t:+.1f},n={n:,},當天{col} {d[d.g == k][col].median():+.3f})")
        mm = d[d.g.isin([0, 2])].groupby(["ym", "g"]).post5.mean().unstack().dropna()
        df_ = mm[2] - mm[0]
        td = df_.mean() / (df_.std(ddof=1) / np.sqrt(len(df_)))
        res[(info, col)] = (df_.mean(), td)
        say(f"  {nm}：" + "｜".join(parts) + f"｜高−低 {df_.mean():+.2%}（t={td:+.2f}，{len(df_)} 個月）")
        for a_, b_ in [("2015", "2019"), ("2020", "2025")]:
            x = d[(d.Date >= a_) & (d.Date <= b_ + "-12-31")]
            say(f"      {a_}~{b_}：低 {x[x.g == 0].post5.mean():+.2%}／高 {x[x.g == 2].post5.mean():+.2%}")
    d = S.dropna(subset=["M", "rs_dev", "mktcap"]).copy()
    for c in ["M", "rs_dev", "x"]:
        d["z_" + c] = d.groupby("year")[c].transform(lambda v: (v.rank(pct=True) - 0.5))
    d["z_size"] = d.groupby("Date").mktcap.rank(pct=True).fillna(0.5) - 0.5 if False else d.groupby("year").mktcap.transform(lambda v: v.rank(pct=True) - 0.5)
    r = sm.OLS(d.post5 * 100, sm.add_constant(d[["z_M", "z_rs_dev", "z_x", "z_size"]])).fit(cov_type="cluster", cov_kwds={"groups": d.ym.astype(str)})
    say(f"  迴歸（之後 5 天 %；自變數為當年內的百分位排名−0.5，係數＝最高到最低的差）：融資增減 {r.params['z_M']:+.2f}（t={r.tvalues['z_M']:+.2f}）｜散戶成交比重異常 {r.params['z_rs_dev']:+.2f}（t={r.tvalues['z_rs_dev']:+.2f}）"
        f"｜當天漲幅 {r.params['z_x']:+.2f}（t={r.tvalues['z_x']:+.2f}）｜規模 {r.params['z_size']:+.2f}（t={r.tvalues['z_size']:+.2f}）｜n={len(d):,}")
    res[(info, "reg")] = (r.params["z_M"], r.tvalues["z_M"], r.params["z_rs_dev"], r.tvalues["z_rs_dev"])
    if not OUTS and "ptt_n" in S:
        d = S[S.Date <= "2024-12-31"]
        a1, t1, n1 = tm(d[d.ptt_n > 0].post5, d[d.ptt_n > 0].ym)
        a0, t0, n0 = tm(d[d.ptt_n == 0].post5, d[d.ptt_n == 0].ym)
        say(f"  PTT（2015~2024）：當天有個股留言 {a1:+.2%}（t={t1:+.1f}，n={n1:,}）｜沒有 {a0:+.2%}（t={t0:+.1f}，n={n0:,}）")

if OUTS:
    # HYP（2026-10-10 看過 265 檔結果後寫定，見 PROJECT_STATUS.md）：兩個主要檢定，Bonferroni |t|>2.24（p<0.025）
    a = res[("沒消息", "M")]
    b = res[("沒消息", "rs_dev")]
    say(f"\n→ C1 沒消息大漲：融資增加最多的一組之後 5 天 比 最少的一組更差（高−低<0）：{a[0]:+.2%}，t={a[1]:+.2f} → {'通過' if a[1] < -2.24 else '未通過'}")
    say(f"→ C2 沒消息大漲：散戶成交比重異常最高的一組 比 最低的一組更差（高−低<0）：{b[0]:+.2%}，t={b[1]:+.2f} → {'通過' if b[1] < -2.24 else '未通過'}")
open(f"retail_chase_reversal{TAG}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
