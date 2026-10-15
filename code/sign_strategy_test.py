# -*- coding: utf-8 -*-
"""
最簡單的交易規則回測：「預測會漲就買、預測會跌就賣」（日頻，隔天開盤進場、收盤出場＝當沖，2018~2024）
  預測：daytrade_test.py 存好的 XGBoost 上漲機率（逐年滾動、樣本外），p > 0.5 視為預測會漲。
  A 多空版：p>0.5 開盤買、收盤賣；p<0.5 開盤先賣、收盤買回（當沖先賣後買）。
  B 只做多：p>0.5 才買，否則不交易（空手）。
  每天把所有部位等權平均成組合報酬；每筆部位都是一次當沖來回，扣成本 32 bp（手續費六折）或 44 bp（無折扣）。
python sign_strategy_test.py
"""
import pickle

import numpy as np
import pandas as pd

allraw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                     usecols=["stock_code", "Date", "Open", "Close"], parse_dates=["Date"]
                     ).sort_values(["stock_code", "Date"]).reset_index(drop=True)
ga = allraw.groupby("stock_code")
r_next = (ga.Close.shift(-1) / ga.Open.shift(-1) - 1).values   # 隔天開盤→收盤
rows = []
for Y in range(2018, 2025):
    o = pickle.load(open(f"daytrade_preds/B_{Y}.pkl", "rb"))
    rows.append(pd.DataFrame({"i": o["idx"], **{k: v for k, v in o["P"].items() if k in ("技術面", "+融資融券", "+PTT")}}))
pr = pd.concat(rows)
pr["Date"] = allraw.Date.values[pr.i]
pr["r"] = r_next[pr.i]
pr = pr.dropna(subset=["r"])
COSTS = {"不扣成本": 0.0, "扣32bp": 0.0032, "扣44bp": 0.0044}
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def stats(daily):
    daily = daily.fillna(0)
    ann = daily.mean() * 250
    sharpe = daily.mean() / daily.std() * np.sqrt(250) if daily.std() > 0 else np.nan
    cum = (1 + daily).prod() - 1
    return ann, sharpe, cum


say(f"樣本：{pr.Date.min().date()}~{pr.Date.max().date()}，{pr.Date.nunique()} 個交易日，{len(pr):,} 筆股票日；"
    f"全體隔天開盤→收盤平均 {pr.r.mean()*1e4:+.1f} bp")
for arm in ["技術面", "+融資融券", "+PTT"]:
    p = pr[arm]
    up = p > 0.5
    hit = ((pr.r > 0) == up).mean()
    say(f"\n=== 模型：{arm} ===  預測會漲的比例 {up.mean():.1%}｜方向猜對率 {hit:.1%}")
    # A 多空版
    pos_ret = np.where(up, pr.r, -pr.r)
    for cname, c in COSTS.items():
        daily = pd.Series(pos_ret - c, index=pr.index).groupby(pr.Date).mean()
        ann, sh, cum = stats(daily)
        say(f"  A 多空版 {cname:<5s} 日均 {daily.mean()*1e4:+6.1f} bp｜年化 {ann:+7.1%}｜Sharpe {sh:+5.2f}｜7年累積 {cum:+9.1%}")
    # B 只做多
    for cname, c in COSTS.items():
        d = pr[up]
        daily = (d.r - c).groupby(d.Date).mean().reindex(sorted(pr.Date.unique())).fillna(0)
        ann, sh, cum = stats(daily)
        say(f"  B 只做多 {cname:<5s} 日均 {daily.mean()*1e4:+6.1f} bp｜年化 {ann:+7.1%}｜Sharpe {sh:+5.2f}｜7年累積 {cum:+9.1%}")
    # 逐年（多空版，扣32bp）
    daily = pd.Series(pos_ret - 0.0032, index=pr.index).groupby(pr.Date).mean()
    yr = daily.groupby(daily.index.year).mean() * 250
    say("  A 多空版扣32bp 逐年年化：" + " ".join(f"{y}:{v:+.0%}" for y, v in yr.items()))
say("\n說明：每筆部位都是一次當沖來回；32bp≈證交稅0.15%（當沖減半）＋手續費0.1425%×2×六折，44bp為手續費無折扣。"
    "\n未計滑價、市場衝擊、漲跌停無法成交、借券/平盤下不得放空等限制；等權持有所有有訊號的股票（每天約上百檔），實務上做不到。")
open("sign_strategy_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
