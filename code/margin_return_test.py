# -*- coding: utf-8 -*-
"""
融資融券能不能「預測報酬」（不只漲跌方向）—— 日頻，隔天開盤→收盤報酬（＝當沖型交易），2018~2024
  (a) 單一變數排序：每天依 融資增減 / 融券增減 / 券資比 把股票分五組，看隔天開盤→收盤平均報酬（等權）
  (b) 模型排序：用 daytrade_test.py 存好的 XGBoost 預測（技術面 vs +融資融券），每天取預測前20%/後20%，看報酬差
  報酬為毛報酬（未扣成本）；台股當沖來回成本約 0.32%（六折）~0.44%（無折扣）＝32~44 bp。
python margin_return_test.py
"""
import pickle

import numpy as np
import pandas as pd

raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                  usecols=["stock_code", "Date", "Open", "Close", "abnormal_price", "margin_chg", "short_chg",
                           "short_margin_ratio", "margin_eligible"], parse_dates=["Date"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = raw.groupby("stock_code")
nO, nC = g.Open.shift(-1), g.Close.shift(-1)
abn = raw.abnormal_price.fillna(0)
ok = (abn == 0) & (g.abnormal_price.shift(-1).fillna(0) == 0) & nO.notna() & nC.notna() & (nO > 0)
raw["r_next"] = ((nC / nO - 1) * 1e4).where(ok)          # 隔天開盤→收盤，bp
raw = raw[(raw.Date >= "2018-01-01") & (raw.Date <= "2024-12-31")]
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


say("隔天開盤→收盤報酬（bp，等權，毛報酬）｜2018~2024｜當沖來回成本約 32~44 bp")
say(f"全部股票平均：{raw.r_next.mean():+.1f} bp/天（台股盤中平均為負）\n")

# (a) 單一變數五分位
say("(a) 單一變數排序：五分位（1=最低 … 5=最高）")
for col, name in [("margin_chg", "融資增減"), ("short_chg", "融券增減"), ("short_margin_ratio", "券資比")]:
    d = raw[(raw.margin_eligible == 1) & raw[col].notna() & raw.r_next.notna()].copy()
    d["q"] = d.groupby("Date")[col].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if len(x) >= 25 else np.nan)
    daily = d.groupby(["Date", "q"]).r_next.mean().unstack()
    spread = daily[5] - daily[1]
    say(f"  {name:<6s} 各組 " + " ".join(f"{daily[k].mean():+.1f}" for k in range(1, 6))
        + f"｜最高−最低 {spread.mean():+.1f} bp/天（t={tstat(spread):+.2f}，{spread.notna().sum()} 天）")

# (b) 模型排序
say("\n(b) 模型排序（XGBoost 預測隔天開盤→收盤上漲機率；每天前20% / 後20%）")
rows = []
for Y in range(2018, 2025):
    o = pickle.load(open(f"daytrade_preds/B_{Y}.pkl", "rb"))
    idx, P = o["idx"], o["P"]
    full = pd.read_csv  # placeholder to keep flake quiet
    rows.append(pd.DataFrame({"i": idx, "p_tech": P["技術面"], "p_mg": P["+融資融券"], "p_ptt": P["+PTT"]}))
pr = pd.concat(rows)
allraw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "Open", "Close", "abnormal_price"],
                     parse_dates=["Date"]).sort_values(["stock_code", "Date"]).reset_index(drop=True)
ga = allraw.groupby("stock_code")
r_all = ((ga.Close.shift(-1) / ga.Open.shift(-1) - 1) * 1e4)
pr["Date"] = allraw.Date.values[pr.i]
pr["r"] = r_all.values[pr.i]
pr = pr.dropna(subset=["r"])
out = {}
for col, name in [("p_tech", "技術面"), ("p_mg", "+融資融券"), ("p_ptt", "+PTT")]:
    q = pr.groupby("Date")[col].rank(pct=True)
    top = pr[q > 0.8].groupby("Date").r.mean()
    bot = pr[q <= 0.2].groupby("Date").r.mean()
    allm = pr.groupby("Date").r.mean()
    out[name] = {"ls": top - bot, "top_ex": top - allm, "top": top}
    say(f"  {name:<8s} 前20% {top.mean():+.1f}｜後20% {bot.mean():+.1f}｜多空 {(top - bot).mean():+.1f} bp/天（t={tstat(top - bot):+.2f}）"
        f"｜前20%−全體 {(top - allm).mean():+.1f}（t={tstat(top - allm):+.2f}）")
for name in ["+融資融券", "+PTT"]:
    d = (out[name]["ls"] - out["技術面"]["ls"]).dropna()
    d2 = (out[name]["top"] - out["技術面"]["top"]).dropna()
    say(f"  {name} 比技術面多出：多空 {d.mean():+.2f} bp/天（t={tstat(d):+.2f}）｜只做多前20% {d2.mean():+.2f} bp/天（t={tstat(d2):+.2f}）")
say("\n判讀：當沖一次來回成本約 32~44 bp；『前20%』若要獲利，平均報酬需高於成本。")
open("margin_return_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
