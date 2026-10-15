# -*- coding: utf-8 -*-
"""
市場層級 PTT 情緒的 2025 年「真正樣本外」檢驗（事先寫定的假說，來自 2020-2024 的發現）
  H1：情緒上升日（dS＝今天淨情緒 − 前20日平均 > 0）的隔天，加權指數收盤→收盤報酬高於其他日子
      （2015-2024 全期差 +8.5 bp，t=2.0；2015-2019 −6.0；2020-2024 +22.7 bp，t=3.4）
  H2：規則「dS>0 做多、<0 做空」／「dS>0 才做多」在 2025 年優於買進持有
資料：2015-2024 market_sentiment/chat_labels.csv ＋ 2025 market_sentiment/chat_labels_2025.csv（同一個 Breeze 分類器）
      加權指數：2015-2024 taiex_official.csv；2025 由證交所 FMTQIK（chip_raw/calendar/2025*.json）取收盤
注意：2025 留言只做簡化清理（2015-2024 用完整清理），情緒水準可能有小偏移；dS 用「相對前20日」可減少這個影響。
python market_ptt_2025_test.py
"""
import glob
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def roc(s):
    y, m, d = s.strip().split("/")
    return pd.Timestamp(int(y) + 1911, int(m), int(d))


t25 = pd.Series({roc(r[0]): float(r[4].replace(",", "")) for f in glob.glob("chip_raw/calendar/2025*.json")
                 for r in json.load(open(f, encoding="utf-8"))["data"]}).sort_index()
cal = pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"]).set_index("Date").taiex
close = pd.concat([cal[cal.index <= "2024-12-31"], t25]).sort_index()
close = close[~close.index.duplicated()]

lab = pd.concat([pd.read_csv("market_sentiment/chat_labels.csv", dtype=str, keep_default_na=False, usecols=["Date", "text_sentiment"]),
                 pd.read_csv("market_sentiment/chat_labels_2025.csv", dtype=str, keep_default_na=False, usecols=["Date", "text_sentiment"])])
lab["Date"] = pd.to_datetime(lab.Date)
shares = lab.groupby(lab.Date.dt.year).text_sentiment.value_counts(normalize=True).unstack().round(3)
say("各年分類比例（正/負/中立）：\n" + shares.loc[2020:2025].to_string())
p = lab.text_sentiment.eq("positive").groupby(lab.Date).sum()
n = lab.text_sentiment.eq("negative").groupby(lab.Date).sum()
D = pd.DataFrame({"close": close})
D["S"] = ((p - n) / (p + n)).reindex(D.index)
D["dS"] = D.S - D.S.shift(1).rolling(20, min_periods=10).mean()
D["r0"] = D.close.pct_change()
D["y"] = D.close.shift(-1) / D.close - 1
D = D.dropna(subset=["dS", "r0", "y"])
D["up"] = (D.dS > 0).astype(int)
D["mom"] = (D.r0 > 0).astype(int)


def h1(d, label):
    res = sm.OLS(d.y * 1e4, sm.add_constant(d[["up"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    res2 = sm.OLS(d.y * 1e4, sm.add_constant(d[["up", "mom"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    say(f"  {label:<10s} n={len(d):>4}｜情緒上升日隔天 {d.y[d.up == 1].mean() * 1e4:+6.1f} bp vs 其他 {d.y[d.up == 0].mean() * 1e4:+6.1f} bp"
        f"｜差 {res.params['up']:+6.1f}（t={res.tvalues['up']:+.2f}）｜控制今天漲跌 差 {res2.params['up']:+6.1f}（t={res2.tvalues['up']:+.2f}）")


say("\nH1：情緒上升日的隔天報酬（收盤→收盤，bp；Newey-West 5）")
h1(D[D.index.year <= 2019], "2015-2019")
h1(D[(D.index.year >= 2020) & (D.index.year <= 2024)], "2020-2024")
h1(D[D.index.year == 2025], "2025（樣本外）")

say("\nH2：2025 年交易規則（今天收盤進場、隔天收盤出場；台指期粗估成本 1 bp/來回）")
d = D[D.index.year == 2025]
for name, sig in [("dS>0 做多、<0 做空", np.sign(d.dS)), ("dS>0 才做多", (d.dS > 0).astype(float)),
                  ("今天漲就做多、跌就做空（價格對照）", np.sign(d.r0)), ("買進持有", pd.Series(1.0, index=d.index))]:
    for cost in (0.0, 0.0001):
        net = sig * d.y - cost * (sig != 0)
        say(f"  {name:<20s} 成本{cost * 1e4:.0f}bp：日均 {net.mean() * 1e4:+5.1f} bp｜2025 累積 {(1 + net).prod() - 1:+6.1%}"
            f"｜Sharpe {net.mean() / net.std() * np.sqrt(250):+.2f}")
open("market_ptt_2025_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
