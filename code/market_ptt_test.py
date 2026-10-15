# -*- coding: utf-8 -*-
"""
大盤層級：只用 PTT 閒聊串情緒（收盤前可得）預測隔天加權指數（漲跌方向與報酬）
  情緒：market_sentiment/chat_labels.csv 依「前一交易日收盤後 ~ 今天收盤」的留言（每天抽樣≤300則，Breeze 三分類）
        S＝(正−負)/(正+負)；dS＝S − 前20日平均（相對變化）
  目標：隔天 收盤→收盤、隔夜（收盤→09:01）、盤中（09:01→收盤）
  1) 迴歸（Newey-West 5）：隔天報酬 ~ dS + 今天報酬 + 過去5日報酬
  2) 交易規則（台指期，成本約 1 bp/來回，粗估）：dS>0 做多、<0 做空，今天收盤進場、隔天收盤出場；與「今天漲就做多」對照
  3) 逐年 walk-forward 邏輯迴歸 AUC（2018-2024）：只用價格 vs 價格＋情緒
python market_ptt_test.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


cal = pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"]).set_index("Date").sort_index()
im = pd.read_csv("market_sentiment/taiex_intraday_1min.csv", parse_dates=["Date"])
op = im[im.time.str[:5] == "09:01"].set_index("Date").taiex
c = pd.read_csv("market_sentiment/chat_labels.csv", dtype=str, keep_default_na=False, usecols=["Date", "text_sentiment"])
c["Date"] = pd.to_datetime(c.Date)
p = c.text_sentiment.eq("positive").groupby(c.Date).sum()
n = c.text_sentiment.eq("negative").groupby(c.Date).sum()
D = pd.DataFrame({"close": cal.taiex, "open": op}).loc["2015-01-01":"2024-12-31"]
D["S"] = ((p - n) / (p + n)).reindex(D.index)
D["dS"] = D.S - D.S.shift(1).rolling(20, min_periods=10).mean()
D["r0"] = D.close.pct_change()
D["past5"] = D.close / D.close.shift(5) - 1
D["y_cc"] = D.close.shift(-1) / D.close - 1
D["y_on"] = D.open.shift(-1) / D.close - 1
D["y_id"] = D.close.shift(-1) / D.open.shift(-1) - 1
D = D.dropna(subset=["dS", "r0", "past5", "y_cc"])
say(f"樣本 {D.index.min().date()}~{D.index.max().date()}，{len(D)} 個交易日；dS 與今天大盤報酬相關 {D.dS.corr(D.r0):+.3f}")

say("\n1) 迴歸（bp；Newey-West 5）：隔天報酬 ~ 情緒變化 dS(1σ) + 今天報酬 + 過去5日報酬")
D["dS_z"] = (D.dS - D.dS.mean()) / D.dS.std()
for y, lab in [("y_cc", "收盤→收盤"), ("y_on", "隔夜"), ("y_id", "盤中")]:
    d = D.dropna(subset=[y])
    for xs, xl in [(["dS_z"], "只有情緒"), (["dS_z", "r0", "past5"], "＋今天與過去5日報酬")]:
        res = sm.OLS(d[y] * 1e4, sm.add_constant(d[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        say(f"  {lab:<6s}｜{xl:<10s} 情緒每 1σ {res.params['dS_z']:+5.1f} bp（t={res.tvalues['dS_z']:+.2f}）"
            + (f"｜今天報酬 t={res.tvalues['r0']:+.2f}" if "r0" in xs else ""))

say("\n2) 交易規則：今天收盤進場、隔天收盤出場（台指期粗估成本 1 bp/來回）")
for lab, sig in [("情緒 dS>0 做多、<0 做空", np.sign(D.dS)), ("今天漲就做多、跌就做空（價格動能對照）", np.sign(D.r0)),
                 ("情緒 dS>0 才做多（只做多）", (D.dS > 0).astype(float)), ("每天都做多（買進持有）", pd.Series(1.0, index=D.index))]:
    pr = sig * D.y_cc
    for cost in (0.0, 0.0001):
        net = pr - cost * (sig != 0)
        ann = net.mean() * 250
        sh = net.mean() / net.std() * np.sqrt(250)
        say(f"  {lab:<26s} 成本{cost * 1e4:.0f}bp：猜對 {((sig * D.y_cc) > 0)[sig != 0].mean():.1%}｜日均 {net.mean() * 1e4:+5.1f} bp｜年化 {ann:+6.1%}｜Sharpe {sh:+.2f}｜十年累積 {(1 + net).prod() - 1:+.0%}")

say("\n3) 逐年 walk-forward 邏輯迴歸（預測隔天收盤→收盤漲跌）AUC")
D["y"] = (D.y_cc > 0).astype(int)
D["absr0"] = D.r0.abs()
FE = {"只用價格": ["r0", "past5", "absr0"], "價格＋情緒": ["r0", "past5", "absr0", "dS", "S"]}
for name, cols in FE.items():
    P, Yv = [], []
    for Y in range(2018, 2025):
        tr, te = D.index.year < Y, D.index.year == Y
        sc = StandardScaler().fit(D.loc[tr, cols])
        m = LogisticRegression(C=0.5).fit(sc.transform(D.loc[tr, cols]), D.y[tr])
        P.append(m.predict_proba(sc.transform(D.loc[te, cols]))[:, 1]); Yv.append(D.y[te].values)
    say(f"  {name:<6s} AUC {roc_auc_score(np.concatenate(Yv), np.concatenate(P)):.4f}")
open("market_ptt_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
