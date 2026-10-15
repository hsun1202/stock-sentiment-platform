# -*- coding: utf-8 -*-
"""新聞情緒 20 日漂移的穩健性檢驗：日曆時間投資組合（calendar-time portfolio）
每個交易日 d：持有「進場日在過去 1~20 個交易日內、且 LLM 判為 好（或 壞）」的股票（等權），
當日報酬（收盤→收盤）減 267 檔等權平均 → 每日超額報酬序列；Newey-West(20) t 值；逐年拆開。
進場日同 news_sentiment_analysis.py：發文後第一個能成交的開盤；持有期從進場日收盤後開始算（避開進場當天）。
python news_drift_robust.py"""
import numpy as np, pandas as pd, statsmodels.api as sm
ns = pd.read_csv("news_sentiment/news_scores.csv", dtype={"stock_codes": str}, keep_default_na=False)
ns["post_time"] = pd.to_datetime(ns.post_time)
px = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "Close", "abnormal_price"])
px["Date"] = pd.to_datetime(px.Date); px = px.sort_values(["stock_code", "Date"])
px["r"] = px.groupby("stock_code").Close.pct_change(); px.loc[px.abnormal_price == 1, "r"] = np.nan
px["ex"] = px.r - px.groupby("Date").r.transform("mean")
CAL = np.array(sorted(px.Date.unique()), dtype="datetime64[ns]"); pos = {d: k for k, d in enumerate(pd.DatetimeIndex(CAL))}
ev = ns[ns.stock_codes != ""].assign(stock_code=lambda d: d.stock_codes.str.split(",")).explode("stock_code")
i = np.searchsorted(CAL + np.timedelta64(9 * 3600, "s"), ev.post_time.values, side="left")
ev = ev[i < len(CAL)].copy(); ev["k"] = i[i < len(CAL)]
strong_bad_cut = ev.loc[ev.news_label == "壞", "news_score"].median()
ex = px.pivot(index="Date", columns="stock_code", values="ex").reindex(pd.DatetimeIndex(CAL))
for name, sel in (("好", ev.news_label == "好"), ("壞", ev.news_label == "壞"), ("強烈壞(分數低於壞新聞中位數)", (ev.news_label == "壞") & (ev.news_score <= strong_bad_cut))):
    hold = np.zeros(ex.shape, dtype=bool); cols = {c: j for j, c in enumerate(ex.columns)}
    for k, c in zip(ev.loc[sel, "k"], ev.loc[sel, "stock_code"]):
        if c in cols: hold[k + 1:min(k + 21, len(CAL)), cols[c]] = True   # 進場日之後的第1~20個交易日
    v = np.where(hold, ex.values, np.nan)
    port = pd.Series(np.nanmean(v, axis=1), index=ex.index).loc["2015-01-01":"2024-12-31"]
    n_hold = pd.Series(hold.sum(1), index=ex.index).loc[port.index]
    p = port[n_hold > 0]
    m = sm.OLS(p.values, np.ones(len(p))).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
    yr = p.groupby(p.index.year).agg(["mean", "count"])
    print(f"[{name}] 每日平均超額 {p.mean():+.3%}（約每20日 {p.mean()*20:+.2%}），NW t={m.tvalues[0]:+.2f}，{len(p)}天、平均同時持有 {n_hold[n_hold>0].mean():.1f} 檔")
    print("   各年(每日平均×20)：" + " ".join(f"{y}:{r['mean']*20:+.2%}" for y, r in yr.iterrows()))
