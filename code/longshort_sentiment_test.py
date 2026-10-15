# -*- coding: utf-8 -*-
"""
橫截面多空組合檢驗（Baker & Wurgler式investor sentiment標準檢定法）：
不問「這檔股票會不會漲」，而是問「情緒偏多的股票，隔天表現有沒有比情緒偏空的股票好」。
每天把當天有討論的股票依情緒因子分成「偏多組」(>0)和「偏空組」(<0)，算兩組隔日
報酬的價差(long-short spread)，累積成時間序列後做t檢定。這個設計會自動抵銷當天
大盤整體漲跌(跟直接比較兩組的相對表現，不是各自的絕對方向)，理論上比之前測的
「絕對漲跌方向」更容易偵測到微弱的情緒訊號(如果存在的話)。

同時測 lag0(同日) 和 lag1/lag2(落後)，跟之前的邏輯一致：如果只有lag0顯著、
落後就消失，一樣要懷疑是洩漏而非真訊號。

python longshort_sentiment_test.py
"""
import numpy as np
import pandas as pd
from scipy import stats

MIN_PER_SIDE = 2
df = pd.read_csv("final_dataset_267_stitched/all_stocks_final_dataset_267.csv")
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
df["ret_fwd"] = g["Close"].shift(-1) / df["Close"] - 1  # 隔日報酬(明天收盤/今天收盤-1)
df["date"] = pd.to_datetime(df["Date"])

fac = df["daily_sentiment_factor"]
has_disc = df["total_comments"] > 0
for lag_name, lag in (("lag0", 0), ("lag1", 1), ("lag2", 2)):
    if lag == 0:
        df[f"fac_{lag_name}"] = fac.where(has_disc)
    else:
        df[f"fac_{lag_name}"] = fac.where(has_disc).groupby(df["stock_code"]).shift(lag)

te = df[(df.date >= "2023-01-01") & df.ret_fwd.notna()].copy()
print(f"測試期股票日: {len(te)}  範圍 {te.Date.min()}~{te.Date.max()}", flush=True)

results = {}
for lag_name in ("fac_lag0", "fac_lag1", "fac_lag2"):
    rows = []
    for d, sub in te.groupby("Date"):
        s = sub.dropna(subset=[lag_name])
        long_ = s[s[lag_name] > 0]
        short_ = s[s[lag_name] < 0]
        if len(long_) >= MIN_PER_SIDE and len(short_) >= MIN_PER_SIDE:
            rows.append({
                "Date": d, "n_long": len(long_), "n_short": len(short_),
                "long_ret": long_.ret_fwd.mean(), "short_ret": short_.ret_fwd.mean(),
                "spread": long_.ret_fwd.mean() - short_.ret_fwd.mean(),
            })
    r = pd.DataFrame(rows)
    n = len(r)
    mean, se = r.spread.mean(), r.spread.std(ddof=1) / np.sqrt(n)
    t_stat, p_val = stats.ttest_1samp(r.spread, 0.0)
    ann = mean / r.spread.std(ddof=1) * np.sqrt(252) if r.spread.std(ddof=1) > 0 else float("nan")
    results[lag_name] = r
    print(f"\n=== {lag_name} ===  可交易天數 {n}/{len(te.Date.unique())}  平均每邊股票數 long={r.n_long.mean():.1f} short={r.n_short.mean():.1f}")
    print(f"  多空價差 平均 {mean*100:+.4f}%/天  95%CI [{(mean-1.96*se)*100:+.4f}%, {(mean+1.96*se)*100:+.4f}%]")
    print(f"  t={t_stat:.3f}  p={p_val:.4f}  年化Sharpe(近似) {ann:.3f}")
    print(f"  多方平均報酬 {r.long_ret.mean()*100:+.4f}%/天  空方平均報酬 {r.short_ret.mean()*100:+.4f}%/天")

print("\n完成")
import json
out = {k: {"n_days": len(v), "mean_spread": float(v.spread.mean()), "t_stat": float(stats.ttest_1samp(v.spread, 0.0)[0]),
           "p_value": float(stats.ttest_1samp(v.spread, 0.0)[1])} for k, v in results.items()}
json.dump(out, open("longshort_sentiment_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
