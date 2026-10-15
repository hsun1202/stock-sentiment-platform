# -*- coding: utf-8 -*-
"""
PTT 討論量暴增（爆量）事件：之後報酬怎麼走？依情緒正負分開（Da-Engelberg-Gao 2011：關注度 → 短期推升 → 之後反轉）
  爆量定義：今天收盤前的留言數 ≥5 則，且 log(1+留言數) 比該股過去 60 天平均高 2 個標準差以上（只用過去資料）
  報酬：每日超額（個股 − 當天全體平均），事件日 0＝爆量當天收盤；看第 −20 ~ +20 天累積超額
  分組：爆量且淨情緒 ≥0.3（偏多）／≤ −0.3（偏空）／其他
輸出：ptt_attention_spike.log、ptt_attention_spike.png
python ptt_attention_spike.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Close", "abnormal_price", "ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n",
                          "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
for c in ["ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"]:
    df[c] = df[c].fillna(0)
n = df.ptt_pre_n + df.ptt_intra_n
pos = df.ptt_pre_pos + df.ptt_intra_pos
neg = df.ptt_pre_neg + df.ptt_intra_neg
df["net"] = (pos - neg) / (pos + neg).replace(0, np.nan)
la = np.log1p(n)
mu = la.groupby(df.stock_code).transform(lambda s: s.shift(1).rolling(60, min_periods=40).mean())
sd = la.groupby(df.stock_code).transform(lambda s: s.shift(1).rolling(60, min_periods=40).std())
df["z"] = (la - mu) / sd.replace(0, np.nan)
abn = df.abnormal_price.fillna(0) != 0
r = (df.Close / g.Close.shift(1) - 1).where(~abn & ~g.abnormal_price.shift(1).fillna(0).ne(0))
df["x"] = r - r.groupby(df.Date).transform("mean")
gx = df.groupby("stock_code").x
K = list(range(-20, 21))
for k in K:
    df[f"x{k}"] = gx.shift(-k)
spike = (n >= 5) & (df.z >= 2)
say(f"爆量事件 {int(spike.sum()):,} 個（{df.loc[spike, 'stock_code'].nunique()} 檔）；占全部股票日 {spike.mean():.2%}")
groups = {"爆量＋偏多（淨≥0.3）": spike & (df.net >= 0.3), "爆量＋偏空（淨≤−0.3）": spike & (df.net <= -0.3),
          "爆量＋中性": spike & (df.net.abs() < 0.3)}
fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
colors = ["#A83A2C", "#3E6B4F", "#9C9184"]
for (name, m), col in zip(groups.items(), colors):
    d = df.loc[m, [f"x{k}" for k in K]]
    mean = d.mean()
    se = d.std() / np.sqrt(d.notna().sum())
    post = mean[[f"x{k}" for k in range(1, 21)]]
    post_cum_se = np.sqrt((se[[f"x{k}" for k in range(1, 21)]] ** 2).cumsum())
    say(f"  {name:<14s} {int(m.sum()):>5,} 個｜事件前20天 {mean[[f'x{k}' for k in range(-20, 0)]].sum() * 100:+.2f}%｜當天 {mean['x0'] * 100:+.2f}%"
        f"｜之後 1天 {post.iloc[0] * 100:+.2f}%（t={post.iloc[0] / se['x1']:+.1f}）、5天 {post.iloc[:5].sum() * 100:+.2f}%（t={post.iloc[:5].sum() / post_cum_se.iloc[4]:+.1f}）"
        f"、20天 {post.sum() * 100:+.2f}%（t={post.sum() / post_cum_se.iloc[19]:+.1f}）")
    cum = mean.cumsum() - mean.cumsum()["x-1"]
    ax.plot(K, cum.values * 100, marker="o", ms=3, label=f"{name}（{int(m.sum()):,}）", color=col)
ax.axvline(0, color="#26221D", lw=0.8, ls="--")
ax.axhline(0, color="#9C9184", lw=0.6)
ax.set_xlabel("相對爆量日的交易日（0＝爆量當天收盤）")
ax.set_ylabel("累積超額報酬（%，以第 −1 天為 0）")
ax.set_title("PTT 討論爆量事件前後的累積超額報酬（2015–2024）")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("ptt_attention_spike.png")
say("  t 值為簡化估計（未處理同日事件的相關），僅供參考；圖：ptt_attention_spike.png")
open("ptt_attention_spike.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
