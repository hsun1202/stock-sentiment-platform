# -*- coding: utf-8 -*-
"""
PTT 情緒事件研究＋加強控制（接續 ptt_cc_deepdive.py）
  A) 加強控制：除了今天漲跌，再控制過去 5/20/60 天超額報酬、20 天波動、量能、留言量、規模、漲跌停 → 淨情緒還剩多少？
  B) 事件研究：以「留言日收盤」為第 0 天，強正（≥3則表態、淨≥0.5）vs 強負（淨≤−0.5），看第 −20 ~ +20 天的累積超額報酬
     → 若事件前就大漲/大跌（情緒跟著價格走）、事件後持續，代表情緒是落後指標＋動能
  輸出：ptt_event_study.log、ptt_event_car.csv、ptt_event_car.png
python ptt_event_study.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Close", "Volume", "close_unadj", "abnormal_price",
                          "ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
for c in ["ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"]:
    df[c] = df[c].fillna(0)
pos = df.ptt_pre_pos + df.ptt_intra_pos
neg = df.ptt_pre_neg + df.ptt_intra_neg
df["n_expr"] = pos + neg
df["att"] = np.log1p(df.ptt_pre_n + df.ptt_intra_n)
df["net"] = (pos - neg) / df.n_expr.replace(0, np.nan)
abn = df.abnormal_price.fillna(0) != 0
r = (df.Close / g.Close.shift(1) - 1).where(~abn & ~g.abnormal_price.shift(1).fillna(0).ne(0))
df["r"] = r
df["x"] = r - r.groupby(df.Date).transform("mean")          # 每日超額
gx = df.groupby("stock_code").x
for k in range(-20, 21):                                       # 第 k 天的超額（k>0 未來、k<=0 過去含當天）
    df[f"x{k}"] = gx.shift(-k)
limit = np.where(df.Date < "2015-06-01", 0.07, 0.10)
df["lim_up"] = (df.r >= limit - 0.005).astype(int)
df["lim_dn"] = (df.r <= -limit + 0.005).astype(int)
df["past5"] = df[[f"x{k}" for k in range(-5, 0)]].sum(axis=1, min_count=4)
df["past20"] = df[[f"x{k}" for k in range(-20, 0)]].sum(axis=1, min_count=15)
df["past60"] = gx.transform(lambda s: s.shift(1).rolling(60, min_periods=40).sum())
df["vol20"] = gx.transform(lambda s: s.rolling(20, min_periods=10).std())
vol20 = g.Volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
df["volr"] = np.log((df.Volume + 1) / (vol20 + 1))
dv = df.close_unadj * df.Volume.where(df.Volume > 0)
df["size_r"] = dv.groupby(df.stock_code).transform(lambda s: s.rolling(60, min_periods=20).median()).groupby(df.Date).rank(pct=True)
df["x_abs0"] = df.x0.abs()
df["x2_5"] = df[[f"x{k}" for k in range(2, 6)]].sum(axis=1, min_count=4)
df["x6_20"] = df[[f"x{k}" for k in range(6, 21)]].sum(axis=1, min_count=12)

S = df[(df.n_expr >= 1) & df.x0.notna() & df.x1.notna()].copy()
say(f"樣本：有表態留言的股票日 {len(S):,}")
C1 = ["x0", "x_abs0", "lim_up", "lim_dn", "volr", "att", "size_r"]
C2 = C1 + ["past5", "past20", "past60", "vol20"]
say("\nA) 淨情緒係數（bp；依交易日群聚）：控制只有今天 vs 加上過去 5/20/60 天報酬與波動")
for ycol, lab in [("x1", "隔天"), ("x2_5", "第2~5天"), ("x6_20", "第6~20天")]:
    for ctrl, cl in [(C1, "今天＋漲停等"), (C2, "＋過去報酬與波動")]:
        d = S.dropna(subset=[ycol] + ctrl + ["net"])
        res = sm.OLS(d[ycol] * 1e4, sm.add_constant(d[["net"] + ctrl])).fit(cov_type="cluster", cov_kwds={"groups": d.Date.dt.strftime("%Y%m%d")})
        say(f"  {lab:<6s}｜{cl:<12s} n={len(d):,}｜淨情緒 {res.params['net']:+6.2f}（t={res.tvalues['net']:+.2f}）"
            f"｜過去20天 {res.params.get('past20', np.nan) * 1:+.1f}")
say(f"\n  淨情緒與過去報酬的相關：過去5天 {S.net.corr(S.past5):+.3f}、過去20天 {S.net.corr(S.past20):+.3f}、今天 {S.net.corr(S.x0):+.3f}")

# B) 事件研究
say("\nB) 事件研究：累積超額報酬（%），第 0 天＝留言日（收盤）")
groups = {"強正（≥3則、淨≥0.5）": (S.n_expr >= 3) & (S.net >= 0.5), "強負（≥3則、淨≤−0.5）": (S.n_expr >= 3) & (S.net <= -0.5),
          "中性/混合": (S.n_expr >= 3) & (S.net.abs() < 0.5)}
K = list(range(-20, 21))
rows = []
for name, m in groups.items():
    d = S[m]
    mean = d[[f"x{k}" for k in K]].mean()
    se = d[[f"x{k}" for k in K]].std() / np.sqrt(d[[f"x{k}" for k in K]].notna().sum())
    pre = mean[[f"x{k}" for k in range(-20, 0)]].cumsum()
    car_pre = mean[[f"x{k}" for k in range(-20, 1)]][::-1].cumsum()[::-1]   # 從 k 到 0 的累積（事件前漲多少）
    post = mean[[f"x{k}" for k in range(1, 21)]].cumsum()
    for k in K:
        rows.append({"group": name, "k": k, "mean_bp": mean[f"x{k}"] * 1e4, "se_bp": se[f"x{k}"] * 1e4})
    say(f"  {name:<16s} {len(d):,} 筆｜事件前20天累積 {mean[[f'x{k}' for k in range(-20, 0)]].sum() * 100:+.2f}%｜"
        f"當天 {mean['x0'] * 100:+.2f}%｜事件後 1天 {post.iloc[0] * 100:+.2f}%、5天 {post.iloc[4] * 100:+.2f}%、20天 {post.iloc[19] * 100:+.2f}%")
car = pd.DataFrame(rows)
car.to_csv("ptt_event_car.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
colors = {"強正（≥3則、淨≥0.5）": "#A83A2C", "強負（≥3則、淨≤−0.5）": "#3E6B4F", "中性/混合": "#9C9184"}
for name, d in car.groupby("group", sort=False):
    d = d.sort_values("k")
    cum = (d.mean_bp / 100).cumsum() - (d.mean_bp / 100).cumsum()[d.k == -1].values[0]   # 以第 −1 天為 0
    ax.plot(d.k, cum, marker="o", ms=3, label=name, color=colors[name])
ax.axvline(0, color="#26221D", lw=0.8, ls="--")
ax.axhline(0, color="#9C9184", lw=0.6)
ax.set_xlabel("相對留言日的交易日（0＝留言當天收盤）")
ax.set_ylabel("累積超額報酬（%，以第 −1 天為 0）")
ax.set_title("PTT 個股情緒事件前後的累積超額報酬（2015–2024）")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("ptt_event_car.png")
say("  圖：ptt_event_car.png（以第 −1 天為 0，所以第 0 天的跳動＝留言當天的漲跌）")
open("ptt_event_study.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
