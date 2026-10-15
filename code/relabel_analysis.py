# -*- coding: utf-8 -*-
"""
Claude 重新標註 vs 原本的 Breeze 分類器：更準的標籤會不會改變 PTT 情緒的結論？
  讀 relabel/sample.csv 與 relabel/labels_XX.txt（已標的批次都會用到；p=看多、n=看空、u=中立）
  1) 留言層級：一致率、混淆矩陣、方向相反比例
  2) 股票日層級（同一批留言）：Claude 淨情緒 vs Breeze 淨情緒 的相關與斜率 b（測量誤差的衰減係數）
     → 真實係數 ≈ 觀察係數 ÷ b；用全樣本的觀察值推估「分類器像 Claude 一樣準」時的效果
  3) 同一批股票日直接比較：兩種淨情緒與「今天超額報酬」「隔天隔夜／隔天超額報酬」的關係（樣本小，僅供參考）
python relabel_analysis.py
"""
import glob
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score, confusion_matrix

warnings.filterwarnings("ignore")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


MAP = {"p": "positive", "n": "negative", "u": "neutral"}
lab = {}
for f in sorted(glob.glob("relabel/labels_*.txt")):
    for i, l in re.findall(r"(\d+):([pnu])", open(f, encoding="utf-8").read()):
        lab[int(i)] = MAP[l]
s = pd.read_csv("relabel/sample.csv", dtype={"stock_code": str}, parse_dates=["Date"])
s["claude"] = s.id.map(lab)
d = s.dropna(subset=["claude"]).copy()
L = ["positive", "negative", "neutral"]
say(f"已標註 {len(d):,} / {len(s):,} 則（{len(glob.glob('relabel/labels_*.txt'))} 批）；涵蓋股票日 {d.groupby(['stock_code', 'Date']).ngroups}")
say(f"分布：Claude {d.claude.value_counts(normalize=True).reindex(L).round(3).to_dict()}")
say(f"      Breeze {d.breeze.value_counts(normalize=True).reindex(L).round(3).to_dict()}")
cm = confusion_matrix(d.claude, d.breeze, labels=L)
say(f"\n1) 留言層級：一致率 {(d.claude == d.breeze).mean():.1%}｜Kappa {cohen_kappa_score(d.claude, d.breeze):.2f}")
say("   列＝Claude、欄＝Breeze（正/負/中）：")
for i, l in enumerate(L):
    say(f"     Claude {l:<9s} {cm[i].tolist()} → Breeze 判對 {cm[i, i] / max(cm[i].sum(), 1):.0%}")
both = d[d.claude.isin(L[:2]) & d.breeze.isin(L[:2])]
say(f"   兩邊都有表態的 {len(both)} 則中，方向相反 {(both.claude != both.breeze).mean():.1%}")
for b_lab in L[:2]:
    x = d[d.breeze == b_lab]
    say(f"   Breeze 判「{b_lab}」的 {len(x)} 則：Claude 認為正 {(x.claude == 'positive').mean():.0%}、負 {(x.claude == 'negative').mean():.0%}、中立 {(x.claude == 'neutral').mean():.0%}")


def net(g, col):
    p, n = (g[col] == "positive").sum(), (g[col] == "negative").sum()
    return (p - n) / (p + n) if p + n > 0 else np.nan


def net2(g, col):          # 把中立也算進分母，沒有表態＝0
    return ((g[col] == "positive").sum() - (g[col] == "negative").sum()) / len(g)


G = d.groupby(["stock_code", "Date"])
day = pd.DataFrame({"n": G.size(), "c": G.apply(lambda g: net(g, "claude")), "b": G.apply(lambda g: net(g, "breeze")),
                    "c2": G.apply(lambda g: net2(g, "claude")), "b2": G.apply(lambda g: net2(g, "breeze"))}).reset_index()
say(f"\n2) 股票日層級（{len(day)} 個股票日，每日平均 {day.n.mean():.1f} 則已標留言）")
for a, b_, name in [("c", "b", "淨情緒 (正−負)/(正+負)"), ("c2", "b2", "淨情緒 (正−負)/全部留言")]:
    x = day.dropna(subset=[a, b_])
    slope = np.cov(x[a], x[b_])[0, 1] / np.var(x[b_], ddof=1)
    r = x[a].corr(x[b_])
    se = np.sqrt((1 - r ** 2) * x[a].var() / x[b_].var() / (len(x) - 2))
    say(f"   {name}：n={len(x)}｜相關 {r:.2f}｜衰減係數 b（Claude 對 Breeze 的斜率）= {slope:.2f} ± {1.96 * se:.2f}"
        f"｜方向一致（同號）{(np.sign(x[a]) == np.sign(x[b_])).mean():.0%}")
    if a == "c":
        B, SE = slope, se
say("\n   推估「分類器像 Claude 一樣準」時的效果（真實係數 ≈ 觀察值 ÷ b；觀察值來自 2015-2024 全樣本 29,000 個股票日）")
for labn, beta, t in [("隔夜（控制今天/漲跌停/留言量）", 4.49, 3.78), ("隔夜（再控制過去5/20天）", 2.56, 2.18), ("隔天收盤→收盤", 4.16, 1.84)]:
    hi = beta + 1.96 * beta / t
    lo_b = max(B - 1.96 * SE, 0.05)
    say(f"     {labn:<18s} 觀察 {beta:+.1f} bp → 校正後 {beta / B:+.1f} bp；保守上界（觀察值上界÷b 下界）{hi / lo_b:+.1f} bp｜全負→全正為 2 倍")
say("     對照成本：當沖來回 32~44 bp、一般交易 47~59 bp")

# 3) 直接比較（同一批股票日）
df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Open", "Close", "abnormal_price"]).sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
df["r0"] = (df.Close / g.Close.shift(1) - 1).where(~abn & ~g.abnormal_price.shift(1).fillna(0).ne(0))
okn = g.abnormal_price.shift(-1).fillna(0).eq(0)
df["on"] = (g.Open.shift(-1) / df.Close - 1).where(okn)
df["r1"] = (g.Close.shift(-1) / df.Close - 1).where(okn)
for c in ["r0", "on", "r1"]:
    df["x_" + c] = (df[c] - df.groupby("Date")[c].transform("mean")) * 1e4
m = day.merge(df[["stock_code", "Date", "x_r0", "x_on", "x_r1"]], on=["stock_code", "Date"], how="left")
say(f"\n3) 同一批股票日直接比較（n≈{m.dropna(subset=['c', 'b', 'x_r0']).shape[0]}；樣本小，標準誤大，僅供參考）")
for y, yl in [("x_r0", "今天超額（描述力：情緒是否跟著今天漲跌）"), ("x_on", "隔天隔夜超額（預測）"), ("x_r1", "隔天收盤→收盤超額（預測）")]:
    x = m.dropna(subset=["c", "b", y])
    out = []
    for col, nm in [("b", "Breeze"), ("c", "Claude")]:
        res = sm.OLS(x[y], sm.add_constant(x[[col]])).fit(cov_type="HC1")
        out.append(f"{nm} 係數 {res.params[col]:+7.1f} bp（t={res.tvalues[col]:+.2f}）、相關 {x[col].corr(x[y]):+.3f}")
    say(f"   {yl}：" + "｜".join(out))
open("relabel_analysis.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
