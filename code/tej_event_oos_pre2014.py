# -*- coding: utf-8 -*-
"""
用完全沒看過的年份（2013 年以前）驗證事件研究的結果。
背景：tej_event_study.py 在 2015~2025 的資料上（探索性）看到「庫藏股公告後第 2~5 天還有 +0.65% 的超額報酬」。
      那十一年的資料全部看過了，所以請使用者另外匯出 2013 年以前的 TEJ 還原股價（tej_raw/TEJ20261008113623.zip），在跑之前先寫定假說。

事先寫定的假說（2026-10-08，看到這批股價的任何報酬數字之前寫下；主要檢定 1 個，雙尾 p<0.05）：
  B1：公告日在這批股價涵蓋期間（到 2013-12-31 為止）的庫藏股事件，公告後第 2~5 個交易日的累積超額報酬平均 > 0
      （超額＝個股 − 當天全市場四碼普通股等權平均；t 依公告月份群聚：先算每月平均再對月平均做 t）。
  次要（先寫定方向，不算主要）：庫藏股第 0~1 天 > 0、第 6~20 天 > 0、公告前 20 天下跌的子樣本第 2~5 天 > 0；
      減資第 0~1 天 < 0；無償配股第 0~1 天 > 0；現金股利（配息率高於中位數）第 0~1 天 > 0；現增第 0~1 天 < 0。
作法與 tej_event_study.py 相同（第 0 天＝公告日，非交易日順延，最多 7 天），唯一差別是報酬直接用 TEJ 的「報酬率％」欄。
python tej_event_oos_pre2014.py
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

LOG = []
PKL = "tej_raw/tej_price_daily_pre2014.pkl"


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


if not os.path.exists(PKL):
    z = zipfile.ZipFile("tej_raw/TEJ20261008113623.zip")
    use = ["證券代碼", "年月日", "開盤價(元)", "收盤價(元)", "成交量(千股)", "報酬率％", "市值(百萬元)", "漲跌停", "注意股票(A)", "處置股票(D)", "全額交割(Y)", "市場別"]
    parts = []
    with z.open(z.infolist()[0]) as f:
        for ch in pd.read_csv(io.TextIOWrapper(f, encoding="utf-16"), sep="\t", usecols=use, dtype=str, na_values=["."], chunksize=500_000):
            ch["code"] = ch["證券代碼"].str.strip().str.split().str[0]
            ch = ch[ch.code.str.match(r"^[1-9]\d{3}$", na=False)]
            parts.append(ch.drop(columns=["證券代碼"]))
    p = pd.concat(parts, ignore_index=True)
    p["date"] = pd.to_datetime(p["年月日"].str.strip(), format="%Y%m%d")
    for c in ["開盤價(元)", "收盤價(元)", "成交量(千股)", "報酬率％", "市值(百萬元)"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.drop(columns=["年月日"]).sort_values(["code", "date"]).reset_index(drop=True)
    p.to_pickle(PKL)
p = pd.read_pickle(PKL)
p = p[p.date <= "2013-12-31"]
say(f"[股價] {PKL}：{len(p):,} 列、{p.code.nunique()} 檔、{p.date.min().date()}~{p.date.max().date()}；市場別 {p['市場別'].str.strip().value_counts().to_dict()}")

df = p[["code", "date", "報酬率％"]].rename(columns={"code": "stock_code", "date": "Date"}).copy()
df["r"] = df["報酬率％"] / 100
df["x"] = df.r - df.groupby("Date").r.transform("mean")
df["sd"] = df.groupby("stock_code").x.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
df["big"] = ((df.x / df.sd).abs() > 3) & (df.x.abs() > 0.03)
df["pos"] = df.groupby("stock_code").cumcount()
df["key"] = df.Date

e = pd.read_csv("tej_raw/tej_events.csv", dtype={"stock_code": str}, parse_dates=["event_date", "ann_date"])
E = e[(e.ann_date >= df.Date.min()) & (e.ann_date <= "2013-12-31") & e.stock_code.str.match(r"^[1-9]\d{3}$", na=False)].sort_values("ann_date")
m = pd.merge_asof(E, df[["stock_code", "key", "pos"]].sort_values("key"), left_on="ann_date", right_on="key", by="stock_code", direction="forward")
m = m[m.pos.notna() & ((m.key - m.ann_date).dt.days <= 7)].copy()
m["pos"] = m.pos.astype(int)
m = m.drop_duplicates(["stock_code", "kind", "key"])
say(f"[事件] 公告日在 {df.Date.min().date()}~2013-12-31 且對得到股價的：" + "、".join(f"{k} {n:,}" for k, n in m.kind.value_counts().items()))
X = df.set_index(["stock_code", "pos"])[["x", "big"]]
REL = list(range(-20, 21))
W = pd.DataFrame({k: X.x.reindex(pd.MultiIndex.from_arrays([m.stock_code, m.pos + k])).values for k in REL}, index=m.index)
for k in (0, 1):
    m[f"big{k}"] = X.big.reindex(pd.MultiIndex.from_arrays([m.stock_code, m.pos + k])).values
m["pre"] = W[[k for k in REL if k <= -1]].sum(axis=1, min_count=15)
m["d01"] = W[[0, 1]].sum(axis=1, min_count=2)
m["post2_5"] = W[[2, 3, 4, 5]].sum(axis=1, min_count=3)
m["post6_20"] = W[[k for k in REL if 6 <= k <= 20]].sum(axis=1, min_count=12)
m["bigany"] = m.big0.fillna(False).astype(bool) | m.big1.fillna(False).astype(bool)
m["ym"] = m.key.dt.to_period("M")


def ts(s, col):
    d = s[[col, "ym"]].dropna()
    mm = d.groupby("ym")[col].mean()
    t = mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 5 else np.nan
    return d[col].mean(), mm.mean(), t, 2 * stats.t.sf(abs(t), len(mm) - 1) if len(mm) > 5 else np.nan, len(d), len(mm)


def line(lab, s):
    out = f"  {lab:<18s} n={len(s):>5,}｜當天或隔天大波動 {s.bigany.mean():.1%}"
    for col, nm in [("pre", "前20天"), ("d01", "第0~1天"), ("post2_5", "第2~5天"), ("post6_20", "第6~20天")]:
        a, _, t, _, _, _ = ts(s, col)
        out += f"｜{nm} {a:+.2%}(t={t:+.1f})"
    say(out)


say("\n===== 主要檢定 B1：庫藏股公告後第 2~5 天 =====")
bb = m[m.kind == "庫藏股"]
a, am, t, pv, n, nm = ts(bb, "post2_5")
say(f"  n={n:,} 件、{nm} 個月｜平均 {a:+.2%}（月平均的平均 {am:+.2%}）｜t={t:+.2f}，p={pv:.4f} → {'通過' if (pv < 0.05 and am > 0) else '未通過'}"
    f"（參考：2015~2025 為 +0.65%，t=3.0；交易成本來回約 0.47~0.59%）")
for a_, b_ in [("2001", "2007"), ("2008", "2013")]:
    s = bb[(bb.key >= a_) & (bb.key <= b_ + "-12-31")]
    x = ts(s, "post2_5")
    say(f"    {a_}~{b_}：n={x[4]:,}｜{x[0]:+.2%}（t={x[2]:+.2f}）")

say("\n===== 次要：各類事件公告日前後（同 2015~2025 的表）=====")
for k in ["庫藏股", "減資", "庫藏股減資", "無償配股", "現金股利", "現增", "危機出事", "MSCI納入", "MSCI剔除"]:
    s = m[m.kind == k]
    if len(s) >= 20:
        line(k, s)
line("庫藏股：公告前20天跌", bb[bb.pre < 0]); line("庫藏股：公告前20天漲", bb[bb.pre >= 0])
s = m[m.kind == "現金股利"]
if s.rate.notna().sum() > 100:
    line("現金股利：高於中位數", s[s.rate > s.rate.median()]); line("現金股利：低於中位數", s[s.rate <= s.rate.median()])
s = m[m.kind == "減資"]
if s.rate.notna().sum() > 50:
    line("減資：幅度≥20%", s[s.rate.abs() >= 20]); line("減資：幅度<20%", s[s.rate.abs() < 20])

say("\n===== 庫藏股：公告後逐日平均超額報酬（%）=====")
wb = W.loc[bb.index]
say("  " + " ".join(f"{k:+d}:{wb[k].mean() * 100:+.2f}" for k in range(-3, 11)))
m.drop(columns=["key"]).to_csv("tej_event_oos_pre2014_events.csv", index=False, encoding="utf-8-sig")
open("tej_event_oos_pre2014.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
