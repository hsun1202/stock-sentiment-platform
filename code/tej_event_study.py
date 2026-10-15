# -*- coding: utf-8 -*-
"""
TEJ「特殊事件日期資料庫」事件研究：減資、股利、庫藏股、現增、危機出事、MSCI 納入/剔除 的「公告日」前後，股價怎麼動？
（新方向「大波動日 → 是什麼在影響」的第一步：先用有明確日期與類別的事件，量每一類事件的平均影響。探索性。）
資料：tej_raw/20261008232334DataExport.csv（使用者 2026-10-08 匯出，18 個類別、全部歷史）→ 整理成 tej_raw/tej_events.csv
      股價：final_dataset_v3_tejall25（全市場 1,154 檔、2015-2025、TEJ 還原價）。
超額報酬＝個股 − 當天全市場等權平均。第 0 天＝公告日（非交易日則順延到下一個交易日）；公告多在收盤後，所以反應常落在第 +1 天。
大波動日定義同 bigmove_news.py：|超額| > 自己過去 60 天標準差 3 倍且 > 3%。
python tej_event_study.py
"""
import numpy as np
import pandas as pd

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


e = pd.read_csv("tej_raw/20261008232334DataExport.csv", encoding="cp950", dtype=str)
e["stock_code"] = e["公司及名稱"].str.strip().str.split().str[0]
e["kind"] = e["名稱"].str.replace(r"^(上市|下市)", "", regex=True).str.replace(r"\(.*\)", "", regex=True).str.replace("日", "")
for a, b in [("事件日", "event_date"), ("公告日", "ann_date"), ("股東會日", "agm_date")]:
    e[b] = pd.to_datetime(e[a], errors="coerce")
e["rate"] = pd.to_numeric(e["增減資率"], errors="coerce")
e = e[["stock_code", "kind", "代號", "event_date", "ann_date", "agm_date", "分配年度", "rate", "備註說明"]].rename(columns={"代號": "tej_group", "分配年度": "fiscal_year", "備註說明": "note"})
e.to_csv("tej_raw/tej_events.csv", index=False, encoding="utf-8-sig")
say(f"[事件] 全部 {len(e):,} 筆（{e.event_date.min().date()}~{e.event_date.max().date()}）；有公告日的 {e.ann_date.notna().mean():.0%}")
E = e[(e.ann_date >= "2015-01-01") & (e.ann_date <= "2025-12-31") & e.stock_code.str.match(r"^[1-9]\d{3}$", na=False)].copy()
say("  2015~2025 有公告日的筆數：" + "、".join(f"{k} {n:,}" for k, n in E.kind.value_counts().items()))
say("  公告日到事件日的間隔（天，中位數）：" + "、".join(f"{k} {int(v)}" for k, v in (E.event_date - E.ann_date).dt.days.groupby(E.kind).median().dropna().items()))

df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Close", "abnormal_price", "in265"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
bad = (df.abnormal_price.fillna(0) != 0) | g.abnormal_price.shift(1).fillna(0).ne(0)
df["r"] = (df.Close / g.Close.shift(1) - 1).where(~bad)
df["x"] = df.r - df.groupby("Date").r.transform("mean")
df["sd"] = df.groupby("stock_code").x.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
df["big"] = ((df.x / df.sd).abs() > 3) & (df.x.abs() > 0.03)
df["pos"] = g.cumcount()
base_big = df.big[df.sd.notna()].mean()
say(f"[股價] 全市場 {df.stock_code.nunique()} 檔、{len(df):,} 股票日；一般日大波動機率 {base_big:.2%}")

# 公告日 → 第一個 >= 公告日的交易日（各股自己的交易日）
df["key"] = df.Date
E = E.sort_values("ann_date")
m = pd.merge_asof(E, df[["stock_code", "key", "pos"]].sort_values("key"), left_on="ann_date", right_on="key", by="stock_code", direction="forward")
m = m[m.pos.notna() & ((m.key - m.ann_date).dt.days <= 7)].copy()
m["pos"] = m.pos.astype(int)
m = m.drop_duplicates(["stock_code", "kind", "key"])
X = df.set_index(["stock_code", "pos"])[["x", "big"]]
REL = list(range(-20, 21))
arr = {}
for k in REL:
    idx = pd.MultiIndex.from_arrays([m.stock_code, m.pos + k])
    arr[k] = X.x.reindex(idx).values
    if k in (0, 1):
        m[f"big{k}"] = X.big.reindex(idx).values
W = pd.DataFrame(arr, index=m.index)
m["pre"] = W[[k for k in REL if -20 <= k <= -1]].sum(axis=1, min_count=15)
m["pre5"] = W[[-5, -4, -3, -2, -1]].sum(axis=1, min_count=4)
m["d0"], m["d1"] = W[0], W[1]
m["d01"] = W[[0, 1]].sum(axis=1, min_count=2)
m["post2_5"] = W[[2, 3, 4, 5]].sum(axis=1, min_count=3)
m["post6_20"] = W[[k for k in REL if 6 <= k <= 20]].sum(axis=1, min_count=12)
m["bigany"] = (m.big0.fillna(False).astype(bool) | m.big1.fillna(False).astype(bool))


def ts(s, by):
    """依公告月份群聚的 t：先算每月平均，再對月平均做 t"""
    d = pd.DataFrame({"v": s, "m": by}).dropna()
    mm = d.groupby("m").v.mean()
    return d.v.mean(), (mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 5 else np.nan), len(d)


def line(lab, s):
    ym = s.key.dt.to_period("M")
    out = f"  {lab:<16s} n={len(s):>5,}｜公告當天或隔天大波動 {s.bigany.mean():.1%}（一般約 {1 - (1 - base_big) ** 2:.1%}）"
    for col, nm in [("pre", "前20天"), ("d01", "第0~1天"), ("post2_5", "第2~5天"), ("post6_20", "第6~20天")]:
        a, t, _ = ts(s[col], ym)
        out += f"｜{nm} {a:+.2%}(t={t:+.1f})"
    say(out)


say("\n===== 各類事件公告日前後的累積超額報酬（t 依公告月份群聚）=====")
for k in ["減資", "庫藏股減資", "庫藏股", "現金股利", "無償配股", "現增", "危機出事", "MSCI納入", "MSCI剔除"]:
    s = m[m.kind == k]
    if len(s) >= 20:
        line(k, s)

say("\n===== 細分（看得出方向的幾類）=====")
s = m[m.kind == "減資"]
if s.rate.notna().sum() > 50:
    q = s.rate.abs()
    line("減資：幅度≥20%", s[q >= 20]); line("減資：幅度<20%", s[q < 20])
s = m[m.kind == "現金股利"]
if s.rate.notna().sum() > 100:
    med = s.rate.median()
    line(f"現金股利：高於中位數", s[s.rate > med]); line(f"現金股利：低於中位數", s[s.rate <= med])
for k in ["庫藏股", "現金股利", "減資"]:
    s = m[m.kind == k]
    line(f"{k}：公告前20天跌", s[s.pre < 0]); line(f"{k}：公告前20天漲", s[s.pre >= 0])
for k in ["庫藏股", "現金股利"]:
    s = m[m.kind == k]
    line(f"{k}：2015-2019", s[s.key.dt.year <= 2019]); line(f"{k}：2020-2025", s[s.key.dt.year >= 2020])

say("\n===== 事件能解釋多少大波動日？=====")
bigdays = df[df.big & df.sd.notna()][["stock_code", "pos", "Date", "x"]]
ev = pd.concat([m[["stock_code", "pos", "kind"]], m[["stock_code", "pos", "kind"]].assign(pos=lambda z: z.pos + 1)])
hit = bigdays.merge(ev, on=["stock_code", "pos"], how="left")
cov = hit.drop_duplicates(["stock_code", "pos"]).kind.notna().mean()
say(f"  全市場大波動日 {len(bigdays):,} 個；其中是這九類事件公告當天或隔天的 {cov:.1%}")
say("  各類占大波動日的比例：" + "、".join(f"{k} {v / len(bigdays):.2%}" for k, v in hit.kind.value_counts().items()))
m.drop(columns=[c for c in m.columns if c in ("key",)]).to_csv("tej_event_study_events.csv", index=False, encoding="utf-8-sig")
open("tej_event_study.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
