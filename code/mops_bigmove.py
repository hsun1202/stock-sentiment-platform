# -*- coding: utf-8 -*-
"""
大波動日 → 回頭看公司重大訊息 → 找類別與關鍵字規律 → 看是什麼在影響（使用者 2026-10-08 提出的新方向；用公開資訊觀測站重大訊息，每則有時分秒）
資料：mops_material/*.jsonl（crawl_mops_material.py；265 檔、2015~2025 的重大訊息主旨）＋ final_dataset_v3_tejall25（TEJ 還原價；只取原 265 檔）。
作法：
  影響日：盤前（前一交易日 13:30 後 ~ 當天 09:00 前）與盤中（09:00~13:30）發布的訊息，影響日＝當天；收盤後發布的，影響日＝下一個交易日。
  超額報酬＝個股 − 當天全市場等權；大波動日＝|超額| > 自己過去 60 天標準差 3 倍且 > 3%。
  探索與驗證分開：類別規則與關鍵字只在 2015~2019 上找；2020~2025 只拿來驗「同一個類別/同一個詞」方向是否一致（表中兩段並列）。
  t 值依影響日的月份群聚（先算每月平均，再對月平均做 t）。
輸出：mops_bigmove.log、mops_bigmove_matched.csv
python mops_bigmove.py
"""
import glob
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

import sys

LOG = []
OUTS = len(sys.argv) > 1 and sys.argv[1] == "out"
TAG = "_out889" if OUTS else ""


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------- 股價 ----------------
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "stock_name", "Date", "Close", "abnormal_price", "in265"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
bad = (df.abnormal_price.fillna(0) != 0) | g.abnormal_price.shift(1).fillna(0).ne(0)
df["r"] = (df.Close / g.Close.shift(1) - 1).where(~bad)
df["x"] = df.r - df.groupby("Date").r.transform("mean")           # 市場＝全市場等權
df = df[df.in265 == (0 if OUTS else 1)].reset_index(drop=True)      # out＝只看不在原 265 檔內的股票（驗證用）
g = df.groupby("stock_code")
df["sd"] = g.x.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
df["big"] = ((df.x / df.sd).abs() > 3) & (df.x.abs() > 0.03)
df["up"], df["dn"] = df.big & (df.x > 0), df.big & (df.x < 0)
df["pos"] = g.cumcount()
for k in range(1, 6):
    df[f"x_p{k}"] = g.x.shift(-k)
df["x_m1"] = g.x.shift(1)
df["post1_4"] = df[[f"x_p{k}" for k in range(1, 5)]].sum(axis=1, min_count=3)
df["pre20"] = g.x.transform(lambda s: s.rolling(20, min_periods=15).sum().shift(1))
V = df[df.sd.notna()].copy()
base = V.big.mean()
say(f"[股價] {V.stock_code.nunique()} 檔、{len(V):,} 股票日（{V.Date.min().date()}~{V.Date.max().date()}）；大波動日 {int(V.big.sum()):,}（{base:.2%}；大漲 {int(V.up.sum()):,}、大跌 {int(V.dn.sum()):,}）")

# ---------------- 重大訊息 ----------------
rows = []
for f in glob.glob("mops_material/*.jsonl"):
    with open(f, encoding="utf-8") as fh:
        rows += [json.loads(line) for line in fh if line.strip()]
nw = pd.DataFrame(rows).drop_duplicates(["code", "date", "time", "title"])
nw["t"] = pd.to_datetime(nw.date + " " + nw.time, errors="coerce")
nw = nw[nw.t.notna()].rename(columns={"code": "stock_code"})
say(f"[重大訊息] {len(nw):,} 則、{nw.stock_code.nunique()} 檔（{nw.t.min().date()}~{nw.t.max().date()}）；每檔每年中位數 {nw.groupby(['stock_code', nw.t.dt.year]).size().median():.0f} 則")
cal = np.array(sorted(df.Date.unique()), dtype="datetime64[ns]")
d0 = nw.t.dt.normalize().values.astype("datetime64[ns]")
i = np.searchsorted(cal, d0)
is_td = (i < len(cal)) & (cal[np.minimum(i, len(cal) - 1)] == d0)
hm = (nw.t.dt.hour * 60 + nw.t.dt.minute).values
slot = np.where(is_td & (hm < 540), "盤前", np.where(is_td & (hm < 810), "盤中", "收盤後"))
imp = np.where((slot == "收盤後") & is_td, i + 1, i)
nw["slot"] = slot
nw = nw[imp < len(cal)].copy()
nw["Date"] = cal[imp[imp < len(cal)]]
say("  發布時段：" + "、".join(f"{k} {v:.0%}" for k, v in nw.slot.value_counts(normalize=True).items()))

# ---------------- 類別（規則只依 2015~2019 的標題歸納；重大訊息主旨很制式）----------------
CATS = [
    ("注意交易資訊", r"注意交易資訊|達公布注意|注意股票"),
    ("澄清媒體報導", r"澄清|媒體報導|報載|說明.*報導"),
    ("庫藏股", r"買回本公司股份|庫藏股|買回股份"),
    ("減資-彌補虧損", r"減資.*彌補虧損|彌補虧損.*減資"),
    ("減資-現金/其他", r"減資"),
    ("現金增資/私募/可轉債", r"現金增資|私募|可轉換公司債|發行.*公司債|海外存託憑證"),
    ("股利分派/配股", r"股利|盈餘分[派配]|配息|除息|除權|盈餘轉增資|資本公積轉增資|無償配股"),
    ("自結損益/營收", r"自結|營業收入|營收|損益|每股盈餘"),
    ("財報通過/提報", r"財務報告|財務報表|合併財報|財報"),
    ("法人說明會", r"法人說明會|法說會|業績發表|投資論壇|受邀參加"),
    ("併購/公開收購/股份交換", r"併購|公開收購|股份轉換|股份交換|吸收合併|簡易合併|合併案|分割|收購"),
    ("取得或處分資產", r"取得|處分"),
    ("背書保證/資金貸與", r"背書保證|資金貸與"),
    ("訴訟/裁罰/檢調", r"訴訟|裁罰|罰鍰|檢調|搜索|起訴|判決|仲裁|假扣押|違反"),
    ("災害/停工/事故", r"火災|火警|爆炸|停工|災害|地震|颱風|事故|罷工|停電"),
    ("人事異動", r"董事長|總經理|財務主管|會計主管|發言人|會計師|獨立董事|董事(?!會)|監察人|經理人|稽核主管|辭任|解任|異動"),
    ("股東會", r"股東常會|股東臨時會|股東會"),
    ("訂單/合約/新產品", r"訂單|合約|契約|簽[訂署]|得標|新藥|臨床|許可證|專利|授權"),
    ("董事會決議(其他)", r"董事會"),
    ("更正/補充", r"更正|補充|更補"),
]
CRE = [(n, re.compile(p)) for n, p in CATS]


def cat(t):
    for n, r in CRE:
        if r.search(t):
            return n
    return "其他"


nw["cat"] = nw.title.map(cat)
nw["sub"] = nw.title.str.startswith("代") | nw.title.str.contains(r"^本公司代|代子公司|代重要子公司", regex=True)
key = ["stock_code", "Date"]
N = nw.merge(V[key + ["stock_name", "x", "x_m1", "big", "up", "dn", "post1_4", "pre20"]], on=key, how="inner")
N["ym"] = N.Date.dt.to_period("M")
N["half"] = np.where(N.Date.dt.year <= 2019, "2015-19", "2020-25")
say(f"  對得到股價的 {len(N):,} 則；其中代子公司公告 {N['sub'].mean():.0%}")

# ---------------- 1) 覆蓋率 ----------------
cnt = N.groupby(key + ["slot"]).size().unstack(fill_value=0).reindex(columns=["盤前", "盤中", "收盤後"], fill_value=0).reset_index()
V = V.merge(cnt, on=key, how="left").fillna({"盤前": 0, "盤中": 0, "收盤後": 0})
V["any"] = (V["盤前"] + V["盤中"] + V["收盤後"]) > 0
samed = nw[nw.slot == "收盤後"].assign(Date=lambda z: z.t.dt.normalize()).groupby(key).size().rename("當天收盤後").reset_index()
V = V.merge(samed, on=key, how="left").fillna({"當天收盤後": 0})
say("\n===== 1) 大波動日有沒有重大訊息？（影響日＝當天；收盤後的訊息算在下一個交易日）=====")
for lab, m in [("一般日", ~V.big), ("大漲日", V.up), ("大跌日", V.dn)]:
    s = V[m]
    say(f"  {lab}（{len(s):,}）：有任何影響當天的訊息 {s['any'].mean():.1%}（前一天收盤後 {(s['收盤後'] > 0).mean():.1%}、盤前 {(s['盤前'] > 0).mean():.1%}、盤中 {(s['盤中'] > 0).mean():.1%}）"
        f"｜當天收盤後才發布 {(s['當天收盤後'] > 0).mean():.1%}")
say("\n===== 2) 有訊息的日子，大波動的機率 =====")
for lab, m in [("沒有訊息", ~V["any"]), ("有訊息（影響當天）", V["any"]), ("其中盤中發布", V["盤中"] > 0)]:
    s = V[m]
    say(f"  {lab}（{len(s):,}）：大波動 {s.big.mean():.2%}（{s.big.mean() / base:.1f} 倍）｜大漲 {s.up.mean():.2%}、大跌 {s.dn.mean():.2%}｜超額絕對值平均 {s.x.abs().mean():.2%}")
s = V[V["當天收盤後"] > 0]
say(f"  當天收盤後才發布訊息的日子（{len(s):,}）：當天（訊息發布前）就大波動 {s.big.mean():.2%}（{s.big.mean() / base:.1f} 倍）→ 股價先動、或訊息在回應股價")


# ---------------- 3) 各類別 ----------------
def tm(v, ym):
    d = pd.DataFrame({"v": v, "m": ym}).dropna()
    if len(d) < 20:
        return np.nan, np.nan
    mm = d.groupby("m").v.mean()
    return d.v.mean(), (mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 5 else np.nan)


say("\n===== 3) 各類重大訊息：影響日當天的超額報酬（去掉代子公司公告；同一股票日同類別只算一次）=====")
M = N[~N["sub"]].drop_duplicates(key + ["cat"])
say(f"  {'類別':<14s}{'則數':>7s}｜大波動機率｜影響日超額 全期(t)｜2015-19 → 2020-25｜之後4天(t)｜發布前20天")
tab = []
for c in [n for n, _ in CATS] + ["其他"]:
    s = M[M["cat"] == c]
    if len(s) < 30:
        continue
    a, t = tm(s.x, s.ym)
    a1, t1 = tm(s[s.half == "2015-19"].x, s[s.half == "2015-19"].ym)
    a2, t2 = tm(s[s.half == "2020-25"].x, s[s.half == "2020-25"].ym)
    p, tp_ = tm(s.post1_4, s.ym)
    tab.append((c, len(s), s.big.mean(), s.up.mean(), s.dn.mean(), a, t, a1, t1, a2, t2, p, tp_, s.pre20.mean(), s.x.abs().mean()))
for r in sorted(tab, key=lambda z: -z[2]):
    say(f"  {r[0]:<14s}{r[1]:>7,}｜{r[2]:.1%}（漲{r[3]:.1%}/跌{r[4]:.1%}）｜{r[5]:+.2%}（{r[6]:+.1f}）｜{r[7]:+.2%}（{r[8]:+.1f}）→ {r[9]:+.2%}（{r[10]:+.1f}）｜{r[11]:+.2%}（{r[12]:+.1f}）｜{r[13]:+.2%}")

# ---------------- 4) 關鍵字：2015-19 找、2020-25 驗 ----------------
names = [n for n in df.stock_name.dropna().unique() if len(str(n)) >= 2]
name_re = re.compile("|".join(sorted((re.escape(n) for n in names), key=len, reverse=True)))


def grams(title):
    t = name_re.sub("", title)
    out = set()
    for seg in re.split(r"[^一-鿿]+", t):
        for n in (2, 3, 4):
            for k in range(len(seg) - n + 1):
                out.add(seg[k:k + n])
    return out


K = N[~N["sub"]].drop_duplicates(key + ["title"]).copy()
K["g"] = K.title.map(grams)
A, B = K[K.half == "2015-19"], K[K.half == "2020-25"]


def discover(mask_col, topn=18, min_c=15):
    a, b = A[A[mask_col]].g.tolist(), A[~A.big].g.tolist()
    ca, cb = Counter(w for s in a for w in s), Counter(w for s in b for w in s)
    rows_ = []
    for w, c in ca.items():
        if c >= min_c:
            pa, pb = (c + 0.5) / (len(a) + 1), (cb.get(w, 0) + 0.5) / (len(b) + 1)
            z = (np.log(pa / (1 - pa)) - np.log(pb / (1 - pb))) / np.sqrt(1 / (c + 0.5) + 1 / (cb.get(w, 0) + 0.5))
            if pa / pb >= 1.8:
                rows_.append((w, c, pa / pb, z))
    rows_.sort(key=lambda r: -r[3])
    keep = []
    for r in rows_:
        if not any((r[0] in k[0]) or (k[0] in r[0]) for k in keep):
            keep.append(r)
        if len(keep) >= topn:
            break
    return keep


for lab, col in [("大漲日", "up"), ("大跌日", "dn")]:
    say(f"\n===== 4) {lab}：2015~2019 找到的詞 → 2020~2025 驗證（含該詞的訊息，影響日的超額報酬與大波動機率）=====")
    for w, c, mult, z in discover(col):
        a_ = A[A.g.map(lambda s: w in s)]
        b_ = B[B.g.map(lambda s: w in s)]
        if len(b_) < 15:
            continue
        say(f"  「{w}」 2015-19：{len(a_):,} 則、{lab}的 {mult:.1f} 倍、當天超額 {a_.x.mean():+.2%}｜2020-25：{len(b_):,} 則、當天超額 {b_.x.mean():+.2%}（t={tm(b_.x, b_.ym)[1]:+.1f}）、"
            f"大漲 {b_.up.mean():.1%}/大跌 {b_.dn.mean():.1%}（一般 {V.up.mean():.1%}/{V.dn.mean():.1%}）")

# ---------------- 5) 例子 ----------------
say("\n===== 5) 例子：2020 年以後波動最大、且有重大訊息的 20 個股票日 =====")
ex = N[(~N["sub"]) & N.big & (N.Date >= "2020-01-01")].sort_values("x", key=lambda s: s.abs(), ascending=False).drop_duplicates(key).head(20)
for _, r in ex.iterrows():
    say(f"  {r.Date.date()} {r.stock_code} {r.stock_name} 超額 {r.x:+.1%}｜[{r.slot} {str(r.t)[5:16]}]〔{r['cat']}〕{r.title[:40]}")
N.to_csv(f"mops_bigmove_matched{TAG}.csv", index=False, encoding="utf-8-sig")
open(f"mops_bigmove{TAG}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
