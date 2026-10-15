# -*- coding: utf-8 -*-
"""
剩下四分之三的大波動日是什麼？——把每個大波動日依序歸因，並檢驗「沒有消息的大波動之後會不會反轉」。
資料：final_dataset_v3_tejall25（全市場、TEJ 還原價）、tej_raw/tej_基本資料.csv（TEJ 子產業 249 類；同類不足 4 檔者改用 TEJ 產業 97 類）、
      mops_bigmove_matched*.csv（公司重大訊息對到的影響日）、cnyes_bigmove_matched*.csv（鉅亨網新聞對到的影響日）。
大波動日：|超額| > 自己過去 60 天標準差 3 倍且 > 3%（超額＝個股−全市場等權）。
歸因順序（每個大波動日只歸到第一個符合的）：
  1 公司公告：有影響當天的重大訊息
  2 媒體新聞：有影響當天的鉅亨網個股新聞
  3 事後才有消息：當天收盤後（到下一個交易日開盤前）才出現公告或新聞
  4 同產業一起動：同產業其他股票（至少 3 檔）當天「同方向大波動」的比例 ≥ 25%，或同業平均超額同方向且絕對值 ≥ 2%
  5 延續前一天：前一個交易日也是同方向的大波動
  6 找不到
用法：python bigmove_decompose.py        # 原 265 檔（探索）
      python bigmove_decompose.py out    # 另外 889 檔（驗證）

事先寫定、要在 889 檔上驗的主要檢定（2026-10-10，跑 889 檔之前寫下；3 個，Bonferroni |t|>2.43；t 依月份群聚）：
  「有消息」＝歸因 1 或 2；「沒消息」＝歸因 4、5、6（歸因 3 兩邊都不算）。看大波動日之後第 1~5 個交易日的累積超額報酬。
  D1：沒消息的大漲日，之後 5 天累積超額 < 0（反轉）。
  D2：大漲日「有消息 − 沒消息」的之後 5 天累積超額 > 0（有消息的比較不反轉）。
  D3：沒消息的大跌日，之後 5 天累積超額 > 0（反彈）。
  追加（2026-10-10，看過 265 檔的探索結果之後、跑 889 檔之前寫下；合計 6 個主要檢定，門檻改為 |t|>2.64＝p<0.0083）：
  D4：有消息的大漲日，之後 5 天累積超額 > 0（續漲）。265 檔為 +1.09%。
  D5：有消息的大跌日，之後 5 天累積超額 < 0（續跌）。265 檔為 −1.33%。
  D6：大跌日「有消息 − 沒消息」的之後 5 天累積超額 < 0。265 檔為 −1.15%。
  （265 檔的探索結果：沒消息的大漲/大跌之後 5 天約為 0，沒有明顯反轉 → D1、D3 預期不會過，照原樣檢定、照實報。）
  文獻依據：Chan (2003, JFE)「有新聞的股價變動會延續、沒新聞的會反轉」；Savor (2012, JFE)。
"""
import sys

import numpy as np
import pandas as pd

LOG = []
OUTS = len(sys.argv) > 1 and sys.argv[1] == "out"
TAG = "_out889" if OUTS else ""


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "stock_name", "Date", "Close", "abnormal_price", "in265", "mktcap"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
bad = (df.abnormal_price.fillna(0) != 0) | g.abnormal_price.shift(1).fillna(0).ne(0)
df["r"] = (df.Close / g.Close.shift(1) - 1).where(~bad)
df["x"] = df.r - df.groupby("Date").r.transform("mean")
df["sd"] = g.x.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
df["big"] = (((df.x / df.sd).abs() > 3) & (df.x.abs() > 0.03)).fillna(False)
df["sgn"] = np.sign(df.x).fillna(0) * df.big
# 產業（全市場一起算同業，再取子樣本）
b = pd.read_csv("tej_raw/tej_基本資料.csv", encoding="cp950", dtype=str)
b["stock_code"] = b["代號"].str.strip()
b = b.drop_duplicates("stock_code").set_index("stock_code")
sub, ind = b["TEJ子產業名"].reindex(df.stock_code).values, b["TEJ產業名"].reindex(df.stock_code).values
nsub = pd.Series(sub).map(pd.Series(sub[~df.duplicated("stock_code").values]).value_counts())
df["grp"] = np.where(nsub.values >= 4, sub, ind)
gi = df.groupby(["Date", "grp"])
n = gi.x.transform("count")
df["peer_n"] = n - df.x.notna()
df["peer_x"] = (gi.x.transform("sum") - df.x.fillna(0)) / df.peer_n.replace(0, np.nan)
up_n, dn_n = gi.sgn.transform(lambda s: (s > 0).sum()), gi.sgn.transform(lambda s: (s < 0).sum())
df["peer_same"] = np.where(df.sgn > 0, up_n - 1, np.where(df.sgn < 0, dn_n - 1, 0)) / df.peer_n.replace(0, np.nan)
df["prev_same"] = (g.sgn.shift(1) == df.sgn) & (df.sgn != 0)
for k in range(1, 6):
    df[f"p{k}"] = g.x.shift(-k)
df["post5"] = df[[f"p{k}" for k in range(1, 6)]].sum(axis=1, min_count=4)
df["post1"] = df.p1
df["post2_5"] = df[[f"p{k}" for k in range(2, 6)]].sum(axis=1, min_count=3)
df["post6_20"] = g.x.transform(lambda s: s.shift(-20).rolling(15, min_periods=12).sum())
df["nxt"] = g.Date.shift(-1)
df = df[df.in265 == (0 if OUTS else 1)]
key = ["stock_code", "Date"]


def flag(path, name):
    try:
        m = pd.read_csv(path, dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "Date", "slot"])
    except FileNotFoundError:
        say(f"（找不到 {path}）")
        return
    a = m[key].drop_duplicates().assign(**{name: True})
    out = df.merge(a, on=key, how="left")
    # 當天收盤後才發布：下一個交易日的影響日、且時段為「收盤後」
    late = m[m.slot == "收盤後"][key].drop_duplicates().rename(columns={"Date": "nxt"}).assign(**{name + "_late": True})
    out = out.merge(late, on=["stock_code", "nxt"], how="left")
    return out.fillna({name: False, name + "_late": False})


df = flag(f"mops_bigmove_matched{TAG}.csv", "ann")
df = flag(f"cnyes_bigmove_matched{TAG}.csv", "news")
B = df[df.big & df.sd.notna()].copy()
ind_co = (B.peer_n >= 3) & ((B.peer_same >= 0.25) | ((np.sign(B.peer_x) == B.sgn) & (B.peer_x.abs() >= 0.02)))
B["why"] = np.select([B.ann, B.news, B.ann_late | B.news_late, ind_co, B.prev_same],
                     ["1 公司公告", "2 媒體新聞", "3 事後才有消息", "4 同產業一起動", "5 延續前一天"], "6 找不到")
B["ym"] = B.Date.dt.to_period("M")
say(f"[{'驗證：另外 889 檔' if OUTS else '探索：原 265 檔'}] 大波動日 {len(B):,}（大漲 {int((B.sgn > 0).sum()):,}、大跌 {int((B.sgn < 0).sum()):,}）；產業分組 {B.grp.nunique()} 類")
say("\n===== 1) 大波動日的歸因（依序、每天只算一類）=====")
t = pd.crosstab(B.why, np.where(B.sgn > 0, "大漲", "大跌"), normalize="columns")
c = pd.crosstab(B.why, np.where(B.sgn > 0, "大漲", "大跌"))
for w in t.index:
    say(f"  {w:<10s} 大漲 {t.loc[w, '大漲']:.1%}（{c.loc[w, '大漲']:,}）｜大跌 {t.loc[w, '大跌']:.1%}（{c.loc[w, '大跌']:,}）")
say(f"  （不分先後）同產業一起動的比例：大漲 {ind_co[B.sgn > 0].mean():.1%}、大跌 {ind_co[B.sgn < 0].mean():.1%}；延續前一天：大漲 {B.prev_same[B.sgn > 0].mean():.1%}、大跌 {B.prev_same[B.sgn < 0].mean():.1%}")
big_sz = B.mktcap.groupby(B.Date).rank(pct=True)
say("  依年份「找不到」的比例：" + " ".join(f"{y}:{v:.0%}" for y, v in (B.why == "6 找不到").groupby(B.Date.dt.year).mean().items()))


def tm(v, ym):
    z = pd.DataFrame({"v": v, "m": ym}).dropna()
    if len(z) < 20:
        return np.nan, np.nan, len(z)
    mm = z.groupby("m").v.mean()
    return z.v.mean(), mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))), len(z)


say("\n===== 2) 大波動日之後怎麼走？（累積超額報酬；t 依月份群聚）=====")
for sg, lab in [(1, "大漲"), (-1, "大跌")]:
    say(f"  --- {lab}日（當天超額中位數 {B[B.sgn == sg].x.median():+.1%}）---")
    for w in sorted(B.why.unique()):
        s = B[(B.sgn == sg) & (B.why == w)]
        if len(s) < 30:
            continue
        out = f"  {w:<10s} n={len(s):>6,}"
        for col, nm in [("post1", "隔天"), ("post2_5", "第2~5天"), ("post5", "之後5天"), ("post6_20", "第6~20天")]:
            a, tt, _ = tm(s[col], s.ym)
            out += f"｜{nm} {a:+.2%}(t={tt:+.1f})"
        say(out)
B["info"] = np.where(B.why.isin(["1 公司公告", "2 媒體新聞"]), "有消息", np.where(B.why == "3 事後才有消息", "事後", "沒消息"))
say("\n===== 3) 有消息 vs 沒消息（主要檢定）=====")
res = {}
for sg, lab in [(1, "大漲"), (-1, "大跌")]:
    s = B[B.sgn == sg]
    a1, t1, n1 = tm(s[s["info"] == "有消息"].post5, s[s["info"] == "有消息"].ym)
    a0, t0, n0 = tm(s[s["info"] == "沒消息"].post5, s[s["info"] == "沒消息"].ym)
    mm = s[s["info"] != "事後"].groupby(["ym", "info"]).post5.mean().unstack().dropna()
    d = mm["有消息"] - mm["沒消息"]
    td = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    res[lab] = (a1, t1, a0, t0, d.mean(), td)
    say(f"  {lab}日之後 5 天：有消息 {a1:+.2%}（t={t1:+.1f}，n={n1:,}）｜沒消息 {a0:+.2%}（t={t0:+.1f}，n={n0:,}）｜有消息−沒消息 {d.mean():+.2%}（t={td:+.1f}，{len(d)} 個月）")
    for a_, b_ in [("2015", "2019"), ("2020", "2025")]:
        x = s[(s.Date >= a_) & (s.Date <= b_ + "-12-31")]
        say(f"      {a_}~{b_}：有消息 {x[x['info'] == '有消息'].post5.mean():+.2%}｜沒消息 {x[x['info'] == '沒消息'].post5.mean():+.2%}")
if OUTS:
    u, dn_ = res["大漲"], res["大跌"]
    TH = 2.64
    ok = lambda c: "通過" if c else "未通過"
    say(f"  → D1 沒消息大漲之後<0：{u[2]:+.2%}，t={u[3]:+.2f} → {ok(u[2] < 0 and u[3] < -TH)}")
    say(f"  → D2 大漲 有消息−沒消息>0：{u[4]:+.2%}，t={u[5]:+.2f} → {ok(u[5] > TH)}")
    say(f"  → D3 沒消息大跌之後>0：{dn_[2]:+.2%}，t={dn_[3]:+.2f} → {ok(dn_[2] > 0 and dn_[3] > TH)}")
    say(f"  → D4 有消息大漲之後>0：{u[0]:+.2%}，t={u[1]:+.2f} → {ok(u[0] > 0 and u[1] > TH)}")
    say(f"  → D5 有消息大跌之後<0：{dn_[0]:+.2%}，t={dn_[1]:+.2f} → {ok(dn_[0] < 0 and dn_[1] < -TH)}")
    say(f"  → D6 大跌 有消息−沒消息<0：{dn_[4]:+.2%}，t={dn_[5]:+.2f} → {ok(dn_[5] < -TH)}（門檻 |t|>{TH}）")
B[key + ["stock_name", "x", "why", "info", "peer_n", "peer_x", "peer_same", "post1", "post5", "post6_20"]].to_csv(f"bigmove_decompose{TAG}.csv", index=False, encoding="utf-8-sig")
open(f"bigmove_decompose{TAG}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
