# -*- coding: utf-8 -*-
"""
注意股／處置股事件研究（散戶熱炒的股票被交易所點名之後，股價怎麼走？）
資料：TEJ 調整股價(日) 的「注意股票(A)」「處置股票(D)」旗標。目前只有 2001~2013 那份匯出檔有這兩欄（tej_raw/tej_price_daily_pre2014.pkl）。
用法：python attention_disposition_study.py            # 2001~2013：探索（找規律）
      python attention_disposition_study.py post       # 2014~2025：驗證（要先請使用者重匯含旗標的股價 → tej_raw/tej_price_daily_post2014_flags.pkl）
定義：超額＝個股 − 當天全市場四碼普通股等權平均（用 TEJ 報酬率欄）。
  注意股事件：旗標 A 出現且前一個交易日沒有 A（一波的第一天）；依事件前 5 天累積超額分「漲上來的」（>0）與「跌下去的」（<0）。
  處置股事件：一段連續 D 的第一天＝處置開始（d0），最後一天＝處置結束；看 處置前 10 天、處置期間、處置結束後 5／20 天。
t 依事件月份群聚（先算月平均，再對月平均做 t）。
"""
import sys

import numpy as np
import pandas as pd

LOG = []
POST = len(sys.argv) > 1 and sys.argv[1] == "post"
PKL = "tej_raw/tej_price_daily_post2014_flags.pkl" if POST else "tej_raw/tej_price_daily_pre2014.pkl"
TAG = "post2014" if POST else "pre2014"


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


p = pd.read_pickle(PKL).sort_values(["code", "date"]).reset_index(drop=True)
d = pd.DataFrame({"code": p.code, "date": p.date, "r": p["報酬率％"] / 100,
                  "A": p["注意股票(A)"].notna().values, "D": p["處置股票(D)"].notna().values})
d["x"] = d.r - d.groupby("date").r.transform("mean")
g = d.groupby("code")
d["pos"] = g.cumcount()
cs = g.x.cumsum().fillna(method="ffill")
d["cx"] = d.x.fillna(0).groupby(d.code).cumsum()          # 累積超額（缺值當 0）
say(f"[資料] {PKL}：{len(d):,} 列、{d.code.nunique()} 檔、{d.date.min().date()}~{d.date.max().date()}；注意日 {int(d.A.sum()):,}、處置日 {int(d.D.sum()):,}")
X = d.set_index(["code", "pos"])


def cum(ev, a, b):
    """事件相對位置 a~b（含）的累積超額＝cx[pos+b] − cx[pos+a−1]"""
    hi = X.cx.reindex(pd.MultiIndex.from_arrays([ev.code, ev.pos + b])).values
    lo = X.cx.reindex(pd.MultiIndex.from_arrays([ev.code, ev.pos + a - 1])).values
    return hi - lo


def tm(v, ym):
    z = pd.DataFrame({"v": v, "m": ym}).dropna()
    if len(z) < 20:
        return np.nan, np.nan, len(z)
    mm = z.groupby("m").v.mean()
    return z.v.mean(), mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))), len(z)


def show(lab, ev, wins):
    out = f"  {lab:<20s} n={len(ev):>5,}"
    for nm, a, b in wins:
        m, t, _ = tm(cum(ev, a, b), ev.ym)
        out += f"｜{nm} {m:+.2%}(t={t:+.1f})"
    say(out)


# ---------------- 注意股 ----------------
A = d[d.A & ~g.A.shift(1).fillna(False).astype(bool)].copy()
A["ym"] = A.date.dt.to_period("M")
A["pre5"] = cum(A, -4, 0)                                   # 含當天的 5 天（旗標日通常就是達標那天）
W = [("前20天", -20, -5), ("最近5天(含當天)", -4, 0), ("第1天", 1, 1), ("第2~5天", 2, 5), ("第6~20天", 6, 20)]
say("\n===== 注意股：一波的第一天為事件日 =====")
show("全部", A, W)
show("漲上來的(近5天>0)", A[A.pre5 > 0], W)
show("　其中近5天>+15%", A[A.pre5 > 0.15], W)
show("跌下去的(近5天<0)", A[A.pre5 < 0], W)
for a_, b_ in ([("2014", "2019"), ("2020", "2025")] if POST else [("2001", "2007"), ("2008", "2013")]):
    s = A[(A.date >= a_) & (A.date <= b_ + "-12-31") & (A.pre5 > 0)]
    show(f"漲上來的 {a_}~{b_}", s, W)

# ---------------- 處置股 ----------------
d["Dstart"] = d.D & ~g.D.shift(1).fillna(False).astype(bool)
d["Dend"] = d.D & ~g.D.shift(-1).fillna(False).astype(bool)
S = d[d.Dstart][["code", "date", "pos"]].copy()
Eend = d[d.Dend][["code", "pos"]].rename(columns={"pos": "epos"})
S = pd.merge_asof(S.sort_values("pos"), Eend.sort_values("epos"), left_on="pos", right_on="epos", by="code", direction="forward")
S["len"] = S.epos - S.pos + 1
S["ym"] = S.date.dt.to_period("M")
S["pre10"] = cum(S, -10, -1)
say(f"\n===== 處置股：{len(S):,} 段（處置期間長度中位數 {S.len.median():.0f} 個交易日）=====")


def dshow(lab, ev):
    hi = X.cx.reindex(pd.MultiIndex.from_arrays([ev.code, ev.epos])).values
    lo = X.cx.reindex(pd.MultiIndex.from_arrays([ev.code, ev.pos - 1])).values
    during = hi - lo
    e2 = ev.assign(pos=ev.epos)
    out = f"  {lab:<20s} n={len(ev):>5,}"
    for nm, v in [("處置前10天", cum(ev, -10, -1)), ("處置期間", during), ("結束後1~5天", cum(e2, 1, 5)), ("結束後6~20天", cum(e2, 6, 20))]:
        m, t, _ = tm(v, ev.ym)
        out += f"｜{nm} {m:+.2%}(t={t:+.1f})"
    say(out)


dshow("全部", S)
dshow("漲上來的(前10天>0)", S[S.pre10 > 0])
dshow("　其中前10天>+30%", S[S.pre10 > 0.30])
dshow("跌下去的(前10天<0)", S[S.pre10 < 0])
for a_, b_ in ([("2014", "2019"), ("2020", "2025")] if POST else [("2001", "2007"), ("2008", "2013")]):
    dshow(f"漲上來的 {a_}~{b_}", S[(S.date >= a_) & (S.date <= b_ + "-12-31") & (S.pre10 > 0)])
open(f"attention_disposition_study_{TAG}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
