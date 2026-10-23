# -*- coding: utf-8 -*-
"""
G 節：財報公布後追價——月營收驚喜（SUR，持有 20 天）、季報盈餘驚喜（SUE，持有 60 天）。
事先寫定見 PREREG_ai_strategies.md G 節。公布日之後第一個交易日開盤買、持有 H 個交易日後開盤賣；每天等權、開盤到開盤；扣成本。
cd D:/專題claude && PYTHONIOENCODING=utf-8 python ai_strategies/surprise_event_backtest.py
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "ai_strategies/"
COST = 0.00471
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def t_(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def perf(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    return (1 + r).prod() ** (12 / len(r)) - 1, r.std() * np.sqrt(12), r.mean() * 12 / (r.std() * np.sqrt(12)), (eq / eq.cummax() - 1).min()


def to_date(s):
    return pd.to_datetime(s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace("/", ""), format="%Y%m%d", errors="coerce")


# ---------- 上市名單（T86 有出現的月份）----------
listed = {}
for f in sorted(glob.glob("chip_raw/t86/*.json")):
    j = json.load(open(f, encoding="utf-8"))
    if j.get("data"):
        listed.setdefault(os.path.basename(f)[:6], set()).update(r[0].strip() for r in j["data"])

# ---------- 日資料：開盤到開盤報酬、流動性 ----------
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str},
                 usecols=["stock_code", "Date", "Open", "Close", "close_unadj", "Volume"], parse_dates=["Date"])
df = df[df.stock_code.str.match(r"^[1-9]\d{3}$")].sort_values(["stock_code", "Date"])
df["dv"] = (df.close_unadj * df.Volume).where(df.Volume > 0)
df["liq60"] = df.groupby("stock_code").dv.transform(lambda x: x.rolling(60, min_periods=20).median())
op = df.pivot(index="Date", columns="stock_code", values="Open")
cl = df.pivot(index="Date", columns="stock_code", values="Close")
liq = df.pivot(index="Date", columns="stock_code", values="liq60")
del df
dates, codes = op.index, op.columns
T, data_end = len(dates), dates[-1]
r = (op.shift(-1) / op - 1)                                         # 第 t 天開盤 → 第 t+1 天開盤
last = op.apply(lambda s: s.last_valid_index())
for c, d in last.items():                                            # 之後下市：最後一天記 收盤 ×0.5
    if d is not None and d < data_end - pd.Timedelta(days=10):
        r.loc[d, c] = cl.loc[d, c] / op.loc[d, c] * 0.5 - 1
lst = pd.DataFrame(False, index=dates, columns=codes)
for ym, grp in pd.Series(dates, index=dates).groupby(dates.strftime("%Y%m")):
    lst.loc[grp.index, lst.columns.isin(listed.get(ym, set()))] = True
liqok = (liq.where(lst).rank(axis=1, pct=True) > 0.2) & lst
R = r.values
ew = pd.Series(np.nanmean(np.where(liqok.values, R, np.nan), axis=1), index=dates)
say(f"[日資料] {dates[0].date()}～{data_end.date()}，{len(codes)} 檔；每天可買約 {liqok.sum(axis=1).loc['2018':].mean():.0f} 檔")
cidx = {c: i for i, c in enumerate(codes)}


def code_of(s):
    return s.astype(str).str.split().str[0]


# ---------- 月營收驚喜 SUR ----------
rv = pd.read_csv("tej_raw/rev/20261022103241.csv", encoding="utf-16", sep="\t", on_bad_lines="skip", low_memory=False, na_values=["."])
rv["code"] = code_of(rv["證券代碼"])
rv = rv[rv.code.str.match(r"^[1-9]\d{3}$")].copy()
rv["per"] = pd.PeriodIndex(pd.to_datetime(rv["年月"].astype(str) + "01", format="%Y%m%d"), freq="M")
rv["pub"] = to_date(rv["營收發布日"])
rv["R"] = rv["以合併為主單月營收(千元)"]
rv = rv.drop_duplicates(["code", "per"], keep="last")
ev_rev = []
for c, g in rv.groupby("code"):
    s = g.set_index("per").R
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    d = s - s.shift(12)
    mu, sd = d.shift(1).rolling(24, min_periods=12).mean(), d.shift(1).rolling(24, min_periods=12).std()
    sur = ((d - mu) / sd.where(sd > 0)).clip(-10, 10)
    gg = g.set_index("per")
    ev_rev.append(pd.DataFrame({"code": c, "pub": gg.pub, "x": sur.reindex(gg.index)}))
ev_rev = pd.concat(ev_rev).dropna().reset_index().rename(columns={"index": "per"})
ev_rev["q"] = 0

# ---------- 季報盈餘驚喜 SUE ----------
fn = pd.read_csv("tej_raw/fin/20261022104526.csv", encoding="utf-16", sep="\t", low_memory=False, na_values=["."])
fn["code"] = code_of(fn["證券代碼"])
fn = fn[fn.code.str.match(r"^[1-9]\d{3}$")].copy()
qend = pd.to_datetime(fn["年月"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
fn["per"] = pd.PeriodIndex(qend, freq="Q")
fn["pub"] = to_date(fn["財報發布日"]).where(lambda x: x >= qend)
fn["N"] = fn["歸屬母公司淨利（損）"]
fn = fn.drop_duplicates(["code", "per"], keep="last")
ev_ern = []
for c, g in fn.groupby("code"):
    s = g.set_index("per").N
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="Q"))
    e = s - s.shift(4)
    mu, sd = e.shift(1).rolling(8, min_periods=4).mean(), e.shift(1).rolling(8, min_periods=4).std()
    sue = ((e - mu) / sd.where(sd > 0)).clip(-10, 10)
    gg = g.set_index("per")
    ev_ern.append(pd.DataFrame({"code": c, "pub": gg.pub, "x": sue.reindex(gg.index), "q": gg.index.quarter}))
ev_ern = pd.concat(ev_ern).dropna().reset_index().rename(columns={"index": "per"})


def prep(ev, nm):
    """對到進場日（公布日之後第一個交易日）、套用上市／流動性／開盤漲停排除、滾動 12 個月門檻分 10 組"""
    ev = ev[ev.code.isin(cidx)].copy()
    ev["ent"] = dates.searchsorted(ev.pub.values, side="right")
    ev = ev[(ev.ent >= 1) & (ev.ent < T - 1)]
    ci = ev.code.map(cidx).values
    ent = ev.ent.values
    ok = liqok.values[ent - 1, ci]                                    # 公布當天（或之前最後一個交易日）收盤時的條件
    gap = op.values[ent, ci] / cl.values[ent - 1, ci] - 1
    ev["gap_up"] = gap >= 0.095
    ev = ev[ok & np.isfinite(gap)].copy()
    ev["m"] = ev.pub.dt.to_period("M")
    bps = {}
    for m in pd.period_range("2015-01", "2025-12", freq="M"):
        past = ev.x[(ev.m < m) & (ev.m >= m - 12)]
        bps[m] = np.quantile(past, np.linspace(0.1, 0.9, 9)) if len(past) >= 200 else None
    ev = ev[ev.m.map(lambda m: bps.get(m) is not None)].copy()
    ev["dec"] = [int(np.searchsorted(bps[m], x, side="right")) + 1 for m, x in zip(ev.m, ev.x)]
    say(f"[{nm}] 事件 {len(ev):,} 筆（2018 後 {int((ev.pub >= '2018-01-01').sum()):,}）；開盤漲幅 ≥9.5% 買不到 {ev.gap_up.mean():.1%}")
    return ev


ev_rev, ev_ern = prep(ev_rev, "月營收"), prep(ev_ern, "季報盈餘")


def port(sets, start="2017-09-01"):
    """sets：[(事件, H, 延後天數)]；每天持有進行中的事件股票（重疊只算一次），等權、開盤到開盤；新部位記 0.471% ×（1／持股數）"""
    cnt = np.zeros((T + 1, len(codes)), dtype=np.int32)
    for ev, H, lag in sets:
        e = ev[(ev.pub >= start) & ~ev.gap_up]
        a = e.ent.values + lag
        ci = e.code.map(cidx).values
        k = a < T
        np.add.at(cnt, (a[k], ci[k]), 1)
        np.add.at(cnt, (np.minimum(a[k] + H, T), ci[k]), -1)
    held = np.cumsum(cnt, axis=0)[:T] > 0
    new = held & ~np.vstack([np.zeros((1, len(codes)), bool), held[:-1]])
    Rh = np.where(held, R, np.nan)
    n = np.isfinite(Rh).sum(axis=1)
    gross = np.where(n > 0, np.nansum(Rh, axis=1) / np.maximum(n, 1), 0.0)
    costd = np.where(n > 0, new.sum(axis=1) / np.maximum(n, 1) * COST, 0.0)
    g = pd.Series(gross, index=dates)
    net = pd.Series(gross - costd, index=dates)
    mon = lambda s: (1 + s.loc["2018-01-01":"2025-12-31"]).groupby(pd.PeriodIndex(s.loc["2018-01-01":"2025-12-31"].index, freq="M")).prod() - 1  # noqa: E731
    return mon(net), mon(g), pd.Series(n, index=dates).loc["2018":].mean()


ewm = (1 + ew.loc["2018-01-01":"2025-12-31"]).groupby(pd.PeriodIndex(ew.loc["2018-01-01":"2025-12-31"].index, freq="M")).prod() - 1
e50 = pd.read_csv("quarterly_vol/etf0050_adj.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
b50 = e50.resample("M").last().pct_change()
b50.index = b50.index.to_period("M")
b50 = b50.reindex(ewm.index)


def verdict(d):
    return "通過" if d.mean() > 0 and t_(d) > 1.96 else "未通過"


def yearly(d):
    return " ".join(f"{y}:{v:+.1%}" for y, v in d.groupby(d.index.year).sum().items())


res = {}
say("\n===== G. 財報公布後追價（上市、扣成本），2018-01～2025-12 =====")
for nm, ev, H in [("月營收驚喜", ev_rev, 20), ("盈餘驚喜", ev_ern, 60)]:
    top, bot = ev[ev.dec == 10], ev[ev.dec == 1]
    net, gross, nh = port([(top, H, 0)])
    net2, _, _ = port([(top, H, 1)])
    bnet, bgross, _ = port([(bot, H, 0)])
    res[nm] = net
    d = (net - ewm).dropna()
    a, v, shp, dd = perf(net)
    say(f"\n[{nm}最高 10%，持有 {H} 天] 平均持股 {nh:.0f} 檔｜年化 {a:+.1%}｜波動 {v:.1%}｜Sharpe {shp:.2f}｜最大回落 {dd:.1%}")
    say(f"  不扣成本 − 一般股票 {(gross - ewm).mean():+.2%}／月（t={t_(gross - ewm):+.2f}）；成本約 {(gross - net).mean():.2%}／月")
    say(f"  扣成本 − 一般股票 分年：{yearly(d)}")
    d50 = (net - b50).dropna()
    say(f"  相對 0050 含息 {d50.mean():+.2%}／月（t={t_(d50):+.2f}）")
    dl = (net2 - ewm).dropna()
    say(f"  晚一天進場 − 一般股票 {dl.mean():+.2%}／月（t={t_(dl):+.2f}）")
    db = (bgross - ewm).dropna()
    say(f"  驚喜最低 10%（壞消息）不扣成本 − 一般股票 {db.mean():+.2%}／月（t={t_(db):+.2f}）")
    d25 = d[d.index.year == 2025]
    say(f"  2025 年 扣成本 − 一般股票 {d25.sum():+.1%}（{len(d25)} 個月）")
    tag = "U1" if H == 20 else "U2"
    say(f"→ {tag}（{nm}最高 10% 扣成本 − 一般股票 > 0）：{d.mean():+.3%}／月，t={t_(d):+.2f}（贏 {(d > 0).mean():.0%} 的月） → {verdict(d)}")

a, v, shp, dd = perf(ewm)
say(f"\n  一般股票（等權）年化 {a:+.1%}｜Sharpe {shp:.2f}｜最大回落 {dd:.1%}；0050 含息 年化 {perf(b50)[0]:+.1%}｜Sharpe {perf(b50)[2]:.2f}｜最大回落 {perf(b50)[3]:.1%}")
cnet, _, cnh = port([(ev_rev[ev_rev.dec == 10], 20, 0), (ev_ern[ev_ern.dec == 10], 60, 0)])
dc = (cnet - ewm).dropna()
say(f"  （次要）兩個訊號合併：平均持股 {cnh:.0f} 檔、年化 {perf(cnet)[0]:+.1%}、− 一般股票 {dc.mean():+.2%}／月（t={t_(dc):+.2f}）")

# ---------- 事件層級：累積超額報酬（相對等權）依驚喜 10 組 ----------
AR = np.nan_to_num(R - ew.values[:, None])
CUM = np.vstack([np.zeros((1, len(codes))), np.cumsum(AR, axis=0)])        # CUM[t] = 第 0..t−1 天加總
say("\n事件層級累積超額報酬（相對可買範圍等權；2018 後、含開盤漲停的事件；先每月平均再算 t）：")
for nm, ev in [("月營收驚喜", ev_rev), ("盈餘驚喜", ev_ern)]:
    e = ev[ev.pub >= "2018-01-01"].copy()
    ci, a = e.code.map(cidx).values, e.ent.values
    for k in (5, 20, 60):
        e[f"car{k}"] = CUM[np.minimum(a + k, T), ci] - CUM[a, ci]
    say(f"  [{nm}] 組別：第 1–5 天｜1–20 天｜1–60 天")
    for dcl in (1, 2, 5, 9, 10):
        z = e[e.dec == dcl]
        mm = z.groupby("m")[["car5", "car20", "car60"]].mean()
        say(f"    第 {dcl:>2} 組（{len(z):,} 筆）  " + "｜".join(f"{mm[c].mean():+.2%}（t={t_(mm[c]):+.1f}）" for c in ["car5", "car20", "car60"]))
    if nm == "盈餘驚喜":
        z = e[e.dec == 10]
        say("    最高組依財報季別 1–60 天：" + "｜".join(f"Q{q} {z[z.q == q].car60.mean():+.2%}（{int((z.q == q).sum())} 筆）" for q in (1, 2, 3, 4)))

pd.DataFrame({"rev_top10": res["月營收驚喜"], "ern_top10": res["盈餘驚喜"], "combined": cnet, "ew": ewm, "etf0050": b50}).to_csv(
    OUT + "surprise_event_returns.csv", encoding="utf-8-sig")
open(OUT + "surprise_event_backtest.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
