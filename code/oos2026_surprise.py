# -*- coding: utf-8 -*-
"""
H 節：G 節（月營收驚喜、盈餘驚喜）的 2026 年樣本外。事先寫定見 PREREG_ai_strategies.md H 節。
2025 以前的事件（決定門檻用）沿用 TEJ 股價；2025-10 起的事件與 2026 年的報酬改用證交所公開資料（fetch_twse_2026.py 抓的）。
cd D:/專題claude && PYTHONIOENCODING=utf-8 python ai_strategies/oos2026_surprise.py
"""
import numpy as np
import pandas as pd

src = open("ai_strategies/surprise_event_backtest.py", encoding="utf-8").read().split("ev_rev, ev_ern = prep(")[0]
exec(src)                                   # TEJ 面板（dates/op/cl/R/ew/liqok）、原始事件 ev_rev/ev_ern、prep()、say、t_、perf、COST
raw_rev, raw_ern = ev_rev.copy(), ev_ern.copy()
tej_rev, tej_ern = prep(raw_rev, "月營收（TEJ 股價）"), prep(raw_ern, "季報盈餘（TEJ 股價）")
L.clear()
TEJ_R, TEJ_dates, TEJ_codes = R, dates, codes

# ---------- 公開資料面板 ----------
px = pd.read_csv("ai_strategies/twse_2026_prices.csv", dtype={"code": str}, parse_dates=["date"])
ex = pd.read_csv("ai_strategies/twse_2026_exright.csv", dtype={"code": str}, parse_dates=["date"])
O = px.pivot(index="date", columns="code", values="open").sort_index()
C = px.pivot(index="date", columns="code", values="close").reindex(O.index)
V = px.pivot(index="date", columns="code", values="value").reindex(O.index)
fac = (ex.prev_close / ex.ref_price).where((ex.ref_price > 0) & (ex.prev_close > 0))
F = pd.DataFrame({"date": ex.date, "code": ex.code, "f": fac}).dropna().pivot_table(index="date", columns="code", values="f", aggfunc="last")
F = F.reindex(index=O.index, columns=O.columns).fillna(1.0)
O1 = O.shift(-1) * F.shift(-1)                                      # 隔天開盤（還原當天除權息）
rP = O1 / O - 1
gapn = (O / C.shift(1) - 1).where(F == 1.0)                          # 非除權息日的開盤跳空
rP = rP.where(~(gapn.shift(-1).abs() > 0.105))                       # 減資、停牌恢復等不算
pend = O.index[-1]
for c, d in O.apply(lambda s: s.last_valid_index()).items():
    if d is not None and d < pend - pd.Timedelta(days=10):
        rP.loc[d, c] = C.loc[d, c] / O.loc[d, c] * 0.5 - 1

# 0050（同樣還原除權息）
r50 = rP["0050"].copy()
stk = list(O.columns[O.columns.str.match(r"^[1-9]\d{3}$")])
O, C, F, rP = O[stk], C[stk], F[stk], rP[stk]
liqP = V[stk].where(V[stk] > 0).rolling(60, min_periods=20).median()
present = O.notna() | C.notna()
liqokP = (liqP.where(present).rank(axis=1, pct=True) > 0.2) & present
pdates, pcodes = O.index, O.columns
TP = len(pdates)
RP = rP.values
ewP = pd.Series(np.nanmean(np.where(liqokP.values, RP, np.nan), axis=1), index=pdates)
pidx = {c: i for i, c in enumerate(pcodes)}
say(f"[公開資料] {pdates[0].date()}～{pend.date()}，{TP} 個交易日、{len(pcodes)} 檔四碼股；除權息 {len(ex)} 筆；每天可買約 {liqokP.sum(axis=1).loc['2026'].mean():.0f} 檔")

# ---------- 驗證：2025-09～12 與 TEJ 重疊 ----------
ov = [d for d in pdates if d in set(TEJ_dates) and d < TEJ_dates[-1]]
cc = [c for c in pcodes if c in set(TEJ_codes)]
a = pd.DataFrame(TEJ_R, index=TEJ_dates, columns=TEJ_codes).loc[ov, cc].stack()
b = rP.loc[ov, cc].stack()
j = pd.concat([a.rename("tej"), b.rename("pub")], axis=1).dropna()
say(f"[驗證 2025-09～12 重疊] {len(j):,} 個股票日：開盤到開盤報酬 相關 {j.tej.corr(j.pub):.4f}｜差異 < 0.1% 的比例 {((j.tej - j.pub).abs() < 0.001).mean():.1%}｜差異中位數 {(j.tej - j.pub).abs().median():.5f}")
ewT = ew.loc[ov]
say(f"  等權基準每日 相關 {ewT.corr(ewP.loc[ov]):.4f}；10～12 月累積 TEJ {(1 + ewT.loc['2025-10':]).prod() - 1:+.2%} vs 公開 {(1 + ewP.loc[ov].loc['2025-10':]).prod() - 1:+.2%}")


# ---------- 2025-10 起的事件改在公開資料面板上判斷 ----------
def prep_pub(ev, past, nm):
    ev = ev[(ev.pub >= "2025-10-01") & ev.code.isin(pidx)].copy()
    ev["ent"] = pdates.searchsorted(ev.pub.values, side="right")
    ev = ev[(ev.ent >= 1) & (ev.ent < TP - 1)]
    ci, ent = ev.code.map(pidx).values, ev.ent.values
    ok = liqokP.values[ent - 1, ci]
    gap = O.values[ent, ci] * F.values[ent, ci] / C.values[ent - 1, ci] - 1
    ev["gap_up"] = gap >= 0.095
    ev = ev[ok & np.isfinite(gap)].copy()
    ev["m"] = ev.pub.dt.to_period("M")
    hist = pd.concat([past.loc[past.m < pd.Period("2025-10", "M"), ["m", "x"]], ev[["m", "x"]]])
    bps = {m: np.quantile(hist.x[(hist.m < m) & (hist.m >= m - 12)], np.linspace(0.1, 0.9, 9)) for m in ev.m.unique()}
    ev["dec"] = [int(np.searchsorted(bps[m], x, side="right")) + 1 for m, x in zip(ev.m, ev.x)]
    say(f"[{nm}] 2025-10 起事件 {len(ev):,} 筆（2026 年 {int((ev.pub >= '2026-01-01').sum()):,}）；最高 10% 門檻 2026-01 {bps[pd.Period('2026-01', 'M')][8]:.2f}；開盤買不到 {ev.gap_up.mean():.1%}")
    return ev


pub_rev, pub_ern = prep_pub(raw_rev, tej_rev, "月營收"), prep_pub(raw_ern, tej_ern, "季報盈餘")


def portP(sets):
    cnt = np.zeros((TP + 1, len(pcodes)), dtype=np.int32)
    for ev, H in sets:
        e = ev[~ev.gap_up]
        a, ci = e.ent.values, e.code.map(pidx).values
        np.add.at(cnt, (a, ci), 1)
        np.add.at(cnt, (np.minimum(a + H, TP), ci), -1)
    held = np.cumsum(cnt, axis=0)[:TP] > 0
    new = held & ~np.vstack([np.zeros((1, len(pcodes)), bool), held[:-1]])
    Rh = np.where(held, RP, np.nan)
    n = np.isfinite(Rh).sum(axis=1)
    gross = np.where(n > 0, np.nansum(Rh, axis=1) / np.maximum(n, 1), 0.0)
    net = pd.Series(gross - np.where(n > 0, new.sum(axis=1) / np.maximum(n, 1) * COST, 0.0), index=pdates)
    return net, pd.Series(gross, index=pdates), pd.Series(n, index=pdates)


def monthly(s):
    s = s.loc["2026-01-01":].dropna()
    return (1 + s).groupby(pd.PeriodIndex(s.index, freq="M")).prod() - 1


ewm, b50m = monthly(ewP), monthly(r50)
full = ewm.index[ewm.index <= pd.Period("2026-09", "M")]
say("\n===== H. 2026 年樣本外（公開資料、扣成本）：主要檢定用 2026-01～09，10 月只到 22 日另外報 =====")
say(f"  一般股票（等權）1～9 月累積 {(1 + ewm[full]).prod() - 1:+.1%}；0050 含息 {(1 + b50m[full]).prod() - 1:+.1%}")
out = {}
for tag, nm, ev, H in [("O1", "月營收驚喜", pub_rev, 20), ("O2", "盈餘驚喜", pub_ern, 60)]:
    net, gross, n = portP([(ev[ev.dec == 10], H)])
    bnet, bgross, _ = portP([(ev[ev.dec == 1], H)])
    nm_, gm_ = monthly(net), monthly(gross)
    out[nm] = nm_
    d = (nm_ - ewm)[full]
    say(f"\n[{nm}最高 10%，持有 {H} 天] 2026 平均持股 {n.loc['2026'].mean():.0f} 檔｜1～9 月累積 {(1 + nm_[full]).prod() - 1:+.1%}（一般股票 {(1 + ewm[full]).prod() - 1:+.1%}）")
    say("  每月 扣成本 − 一般股票：" + " ".join(f"{p.month}月 {v:+.1%}" for p, v in (nm_ - ewm).items()))
    say(f"  不扣成本 − 一般股票 {(gm_ - ewm)[full].mean():+.2%}／月；相對 0050 {(nm_ - b50m)[full].mean():+.2%}／月（t={t_((nm_ - b50m)[full]):+.2f}）")
    db = (monthly(bgross) - ewm)[full]
    say(f"  驚喜最低 10%（壞消息）不扣成本 − 一般股票 {db.mean():+.2%}／月（t={t_(db):+.2f}，{int((db < 0).sum())}/{len(db)} 個月為負）")
    say(f"→ {tag}（{nm}最高 10% 扣成本 − 一般股票 > 0，2026-01～09）：{d.mean():+.3%}／月，t={t_(d):+.2f}，{int((d > 0).sum())}/{len(d)} 個月為正"
        f" → {'通過' if d.mean() > 0 and t_(d) > 1.86 else ('方向一致（未達顯著）' if d.mean() > 0 and (d > 0).sum() > len(d) / 2 else '未通過')}")
cnet, _, _ = portP([(pub_rev[pub_rev.dec == 10], 20), (pub_ern[pub_ern.dec == 10], 60)])
dc = (monthly(cnet) - ewm)[full]
say(f"\n  （次要）兩個合併：{dc.mean():+.2%}／月（t={t_(dc):+.2f}，{int((dc > 0).sum())}/{len(dc)} 個月為正）")

# ---------- 事件層級（2026 年公布、相對等權、先每月平均）----------
AR = np.nan_to_num(RP - ewP.values[:, None])
CUM = np.vstack([np.zeros((1, len(pcodes))), np.cumsum(AR, axis=0)])
say("\n事件層級累積超額報酬（2026 年公布、含開盤漲停事件；持有期超過 10-22 的只算到 10-22）：")
for nm, ev in [("月營收驚喜", pub_rev), ("盈餘驚喜", pub_ern)]:
    e = ev[ev.pub >= "2026-01-01"].copy()
    ci, a = e.code.map(pidx).values, e.ent.values
    for k in (5, 20):
        e[f"car{k}"] = CUM[np.minimum(a + k, TP), ci] - CUM[a, ci]
    for dcl in (1, 10):
        z = e[e.dec == dcl]
        mm = z.groupby("m")[["car5", "car20"]].mean()
        say(f"  [{nm}] 第 {dcl:>2} 組（{len(z):,} 筆、{len(mm)} 個月）第 1–5 天 {mm.car5.mean():+.2%}（t={t_(mm.car5):+.1f}）｜1–20 天 {mm.car20.mean():+.2%}（t={t_(mm.car20):+.1f}）")

pd.DataFrame({"rev_top10": out["月營收驚喜"], "ern_top10": out["盈餘驚喜"], "ew": ewm, "etf0050": b50m}).to_csv(OUT + "oos2026_surprise_returns.csv", encoding="utf-8-sig")
open(OUT + "oos2026_surprise.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
