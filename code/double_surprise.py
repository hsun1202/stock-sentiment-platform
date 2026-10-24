# -*- coding: utf-8 -*-
"""
J 節：雙重驚喜（季報 SUE 最高 10% 且 季營收驚喜 SURq 最高 10%）vs 只有盈餘驚喜。事先寫定見 PREREG_ai_strategies.md J 節。
2018–2025 用 TEJ 股價（同 G 節）；2026 用 H 節的證交所公開資料。
cd D:/專題claude && PYTHONIOENCODING=utf-8 python ai_strategies/double_surprise.py
"""
import numpy as np
import pandas as pd

osrc = open("ai_strategies/oos2026_surprise.py", encoding="utf-8").read().split("def portP(")[0]
exec(osrc)                                  # TEJ 面板與 tej_ern（G 節分組）、公開資料面板與 pub_ern、rv（TEJ 月營收）
bsrc = open("ai_strategies/surprise_event_backtest.py", encoding="utf-8").read()
exec("def port(" + bsrc.split("def port(")[1].split("\newm = ")[0])      # G 節的組合函式（TEJ 面板）
exec("def portP(" + open("ai_strategies/oos2026_surprise.py", encoding="utf-8").read().split("def portP(")[1].split("\ndef monthly(")[0])
L.clear()

# ---------- 季營收驚喜 SURq（第三個月營收公布後才知道）----------
q = rv.dropna(subset=["R"]).copy()
q["qper"] = q.per.dt.asfreq("Q", how="end") if hasattr(q.per.dt, "asfreq") else q.per.apply(lambda p: p.asfreq("Q"))
grp = q.groupby(["code", "qper"])
Q = pd.DataFrame({"Rq": grp.R.sum(), "n": grp.R.size(),
                  "pub3": grp.apply(lambda g: g.loc[g.per.dt.month.isin([3, 6, 9, 12]), "pub"].max())}).reset_index()
Q = Q[Q.n == 3]
sq = []
for c, g in Q.groupby("code"):
    s = g.set_index("qper").Rq
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="Q"))
    e = s - s.shift(4)
    mu, sd = e.shift(1).rolling(8, min_periods=4).mean(), e.shift(1).rolling(8, min_periods=4).std()
    gg = g.set_index("qper")
    sq.append(pd.DataFrame({"code": c, "per": gg.index, "surq": ((e - mu) / sd.where(sd > 0)).clip(-10, 10).reindex(gg.index).values,
                            "pub3": gg.pub3.values}))
SQ = pd.concat(sq, ignore_index=True)


def attach(ev):
    ev = ev.merge(SQ, on=["code", "per"], how="left")
    ev.loc[~(ev.pub3 <= ev.pub), "surq"] = np.nan                        # 第三個月營收要在季報公布前就已公布
    return ev


def surq_groups(ev, hist):
    """依前 12 個月（hist 內）符合條件事件的 SURq 分位數，標出最高／最低 10%"""
    th = {}
    for m in ev.m.unique():
        past = hist.surq[(hist.m < m) & (hist.m >= m - 12)].dropna()
        th[m] = (np.quantile(past, 0.1), np.quantile(past, 0.9)) if len(past) >= 200 else (np.nan, np.nan)
    lo = ev.m.map(lambda m: th[m][0])
    hi = ev.m.map(lambda m: th[m][1])
    ev["sq_top"], ev["sq_bot"] = ev.surq > hi, ev.surq < lo
    return ev


def car_cmp(ev, CUMm, Tn, idx, H, a_mask, b_mask, lab_a, lab_b):
    ci, a = ev.code.map(idx).values, ev.ent.values
    ev = ev.assign(car=CUMm[np.minimum(a + H, Tn), ci] - CUMm[a, ci])
    A = ev[a_mask(ev)].groupby("m").car.mean()
    B = ev[b_mask(ev)].groupby("m").car.mean()
    d = (A - B).dropna()
    say(f"  {lab_a} {int(a_mask(ev).sum()):,} 筆 第 1–{H} 天 {A.mean():+.2%}｜{lab_b} {int(b_mask(ev).sum()):,} 筆 {B.mean():+.2%}｜"
        f"差 {d.mean():+.2%}（t={t_(d):+.2f}，{len(d)} 個公布月、{int((d > 0).sum())} 個為正）")
    return d


# ================= 2018–2025（TEJ 股價）=================
E = attach(tej_ern)
E = surq_groups(E, E)
E8 = E[(E.pub >= "2018-01-01") & (E.pub <= "2025-12-31")]
AR = np.nan_to_num(R - ew.values[:, None])
CUM = np.vstack([np.zeros((1, len(codes))), np.cumsum(AR, axis=0)])
say("===== J. 雙重驚喜，2018–2025（TEJ 股價、相對可買範圍等權）=====")
say(f"  季報事件 {len(E8):,} 筆；有 SURq（第三個月營收已在季報前公布）{E8.surq.notna().mean():.1%}；盈餘最高 10% 中雙重驚喜占 "
    f"{(E8[E8.dec == 10].sq_top).mean():.0%}")
dD = car_cmp(E8, CUM, T, cidx, 60, lambda z: (z.dec == 10) & z.sq_top, lambda z: (z.dec == 10) & z.surq.notna() & ~z.sq_top,
             "雙重驚喜", "只有盈餘驚喜")
say(f"→ D1（雙重驚喜 − 只有盈餘驚喜，第 1–60 天累積超額 > 0）：{dD.mean():+.2%}，t={t_(dD):+.2f} → {'通過' if dD.mean() > 0 and t_(dD) > 1.96 else '未通過'}")
car_cmp(E8, CUM, T, cidx, 60, lambda z: (z.dec == 1) & z.sq_bot, lambda z: (z.dec == 1) & z.surq.notna() & ~z.sq_bot, "（次要）雙重壞消息", "只有盈餘壞消息")
car_cmp(E8, CUM, T, cidx, 20, lambda z: (z.dec == 10) & z.sq_top, lambda z: (z.dec == 10) & z.surq.notna() & ~z.sq_top, "（次要）雙重驚喜", "只有盈餘驚喜")

dbl = E[(E.dec == 10) & E.sq_top]
netD, _, nD = port([(dbl, 60, 0)])
netS, _, nS = port([(E[E.dec == 10], 60, 0)])
ewm8 = (1 + ew.loc["2018-01-01":"2025-12-31"]).groupby(pd.PeriodIndex(ew.loc["2018-01-01":"2025-12-31"].index, freq="M")).prod() - 1
d1, d2 = (netD - ewm8).dropna(), (netD - netS).dropna()
say(f"  （次要）組合：雙重驚喜平均 {nD:.0f} 檔、年化 {perf(netD)[0]:+.1%}｜− 一般股票 {d1.mean():+.2%}／月（t={t_(d1):+.2f}）｜"
    f"− G 節盈餘驚喜組合（{nS:.0f} 檔）{d2.mean():+.2%}／月（t={t_(d2):+.2f}）")

# ================= 2026（公開資料）=================
say("\n===== 2026 年（H 節公開資料；只報數字）=====")
P26 = attach(pub_ern)
hist = pd.concat([E[E.m < pd.Period("2025-10", "M")][["m", "surq"]], P26[["m", "surq"]]])
P26 = surq_groups(P26, hist)
P26e = P26[P26.pub >= "2026-01-01"]
ARp = np.nan_to_num(RP - ewP.values[:, None])
CUMp = np.vstack([np.zeros((1, len(pcodes))), np.cumsum(ARp, axis=0)])
car_cmp(P26e, CUMp, TP, pidx, 20, lambda z: (z.dec == 10) & z.sq_top, lambda z: (z.dec == 10) & z.surq.notna() & ~z.sq_top,
        "雙重驚喜", "只有盈餘驚喜")
car_cmp(P26e, CUMp, TP, pidx, 60, lambda z: (z.dec == 10) & z.sq_top, lambda z: (z.dec == 10) & z.surq.notna() & ~z.sq_top,
        "雙重驚喜", "只有盈餘驚喜（60 天，超過 10-22 只算到 10-22）")
open(OUT + "double_surprise.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
