# -*- coding: utf-8 -*-
"""
G 節結果的事後檢查（看到結果後才加，要如實標明）：
1) TEJ 營收發布日 vs 公開資訊觀測站重大訊息裡的「營收」公告時間（有沒有用到公布前的資訊）
2) 公布前（進場前 5 天）是否已經漲；3) 依市值分大型／其他。
cd D:/專題claude && PYTHONIOENCODING=utf-8 python ai_strategies/surprise_event_checks.py
"""
import glob
import json

import numpy as np
import pandas as pd

src = open("ai_strategies/surprise_event_backtest.py", encoding="utf-8").read().split("\nres = {}")[0]
exec(src)

L.clear()

# ---------- 1) MOPS 重大訊息中的營收公告 ----------
rows = []
for f in glob.glob("mops_material/*.jsonl"):
    for ln in open(f, encoding="utf-8"):
        j = json.loads(ln)
        if "營收" in j["title"]:
            rows.append((j["code"], j["date"], j["time"], j["title"]))
mo = pd.DataFrame(rows, columns=["code", "date", "time", "title"])
mo["date"] = pd.to_datetime(mo.date)
say(f"[MOPS 重大訊息含「營收」] {len(mo):,} 則、{mo.code.nunique()} 檔")
e = ev_rev[["code", "per", "pub"]].copy()
e["pend"] = e.per.dt.end_time.dt.normalize()
m = e.merge(mo, on="code")
m = m[(m.date > m.pend) & (m.date <= m.pend + pd.Timedelta(days=45))]
first = m.sort_values(["date", "time"]).groupby(["code", "per"]).first().reset_index()
first["diff"] = (first.date - first.pub).dt.days
say(f"  對得到的營收事件 {len(first):,} 筆：MOPS 第一則營收公告日 − TEJ 營收發布日")
say("  " + "｜".join(f"{k}: {v:.1%}" for k, v in pd.cut(first["diff"], [-99, -2, -1, 0, 1, 99]).value_counts(normalize=True).sort_index().items()))
same = first[first["diff"] == 0]
say(f"  同一天的公告時間：收盤（13:30）前 {(same.time < '13:30:00').mean():.1%}、收盤後 {(same.time >= '13:30:00').mean():.1%}")
early = first[first["diff"] < 0]
say(f"  MOPS 比 TEJ 早的（我們會晚買，保守）{len(early):,} 筆；MOPS 比 TEJ 晚的（可能偷看）{int((first['diff'] > 0).sum()):,} 筆")

# ---------- 2) 公布前的走勢、3) 大型股 vs 其他 ----------
AR = np.nan_to_num(R - ew.values[:, None])
CUM = np.vstack([np.zeros((1, len(codes))), np.cumsum(AR, axis=0)])
mk = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "mktcap"], parse_dates=["Date"])
mk = mk.pivot(index="Date", columns="stock_code", values="mktcap").reindex(index=dates, columns=codes)
mrank = mk.where(liqok).rank(axis=1, ascending=False).values
for nm, ev, H in [("月營收驚喜", ev_rev, 20), ("盈餘驚喜", ev_ern, 60)]:
    z = ev[(ev.pub >= "2018-01-01") & (ev.dec == 10)].copy()
    ci, a = z.code.map(cidx).values, z.ent.values
    z["pre5"] = CUM[a, ci] - CUM[np.maximum(a - 5, 0), ci]
    z["d0"] = CUM[a, ci] - CUM[a - 1, ci]                      # 公布日（或前一個交易日）開盤 → 進場日開盤
    z["carH"] = CUM[np.minimum(a + H, T), ci] - CUM[a, ci]
    z["big"] = mrank[a - 1, ci] <= 150
    g = lambda s: f"{s.groupby(z.m).mean().mean():+.2%}（t={t_(s.groupby(z.m).mean()):+.1f}）"  # noqa: E731
    say(f"\n[{nm}最高 10%] 進場前 5 天 {g(z.pre5)}｜進場前一天開盤→進場開盤 {g(z.d0)}｜持有 {H} 天 {g(z.carH)}")
    for b, lab in [(True, "市值前 150 大"), (False, "其他")]:
        s = z[z.big == b]
        say(f"  {lab}（{len(s):,} 筆）持有 {H} 天超額 {s.carH.groupby(s.m).mean().mean():+.2%}（t={t_(s.carH.groupby(s.m).mean()):+.1f}）")
    big_ev = ev[ev.code.isin(codes) & (ev.dec == 10)]
    ok_big = mrank[big_ev.ent.values - 1, big_ev.code.map(cidx).values] <= 150
    netb, _, nb = port([(big_ev[ok_big], H, 0)])
    ewb = pd.Series(np.nanmean(np.where(mrank <= 150, R, np.nan), axis=1), index=dates).loc["2018-01-01":"2025-12-31"]
    ewbm = (1 + ewb).groupby(pd.PeriodIndex(ewb.index, freq="M")).prod() - 1
    d = (netb - ewbm).dropna()
    d50 = (netb - b50).dropna()
    say(f"  只買市值前 150 大的最高 10%：平均持股 {nb:.0f} 檔、年化 {perf(netb)[0]:+.1%}｜扣成本 − 前 150 大等權 {d.mean():+.2%}／月（t={t_(d):+.2f}）｜− 0050 {d50.mean():+.2%}／月（t={t_(d50):+.2f}）")
open(OUT + "surprise_event_checks.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
