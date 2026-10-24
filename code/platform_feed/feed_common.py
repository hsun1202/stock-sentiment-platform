# -*- coding: utf-8 -*-
"""月營收驚喜、盈餘驚喜兩份名單共用的部分：股價面板（交易日、上市、流動性）、門檻與分組、輸出 JSON。"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(BASE, "out")
DISCLAIMER = "學術研究用。名單依歷史統計規則產生（2018–2025 回測與 2026 樣本外贏一般股票，但沒有贏 0050），不保證未來報酬，不構成投資建議。"


def to_date(s):
    return pd.to_datetime(s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace("/", ""), format="%Y%m%d", errors="coerce")


def load_market():
    """證交所公開行情（run_feed.py 每天補）→ 交易日、可買（上市且近 60 日成交金額中位數不在最低 20%）、含未來估計的交易日曆。"""
    px = pd.read_csv(os.path.join(BASE, "data/prices.csv"), dtype={"code": str})
    px = px[px.code.str.match(r"^[1-9]\d{3}$")].drop_duplicates(["code", "date"], keep="last")
    px["date"] = pd.to_datetime(px.date)
    V = px.pivot(index="date", columns="code", values="value").sort_index()
    dates = V.index
    liq = V.where(V > 0).rolling(60, min_periods=20).median()
    present = V.notna()
    liqok = (liq.where(present).rank(axis=1, pct=True) > 0.2) & present
    future = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=120)    # 未來交易日先用平日估（國定假日會差幾天）
    return {"dates": dates, "liqok": liqok, "cal": dates.append(future), "asof": dates[-1]}


def classify(ev, mk, H, min_n, labels):
    """ev 要有 code、pub、x（驚喜分數）。套用可買條件、前 12 個月門檻分 10 組、進出場日、是否在持有期內。"""
    dates, liqok, cal, asof = mk["dates"], mk["liqok"], mk["cal"], mk["asof"]
    ev = ev[ev.pub >= dates[0] + pd.Timedelta(days=30)].copy()                 # 流動性要 20 天以上的股價
    ev["ent_i"] = cal.searchsorted(ev.pub.values, side="right")
    li = dates.searchsorted(ev.pub.values, side="right") - 1
    ok = np.array([bool(liqok.iat[i, liqok.columns.get_loc(c)]) if c in liqok.columns and i >= 0 else False for i, c in zip(li, ev.code)])
    ev = ev[ok].copy()
    ev["entry"] = cal[ev.ent_i.values]
    ev["exit"] = cal[np.minimum(ev.ent_i.values + H, len(cal) - 1)]
    ev["m"] = ev.pub.dt.to_period("M")

    def bps(m):
        past = ev.x[(ev.m < m) & (ev.m >= m - 12)]
        return np.quantile(past, np.linspace(0.1, 0.9, 9)) if len(past) >= min_n else None
    B = {m: bps(m) for m in ev.m.unique()}
    ev = ev[ev.m.map(lambda m: B[m] is not None)].copy()
    ev["dec"] = [int(np.searchsorted(B[m], x, side="right")) + 1 for m, x in zip(ev.m, ev.x)]
    ev["label"] = np.select([ev.dec == 10, ev.dec == 1], labels, "")
    ev["active"] = (ev.entry <= asof + pd.Timedelta(days=1)) & (ev.exit > asof)
    ev["est_exit"] = ev.exit > dates[-1]
    return ev, B


def dump(name, ev, B, mk, recent_days, rule, extra_cols, rec_extra, log, title):
    """輸出 out/{name}.json（前端用）與 .csv。rec_extra(r) 回傳每筆額外欄位。"""
    asof = mk["asof"]
    cur_m = max(B)
    recent = ev[ev.pub >= asof - pd.Timedelta(days=recent_days)].sort_values(["dec", "x"], ascending=False)
    os.makedirs(OUT, exist_ok=True)
    recent[["code", "name", "per", "pub", "entry", "exit", "est_exit"] + extra_cols + ["x", "dec", "label", "active"]].rename(
        columns={"x": "surprise"}).to_csv(os.path.join(OUT, f"{name}.csv"), index=False, encoding="utf-8-sig")

    def rec(z):
        return [{"code": r.code, "name": r.name, "period": str(r.per), "announce_date": f"{r.pub:%Y-%m-%d}",
                 "entry_date": f"{r.entry:%Y-%m-%d}", "exit_date": f"{r.exit:%Y-%m-%d}", "exit_is_estimate": bool(r.est_exit),
                 **rec_extra(r), "surprise": round(float(r.x), 3), "decile": int(r.dec), "label": r.label} for r in z.itertuples()]
    out = {"asof": f"{asof:%Y-%m-%d}", "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "rule": rule,
           "thresholds": {"month": str(cur_m), "p10": round(float(B[cur_m][0]), 3), "p90": round(float(B[cur_m][8]), 3)},
           "surprise_active": rec(recent[(recent.dec == 10) & recent.active]),
           "bad_news_active": rec(recent[(recent.dec == 1) & recent.active].sort_values("x")),
           "all_recent": rec(recent), "disclaimer": DISCLAIMER}
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[{title} {asof:%Y-%m-%d}] 近 {recent_days} 天公布 {len(recent)} 家；持有中：驚喜 {len(out['surprise_active'])} 檔、"
        f"壞消息 {len(out['bad_news_active'])} 檔；{cur_m} 門檻 p90={out['thresholds']['p90']}、p10={out['thresholds']['p10']}")
    return out
