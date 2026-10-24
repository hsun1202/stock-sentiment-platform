# -*- coding: utf-8 -*-
"""
平台用「月營收驚喜」名單：規則同 PREREG G／H 節（已用 2018–2025 回測與 2026 樣本外驗證）。
  SUR＝(d_t − 前 24 個月 d 的平均) ÷ 前 24 個月 d 的標準差，d_t＝單月營收 − 去年同月（至少 12 個月）
  公布日之後第一個交易日開盤進場、持有 20 個交易日；最高 10%＝「營收驚喜」、最低 10%＝「營收壞消息（避開）」
  門檻：前 12 個月、上市、近 60 日成交金額中位數不在最低 20% 的事件的第 10／90 百分位
營收歷史：TEJ 匯出（到 2026-09，含發布日）＋公開資訊觀測站每日快照（mops_revenue.py，之後的月份）。
股價：證交所公開資料 platform_feed/data/prices.csv（由 ai_strategies/twse_2026_prices.csv 起頭，run_feed.py 每天補）。
輸出 platform_feed/out/revenue_signals.json（給前端）、revenue_signals.csv。
cd D:/專題claude && python platform_feed/revenue_signals.py
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(BASE, "out")
H = 20
DISCLAIMER = "學術研究用。名單依歷史統計規則產生（2018–2025 回測與 2026 樣本外贏一般股票，但沒有贏 0050），不保證未來報酬，不構成投資建議。"


def to_date(s):
    return pd.to_datetime(s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace("/", ""), format="%Y%m%d", errors="coerce")


def build(log=print):
    # ---------- 營收歷史 ----------
    rv = pd.read_csv(os.path.join(ROOT, "tej_raw/rev/20261022103241.csv"), encoding="utf-16", sep="\t", on_bad_lines="skip",
                     low_memory=False, na_values=["."], usecols=["證券代碼", "年月", "營收發布日", "以合併為主單月營收(千元)"])
    rv["code"] = rv["證券代碼"].astype(str).str.split().str[0]
    rv = rv[rv.code.str.match(r"^[1-9]\d{3}$")]
    T = pd.DataFrame({"code": rv.code, "per": pd.PeriodIndex(pd.to_datetime(rv["年月"].astype(str) + "01", format="%Y%m%d"), freq="M"),
                      "R": rv["以合併為主單月營收(千元)"], "pub": to_date(rv["營收發布日"])}).drop_duplicates(["code", "per"], keep="last")
    tej_end = T.per.max()
    P = pd.read_csv(os.path.join(BASE, "data/revenue_public.csv"), dtype={"code": str})
    P = pd.DataFrame({"code": P.code, "name": P.name, "per": pd.PeriodIndex(P.per, freq="M"), "R": P.R,
                      "pub": pd.to_datetime(P.first_seen, errors="coerce")})
    names = P.drop_duplicates("code", keep="last").set_index("code").name
    newP = P[P.per > tej_end]
    A = pd.concat([T, newP[["code", "per", "R", "pub"]]], ignore_index=True).drop_duplicates(["code", "per"], keep="first")
    ev = []
    for c, g in A.groupby("code"):
        s = g.set_index("per").R
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
        d = s - s.shift(12)
        mu, sd = d.shift(1).rolling(24, min_periods=12).mean(), d.shift(1).rolling(24, min_periods=12).std()
        gg = g.set_index("per")
        ev.append(pd.DataFrame({"code": c, "pub": gg.pub, "R": gg.R, "yoy": (s / s.shift(12) - 1).reindex(gg.index),
                                "sur": ((d - mu) / sd.where(sd > 0)).clip(-10, 10).reindex(gg.index)}))
    ev = pd.concat(ev).reset_index().rename(columns={"index": "per"}).dropna(subset=["sur", "pub"])

    # ---------- 股價：交易日、上市、流動性 ----------
    px = pd.read_csv(os.path.join(BASE, "data/prices.csv"), dtype={"code": str})       # run_feed.py 每天補
    px = px[px.code.str.match(r"^[1-9]\d{3}$")].drop_duplicates(["code", "date"], keep="last")
    px["date"] = pd.to_datetime(px.date)
    V = px.pivot(index="date", columns="code", values="value").sort_index()
    dates = V.index
    liq = V.where(V > 0).rolling(60, min_periods=20).median()
    present = V.notna()
    liqok = (liq.where(present).rank(axis=1, pct=True) > 0.2) & present
    asof = dates[-1]
    future = pd.bdate_range(asof + pd.Timedelta(days=1), periods=80)        # 未來交易日先用平日估（國定假日會差幾天）
    cal = dates.append(future)

    ev = ev[ev.pub >= dates[0] + pd.Timedelta(days=30)].copy()             # 流動性要 20 天以上的股價
    ev["ent_i"] = cal.searchsorted(ev.pub.values, side="right")
    li = dates.searchsorted(ev.pub.values, side="right") - 1
    ok = np.array([bool(liqok.iat[i, liqok.columns.get_loc(c)]) if c in liqok.columns and i >= 0 else False for i, c in zip(li, ev.code)])
    ev = ev[ok].copy()
    ev["entry"] = cal[ev.ent_i.values]
    ev["exit"] = cal[np.minimum(ev.ent_i.values + H, len(cal) - 1)]
    ev["m"] = ev.pub.dt.to_period("M")

    # ---------- 門檻（前 12 個月）與分組 ----------
    def bps(m):
        past = ev.sur[(ev.m < m) & (ev.m >= m - 12)]
        return np.quantile(past, np.linspace(0.1, 0.9, 9)) if len(past) >= 2000 else None
    B = {m: bps(m) for m in ev.m.unique()}
    ev = ev[ev.m.map(lambda m: B[m] is not None)].copy()
    ev["dec"] = [int(np.searchsorted(B[m], x, side="right")) + 1 for m, x in zip(ev.m, ev.sur)]
    ev["label"] = np.select([ev.dec == 10, ev.dec == 1], ["營收驚喜", "營收壞消息（避開）"], "")
    ev["active"] = (ev.entry <= asof + pd.Timedelta(days=1)) & (ev.exit > asof)
    ev["est_exit"] = ev.exit > dates[-1]
    ev["name"] = ev.code.map(names)

    cur_m = max(B)
    recent = ev[ev.pub >= asof - pd.Timedelta(days=45)].sort_values(["dec", "sur"], ascending=False)
    col = ["code", "name", "per", "pub", "entry", "exit", "est_exit", "R", "yoy", "sur", "dec", "label", "active"]
    os.makedirs(OUT, exist_ok=True)
    recent[col].to_csv(os.path.join(OUT, "revenue_signals.csv"), index=False, encoding="utf-8-sig")
    rec = lambda z: [{"code": r.code, "name": r.name, "revenue_month": str(r.per), "announce_date": f"{r.pub:%Y-%m-%d}",  # noqa: E731
                      "entry_date": f"{r.entry:%Y-%m-%d}", "exit_date": f"{r.exit:%Y-%m-%d}", "exit_is_estimate": bool(r.est_exit),
                      "revenue_k": float(r.R), "yoy": None if pd.isna(r.yoy) else round(float(r.yoy), 4), "surprise": round(float(r.sur), 3),
                      "decile": int(r.dec), "label": r.label} for r in z.itertuples()]
    out = {"asof": f"{asof:%Y-%m-%d}", "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
           "rule": "公布日後第一個交易日開盤進場、持有 20 個交易日；最高 10%＝營收驚喜、最低 10%＝營收壞消息（上市股）",
           "thresholds": {"month": str(cur_m), "p10": round(float(B[cur_m][0]), 3), "p90": round(float(B[cur_m][8]), 3)},
           "surprise_active": rec(recent[(recent.dec == 10) & recent.active]),
           "bad_news_active": rec(recent[(recent.dec == 1) & recent.active].sort_values("sur")),
           "all_recent": rec(recent), "disclaimer": DISCLAIMER}
    json.dump(out, open(os.path.join(OUT, "revenue_signals.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[營收驚喜名單 {asof:%Y-%m-%d}] 近 45 天公布 {len(recent)} 家；持有中：驚喜 {len(out['surprise_active'])} 檔、壞消息 {len(out['bad_news_active'])} 檔；"
        f"{cur_m} 門檻 p90={out['thresholds']['p90']}、p10={out['thresholds']['p10']}")
    return ev


if __name__ == "__main__":
    build()
