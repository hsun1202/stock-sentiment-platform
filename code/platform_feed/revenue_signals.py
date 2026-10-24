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
import os

import pandas as pd

from feed_common import BASE, ROOT, classify, dump, load_market, to_date

H = 20


def build(log=print):
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
    A = pd.concat([T, P[P.per > tej_end][["code", "per", "R", "pub"]]], ignore_index=True).drop_duplicates(["code", "per"], keep="first")
    ev = []
    for c, g in A.groupby("code"):
        s = g.set_index("per").R
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
        d = s - s.shift(12)
        mu, sd = d.shift(1).rolling(24, min_periods=12).mean(), d.shift(1).rolling(24, min_periods=12).std()
        gg = g.set_index("per")
        ev.append(pd.DataFrame({"code": c, "pub": gg.pub, "R": gg.R, "yoy": (s / s.shift(12) - 1).reindex(gg.index),
                                "x": ((d - mu) / sd.where(sd > 0)).clip(-10, 10).reindex(gg.index)}))
    ev = pd.concat(ev).reset_index().rename(columns={"index": "per"}).dropna(subset=["x", "pub"])

    mk = load_market()
    ev, B = classify(ev, mk, H, 2000, ["營收驚喜", "營收壞消息（避開）"])
    ev["name"] = ev.code.map(names)
    dump("revenue_signals", ev, B, mk, 45, "公布日後第一個交易日開盤進場、持有 20 個交易日；最高 10%＝營收驚喜、最低 10%＝營收壞消息（上市股）",
         ["R", "yoy"], lambda r: {"revenue_k": float(r.R), "yoy": None if pd.isna(r.yoy) else round(float(r.yoy), 4)}, log, "營收驚喜名單")
    return ev


if __name__ == "__main__":
    build()
