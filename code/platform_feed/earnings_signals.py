# -*- coding: utf-8 -*-
"""
平台用「盈餘驚喜」名單：規則同 PREREG G／H 節。
  SUE＝(e_q − 前 8 季 e 的平均) ÷ 前 8 季 e 的標準差，e_q＝單季歸屬母公司淨利 − 去年同季（至少 4 季）
  公布日之後第一個交易日開盤進場、持有 60 個交易日；最高 10%＝「盈餘驚喜」、最低 10%＝「盈餘壞消息（避開）」
  門檻：前 12 個月、上市、近 60 日成交金額中位數不在最低 20% 的事件的第 10／90 百分位
季報歷史：TEJ 匯出（單季，含財報發布日）＋公開資訊觀測站每日快照（mops_earnings.py；年初累計 − 同年前幾季 ＝ 單季）。
輸出 platform_feed/out/earnings_signals.json（給前端）、earnings_signals.csv。
cd D:/專題claude && python platform_feed/earnings_signals.py
"""
import os

import numpy as np
import pandas as pd

from feed_common import BASE, ROOT, classify, dump, load_market, to_date

H = 60


def build(log=print):
    fn = pd.read_csv(os.path.join(ROOT, "tej_raw/fin/20261022104526.csv"), encoding="utf-16", sep="\t", low_memory=False, na_values=["."],
                     usecols=["證券代碼", "年月", "財報發布日", "歸屬母公司淨利（損）"])
    fn["code"] = fn["證券代碼"].astype(str).str.split().str[0]
    fn = fn[fn.code.str.match(r"^[1-9]\d{3}$")].copy()
    qend = pd.to_datetime(fn["年月"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
    T = pd.DataFrame({"code": fn.code, "per": pd.PeriodIndex(qend, freq="Q"), "N": fn["歸屬母公司淨利（損）"],
                      "pub": to_date(fn["財報發布日"]).where(lambda x: x >= qend)}).drop_duplicates(["code", "per"], keep="last")
    have = set(zip(T.code, T.per))
    P = pd.read_csv(os.path.join(BASE, "data/earnings_public.csv"), dtype={"code": str})
    P["per"] = [pd.Period(year=int(y), quarter=int(s), freq="Q") for y, s in zip(P.year, P.season)]
    names = P.drop_duplicates("code", keep="last").set_index("code").name
    single = T.set_index(["code", "per"]).N.to_dict()
    add = []
    for r in P.sort_values("per").itertuples():                                 # 年初累計 → 單季（減掉同年前幾季）
        if (r.code, r.per) in have:
            continue
        prev = [single.get((r.code, pd.Period(year=r.per.year, quarter=k, freq="Q")), np.nan) for k in range(1, r.per.quarter)]
        n = r.ni_ytd - np.sum(prev) if np.all(np.isfinite(prev)) else np.nan
        single[(r.code, r.per)] = n
        add.append({"code": r.code, "per": r.per, "N": n, "pub": pd.to_datetime(r.first_seen, errors="coerce")})
    A = pd.concat([T, pd.DataFrame(add, columns=["code", "per", "N", "pub"])], ignore_index=True)
    ev = []
    for c, g in A.groupby("code"):
        s = g.set_index("per").N
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="Q"))
        e = s - s.shift(4)
        mu, sd = e.shift(1).rolling(8, min_periods=4).mean(), e.shift(1).rolling(8, min_periods=4).std()
        gg = g.set_index("per")
        ev.append(pd.DataFrame({"code": c, "pub": gg.pub, "N": gg.N, "N_ly": s.shift(4).reindex(gg.index),
                                "x": ((e - mu) / sd.where(sd > 0)).clip(-10, 10).reindex(gg.index)}))
    ev = pd.concat(ev).reset_index().rename(columns={"index": "per"}).dropna(subset=["x", "pub"])

    mk = load_market()
    ev, B = classify(ev, mk, H, 1500, ["盈餘驚喜", "盈餘壞消息（避開）"])
    ev["name"] = ev.code.map(names)
    dump("earnings_signals", ev, B, mk, 100, "季報公布日後第一個交易日開盤進場、持有 60 個交易日；最高 10%＝盈餘驚喜、最低 10%＝盈餘壞消息（上市股）",
         ["N", "N_ly"], lambda r: {"net_income_k": float(r.N), "net_income_last_year_k": None if pd.isna(r.N_ly) else float(r.N_ly)},
         log, "盈餘驚喜名單")
    return ev


if __name__ == "__main__":
    build()
