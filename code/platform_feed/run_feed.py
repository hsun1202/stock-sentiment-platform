# -*- coding: utf-8 -*-
"""
平台資料每日更新（收盤後跑）：①補證交所每日行情 ②公開資訊觀測站月營收快照 ③重算營收驚喜名單。
行情用 papertrade/tw_data.py 的抓取與快取（同一份 raw 快取，不會重複請求），但寫到自己的 platform_feed/data/prices.csv。
紀錄 platform_feed/run.log。
cd D:/專題claude && python platform_feed/run_feed.py
"""
import datetime as dt
import os
import sys
import traceback

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "papertrade"))
import mops_revenue  # noqa: E402
import revenue_signals  # noqa: E402
import tw_data  # noqa: E402

LOG = os.path.join(BASE, "run.log")
PX = os.path.join(BASE, "data", "prices.csv")


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def update_prices():
    px = pd.read_csv(PX, dtype={"code": str})
    last = px.date.max()
    days = [d for d in tw_data.trading_days(last, dt.date.today().isoformat()) if d > last]
    add = [q for q in (tw_data.daily_quotes(d) for d in days) if q is not None]
    if add:
        px = pd.concat([px] + add, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
        px.to_csv(PX, index=False, encoding="utf-8-sig")
    log(f"[行情] 原本到 {last}，補 {len(add)} 天 → 到 {px.date.max()}")


if __name__ == "__main__":
    for step in (update_prices, lambda: mops_revenue.update(log=log), lambda: revenue_signals.build(log=log)):
        try:
            step()
        except Exception:
            log("錯誤：" + traceback.format_exc().replace("\n", " | "))
    log("DONE")
