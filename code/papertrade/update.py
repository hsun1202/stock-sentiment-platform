# -*- coding: utf-8 -*-
"""把還沒抓過的交易日補進 data/prices.csv，並更新 data/exright.csv。第一次執行往回補 BOOT_DAYS 個交易日。"""
import os

import pandas as pd

import tw_data as T

BASE = T.BASE
DATA = os.path.join(BASE, "data")
PRICES = os.path.join(DATA, "prices.csv")
EXR = os.path.join(DATA, "exright.csv")
BOOT_DAYS = 90


def load_prices():
    if not os.path.exists(PRICES):
        return pd.DataFrame(columns=["code", "name", "date", "volume", "value", "open", "high", "low", "close"])
    return pd.read_csv(PRICES, dtype={"code": str})


def update(log=print):
    os.makedirs(DATA, exist_ok=True)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    have = load_prices()
    start = (pd.Timestamp(today) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    days = T.trading_days(start, today)
    if have.empty:
        days = days[-BOOT_DAYS:]
        log(f"[資料] 第一次執行：往回補 {len(days)} 個交易日（{days[0]}～{days[-1]}）")
    done = set(have.date) if not have.empty else set()
    todo = [d for d in days if d not in done and d >= (min(done) if done else days[0])]
    new = []
    for d in todo:
        q = T.daily_quotes(d)
        if q is None:
            log(f"[資料] {d} 行情還沒公布，下次再抓")
            break                         # 之後的日子一定也還沒有
        new.append(q)
    if new:
        allp = pd.concat([have] + new, ignore_index=True).drop_duplicates(["code", "date"], keep="last").sort_values(["date", "code"])
        allp.to_csv(PRICES, index=False, encoding="utf-8-sig")
        log(f"[資料] 新增 {len(new)} 個交易日（到 {new[-1].date.iloc[0]}）；共 {allp.date.nunique()} 天、{allp.code.nunique()} 檔")
        have = allp
    else:
        log("[資料] 沒有新的交易日")
    # 除權息：涵蓋面板全部期間，最近 60 天的那段每次重抓
    lo = have.date.min()
    old_end = (pd.Timestamp(today) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    parts = []
    if lo < old_end:
        parts.append(T.exright(lo, old_end))
    parts.append(T.exright(max(lo, (pd.Timestamp(old_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")), today, refresh=True))
    ex = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "code"])
    ex.to_csv(EXR, index=False, encoding="utf-8-sig")
    return have, ex
