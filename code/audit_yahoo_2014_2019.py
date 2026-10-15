# -*- coding: utf-8 -*-
"""
稽核 Yahoo 股價 2014-06~2019-12：對照證交所官方交易日曆，找出
  (1) 官方有開市但 Yahoo 大多數股票缺資料的日子（例如2020-2024版的2021-04-06）
  (2) Yahoo 有、官方沒開市的日子
  (3) 大量股票「成交量=0且收盤價=前一天」的壞資料日（例如2021-08-17）
Yahoo 原始資料快取在 yahoo_raw_2014_2025.pkl（之後 fix_prices_2015.py 共用，不必重抓）。
python audit_yahoo_2014_2019.py
"""
import os
import pandas as pd
import yfinance as yf
from stock_list_267 import STOCK_LIST_267

CACHE = "yahoo_raw_2014_2025.pkl"
if os.path.exists(CACHE):
    raw = pd.read_pickle(CACHE)
else:
    raw = yf.download([f"{c}.TW" for c, _, _ in STOCK_LIST_267], start="2014-06-01", end="2025-02-01",
                      auto_adjust=False, group_by="ticker", threads=True, progress=False)
    raw.to_pickle(CACHE)
cal = pd.to_datetime(pd.read_csv("chip_raw/trading_days_2014_2019.csv", dtype=str)["date"], format="%Y%m%d")
lo, hi = cal.min(), cal.max()
close = pd.DataFrame({tk: raw[tk]["Close"] for tk in raw.columns.get_level_values(0).unique()})
vol = pd.DataFrame({tk: raw[tk]["Volume"] for tk in raw.columns.get_level_values(0).unique()})
close.index = pd.to_datetime(close.index); vol.index = close.index
c = close.loc[lo:hi]; v = vol.loc[lo:hi]
alive = c.notna().sum()
print(f"Yahoo 有資料的股票 {int((alive > 0).sum())} 檔；期間 {lo.date()}~{hi.date()}，官方交易日 {len(cal)}，Yahoo 日期 {len(c)}")
cnt = c.notna().sum(axis=1)
missing = [d for d in cal if d not in c.index or cnt.get(d, 0) < 0.5 * cnt.median()]
extra = [d for d in c.index if d not in set(cal) and cnt[d] > 0]
print("(1) 官方開市但Yahoo缺(或大半股票缺)：", [d.date().isoformat() for d in missing])
print("(2) Yahoo有但官方沒開市：", [d.date().isoformat() for d in extra])
stale = ((v == 0) & (c == c.shift(1))).sum(axis=1) / cnt.clip(lower=1)
bad = stale[stale > 0.3]
print("(3) 過半股票量=0且價=前日：", {d.date().isoformat(): round(x, 2) for d, x in bad.items()})
print("    一般日子這個比例的中位數：", round(stale.median(), 3), " 90百分位：", round(stale.quantile(.9), 3))
pd.Series([d.date().isoformat() for d in missing + list(bad.index)], name="date").drop_duplicates().to_csv(
    "yahoo_bad_days_2014_2019.csv", index=False)
