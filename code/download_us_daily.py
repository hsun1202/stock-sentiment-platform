# -*- coding: utf-8 -*-
"""下載美股日資料（給「隔夜消息」對照組用）：S&P500、那斯達克、費城半導體、VIX
輸出 market_sentiment/us_daily.csv（Date=美國交易日，欄位為各指數收盤）
python download_us_daily.py
"""
import pandas as pd
import yfinance as yf

TICKERS = {"^GSPC": "spx", "^IXIC": "ndx", "^SOX": "sox", "^VIX": "vix"}
df = yf.download(list(TICKERS), start="2014-11-01", end="2025-01-10", auto_adjust=False, progress=False)["Close"]
df = df.rename(columns=TICKERS)[list(TICKERS.values())]
df.index.name = "Date"
df = df.dropna(how="all")
df.to_csv("market_sentiment/us_daily.csv", encoding="utf-8-sig")
print(df.shape, df.index.min().date(), df.index.max().date())
print(df.isna().sum())
print(df.tail(3))
