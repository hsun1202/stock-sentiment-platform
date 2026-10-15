# -*- coding: utf-8 -*-
"""
把 chip_raw/daytrade/*.json（證交所當日沖銷交易統計）整理成：
  chip_raw/daytrade_stock.csv   每檔每日：stock_code, Date, dt_shares, dt_buy, dt_sell
  chip_raw/daytrade_market.csv  大盤每日：Date, dt_shares, dt_shares_pct, dt_buy, dt_buy_pct, dt_sell, dt_sell_pct
並檢查：與 all_stocks_v3 的成交量對得上嗎（個股當沖股數 ≤ 成交股數？加總後與大盤占比一致嗎？）
python daytrade_build.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd


def num(s):
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


stock_rows, mkt_rows = [], []
files = sorted(glob.glob("chip_raw/daytrade/*.json"))
for f in files:
    d = os.path.basename(f)[:8]
    j = json.load(open(f, encoding="utf-8"))
    t0, t1 = j["tables"][0], j["tables"][1]
    if t0.get("data"):
        r = t0["data"][0]
        mkt_rows.append({"Date": d, "dt_shares": num(r[0]), "dt_shares_pct": num(r[1]), "dt_buy": num(r[2]),
                         "dt_buy_pct": num(r[3]), "dt_sell": num(r[4]), "dt_sell_pct": num(r[5])})
    for r in t1["data"]:
        stock_rows.append((r[0].strip(), d, num(r[3]), num(r[4]), num(r[5])))

S = pd.DataFrame(stock_rows, columns=["stock_code", "Date", "dt_shares", "dt_buy", "dt_sell"])
S["Date"] = pd.to_datetime(S.Date)
M = pd.DataFrame(mkt_rows)
M["Date"] = pd.to_datetime(M.Date)
S.to_csv("chip_raw/daytrade_stock.csv", index=False, encoding="utf-8-sig")
M.to_csv("chip_raw/daytrade_market.csv", index=False, encoding="utf-8-sig")
print(f"{len(files)} 個檔案 → 個股 {len(S):,} 列（{S.stock_code.nunique()} 種代號）、大盤 {len(M)} 日")

# ---- 檢查 ----
A = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "Volume"])
A["Date"] = pd.to_datetime(A.Date)
m = A.merge(S, on=["stock_code", "Date"], how="left")
m["dt_shares"] = m.dt_shares.fillna(0)
m["ratio"] = m.dt_shares / m.Volume
print(f"\n265 檔主資料 {len(A):,} 列，其中有當沖紀錄 {int((m.dt_shares > 0).sum()):,} 列（{(m.dt_shares > 0).mean():.1%}）")
print(f"當沖比率 = 當沖成交股數/成交股數：中位數 {m.ratio.median():.3f}、平均 {m.ratio.mean():.3f}、>1 的列 {(m.ratio > 1).sum():,}、NaN {m.ratio.isna().sum():,}")
m["year"] = m.Date.dt.year
by = m.groupby("year").agg(mean_ratio=("ratio", "mean"), med_ratio=("ratio", "median"), share_rows=("dt_shares", lambda x: (x > 0).mean()))
mk = M.assign(year=M.Date.dt.year).groupby("year").agg(mkt_buy_pct=("dt_buy_pct", "mean"), mkt_shares_pct=("dt_shares_pct", "mean"))
print("\n每年：265檔平均當沖比率 vs 大盤當沖占比（證交所，買進金額%、股數%）")
print(by.join(mk).round(3).to_string())
# 加總一致性：265檔當沖股數加總 / 265檔成交股數 vs 大盤股數占比（只當粗略對照）
agg = m.groupby("Date").agg(dt=("dt_shares", "sum"), vol=("Volume", "sum"))
agg["r"] = agg.dt / agg.vol
print(f"\n265檔合計當沖/成交股數（日平均）{agg.r.mean():.3f}；大盤股數占比平均 {M.dt_shares_pct.mean() / 100:.3f}")
print("大盤統計缺值天數:", int(M[["dt_shares_pct", "dt_buy_pct"]].isna().any(axis=1).sum()))
