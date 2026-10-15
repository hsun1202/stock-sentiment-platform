# -*- coding: utf-8 -*-
"""
延伸還原股價到 2025-12（給「2015-2024 訓練、預測 2025」用）；作法同 fix_prices_2015.py，輸出到新資料夾 prices_adj_2025/（舊的不動）
  1. yfinance 重新下載 267 檔 2014-06 ~ 2026-01（Adj Close / Close 當還原因子）→ yahoo_raw_2014_2026.pkl
  2. 自動稽核：對照證交所官方交易日曆（chip_raw/calendar/*.json，FMTQIK）
       - 官方開市但 Yahoo 沒有的日子 → 用官方日行情 MI_INDEX 補
       - Yahoo 有但官方休市的日子（颱風假資料）→ 刪除
       - 超過 80% 股票成交量為 0 的日子（Yahoo 壞資料，如 2021-08-17）→ 用官方日行情取代
  3. 與舊版 fix_prices_2015.py 的修補清單比對，印出差異
python fix_prices_2025.py
"""
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from stock_list_267 import STOCK_LIST_267

OUT = "prices_adj_2025"
RAW = "yahoo_raw_2014_2026.pkl"
START, END = "2014-06-01", "2026-01-10"
OLD_PATCH = ["2014-12-27", "2016-01-30", "2016-06-04", "2016-09-10", "2017-02-18", "2017-06-03", "2017-09-30",
             "2018-03-31", "2018-12-22", "2019-09-09", "2021-04-06", "2021-08-17"]
OLD_DROP = ["2014-07-23", "2015-07-10", "2015-09-29", "2016-07-08", "2016-09-27", "2016-09-28"]
H = {"User-Agent": "Mozilla/5.0"}
os.makedirs(OUT, exist_ok=True)


def num(x):
    x = str(x).replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return np.nan


def roc(s):
    y, m, d = s.strip().split("/")
    return f"{int(y) + 1911}-{int(m):02d}-{int(d):02d}"


# ---- 官方交易日曆 ----
official = sorted({roc(r[0]) for f in glob.glob("chip_raw/calendar/*.json") for r in json.load(open(f, encoding="utf-8"))["data"]})
official = [d for d in official if START <= d <= "2025-12-31"]
print(f"官方交易日 {official[0]}~{official[-1]}，{len(official)} 天", flush=True)

# ---- Yahoo 原始資料 ----
tickers = [f"{c}.TW" for c, _, _ in STOCK_LIST_267]
if os.path.exists(RAW):
    raw = pd.read_pickle(RAW)
else:
    raw = yf.download(tickers, start=START, end=END, group_by="ticker", auto_adjust=False, threads=True, progress=False)
    raw.to_pickle(RAW)
print(f"Yahoo 原始資料 {raw.shape}，日期 {raw.index.min().date()}~{raw.index.max().date()}", flush=True)
have = [t for t in tickers if t in raw.columns.get_level_values(0)]
closes = pd.DataFrame({t: raw[t]["Close"] for t in have})
vols = pd.DataFrame({t: raw[t]["Volume"] for t in have})
closes.index = closes.index.strftime("%Y-%m-%d")
vols.index = closes.index
cover = closes.notna().mean(axis=1)
ydays = set(closes.index[cover >= 0.5])
off = set(official)
missing = sorted(d for d in off - ydays if d <= "2025-12-31")
extra = sorted(d for d in ydays - off if d <= "2025-12-31")
zero = (vols.eq(0) & closes.notna()).sum(axis=1) / closes.notna().sum(axis=1).replace(0, np.nan)
badvol = sorted(d for d in zero.index[zero > 0.8] if d in off)
PATCH = sorted(set(missing) | set(badvol))
DROP = extra
print(f"官方開市但 Yahoo 缺：{missing}")
print(f"Yahoo 有但官方休市（刪）：{extra}")
print(f"成交量為0的壞資料日（>80%股票）：{badvol}")
print(f"與舊清單比：補－新增 {sorted(set(PATCH) - set(OLD_PATCH))}｜補－舊有但這次沒偵測到 {sorted(set(OLD_PATCH) - set(PATCH))}")
print(f"           刪－新增 {sorted(set(DROP) - set(OLD_DROP))}｜刪－舊有但這次沒偵測到 {sorted(set(OLD_DROP) - set(DROP))}", flush=True)


def official_day(d):
    cache = f"chip_raw/mi_index/{d.replace('-', '')}.json"
    if not os.path.exists(cache) and os.path.exists(f"chip_raw/mi_index_{d.replace('-', '')}.json"):
        cache = f"chip_raw/mi_index_{d.replace('-', '')}.json"
    if os.path.exists(cache):
        j = json.load(open(cache, encoding="utf-8"))
    else:
        j = requests.get(f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d.replace('-', '')}"
                         f"&type=ALLBUT0999&response=json", headers=H, timeout=30).json()
        json.dump(j, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
        time.sleep(4)
    tb = [t for t in j["tables"] if t.get("data") and len(t["data"]) > 100 and "收盤行情" in t.get("title", "")][0]
    rows = {}
    for r in tb["data"]:
        sign = -1 if "-" in re.sub(r"<[^>]+>", "", r[9]) else 1
        close, chg = num(r[8]), num(r[10])
        rows[r[0].strip()] = {"open": num(r[5]), "high": num(r[6]), "low": num(r[7]), "close": close,
                              "volume": num(r[2]), "ref": close - sign * chg if not np.isnan(chg) else np.nan}
    return rows


os.makedirs("chip_raw/mi_index", exist_ok=True)
offd = {d: official_day(d) for d in PATCH}
report = []
for code, name, _ in STOCK_LIST_267:
    tk = f"{code}.TW"
    if tk not in have:
        report.append({"code": code, "status": "yfinance無資料"})
        continue
    d = raw[tk].dropna(subset=["Close"]).copy()
    if d.empty:
        report.append({"code": code, "status": "yfinance無資料"})
        continue
    d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
    d = d[~d.index.isin(DROP) & (d.index <= "2025-12-31")]
    f = d["Adj Close"] / d["Close"]
    out = pd.DataFrame({"Open": d.Open * f, "High": d.High * f, "Low": d.Low * f, "Close": d["Adj Close"],
                        "Volume": d.Volume, "close_unadj": d.Close, "adj_factor": f})
    out = out[~out.index.isin(PATCH)]          # 壞資料日先拿掉，再用官方資料補
    patched = []
    for day in PATCH:
        out = out.sort_index()
        prev = out.index[out.index < day]
        if len(prev) == 0 or day < out.index.min():
            continue
        p = prev.max()
        o = offd[day].get(code)
        if o is None or np.isnan(o["close"]):
            c, k = out.at[p, "Close"], out.at[p, "adj_factor"]
            out.loc[day] = [c, c, c, c, 0.0, out.at[p, "close_unadj"], k]
            patched.append(day + "(無成交)")
            continue
        ret = o["close"] / o["ref"] if o["ref"] and not np.isnan(o["ref"]) else o["close"] / out.at[p, "close_unadj"]
        adj_c = out.at[p, "Close"] * ret
        k = adj_c / o["close"]
        out.loc[day] = [o["open"] * k, o["high"] * k, o["low"] * k, adj_c, o["volume"], o["close"], k]
        patched.append(day)
    out = out.sort_index()
    out.index.name = "Date"
    out.to_csv(os.path.join(OUT, f"{code}_{name}.csv"), encoding="utf-8-sig")
    report.append({"code": code, "status": "ok", "rows": len(out), "first": out.index.min(), "last": out.index.max(),
                   "patched": ",".join(patched)})
rp = pd.DataFrame(report)
rp.to_csv(os.path.join(OUT, "_fix_report.csv"), index=False, encoding="utf-8-sig")
print(rp.status.value_counts().to_dict())
print("無資料:", rp.loc[rp.status != "ok", "code"].tolist())
print("最後日期分布:", rp.loc[rp.status == "ok", "last"].value_counts().head(5).to_dict())
