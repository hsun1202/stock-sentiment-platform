# -*- coding: utf-8 -*-
"""
（2014-06 ~ 2025-01 版，給 2015-2024 十年分析用；原本 fix_prices.py 的 2017~ 版 prices_adj/ 保留不動）
重建267檔「還原股價」(含現金股利還原)，並修補Yahoo壞資料（audit_yahoo_2014_2019.py 對照證交所官方交易日曆找出）：
  - Yahoo 缺的官方開市日（多為週六補班開市日，如2016-01-30、2018-12-22；另有2019-09-09、2021-04-06整天缺）
    -> 用證交所官方日行情插入
  - 2021-08-17：Yahoo成交量=0且收盤價複製前一天 -> 用官方日行情取代
  - Yahoo 有、但證交所颱風休市的日子（2014-07-23、2015-07-10、2015-09-29、2016-07-08、2016-09-27、2016-09-28）
    -> Yahoo 填的是「量0、價照抄前一天」的假資料，刪除
還原方式：Yahoo的 Adj Close / Close 當作還原因子，開高低收同乘該因子。
補的那兩天用官方「漲跌價差」(除息日會自動相對除息參考價)算當日報酬，接在前一日還原價後面。
輸出 prices_adj/{code}_{name}.csv；原始 stocks_267 不動。

python fix_prices.py
"""
import os
import re
import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from stock_list_267 import STOCK_LIST_267

OUT = "prices_adj_2014"
START, END = "2014-06-01", "2025-02-01"
PATCH_DAYS = ["2014-12-27", "2016-01-30", "2016-06-04", "2016-09-10", "2017-02-18", "2017-06-03", "2017-09-30",
              "2018-03-31", "2018-12-22", "2019-09-09", "2021-04-06", "2021-08-17"]
DROP_DAYS = ["2014-07-23", "2015-07-10", "2015-09-29", "2016-07-08", "2016-09-27", "2016-09-28"]
H = {"User-Agent": "Mozilla/5.0"}
os.makedirs(OUT, exist_ok=True)


def num(x):
    x = str(x).replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return np.nan


def official_day(d):
    cache = f"chip_raw/mi_index/{d.replace('-', '')}.json"
    if not os.path.exists(cache) and os.path.exists(f"chip_raw/mi_index_{d.replace('-', '')}.json"):
        cache = f"chip_raw/mi_index_{d.replace('-', '')}.json"  # 2020-2024 版存的舊路徑
    if os.path.exists(cache):
        j = json.load(open(cache, encoding="utf-8"))
    else:
        r = requests.get(f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d.replace('-', '')}"
                         f"&type=ALLBUT0999&response=json", headers=H, timeout=30)
        j = r.json()
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


official = {d: official_day(d) for d in PATCH_DAYS}
print("官方日行情:", {d: len(v) for d, v in official.items()}, flush=True)

tickers = [f"{c}.TW" for c, _, _ in STOCK_LIST_267]
raw = pd.read_pickle("yahoo_raw_2014_2025.pkl")  # audit_yahoo_2014_2019.py 下載的同一份原始資料
report = []
for code, name, _ in STOCK_LIST_267:
    tk = f"{code}.TW"
    if tk not in raw.columns.get_level_values(0):
        report.append({"code": code, "status": "yfinance無資料"})
        continue
    d = raw[tk].dropna(subset=["Close"]).copy()
    if d.empty:
        report.append({"code": code, "status": "yfinance無資料"})
        continue
    d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
    d = d[~d.index.isin(DROP_DAYS)]
    f = d["Adj Close"] / d["Close"]
    out = pd.DataFrame({
        "Open": d.Open * f, "High": d.High * f, "Low": d.Low * f, "Close": d["Adj Close"],
        "Volume": d.Volume, "close_unadj": d.Close, "adj_factor": f,
    })
    patched = []
    for day in PATCH_DAYS:
        out = out.sort_index()
        prev = out.index[out.index < day]
        if len(prev) == 0:
            continue
        p = prev.max()
        o = official[day].get(code)
        if o is None or np.isnan(o["close"]):
            # 當天沒成交(或停牌)：比照Yahoo沒成交日的慣例，價格沿用前一天、成交量0
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
    report.append({"code": code, "status": "ok", "rows": len(out), "first": out.index.min(),
                   "last": out.index.max(), "patched": ",".join(patched)})

rp = pd.DataFrame(report)
rp.to_csv(os.path.join(OUT, "_fix_report.csv"), index=False, encoding="utf-8-sig")
print(rp.status.value_counts().to_dict())
print("無資料:", rp.loc[rp.status != "ok", "code"].tolist())
ok = rp[rp.status == "ok"]
for day in PATCH_DAYS:
    print(f"補了{day}: {ok.patched.fillna('').str.contains(day).sum()} 檔（其中無成交 {ok.patched.fillna('').str.contains(day + '(無成交)', regex=False).sum()} 檔）")
