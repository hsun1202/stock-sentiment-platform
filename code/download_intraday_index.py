# -*- coding: utf-8 -*-
"""下載證交所「每5秒指數統計」MI_5MINS_INDEX（2015-2024 每個交易日一次請求），整理成每分鐘加權指數
→ market_sentiment/taiex_intraday_1min.csv（Date, time, taiex）；可續跑（已下載的日期跳過）。給盤中情緒（Renault 2017 式）檢驗用。
python download_intraday_index.py"""
import os, time, pandas as pd
from download_chip_data import fetch_json
OUT = "market_sentiment/taiex_intraday_1min.csv"
cal = pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"])
days = [d for d in cal.Date if d >= pd.Timestamp("2015-01-01")]
done = set(pd.read_csv(OUT, usecols=["Date"]).Date.unique()) if os.path.exists(OUT) else set()
todo = [d for d in days if d.strftime("%Y-%m-%d") not in done]
print(f"交易日 {len(days)}，待下載 {len(todo)}", flush=True)
t0 = time.time()
for i, d in enumerate(todo, 1):
    j = fetch_json(f"https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_INDEX?date={d.strftime('%Y%m%d')}&response=json")
    if j and j.get("data"):
        x = pd.DataFrame([(r[0], float(r[1].replace(",", ""))) for r in j["data"]], columns=["time", "taiex"])
        x = x[x.time.str[6:8] == "00"]  # 每分鐘整點
        x.insert(0, "Date", d.strftime("%Y-%m-%d"))
        x.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    else:
        print("  無資料", d.date(), None if j is None else j.get("stat"), flush=True)
    if i % 100 == 0:
        print(f"  進度 {i}/{len(todo)}，預估剩 {(time.time()-t0)/i*(len(todo)-i)/60:.0f} 分鐘", flush=True)
    time.sleep(2.5)
print("DONE", flush=True)
