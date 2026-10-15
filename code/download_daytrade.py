# -*- coding: utf-8 -*-
"""
下載證交所「當日沖銷交易標的及成交量值」(TWTB4U)，2015-01-01 ~ 2024-12-31 每個交易日
  -> chip_raw/daytrade/YYYYMMDD.json（含兩張表：大盤當沖統計、每檔當沖成交股數/買賣金額）
交易日曆用 market_sentiment/taiex_official.csv（證交所官方）。已存在的檔案會跳過，中斷後重跑同一指令即可接續。
間隔 DELAY 秒；非 OK 回應重試 3 次，仍失敗記到 download_daytrade_giveup.txt。
python -u download_daytrade.py >> download_daytrade.log 2>&1
"""
import json
import os
import time

import pandas as pd
import requests

DELAY = 2.5
OUT = "chip_raw/daytrade"
URL = "https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U?date={d}&response=json"
H = {"User-Agent": "Mozilla/5.0"}
os.makedirs(OUT, exist_ok=True)

cal = pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"])
days = [d.strftime("%Y%m%d") for d in cal.Date if "2015-01-01" <= d.strftime("%Y-%m-%d") <= "2024-12-31"]
print(f"[start] {time.strftime('%F %T')} 共 {len(days)} 個交易日", flush=True)


def valid(j):
    return j and j.get("stat") == "OK" and j.get("tables") and len(j["tables"]) >= 2 and j["tables"][1].get("data")


def fetch(d):
    wait = 30
    for attempt in range(3):
        try:
            j = requests.get(URL.format(d=d), headers=H, timeout=30).json()
            if valid(j):
                return j, "OK"
            stat = (j or {}).get("stat")
        except Exception as e:
            stat = e.__class__.__name__
        print(f"    [警告] {d} {stat}，{wait}秒後重試（{attempt + 1}/3）", flush=True)
        time.sleep(wait)
        wait *= 2
    return None, stat


done = giveup = 0
t0 = time.time()
for i, d in enumerate(days):
    path = os.path.join(OUT, f"{d}.json")
    if os.path.exists(path):
        done += 1
        continue
    j, stat = fetch(d)
    if j is None:
        giveup += 1
        with open("download_daytrade_giveup.txt", "a", encoding="utf-8") as f:
            f.write(f"{d}\t{stat}\n")
    else:
        json.dump(j, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        done += 1
    if (i + 1) % 100 == 0:
        el = (time.time() - t0) / 60
        print(f"[progress] {i + 1}/{len(days)} 完成 {done} 放棄 {giveup}｜已跑 {el:.0f} 分鐘", flush=True)
    time.sleep(DELAY)

print(f"[DONE] {time.strftime('%F %T')} 完成 {done}/{len(days)}，放棄 {giveup}", flush=True)
