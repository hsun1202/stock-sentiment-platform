# -*- coding: utf-8 -*-
"""
下載 2025 年資料（給「用 2015-2024 訓練、預測 2025」的真正樣本外測試）：
  - 證交所官方交易日曆 FMTQIK（每月）      -> chip_raw/calendar/YYYYMM.json、chip_raw/trading_days_2025.csv
  - 期交所 Put/Call 比（每月；build 的交易日曆 2020 年起用它） -> chip_raw/pcr/YYYYMM.csv
  - 證交所三大法人 T86、融資融券 MI_MARGN（每日） -> chip_raw/t86/、chip_raw/margin/YYYYMMDD.json
已存在的檔案會跳過，中斷後重跑即可接續。PTT 2025 年沒有資料（不在本腳本範圍）。
python -u download_2025.py >> download_2025.log 2>&1
"""
import json
import os
import time

import pandas as pd
import requests

from download_chip_data import fetch_json, H

DELAY = 2.5
MONTHS = pd.period_range("2025-01", "2025-12", freq="M")
SOURCES = {
    "t86": "https://www.twse.com.tw/rwd/zh/fund/T86?date={d}&selectType=ALLBUT0999&response=json",
    "margin": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d}&selectType=ALL&response=json",
}
for k in ("calendar", "pcr", "t86", "margin"):
    os.makedirs(f"chip_raw/{k}", exist_ok=True)


def roc_to_ymd(s):
    y, m, d = s.strip().split("/")
    return f"{int(y) + 1911}{int(m):02d}{int(d):02d}"


print(f"[start] {time.strftime('%F %T')}", flush=True)
days = []
for p in MONTHS:
    path = f"chip_raw/calendar/{p.strftime('%Y%m')}.json"
    if not os.path.exists(path):
        j = fetch_json(f"https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date={p.strftime('%Y%m')}01&response=json")
        if j is None or j.get("stat") != "OK":
            raise RuntimeError(f"交易日曆下載失敗 {p}")
        json.dump(j, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        time.sleep(DELAY)
    days += [roc_to_ymd(r[0]) for r in json.load(open(path, encoding="utf-8"))["data"]]
days = sorted(set(days))
pd.Series(days, name="date").to_csv("chip_raw/trading_days_2025.csv", index=False)
print(f"[交易日曆] {days[0]}~{days[-1]} 共 {len(days)} 天", flush=True)

for p in MONTHS:
    path = f"chip_raw/pcr/{p.strftime('%Y%m')}.csv"
    if os.path.exists(path):
        continue
    start = f"{p.strftime('%Y/%m')}/01"
    end = (p.to_timestamp() + pd.offsets.MonthEnd(0)).strftime("%Y/%m/%d")
    r = requests.post("https://www.taifex.com.tw/cht/3/pcRatioDown", headers=H, timeout=30,
                      data={"queryStartDate": start, "queryEndDate": end})
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(r.content.decode("big5", errors="replace"))
    time.sleep(1.5)
print("[PCR] 完成", flush=True)

todo = [(k, d) for k in SOURCES for d in days if not os.path.exists(f"chip_raw/{k}/{d}.json")]
print(f"[籌碼] 待下載 {len(todo)} 筆", flush=True)
giveup = []
for i, (k, d) in enumerate(todo):
    j = fetch_json(SOURCES[k].format(d=d))
    if j is None or j.get("stat") != "OK":
        giveup.append((k, d, None if j is None else j.get("stat")))
    else:
        json.dump(j, open(f"chip_raw/{k}/{d}.json", "w", encoding="utf-8"), ensure_ascii=False)
    if (i + 1) % 50 == 0:
        print(f"[progress] {i + 1}/{len(todo)}｜放棄 {len(giveup)}｜{time.strftime('%T')}", flush=True)
    time.sleep(DELAY)
print(f"[DONE] {time.strftime('%F %T')}｜放棄 {len(giveup)}：{giveup[:10]}", flush=True)
