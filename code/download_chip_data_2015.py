# -*- coding: utf-8 -*-
"""
補下載 2015-2019 的籌碼面資料＋官方日行情（2020-2024 已由 download_chip_data.py 下載）：
  - 證交所三大法人買賣超日報 T86  -> chip_raw/t86/YYYYMMDD.json
  - 證交所融資融券彙總 MI_MARGN    -> chip_raw/margin/YYYYMMDD.json
  - 證交所每日收盤行情 MI_INDEX    -> chip_raw/mi_index/YYYYMMDD.json（稽核 Yahoo 股價：每月第一個交易日抽樣，
                                     Yahoo 缺漏的日子已由 audit_yahoo_2014_2019.py 對照交易日曆找出、另外抓好）
  - 期交所台指選擇權Put/Call比     -> chip_raw/pcr/YYYYMM.csv
交易日曆改用證交所官方「每月市場成交資訊」FMTQIK（2020-2024 版用 yfinance 大盤日期，漏了2021-04-06）
  -> chip_raw/calendar/YYYYMM.json、chip_raw/trading_days_2014_2019.csv
已存在的檔案會跳過，中斷後重跑同一指令即可接續。證交所每次請求間隔 DELAY 秒。

python download_chip_data_2015.py
"""
import os
import time
import json
import requests
import pandas as pd

from download_chip_data import fetch_json, H

DELAY = 2.5  # 實測回應約1.5~3秒，加上間隔約每4秒1次；原本3.5秒要5小時
OUT = "chip_raw"
CAL_MONTHS = pd.period_range("2014-06", "2019-12", freq="M")
CHIP_START, CHIP_END = "20150101", "20191231"
SOURCES = {
    "t86": "https://www.twse.com.tw/rwd/zh/fund/T86?date={d}&selectType=ALLBUT0999&response=json",
    "margin": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d}&selectType=ALL&response=json",
    "mi_index": "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d}&type=ALLBUT0999&response=json",
}


def roc_to_ymd(s):
    y, m, d = s.strip().split("/")
    return f"{int(y) + 1911}{int(m):02d}{int(d):02d}"


def trading_calendar():
    os.makedirs(os.path.join(OUT, "calendar"), exist_ok=True)
    days = []
    for p in CAL_MONTHS:
        path = os.path.join(OUT, "calendar", f"{p.strftime('%Y%m')}.json")
        if not os.path.exists(path):
            j = fetch_json(f"https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date={p.strftime('%Y%m')}01&response=json")
            if j is None or j.get("stat") != "OK":
                raise RuntimeError(f"交易日曆下載失敗 {p}：{None if j is None else j.get('stat')}")
            json.dump(j, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            time.sleep(DELAY)
        j = json.load(open(path, encoding="utf-8"))
        days += [roc_to_ymd(r[0]) for r in j["data"]]
    days = sorted(set(days))
    pd.Series(days, name="date").to_csv(os.path.join(OUT, "trading_days_2014_2019.csv"), index=False)
    print(f"[交易日曆] {days[0]}~{days[-1]} 共 {len(days)} 個交易日（證交所官方）", flush=True)
    return days


def main():
    days = trading_calendar()
    chip_days = [d for d in days if CHIP_START <= d <= CHIP_END]
    for k in SOURCES:
        os.makedirs(os.path.join(OUT, k), exist_ok=True)
    os.makedirs(os.path.join(OUT, "pcr"), exist_ok=True)

    months = sorted({d[:6] for d in chip_days})
    for m in months:
        path = os.path.join(OUT, "pcr", f"{m}.csv")
        if os.path.exists(path):
            continue
        start = f"{m[:4]}/{m[4:]}/01"
        end = (pd.Timestamp(f"{m[:4]}-{m[4:]}-01") + pd.offsets.MonthEnd(0)).strftime("%Y/%m/%d")
        r = requests.post("https://www.taifex.com.tw/cht/3/pcRatioDown", headers=H, timeout=30,
                          data={"queryStartDate": start, "queryEndDate": end})
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.content.decode("big5", errors="replace"))
        time.sleep(1.5)
    print(f"[PCR] {len(months)} 個月完成", flush=True)

    # 籌碼（t86、margin）先下載完，官方日行情（mi_index）排最後
    todo = [(k, d) for k in ("t86", "margin") for d in chip_days
            if not os.path.exists(os.path.join(OUT, k, f"{d}.json"))]
    todo = sorted(todo, key=lambda kd: (kd[1], kd[0]))
    sample_days = sorted({min(d for d in chip_days if d[:6] == m) for m in months})  # 每月第一個交易日
    todo += [("mi_index", d) for d in sample_days if not os.path.exists(os.path.join(OUT, "mi_index", f"{d}.json"))]
    total = len(chip_days) * 2 + len(sample_days)
    print(f"交易日 {len(chip_days)} 天，待下載 {len(todo)} 筆（已完成 {total - len(todo)} 筆）", flush=True)
    t0 = time.time()
    bad = []
    for i, (k, d) in enumerate(todo, 1):
        j = fetch_json(SOURCES[k].format(d=d))
        if j is None:
            bad.append((k, d))
        else:
            if j.get("stat") not in ("OK", "ok", None):
                bad.append((k, d, j.get("stat")))
            with open(os.path.join(OUT, k, f"{d}.json"), "w", encoding="utf-8") as f:
                json.dump(j, f, ensure_ascii=False)
        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  進度 {i}/{len(todo)}  最新 {k} {d}  已花 {el/60:.1f} 分鐘，預估剩 {el/i*(len(todo)-i)/60:.1f} 分鐘",
                  flush=True)
        time.sleep(DELAY)
    print(f"\n[完成] 異常筆數 {len(bad)}：{bad[:20]}", flush=True)


if __name__ == "__main__":
    main()
