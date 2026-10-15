# -*- coding: utf-8 -*-
"""
下載2020-2024籌碼面／市場情緒資料（原始檔，之後再解析）：
  - 證交所三大法人買賣超日報 T86  -> chip_raw/t86/YYYYMMDD.json
  - 證交所融資融券彙總 MI_MARGN    -> chip_raw/margin/YYYYMMDD.json
  - 期交所台指選擇權Put/Call比     -> chip_raw/pcr/YYYYMM.csv
交易日取自 taiex_2020_2024.csv。已存在的檔案會跳過，中斷後重跑同一指令即可接續。
證交所會封鎖過於頻繁的請求，每次請求間隔 DELAY 秒，被擋時指數退避重試。

python download_chip_data.py
"""
import os
import time
import json
import requests
import pandas as pd

DELAY = 3.5
OUT = "chip_raw"
H = {"User-Agent": "Mozilla/5.0"}
SOURCES = {
    "t86": "https://www.twse.com.tw/rwd/zh/fund/T86?date={d}&selectType=ALLBUT0999&response=json",
    "margin": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d}&selectType=ALL&response=json",
}


def fetch_json(url):
    wait = 60
    for attempt in range(6):
        try:
            r = requests.get(url, headers=H, timeout=30)
            j = r.json()
            return j
        except Exception as e:
            print(f"    [警告] {e.__class__.__name__}，{wait}秒後重試（{attempt + 1}/6）", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 900)
    return None


def main():
    taiex = pd.read_csv("taiex_2020_2024.csv", header=[0, 1], index_col=0)
    dates = [pd.Timestamp(x).strftime("%Y%m%d") for x in taiex.index]
    for k in SOURCES:
        os.makedirs(os.path.join(OUT, k), exist_ok=True)
    os.makedirs(os.path.join(OUT, "pcr"), exist_ok=True)

    months = sorted({d[:6] for d in dates})
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

    todo = [(k, d) for d in dates for k in SOURCES
            if not os.path.exists(os.path.join(OUT, k, f"{d}.json"))]
    print(f"交易日 {len(dates)} 天，待下載 {len(todo)} 筆（已完成 {len(dates) * len(SOURCES) - len(todo)} 筆）", flush=True)
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
            eta = el / i * (len(todo) - i)
            print(f"  進度 {i}/{len(todo)}  最新 {k} {d}  已花 {el/60:.1f} 分鐘，預估剩 {eta/60:.1f} 分鐘", flush=True)
        time.sleep(DELAY)
    print(f"\n[完成] 異常筆數 {len(bad)}：{bad[:20]}", flush=True)


if __name__ == "__main__":
    main()
