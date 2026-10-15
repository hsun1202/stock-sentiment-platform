# -*- coding: utf-8 -*-
"""
抓公開資訊觀測站「重大訊息」（歷史，逐公司逐年；只抓主旨，不抓全文）
  介面：POST https://mops.twse.com.tw/mops/api/t05st01  {"companyId","year"(民國),"month":"all","firstDay":"","lastDay":"","dataType":"1"}
  回傳每則：公司代號、名稱、發言日期(民國)、發言時間(時:分:秒)、主旨
輸出：mops_material/{code}.jsonl（每行一則：code,name,date,time,title,seq）、mops_material/done.txt（已完成的 code,year）、crawl.log
python crawl_mops_material.py [--all]      # 預設 265 檔；--all＝全市場（final_dataset_v3_tejall25 的代碼）
可中斷續跑；速度約每秒 1 次。
"""
import json
import os
import random
import sys
import time

import pandas as pd
import requests

OUT = "mops_material"
YEARS = range(104, 115)          # 2015~2025
URL = "https://mops.twse.com.tw/mops/api/t05st01"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
     "Content-Type": "application/json", "Origin": "https://mops.twse.com.tw", "Referer": "https://mops.twse.com.tw/mops/"}
os.makedirs(OUT, exist_ok=True)
src = "final_dataset_v3_tejall25" if "--all" in sys.argv else "final_dataset_v3_tej"
codes = sorted(pd.read_csv(f"{src}/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code"]).stock_code.unique())
done_f = os.path.join(OUT, "done.txt")
done = set(open(done_f, encoding="utf-8").read().split()) if os.path.exists(done_f) else set()
todo = [(c, y) for c in codes for y in YEARS if f"{c},{y}" not in done]
if "--reverse" in sys.argv:          # 第二支從後面往前抓，與第一支在中間會合
    todo = todo[::-1]


def log(s):
    line = time.strftime("%H:%M:%S ") + s
    print(line, flush=True)
    open(os.path.join(OUT, "crawl.log"), "a", encoding="utf-8").write(line + "\n")


log(f"開始：{len(codes)} 檔 × {len(YEARS)} 年，剩 {len(todo)} 次請求")
sess, n_rows, fails, t0 = requests.Session(), 0, 0, time.time()
for k, (c, y) in enumerate(todo, 1):
    if k % 10 == 1 and os.path.exists(done_f):      # 另一支可能已抓過 → 重新讀完成清單
        done = set(open(done_f, encoding="utf-8").read().split())
    if f"{c},{y}" in done:
        continue
    body = {"companyId": c, "year": str(y), "month": "all", "firstDay": "", "lastDay": "", "dataType": "1"}
    rows, ok = [], False
    for attempt in range(5):
        try:
            r = sess.post(URL, headers=H, json=body, timeout=30)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                j = r.json()
                if j.get("code") == 200:
                    rows = (j.get("result") or {}).get("data") or []
                    ok = True
                    break
                if j.get("code") == 406 or "查無" in str(j.get("message")):      # 該年沒有資料
                    ok = True
                    break
            wait = 20 * (attempt + 1)
            log(f"  {c} {y} 第{attempt + 1}次異常（HTTP {r.status_code}，{r.text[:60]!r}），等 {wait}s")
        except Exception as e:
            wait = 20 * (attempt + 1)
            log(f"  {c} {y} 第{attempt + 1}次例外 {type(e).__name__}，等 {wait}s")
        time.sleep(wait)
    if not ok:
        fails += 1
        if fails >= 10:
            log("連續/累計失敗過多，停止（可重跑續抓）")
            break
        continue
    with open(os.path.join(OUT, f"{c}.jsonl"), "a", encoding="utf-8") as f:
        for d in rows:
            seq = (d[5] or {}).get("parameters", {}).get("serialNumber") if len(d) > 5 and isinstance(d[5], dict) else None
            yy, mm, dd = d[2].split("/")
            f.write(json.dumps({"code": d[0], "name": d[1], "date": f"{int(yy) + 1911}-{mm}-{dd}", "time": d[3],
                                "title": " ".join(str(d[4]).split()), "seq": seq}, ensure_ascii=False) + "\n")
    open(done_f, "a", encoding="utf-8").write(f"{c},{y}\n")
    n_rows += len(rows)
    if k % 100 == 0:
        el = time.time() - t0
        log(f"進度 {k}/{len(todo)}（{k / len(todo):.0%}），累計 {n_rows:,} 則，失敗 {fails}，預估剩 {el / k * (len(todo) - k) / 60:.0f} 分")
    time.sleep(0.8 + random.random() * 0.4)
log(f"結束：本次 {n_rows:,} 則，失敗 {fails}")
