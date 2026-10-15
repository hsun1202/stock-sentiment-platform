# -*- coding: utf-8 -*-
"""
下載期交所「臺指期貨 TX 每日交易行情」（含一般 / 盤後 兩個交易時段），2017-05（夜盤開始）~2024-12，按月請求
  -> chip_raw/taifex_tx/YYYYMM.csv（UTF-8）
  期交所的「盤後」列歸在「次一個交易日」的日期下：該列是前一晚 15:00 ~ 當天 05:00 的夜盤（已確認：開盤價 = 前一個一般盤收盤）。
已存在的檔案會跳過，中斷後重跑同一指令即可接續。
python download_taifex_night.py
"""
import os
import time

import pandas as pd
import requests

URL = "https://www.taifex.com.tw/cht/3/futDataDown"
H = {"User-Agent": "Mozilla/5.0"}
OUT = "chip_raw/taifex_tx"
os.makedirs(OUT, exist_ok=True)
months = pd.period_range("2017-05", "2024-12", freq="M")
ok = bad = 0
for p in months:
    path = os.path.join(OUT, f"{p.strftime('%Y%m')}.csv")
    if os.path.exists(path):
        ok += 1
        continue
    start = f"{p.strftime('%Y/%m')}/01"
    end = (p.to_timestamp() + pd.offsets.MonthEnd(0)).strftime("%Y/%m/%d")
    txt = None
    for attempt in range(4):
        try:
            r = requests.post(URL, data={"down_type": "1", "commodity_id": "TX", "commodity_id2": "",
                                         "queryStartDate": start, "queryEndDate": end}, headers=H, timeout=60)
            t = r.content.decode("big5", errors="replace")
            if t.startswith("交易日期") and t.count("\n") > 5:
                txt = t
                break
            print(f"  [警告] {p} 回應格式異常（{len(t)} 字）重試 {attempt + 1}/4", flush=True)
        except Exception as e:
            print(f"  [警告] {p} {e.__class__.__name__} 重試 {attempt + 1}/4", flush=True)
        time.sleep(15 * (attempt + 1))
    if txt is None:
        bad += 1
        print(f"[失敗] {p}", flush=True)
        continue
    open(path, "w", encoding="utf-8").write(txt)
    ok += 1
    print(f"[ok] {p} {txt.count(chr(10))} 行", flush=True)
    time.sleep(2)
print(f"[DONE] 完成 {ok}/{len(months)}，失敗 {bad}", flush=True)
