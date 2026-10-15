# -*- coding: utf-8 -*-
"""
CMoney 同學會可行性測試（小規模，約 200 次請求）：直接跳到指定交易日的收盤時間，只抓收盤前最後一頁。
  要確認：1) 2015 年起抓不抓得到、每頁幾篇  2) 回傳文章的時間是否都在指定時間之前、跨了多久
          3) 冷門股有沒有資料  4) 作者自己標的多空（bullOrBear）有多少比例
需要 cmoney_token.json 內有未過期的 access_token（由使用者登入後更新）。
輸出：cmoney_probe.log、cmoney_probe.csv
python cmoney_probe.py
"""
import json
import time
from datetime import datetime

import pandas as pd
import requests

BASE = "https://www.cmoney.tw/api/mach/api/Article/Stocks"
STOCKS = [("2330", "台積電"), ("2317", "鴻海"), ("2303", "聯電"), ("2603", "長榮"), ("2002", "中鋼"),
          ("2409", "友達"), ("1101", "台泥"), ("2059", "川湖"), ("1717", "長興"), ("1339", "昭輝")]
DATES = ["2015-03-17", "2015-09-15", "2017-03-15", "2017-09-13", "2019-03-13", "2019-09-11",
         "2021-03-17", "2021-09-15", "2023-03-15", "2023-09-13"]
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


tok = json.load(open("cmoney_token.json", encoding="utf-8-sig"))
if tok.get("expires_at") and tok["expires_at"] < time.time():
    raise SystemExit("cmoney_token.json 的 token 已過期，請重新登入 CMoney 後更新")
s = requests.Session()
s.headers.update({"X-Version": "3.0", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                  "Authorization": f"Bearer {tok['access_token']}"})
rows = []
for code, name in STOCKS:
    for d in DATES:
        close_ms = int(datetime.strptime(d + " 13:30", "%Y-%m-%d %H:%M").timestamp() * 1000)
        cursor = int(close_ms * 1048.576)
        r = s.get(f"{BASE}/{code}/AllLatest", params={"limit": 20, "cursor": cursor}, timeout=20)
        if r.status_code == 401:
            raise SystemExit("401：token 無效或過期")
        row = {"code": code, "name": name, "date": d, "status": r.status_code, "n": 0}
        if r.status_code == 200:
            arts = r.json().get("articles", [])
            ts = [a.get("createTime", 0) for a in arts]
            row["n"] = len(arts)
            if ts:
                row["newest"] = datetime.fromtimestamp(max(ts) / 1000).strftime("%Y-%m-%d %H:%M")
                row["oldest"] = datetime.fromtimestamp(min(ts) / 1000).strftime("%Y-%m-%d %H:%M")
                row["after_cursor"] = sum(t > close_ms for t in ts)
                row["same_day"] = sum(datetime.fromtimestamp(t / 1000).strftime("%Y-%m-%d") == d for t in ts)
                row["span_hours"] = round((max(ts) - min(ts)) / 3.6e6, 1)
                bb = []
                for a in arts:
                    tag = next((t for t in (a.get("content") or {}).get("commodityTags", []) if t.get("key") == code), None)
                    bb.append((tag or {}).get("bullOrBear"))
                row["bull"] = sum(str(b) in ("1", "Bull", "bull") for b in bb)
                row["bear"] = sum(str(b) in ("-1", "2", "Bear", "bear") for b in bb)
                row["bb_values"] = json.dumps(sorted({str(b) for b in bb}), ensure_ascii=False)
                row["med_len"] = int(pd.Series([len(((a.get("content") or {}).get("text") or "")) for a in arts]).median())
        rows.append(row)
        time.sleep(1.0)
R = pd.DataFrame(rows)
R.to_csv("cmoney_probe.csv", index=False, encoding="utf-8-sig")
say(f"請求 {len(R)} 次；非 200：{(R.status != 200).sum()}")
say("\n每檔 × 每年：一頁抓到幾篇／其中當天的幾篇／一頁涵蓋幾小時（兩個抽樣日的平均）")
R["year"] = R.date.str[:4]
for code, name in STOCKS:
    g = R[R.code == code].groupby("year").agg(n=("n", "mean"), same=("same_day", "mean"), span=("span_hours", "mean"))
    say(f"  {code} {name:<4s} " + "｜".join(f"{y}: {v.n:.0f}篇/當天{0 if pd.isna(v.same) else v.same:.0f}/跨{0 if pd.isna(v.span) else v.span:.0f}h" for y, v in g.iterrows()))
if "after_cursor" in R:
    say(f"\n時間檢查：回傳文章晚於指定時間的共 {int(R.after_cursor.fillna(0).sum())} 篇（應為 0）")
    say(f"作者自標多空的取值：{sorted(set(R.bb_values.dropna()))}")
    say(f"內文長度中位數（各頁中位數的中位數）：{R.med_len.median():.0f} 字")
open("cmoney_probe.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
