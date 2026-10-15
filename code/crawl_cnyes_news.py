# -*- coding: utf-8 -*-
"""
抓鉅亨網「台股」分類新聞（媒體新聞；補公開資訊觀測站重大訊息沒有的記者報導、法人觀點）
  介面：GET https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?startAt=<unix>&endAt=<unix>&limit=30&page=N
  實測可回溯到 2015 年；每頁最多 30 則；時間區間太長會被截斷 → 以 3 天為一個視窗逐頁抓（超過 30 頁再切成單日）；分類用 tw_stock_news（台股新聞，不重複）。
  只存：newsId、publishAt(unix 秒)、title、keyword、stock、otherProduct、source、categoryName、summary（不存全文）
輸出：cnyes_news/{year}.jsonl、cnyes_news/done.txt（已完成的視窗起點）、cnyes_news/crawl.log
python crawl_cnyes_news.py [--part 0|1]     # --part：兩支並行時各抓奇/偶數視窗
可中斷續跑。
"""
import datetime as dt
import json
import os
import random
import sys
import time

import requests

OUT = "cnyes_news"
URL = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock_news"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
     "Origin": "https://news.cnyes.com", "Referer": "https://news.cnyes.com/"}
TZ = dt.timezone(dt.timedelta(hours=8))
START, END = dt.datetime(2015, 1, 1, tzinfo=TZ), dt.datetime(2026, 1, 1, tzinfo=TZ)
os.makedirs(OUT, exist_ok=True)
PART = int(sys.argv[sys.argv.index("--part") + 1]) if "--part" in sys.argv else None
wins, t = [], START
while t < END:
    wins.append((int(t.timestamp()), int(min(t + dt.timedelta(days=3), END).timestamp()) - 1))
    t += dt.timedelta(days=3)
done_f = os.path.join(OUT, "done.txt")
done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
todo = [(i, w) for i, w in enumerate(wins) if str(w[0]) not in done and (PART is None or i % 2 == PART)]


def log(s):
    line = time.strftime("%H:%M:%S ") + (f"[{PART}] " if PART is not None else "") + s
    print(line, flush=True)
    open(os.path.join(OUT, "crawl.log"), "a", encoding="utf-8").write(line + "\n")


log(f"開始：共 {len(wins)} 個視窗，剩 {len(todo)} 個")
sess, n_rows, n_req, fails, t0 = requests.Session(), 0, 0, 0, time.time()


def get_page(a, b, page):
    """回傳 (data, last_page)；失敗回 None"""
    global n_req
    for attempt in range(5):
        try:
            r = sess.get(URL, headers=H, params={"startAt": a, "endAt": b, "limit": 30, "page": page}, timeout=30)
            n_req += 1
            if r.status_code == 200:
                it = r.json()["items"]
                return it.get("data") or [], it.get("last_page") or 1
            wait = 90 * (attempt + 1)          # 實測約每分鐘 30 多次就會回 500（statusCode 4008）→ 放慢並久等
            log(f"  {a}~{b} 第 {page} 頁 HTTP {r.status_code}，等 {wait}s")
        except Exception as e:
            wait = 30 * (attempt + 1)
            log(f"  {a}~{b} 第 {page} 頁 例外 {type(e).__name__}，等 {wait}s")
        time.sleep(wait)
    return None


def fetch(a, b):
    """介面最多只給 30 頁（900 則）；超過就把時間區間對半切再抓"""
    x = get_page(a, b, 1)
    if x is None:
        return None
    data, last = x
    if last > 30 and b - a > 86400:          # 介面以「天」為單位篩選，最小只能切到一天
        mid = a + ((b - a + 1) // 86400 // 2 or 1) * 86400 - 1
        l, r = fetch(a, mid), fetch(mid + 1, b)
        return None if (l is None or r is None) else l + r
    rows = list(data)
    for page in range(2, min(last, 30) + 1):
        time.sleep(2.1 + random.random() * 0.6)
        x = get_page(a, b, page)
        if x is None:
            return None
        rows += x[0]
    time.sleep(2.1 + random.random() * 0.6)
    return rows


for k, (i, (a, b)) in enumerate(todo, 1):
    rows = fetch(a, b)
    ok = rows is not None
    if not ok:
        fails += 1
        if fails >= 8:
            log("失敗過多，停止（可重跑續抓）")
            break
        continue
    seen = set()
    year = dt.datetime.fromtimestamp(a, TZ).year
    with open(os.path.join(OUT, f"{year}.jsonl"), "a", encoding="utf-8") as f:
        for d in rows:
            if d["newsId"] in seen:
                continue
            seen.add(d["newsId"])
            f.write(json.dumps({"id": d["newsId"], "t": d["publishAt"], "title": " ".join(str(d.get("title", "")).split()),
                                "kw": d.get("keyword") or [], "stock": d.get("stock") or [], "other": d.get("otherProduct") or [],
                                "src": d.get("source") or "", "cat": d.get("categoryName") or "",
                                "summary": " ".join(str(d.get("summary") or "").split())[:200]}, ensure_ascii=False) + "\n")
    open(done_f, "a").write(f"{a}\n")
    n_rows += len(seen)
    if k % 20 == 0:
        el = time.time() - t0
        log(f"進度 {k}/{len(todo)}（{k / len(todo):.0%}），{n_req:,} 次請求、{n_rows:,} 則，失敗 {fails}，預估剩 {el / k * (len(todo) - k) / 60:.0f} 分")
log(f"結束：{n_req:,} 次請求、{n_rows:,} 則，失敗 {fails}")
