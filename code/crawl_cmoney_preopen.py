# -*- coding: utf-8 -*-
"""
CMoney 補抓：熱門股「開盤前被擠掉」的股票日，各補抓一頁 ——「當天 09:00 之前最新的 20 篇」。
原爬蟲（crawl_cmoney_daily.py）熱門股每天只抓收盤前最後 20 篇，盤中熱的日子開盤前貼文全被擠掉，
造成「t+1 開盤前」特徵偷看到 t+1 盤中熱度（見 cmoney_full/xgb_leak_check.log）。補抓後每個股票日開盤前都至少有最新的一頁。
目標：抓到 ≥20 篇、且最舊一篇已在當天 09:00 之後的股票日（2026-10-19 統計約 14,004 個、152 檔）。
輸出：cmoney_daily_pre/{code}.jsonl（欄位同 cmoney_daily）、cmoney_daily_pre/done.txt（已完成的 code_日期，可續跑）、crawl.log
token：cmoney_token.json 過期時暫停，每 30 秒檢查，使用者更新後自動繼續（Claude 不讀取、不顯示 token 內容，只用來送請求）。
python cmoney_full/crawl_cmoney_preopen.py [--rate 15] [--workers 12]
"""
import argparse
import glob
import json
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

OUT = os.environ.get("PRE_OUT", "cmoney_daily_pre")          # 2025 年：PRE_SRC=cmoney_daily_2025 PRE_OUT=cmoney_daily_pre_2025 PRE_YEAR=2025
SRC = os.environ.get("PRE_SRC", "cmoney_daily")
PRE_YEAR = os.environ.get("PRE_YEAR")
BASE = "https://www.cmoney.tw/api/mach/api/Article/Stocks"
TOKEN_FILE = "cmoney_token.json"
K = 1048.576
ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=float, default=15.0)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--limit", type=int, default=0, help="測試用：只抓前 N 個")
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
lock = threading.Lock()


def log(msg):
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
    with lock:
        open(os.path.join(OUT, "crawl.log"), "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


# ---------- 目標股票日 ----------
if PRE_YEAR:
    cal = pd.Series(sorted(set(pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", usecols=["Date"]).Date)))
    days = [d for d in cal if f"{PRE_YEAR}-01-01" <= d <= f"{PRE_YEAR}-12-31"]
else:
    cal = pd.read_csv("market_sentiment/taiex_official.csv", usecols=["Date"]).Date
    days = [d for d in cal if "2020-01-01" <= d <= "2024-12-31"]
CLOSE = np.array([int(datetime.strptime(d + " 13:30", "%Y-%m-%d %H:%M").timestamp() * 1000) for d in days])
OPEN = np.array([int(datetime.strptime(d + " 09:00", "%Y-%m-%d %H:%M").timestamp() * 1000) for d in days])
targets = []
for f in sorted(glob.glob(f"{SRC}/*.jsonl")):
    code = os.path.basename(f)[:-6]
    t = np.array([json.loads(l)["t"] for l in open(f, encoding="utf-8")], dtype=np.int64)
    if len(t) == 0:
        continue
    i = np.searchsorted(CLOSE, t, side="left")
    ok = i < len(CLOSE)
    g = pd.DataFrame({"i": i[ok], "t": t[ok]}).groupby("i").agg(n=("t", "size"), tmin=("t", "min"))
    for di in g.index[(g.n >= 20) & (g.tmin.values > OPEN[g.index])]:
        targets.append((code, days[di], int(OPEN[di])))
done_path = os.path.join(OUT, "done.txt")
done = set(open(done_path, encoding="utf-8").read().split()) if os.path.exists(done_path) else set()
todo = [x for x in targets if f"{x[0]}_{x[1]}" not in done]
if args.limit:
    todo = todo[:args.limit]
log(f"=== 目標 {len(targets):,} 個股票日（{len({x[0] for x in targets})} 檔）；已完成 {len(done):,}；本次 {len(todo):,}｜限速 {args.rate}/s ===")


# ---------- token 與限速 ----------
class S:
    token, mtime, next_slot, rate, exp = None, None, 0.0, args.rate, 0
    n_req = n_art = n_err = 0
    waiting = False


def token():
    with lock:
        m = os.path.getmtime(TOKEN_FILE)
        if m != S.mtime:
            S.mtime = m
            try:
                d = json.load(open(TOKEN_FILE, encoding="utf-8-sig"))
                S.token, S.exp = d["access_token"], d.get("expires_at", 0)
            except (ValueError, KeyError):         # 檔案格式壞了（例如貼了兩次）→ 當作過期，等使用者修好
                S.exp = 0
                log("[token] cmoney_token.json 格式不正確（要只有一段 {...}），請重新存檔")
        return S.token


def wait_valid_token():
    token()
    if S.exp > time.time() + 60:
        return
    if not S.waiting:
        S.waiting = True
        log("[token] 已過期，暫停；請更新 cmoney_token.json（每 30 秒檢查一次）")
    old = S.mtime
    while True:
        time.sleep(30)
        token()
        if S.mtime != old and S.exp > time.time() + 60:
            break
    S.waiting = False
    log(f"[token] 已更新（剩餘約 {(S.exp - time.time()) / 3600:.1f} 小時），繼續")


def throttle():
    with lock:
        now = time.time()
        slot = max(now, S.next_slot)
        S.next_slot = slot + 1.0 / S.rate
    time.sleep(max(0, slot - time.time()))


def fetch(sess, code, cursor):
    back = 2.0
    while True:
        wait_valid_token()
        throttle()
        try:
            r = sess.get(f"{BASE}/{code}/AllLatest", params={"limit": 20, "cursor": cursor}, timeout=30,
                         headers={"Authorization": f"Bearer {token()}"})
        except requests.RequestException:
            S.n_err += 1
            time.sleep(back)
            back = min(back * 2, 300)
            continue
        S.n_req += 1
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass
        elif r.status_code == 401:
            S.exp = 0                      # 視為過期，進入等待
            continue
        else:
            S.n_err += 1
            if r.status_code in (429, 403):
                S.rate = max(1.0, S.rate / 2)
            log(f"[{code}] HTTP {r.status_code}，{back:.0f} 秒後重試（限速 {S.rate:.1f}/s）")
        time.sleep(back)
        back = min(back * 2, 300)


def worker(q):
    sess = requests.Session()
    sess.headers.update({"X-Version": "3.0", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    while True:
        try:
            code, day, open_ms = q.get_nowait()
        except queue.Empty:
            return
        try:
            data = fetch(sess, code, int(open_ms * K))
            rows = []
            for a in data.get("articles", []):
                ct = a.get("createTime", 0)
                if ct >= open_ms:          # 只要開盤前的
                    continue
                c = a.get("content") or {}
                tag = next((t for t in c.get("commodityTags", []) if t.get("key") == code), None)
                rows.append(json.dumps({"id": a.get("id"), "t": ct, "creator": a.get("creatorId"), "title": (c.get("title") or "")[:200],
                                        "text": (c.get("text") or "")[:600], "comments": a.get("commentCount", 0),
                                        "bb": (tag or {}).get("bullOrBear"), "fill_day": day}, ensure_ascii=False))
            with lock:
                if rows:
                    open(os.path.join(OUT, f"{code}.jsonl"), "a", encoding="utf-8").write("\n".join(rows) + "\n")
                open(done_path, "a", encoding="utf-8").write(f"{code}_{day}\n")
                S.n_art += len(rows)
        except Exception as e:
            log(f"[{code} {day}] 例外 {type(e).__name__}: {str(e)[:120]}（下次重跑會再抓）")


q = queue.Queue()
for x in todo:
    q.put(x)
th = [threading.Thread(target=worker, args=(q,), daemon=True) for _ in range(args.workers)]
for t in th:
    t.start()
t0 = time.time()
while any(t.is_alive() for t in th):
    time.sleep(30)
    log(f"進度：剩 {q.qsize():,}/{len(todo):,}｜請求 {S.n_req:,}｜補到 {S.n_art:,} 篇｜錯誤 {S.n_err}｜{S.n_req / max(1, time.time() - t0):.1f}/s"
        + ("｜等待 token" if S.waiting else ""))
nd = len(set(open(done_path, encoding="utf-8").read().split())) if os.path.exists(done_path) else 0
log(f"=== 結束：完成 {nd:,}/{len(targets):,} ===" + ("（全部完成）" if nd >= len(targets) else "（未完成，可重跑續抓）"))
