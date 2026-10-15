# -*- coding: utf-8 -*-
"""
CMoney 同學會「每個交易日收盤前」抽樣爬蟲（2020-01-01 ~ 2024-12-31，資料集內全部股票）
  作法：每檔股票從 2024 年底往回走。目前位置 T（時間）：
        抓「T 之前最新的一頁（20 篇）」→ 若這頁最舊的文章還沒碰到上一個交易日收盤，就直接跳到上一個交易日收盤（熱門股：每天一頁）；
        否則從這頁最舊的文章接著往回翻（冷門股：連續翻頁，不漏文章）。
  cursor ＝ 毫秒時間 × 1048.576（2026-09-19 實測；2026-10-04 再驗證：回傳文章皆不晚於指定時間）
  需要 cmoney_token.json（使用者登入後自行更新）；401 時暫停並每 30 秒檢查檔案是否更新，不會漏資料。
  可中斷續跑：每檔有 checkpoint；完成的有 .done。
輸出：cmoney_daily/{code}.jsonl（id, t, creator, title, text, comments, bb）、cmoney_daily/crawl.log、cmoney_daily/progress.json
python crawl_cmoney_daily.py [--rate 10] [--workers 8]
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime

import pandas as pd
import requests

from stock_list_267 import STOCK_LIST_267

OUT = "cmoney_daily"
BASE = "https://www.cmoney.tw/api/mach/api/Article/Stocks"
TOKEN_FILE = "cmoney_token.json"
START_MS = int(datetime(2020, 1, 1).timestamp() * 1000)
os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=float, default=10.0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--only", default=None)
args = ap.parse_args()

log_lock = threading.Lock()


def log(msg):
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
    with log_lock:
        with open(os.path.join(OUT, "crawl.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except Exception:
            pass


# ---------- 交易日收盤時間 ----------
cal = pd.read_csv("market_sentiment/taiex_official.csv", usecols=["Date"]).Date
CLOSES = sorted(int(datetime.strptime(d + " 13:30", "%Y-%m-%d %H:%M").timestamp() * 1000) for d in cal if "2020-01-01" <= d <= "2024-12-31")


def prev_close(t_ms):
    """嚴格早於 t_ms 的最後一個交易日收盤；沒有則 None"""
    lo, hi = 0, len(CLOSES)
    while lo < hi:
        mid = (lo + hi) // 2
        if CLOSES[mid] < t_ms:
            lo = mid + 1
        else:
            hi = mid
    return CLOSES[lo - 1] if lo > 0 else None


# ---------- token 與限速 ----------
class State:
    token = None
    mtime = None
    rate = args.rate
    next_slot = 0.0
    paused_401 = False
    n_req = 0
    n_art = 0
    n_err = 0
    n_429 = 0


st = State()
tok_lock, rate_lock, cnt_lock = threading.Lock(), threading.Lock(), threading.Lock()


def load_token(force=False):
    with tok_lock:
        m = os.path.getmtime(TOKEN_FILE)
        if force or m != st.mtime:
            d = json.load(open(TOKEN_FILE, encoding="utf-8-sig"))
            st.token, st.mtime = d["access_token"], m
            left = (d.get("expires_at", 0) - time.time()) / 3600
            log(f"[token] 已載入（剩餘約 {left:.1f} 小時）")
        return st.token


def throttle():
    with rate_lock:
        now = time.time()
        slot = max(now, st.next_slot)
        st.next_slot = slot + 1.0 / st.rate
    time.sleep(max(0, slot - time.time()))


def fetch(sess, code, cursor):
    back = 2.0
    while True:
        throttle()
        try:
            r = sess.get(f"{BASE}/{code}/AllLatest", params={"limit": 20, "cursor": cursor}, timeout=30,
                         headers={"Authorization": f"Bearer {load_token()}"})
        except requests.RequestException:
            with cnt_lock:
                st.n_err += 1
            time.sleep(back)
            back = min(back * 2, 300)
            continue
        with cnt_lock:
            st.n_req += 1
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                time.sleep(back)
                back = min(back * 2, 300)
                continue
        if r.status_code == 401:
            if not st.paused_401:
                st.paused_401 = True
                log("[token] 401：token 過期，暫停，等待 cmoney_token.json 更新（每 30 秒檢查一次）")
            old = st.mtime
            while os.path.getmtime(TOKEN_FILE) == old:
                time.sleep(30)
            load_token()
            st.paused_401 = False
            continue
        with cnt_lock:
            st.n_err += 1
            if r.status_code in (429, 403):
                st.n_429 += 1
                st.rate = max(1.0, st.rate / 2)
        log(f"[{code}] HTTP {r.status_code}，{back:.0f} 秒後重試（目前限速 {st.rate:.1f}/s）")
        time.sleep(back)
        back = min(back * 2, 300)


def crawl(code, name):
    done, ckpt, path = (os.path.join(OUT, f"{code}.{e}") for e in ("done", "ckpt", "jsonl"))
    if os.path.exists(done):
        return
    K = 1048.576
    C = int(CLOSES[-1] * K)                      # 目前位置（cursor）；cursor ≈ 文章時間(毫秒) × 1048.576，但少數文章的 createTime 與 cursor 不一致，
    if os.path.exists(ckpt):                     # 所以「接著往回翻」一律用 API 回傳的 nextCursor，只有「跳到某天收盤」才用時間換算
        C = json.load(open(ckpt))["C"]
    sess = requests.Session()
    sess.headers.update({"X-Version": "3.0", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    n_pages = n_saved = 0
    with open(path, "a", encoding="utf-8") as f:
        while C / K >= START_MS:
            data = fetch(sess, code, C)
            arts = data.get("articles", [])
            nxt = data.get("nextCursor")
            for a in arts:
                ct = a.get("createTime", 0)
                if ct < START_MS:
                    continue
                c = a.get("content") or {}
                tag = next((t for t in c.get("commodityTags", []) if t.get("key") == code), None)
                f.write(json.dumps({"id": a.get("id"), "t": ct, "creator": a.get("creatorId"), "title": (c.get("title") or "")[:200],
                                    "text": (c.get("text") or "")[:600], "comments": a.get("commentCount", 0),
                                    "bb": (tag or {}).get("bullOrBear")}, ensure_ascii=False) + chr(10))
                n_saved += 1
            f.flush()
            n_pages += 1
            with cnt_lock:
                st.n_art += len(arts)
            if not nxt or (not arts and not data.get("hasNext")):
                break
            if nxt >= C:                          # 沒有前進：強制往回 1 秒，避免卡住
                nxt = C - 1048576
            pc = prev_close(C / K)                # 目前位置之前的上一個交易日收盤
            C = int(pc * K) if (pc is not None and nxt / K > pc) else int(nxt)
            with open(ckpt, "w") as k:
                json.dump({"C": C}, k)
    open(done, "w").write(datetime.now().isoformat())
    if os.path.exists(ckpt):
        os.remove(ckpt)
    log(f"[{code} {name}] 完成：本次 {n_pages} 頁、{n_saved} 篇")


def worker(q):
    while True:
        try:
            code, name = q.get_nowait()
        except queue.Empty:
            return
        try:
            crawl(code, name)
        except Exception as e:  # 單一股票出錯不影響其他；checkpoint 保留，可重跑接續
            log(f"[{code} {name}] 例外 {type(e).__name__}: {str(e)[:120]}（保留進度）")


if __name__ == "__main__":
    stocks = [(c, n) for c, n, _ in STOCK_LIST_267]
    if args.only:
        stocks = [x for x in stocks if x[0] in set(args.only.split(","))]
    todo = [x for x in stocks if not os.path.exists(os.path.join(OUT, f"{x[0]}.done"))]
    load_token(force=True)
    log(f"=== 開始：{len(todo)}/{len(stocks)} 檔待爬｜交易日 {len(CLOSES)}｜限速 {st.rate}/s｜workers {args.workers} ===")
    q = queue.Queue()
    for x in todo:
        q.put(x)
    th = [threading.Thread(target=worker, args=(q,), daemon=True) for _ in range(args.workers)]
    for t in th:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in th):
        time.sleep(60)
        nd = sum(os.path.exists(os.path.join(OUT, f"{c}.done")) for c, _ in stocks)
        prog = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stocks_done": nd, "stocks_total": len(stocks), "requests": st.n_req,
                "articles": st.n_art, "errors": st.n_err, "rate_limited": st.n_429, "rate": st.rate, "paused_for_token": st.paused_401,
                "req_per_s": round(st.n_req / (time.time() - t0), 2)}
        json.dump(prog, open(os.path.join(OUT, "progress.json"), "w", encoding="utf-8"), ensure_ascii=False)
        log(f"進度 {nd}/{len(stocks)} 檔｜請求 {st.n_req:,}｜文章 {st.n_art:,}｜錯誤 {st.n_err}｜{prog['req_per_s']}/s" + ("｜等待 token" if st.paused_401 else ""))
    log("=== 全部完成 ===")
