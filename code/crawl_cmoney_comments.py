# -*- coding: utf-8 -*-
"""
補抓CMoney文章底下的留言（crawl_cmoney.py只抓了文章本身，沒抓留言內容）
============================================================

背景：
    crawl_cmoney.py第一輪只存了每篇文章的commentCount（留言「數量」），
    沒有存留言本身的文字內容。跟PTT一樣，很多真實的情緒訊號其實藏在
    留言裡，不是只有原文，所以要另外補這一塊。

API：
    GET /api/mach/api/Article/{articleId}/Comments?startCommentIndex={idx}&fetch=-20
    headers: X-Version: 3.0, Authorization: Bearer {token}
    回應：{comments: [...], remainCount: N, nextCommentIndex: N}
    remainCount>0代表還有更多留言，用nextCommentIndex接著抓下一批。

使用方式：
    先跑完 crawl_cmoney.py，產生 cmoney_267/*.csv（每篇文章的article_id）。
    python crawl_cmoney_comments.py --out-dir cmoney_267_comments
    （同樣支援中斷續傳、同樣的token定期更新機制，跟crawl_cmoney.py共用
    同一份 cmoney_token.json）
"""

import os
import re
import csv
import json
import time
import argparse
import glob
from datetime import datetime

import requests

ARTICLE_BASE = "https://www.cmoney.tw/api/mach/api/Article"
TOKEN_FILE = "cmoney_token.json"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(text):
    return _CONTROL_CHARS_RE.sub("", text or "")


def load_token():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["access_token"], data.get("expires_at")


def build_session():
    s = requests.Session()
    s.headers.update({
        # 注意：Comments端點要求的版本是2.0，不是文章列表端點用的3.0，
        # 兩個端點版本不一樣是實測確認的（用3.0會得到405 UnsupportedApiVersion）。
        "X-Version": "2.0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    return s


_token_state = {"mtime": None}


def refresh_token_if_changed(session):
    try:
        mtime = os.path.getmtime(TOKEN_FILE)
    except OSError:
        return
    if mtime != _token_state["mtime"]:
        token, expires_at = load_token()
        session.headers["Authorization"] = f"Bearer {token}"
        _token_state["mtime"] = mtime
        remaining = (expires_at - time.time()) / 3600 if expires_at else None
        print(f"    [token] 偵測到token檔案更新，已套用新token"
              + (f"（剩餘約{remaining:.1f}小時）" if remaining else ""))


class TokenExpired(Exception):
    pass


class FetchFailed(Exception):
    pass


def fetch_comments_page(session, article_id, start_index, retries=5):
    refresh_token_if_changed(session)
    url = f"{ARTICLE_BASE}/{article_id}/Comments"
    params = {"startCommentIndex": start_index, "fetch": -20}
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            wait = min(3 * (attempt + 1), 30)
            print(f"    [警告] 請求失敗（{attempt+1}/{retries}），{wait}秒後重試：{e}")
            time.sleep(wait)
            continue
        if resp.status_code == 401:
            print("    [Token過期] 等待重新登入...")
            waited = 0
            old_mtime = _token_state["mtime"]
            while waited < 3600:
                time.sleep(10)
                waited += 10
                try:
                    new_mtime = os.path.getmtime(TOKEN_FILE)
                except OSError:
                    continue
                if new_mtime != old_mtime:
                    refresh_token_if_changed(session)
                    break
            else:
                raise TokenExpired()
            continue
        if resp.status_code == 404:
            return {"comments": [], "remainCount": 0}  # 文章可能被刪除了，當作沒有留言
        if resp.status_code != 200:
            wait = min(3 * (attempt + 1), 30)
            print(f"    [警告] 非預期狀態碼 {resp.status_code}，{wait}秒後重試：{resp.text[:150]}")
            time.sleep(wait)
            continue
        return resp.json()
    raise FetchFailed(f"重試{retries}次後仍失敗：article={article_id}")


def load_article_ids(articles_dir):
    """讀取crawl_cmoney.py輸出的每檔股票CSV，收集所有(stock_code, stock_name, article_id)"""
    rows = []
    seen = set()
    for path in glob.glob(os.path.join(articles_dir, "*.csv")):
        base = os.path.basename(path)[:-4]
        code, _, name = base.partition("_")
        with open(path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                aid = row.get("article_id")
                if aid and aid not in seen:
                    seen.add(aid)
                    rows.append((code, name, aid))
    return rows


def crawl_comments_for_article(session, code, article_id, writer, delay):
    start_index = 0
    total = 0
    while True:
        data = fetch_comments_page(session, article_id, start_index)
        comments = data.get("comments", [])
        for c in comments:
            content = c.get("content", {})
            writer.writerow({
                "article_id": article_id,
                "comment_id": c.get("id"),
                "creatorId": c.get("creatorId"),
                "createTime": c.get("createTime"),
                "text": sanitize((content.get("text") or "").replace("\n", " ")),
                "likeCount": (c.get("emojiCount") or {}).get("like", 0),
            })
            total += 1
        remain = data.get("remainCount", 0)
        next_idx = data.get("nextCommentIndex")
        if remain <= 0 or next_idx is None or not comments:
            break
        start_index = next_idx
        time.sleep(delay)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="補抓CMoney文章的留言內容")
    parser.add_argument("--articles-dir", type=str, default="cmoney_267")
    parser.add_argument("--out-dir", type=str, default="cmoney_267_comments")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.json")
    out_path = os.path.join(args.out_dir, "all_comments.csv")

    print("讀取已抓文章的article_id清單...")
    all_articles = load_article_ids(args.articles_dir)
    print(f"共 {len(all_articles)} 篇文章需要抓留言")

    done_ids = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            done_ids = set(json.load(f).get("done_ids", []))
        print(f"偵測到中斷紀錄，已完成 {len(done_ids)} 篇，接續抓剩下的")

    fieldnames = ["article_id", "comment_id", "creatorId", "createTime", "text", "likeCount"]
    file_is_new = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if file_is_new:
        writer.writeheader()

    session = build_session()
    refresh_token_if_changed(session)

    total_comments = 0
    t0 = time.time()
    processed_this_run = 0
    try:
        for i, (code, name, article_id) in enumerate(all_articles, 1):
            if article_id in done_ids:
                continue
            try:
                n = crawl_comments_for_article(session, code, article_id, writer, args.delay)
            except FetchFailed as e:
                print(f"  [警告] {e}，跳過這篇，繼續下一篇（可事後重跑補漏）")
                continue
            total_comments += n
            done_ids.add(article_id)
            processed_this_run += 1
            f.flush()

            if processed_this_run % 200 == 0:
                with open(progress_path, "w", encoding="utf-8") as pf:
                    json.dump({"done_ids": list(done_ids)}, pf)
                elapsed = time.time() - t0
                rate = processed_this_run / elapsed * 60 if elapsed > 0 else 0
                remaining_n = len(all_articles) - len(done_ids)
                eta_min = remaining_n / rate if rate > 0 else 0
                print(f"    進度：已處理 {len(done_ids)}/{len(all_articles)} 篇文章，"
                      f"累積 {total_comments} 則留言，"
                      f"速度約 {rate:.1f} 篇/分鐘，預估剩餘 {eta_min/60:.1f} 小時")

            time.sleep(args.delay)
    except TokenExpired:
        print("\n[Token過期超過1小時] 請重新登入更新cmoney_token.json後重新執行同一指令接續")
    finally:
        f.close()
        with open(progress_path, "w", encoding="utf-8") as pf:
            json.dump({"done_ids": list(done_ids)}, pf)

    elapsed = time.time() - t0
    print(f"\n=== 本次執行結束，處理 {processed_this_run} 篇文章，"
          f"共存 {total_comments} 則留言，耗時 {elapsed/3600:.2f} 小時 ===")
    print(f"整體進度：{len(done_ids)}/{len(all_articles)} 篇文章已完成")
