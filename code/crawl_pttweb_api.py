# -*- coding: utf-8 -*-
"""
PTT Stock板 鏡像站(pttweb.cc)爬蟲：補 PTT 本站已被清掉的舊文章（2015-2019初）
============================================================

為什麼要爬鏡像站：
    PTT 看板有文章數上限，超過時會自動刪掉最舊、沒被板主標記(M)的文章。
    2026-09 實測 Stock 板 2019-03 以前幾乎只剩 [公告]/盤中閒聊串（2015-2018 四年只剩約2,700篇），
    一般討論文已經不在 PTT 上。pttweb.cc 當年有即時備份，文章與留言都還在。

資料來源：pttweb.cc 前端自己用的 twirp JSON API（逆向自其前端JS，不是用猜的）
    列表：POST /twirp/e7pttV1.E7Service/GetPttBoard
          {"pttBoardName":"Stock","paginationOffset":N,"resultCountLimit":100}
          → 回傳 sort_order 在 [N-100, N-1] 的文章（sort_order 是板上文章流水號，隨時間遞增）
    文章：POST /twirp/e7pttV1.E7Service/GetPttArticle
          {"pttBoardName":"Stock","articleAid":"M.xxxx.A.xxx"}
          → 內文(main_text_data)、留言(recommend_data：type +/-/空白、author、text、ip_datetime)

兩階段＋中斷續傳：
    階段一（list）：翻列表，把日期範圍內的文章記到 articles.csv
    階段二（fetch）：逐篇抓文章JSON，原封不動存成 raw_articles.jsonl（之後再轉格式，不怕解析寫錯要重爬）

使用方式：
    python crawl_pttweb_api.py --start 2015-01-01 --end 2019-03-31 --out-dir pttweb_stock_2015_2019
    # 只補PTT本站已刪的文章（列表仍記全部；階段二跳過 --skip-ids 檔案裡已有PTT本站資料的文章ID）
    python crawl_pttweb_api.py --start 2019-04-01 --end 2024-12-31 --out-dir pttweb_stock_2019_2024_deleted \
        --skip-ids pttweb_stock_2019_2024_deleted/ptt_existing_ids.txt
"""

import os
import csv
import json
import time
import argparse
from datetime import datetime, date

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://www.pttweb.cc/twirp/e7pttV1.E7Service/"
BOARD = "Stock"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
}


def build_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(total=5, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


SESSION = build_session()


def call(method, payload, retries=4):
    last = None
    for attempt in range(retries):
        try:
            r = SESSION.post(API + method, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            wait = 5 * (attempt + 1)
            print(f"    [警告] {method} 失敗 ({attempt+1}/{retries})：{e}，{wait}秒後重試")
            time.sleep(wait)
    print(f"    [失敗] {method} 放棄：{payload}（{last}）")
    return None


def aid_date(aid_time_ns):
    return datetime.fromtimestamp(int(aid_time_ns) // 10**9).date()


PAGE = 100  # 實測每頁500篇一次要81秒、100篇只要5秒（伺服器端非線性變慢），所以用100


def list_page(offset, limit=PAGE):
    d = call("GetPttBoard", {"pttBoardName": BOARD, "paginationOffset": offset, "resultCountLimit": limit})
    if d is None:
        return None
    return (d.get("ptt_board") or {}).get("articles", [])


def find_sort_order(target, hi_bound):
    """二分搜尋：第一篇發文日期 >= target 的 sort_order（近似即可，後面會用日期再過濾）。"""
    lo, hi = 1, hi_bound
    while hi - lo > 50:
        mid = (lo + hi) // 2
        arts = list_page(mid, 1)
        time.sleep(0.5)
        if not arts:
            lo = mid
            continue
        if aid_date(arts[0]["aid_time"]) < target:
            lo = mid
        else:
            hi = mid
    return lo


def phase_list(out_dir, start, end, delay):
    path = os.path.join(out_dir, "articles.csv")
    done_marker = os.path.join(out_dir, "list_done.marker")
    if os.path.exists(done_marker):
        print("[階段一] 已完成，跳過")
        return
    board = call("GetPttBoard", {"pttBoardName": BOARD, "paginationOffset": 1, "resultCountLimit": 1})
    max_so = int(board["ptt_board"]["max_article_sort_order"])
    so_start = max(1, find_sort_order(start, max_so) - PAGE)
    so_end = find_sort_order(date.fromordinal(end.toordinal() + 1), max_so) + PAGE
    print(f"[階段一] sort_order 範圍約 {so_start} ~ {so_end}（最大 {max_so}）")

    fields = ["article_aid", "pttweb_article_id", "sort_order", "aid_time", "date", "author",
              "title", "is_marked", "recommend_up_count", "recommend_down_count",
              "recommend_neutral_count", "recommend_total_count"]
    seen = {}
    offset = so_start + PAGE
    # 列表續傳：每頁的結果即時寫進 list_partial.jsonl，progress 記下一個 offset
    partial_path = os.path.join(out_dir, "list_partial.jsonl")
    prog_path = os.path.join(out_dir, "list_progress.json")
    if os.path.exists(prog_path) and os.path.exists(partial_path):
        with open(prog_path, encoding="utf-8") as f:
            prog = json.load(f)
        so_start, so_end, offset = prog["so_start"], prog["so_end"], prog["next_offset"]
        with open(partial_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen[r["article_aid"]] = r
                except ValueError:
                    pass
        print(f"[階段一-續傳] 從 offset {offset} 接續，已記錄 {len(seen)} 篇")
    pf = open(partial_path, "a", encoding="utf-8")
    while offset - PAGE <= so_end:
        arts = list_page(offset)
        if arts is None:
            raise RuntimeError(f"列表抓取失敗 offset={offset}，重新執行即可")
        for a in arts:
            d = aid_date(a["aid_time"])
            if start <= d <= end:
                row = {
                    "article_aid": a["article_aid"], "pttweb_article_id": a.get("article_id", ""),
                    "sort_order": a.get("sort_order", ""), "aid_time": a["aid_time"],
                    "date": d.isoformat(), "author": a.get("author", ""), "title": a.get("title", ""),
                    "is_marked": a.get("is_marked", False),
                    "recommend_up_count": a.get("recommend_up_count", 0),
                    "recommend_down_count": a.get("recommend_down_count", 0),
                    "recommend_neutral_count": a.get("recommend_neutral_count", 0),
                    "recommend_total_count": a.get("recommend_total_count", 0),
                }
                seen[a["article_aid"]] = row
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
        pf.flush()
        with open(prog_path, "w", encoding="utf-8") as f:
            json.dump({"so_start": so_start, "so_end": so_end, "next_offset": offset + PAGE}, f)
        if (offset // PAGE) % 20 == 0:
            print(f"    [列表] offset {offset}/{so_end}：本頁 {len(arts)} 篇，範圍內累計 {len(seen)} 篇")
        offset += PAGE
        time.sleep(delay)
    pf.close()

    rows = sorted(seen.values(), key=lambda r: int(r["sort_order"] or 0))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write(f"completed at {datetime.now().isoformat()}, {len(rows)} articles\n")
    print(f"[階段一完成] {start}~{end} 共 {len(rows)} 篇 → {path}")


def phase_fetch(out_dir, delay, skip_ids_path=None):
    with open(os.path.join(out_dir, "articles.csv"), encoding="utf-8-sig") as f:
        arts = list(csv.DictReader(f))
    if skip_ids_path:
        with open(skip_ids_path, encoding="utf-8") as f:
            skip = {line.strip() for line in f if line.strip()}
        before = len(arts)
        arts = [a for a in arts if a["article_aid"] not in skip]
        print(f"[階段二] 列表 {before} 篇，其中 {before-len(arts)} 篇已在PTT本站資料中（跳過），只抓 {len(arts)} 篇")
    raw_path = os.path.join(out_dir, "raw_articles.jsonl")
    fail_path = os.path.join(out_dir, "fetch_failed.txt")
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["_aid"])
                except (ValueError, KeyError):
                    pass  # 中斷時寫到一半的最後一行，跳過即可（會重抓）
    todo = [a for a in arts if a["article_aid"] not in done]
    print(f"[階段二] 共 {len(arts)} 篇，已抓 {len(done)}，待抓 {len(todo)}")

    t0 = time.time()
    with open(raw_path, "a", encoding="utf-8") as out, open(fail_path, "a", encoding="utf-8") as ff:
        for i, a in enumerate(todo, 1):
            d = call("GetPttArticle", {"pttBoardName": BOARD, "articleAid": a["article_aid"]})
            if d is None or "article" not in d:
                ff.write(a["article_aid"] + "\n")
                ff.flush()
            else:
                d["_aid"] = a["article_aid"]
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                out.flush()
            if i % 200 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 3600
                print(f"    [階段二進度] {i}/{len(todo)}（{a['date']}），{rate:.2f}篇/秒，預估剩 {eta:.1f} 小時")
            time.sleep(delay)
    print("[階段二完成]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="pttweb.cc Stock板鏡像爬蟲（補舊文章）")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2019-03-31")
    p.add_argument("--out-dir", default="pttweb_stock_2015_2019")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--phase", default="both", choices=["list", "fetch", "both"])
    p.add_argument("--skip-ids", default=None, help="文字檔，每行一個文章ID；階段二不抓這些（已有PTT本站資料）")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    print(f"=== pttweb 鏡像爬蟲：{start} ~ {end} → {args.out_dir}/ ===")
    if args.phase in ("list", "both"):
        phase_list(args.out_dir, start, end, args.delay)
    if args.phase in ("fetch", "both"):
        phase_fetch(args.out_dir, args.delay, args.skip_ids)
