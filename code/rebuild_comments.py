# -*- coding: utf-8 -*-
"""
留言資料重建工具
============================================================

背景：
    先前版本的自我修復機制有個bug，誤把 comments.csv 用「link去重」
    處理，導致同一篇文章底下的多則留言被錯誤地只保留一則，
    造成留言資料大量遺失。posts.csv 沒有受影響（依然完整正確）。

做法：
    不去修補原本複雜的階段二判斷邏輯，而是直接寫一支專用工具：
    拿每個股票完好的 posts.csv 當作「文章清單」，重新逐篇抓取
    留言，重建乾淨的 comments.csv。這樣完全不會動到posts.csv，
    也不需要理解階段二原本的複雜跳過邏輯，最安全直接。

    支援斷點續傳：每個股票有自己的進度紀錄檔（記錄posts.csv裡
    處理到第幾列），中斷後重新執行會從該股票中斷的地方接續，
    不會重新抓已經抓過的文章留言。

使用方式：
    python rebuild_comments.py --out-dir batch_output --delay 1.0
"""

import time
import os
import csv
import re
import argparse
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "stock_volume_scanner",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
)
_svs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_svs)
STOCK_LIST = _svs.STOCK_LIST

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}
COOKIES = {"over18": "1"}


def build_session():
    session = requests.Session()
    session.cookies.update(COOKIES)
    session.headers.update(HEADERS)
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = build_session()


def fetch_html(url, retries=3, sleep_base=2.0):
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            print(f"    [警告] 抓取失敗 ({attempt+1}/{retries}): {e}")
            time.sleep(sleep_base * (attempt + 1))
    return None


def parse_comments_only(html):
    """只解析留言部分（文章本身的metadata不需要，posts.csv已經有了）"""
    soup = BeautifulSoup(html, "html.parser")
    main_content = soup.select_one("#main-content")
    if not main_content:
        return []
    pushes = []
    for push in main_content.select("div.push"):
        tag_span = push.select_one("span.push-tag")
        content_span = push.select_one("span.push-content")
        if tag_span and content_span:
            pushes.append({
                "type": tag_span.text.strip(),
                "content": content_span.text.lstrip(": ").strip(),
            })
    return pushes


def rebuild_one_stock(code, name, out_dir, delay):
    posts_path = os.path.join(out_dir, f"{code}_{name}_posts.csv")
    comments_path = os.path.join(out_dir, f"{code}_{name}_comments.csv")
    progress_path = os.path.join(out_dir, f"{code}_{name}.rebuild_progress")

    if not os.path.exists(posts_path):
        print(f"[跳過] {code} {name}：找不到posts.csv")
        return

    with open(posts_path, "r", encoding="utf-8-sig") as f:
        posts = list(csv.DictReader(f))

    if not posts:
        print(f"[跳過] {code} {name}：posts.csv是空的")
        return

    start_idx = 0
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            try:
                start_idx = int(f.read().strip())
            except ValueError:
                start_idx = 0

    if start_idx >= len(posts):
        print(f"[完成] {code} {name}：已全部重建完成（{len(posts)}篇），跳過")
        return

    print(f"[重建] {code} {name}：共{len(posts)}篇，從第{start_idx+1}篇開始")

    comments_fieldnames = ["stock_code", "link", "resolved_date", "push_type", "content"]
    file_is_new = not os.path.exists(comments_path) or start_idx == 0
    mode = "w" if file_is_new else "a"
    cf = open(comments_path, mode, newline="", encoding="utf-8-sig")
    cw = csv.DictWriter(cf, fieldnames=comments_fieldnames)
    if file_is_new:
        cw.writeheader()

    try:
        for i in range(start_idx, len(posts)):
            row = posts[i]
            html = fetch_html(row["link"])
            time.sleep(delay)
            if html:
                pushes = parse_comments_only(html)
                for p in pushes:
                    cw.writerow({
                        "stock_code": code, "link": row["link"],
                        "resolved_date": row.get("resolved_date", ""),
                        "push_type": p["type"], "content": p["content"],
                    })
                cf.flush()

            with open(progress_path, "w", encoding="utf-8") as pf:
                pf.write(str(i + 1))

            if (i + 1) % 100 == 0:
                print(f"    [進度] {code} {name}：{i+1}/{len(posts)} 篇")
    finally:
        cf.close()

    print(f"[完成] {code} {name}：重建完成，共{len(posts)}篇")
    if os.path.exists(progress_path):
        os.remove(progress_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="留言資料重建工具")
    parser.add_argument("--out-dir", type=str, default="batch_output")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    print(f"=== 開始重建 {len(STOCK_LIST)} 檔股票的留言資料 ===\n")
    total_start = time.time()

    for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
        print(f"\n[{i}/{len(STOCK_LIST)}]", end=" ")
        try:
            rebuild_one_stock(code, name, args.out_dir, args.delay)
        except Exception as e:
            print(f"[錯誤] {code} {name}：{e}")
            print(f"[提示] 重新執行同一指令會自動從中斷處接續")
            continue

    total_elapsed = time.time() - total_start
    print(f"\n=== 重建作業結束，總耗時 {total_elapsed/3600:.2f} 小時 ===")
