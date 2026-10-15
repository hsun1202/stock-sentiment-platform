# -*- coding: utf-8 -*-
"""
查詢PTT Stock板全板爬蟲（crawl_ptt_stock_full.py）的即時進度
============================================================
使用方式：
    python check_crawl_full_progress.py
    python check_crawl_full_progress.py --out-dir ptt_stock_full_2020_2024
"""

import os
import csv
import json
import argparse
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="ptt_stock_full_2020_2024")
    args = parser.parse_args()
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        print(f"輸出資料夾 {out_dir}/ 不存在，任務還沒開始跑")
        raise SystemExit(0)

    print(f"=== PTT Stock板全板爬蟲進度（{out_dir}/）===\n")

    # ---------- 階段一：掃描 ----------
    articles_path = os.path.join(out_dir, "all_articles.csv")
    checkpoint_path = os.path.join(out_dir, "scan.checkpoint")
    done_marker_path = os.path.join(out_dir, "scan_done.marker")

    article_count = 0
    if os.path.exists(articles_path):
        with open(articles_path, "r", encoding="utf-8-sig") as f:
            article_count = sum(1 for _ in csv.DictReader(f))

    print("[階段一：掃描列表頁]")
    if os.path.exists(done_marker_path):
        print(f"  狀態：已完整掃描完成")
    elif os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            page = f.read().strip()
        print(f"  狀態：進行中，目前掃描到第 {page} 頁（會持續往舊頁碼走，直到掃到起始日期為止）")
    else:
        print(f"  狀態：尚未開始，或本次執行尚未產生checkpoint")
    print(f"  目前累積記錄文章數：{article_count}\n")

    # ---------- 階段二：抓取內文與留言 ----------
    posts_path = os.path.join(out_dir, "all_posts.csv")
    comments_path = os.path.join(out_dir, "all_comments.csv")
    progress_path = os.path.join(out_dir, "fetch_progress.json")

    total_posts = 0
    if os.path.exists(posts_path):
        with open(posts_path, "r", encoding="utf-8-sig") as f:
            total_posts = sum(1 for _ in csv.DictReader(f))

    total_comments = 0
    if os.path.exists(comments_path):
        with open(comments_path, "r", encoding="utf-8-sig") as f:
            total_comments = sum(1 for _ in csv.DictReader(f))

    print("[階段二：抓取內文與留言]")
    print(f"  累積已抓取文章數：{total_posts}（含使用者ID的留言共 {total_comments} 則）")

    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            p = json.load(f)
        total_done = p["already_done_before_this_run"] + p["done_this_run"]
        pct = total_done / p["total_articles"] * 100 if p["total_articles"] else 0
        print(f"  本次任務目標：{p['total_articles']} 篇（{total_done}/{p['total_articles']}，{pct:.1f}%）")
        print(f"  最後更新時間：{p['updated_at']}")
        print(f"  本次執行開始時間：{p['started_at']}")

        try:
            started = datetime.fromisoformat(p["started_at"])
            updated = datetime.fromisoformat(p["updated_at"])
            elapsed_sec = (updated - started).total_seconds()
            if elapsed_sec > 0 and p["done_this_run"] > 0:
                rate = p["done_this_run"] / elapsed_sec  # 篇/秒
                remaining_sec = p["remaining"] / rate if rate > 0 else 0
                print(f"  本次執行速度：約 {rate*60:.1f} 篇/分鐘，"
                      f"預估剩餘時間：約 {remaining_sec/3600:.1f} 小時")
        except (ValueError, KeyError, ZeroDivisionError):
            pass
    else:
        print("  尚未開始階段二，或progress檔案還沒產生")

    print(f"\n提示：程式每處理20篇文章就會存一次 fetch_progress.json，"
          f"中斷後直接重新執行同一指令即可自動接續，不會重複抓取。"
          f"\n提示：這是全板未篩股票的原始資料，之後股票清單如果再改，"
          f"直接對 all_articles.csv / all_posts.csv 的title欄位做文字比對即可，"
          f"不需要重新爬蟲。")
