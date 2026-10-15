# -*- coding: utf-8 -*-
"""
查詢267檔PTT重爬任務（batch_crawler_267.py）的即時進度
============================================================
使用方式：
    python check_crawler_267_progress.py
    python check_crawler_267_progress.py --out-dir ptt_267_userid
"""

import os
import csv
import json
import glob
import argparse
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="ptt_267_userid")
    args = parser.parse_args()
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        print(f"輸出資料夾 {out_dir}/ 不存在，任務還沒開始跑")
        raise SystemExit(0)

    print(f"=== 267檔PTT重爬進度（{out_dir}/）===\n")

    # ---------- 階段一：掃描 ----------
    matched_path = os.path.join(out_dir, "matched_articles.csv")
    checkpoint_path = os.path.join(out_dir, "scan.checkpoint")
    done_marker_path = os.path.join(out_dir, "scan_done.marker")

    matched_count = 0
    if os.path.exists(matched_path):
        with open(matched_path, "r", encoding="utf-8-sig") as f:
            matched_count = sum(1 for _ in csv.DictReader(f))

    print("[階段一：掃描列表頁]")
    if os.path.exists(done_marker_path):
        print(f"  狀態：已完整掃描完成")
    elif os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            page = f.read().strip()
        print(f"  狀態：進行中，目前掃描到第 {page} 頁（會持續往舊頁碼走，直到掃到起始日期為止）")
    else:
        print(f"  狀態：尚未開始，或本次執行尚未產生checkpoint")
    print(f"  目前累積命中文章數：{matched_count}\n")

    # ---------- 階段二：抓取內文與留言 ----------
    progress_path = os.path.join(out_dir, "fetch_progress.json")
    posts_files = glob.glob(os.path.join(out_dir, "*_posts.csv"))
    comments_files = glob.glob(os.path.join(out_dir, "*_comments.csv"))

    total_posts = 0
    for p in posts_files:
        with open(p, "r", encoding="utf-8-sig") as f:
            total_posts += sum(1 for _ in csv.DictReader(f))

    total_comments = 0
    for c in comments_files:
        with open(c, "r", encoding="utf-8-sig") as f:
            total_comments += sum(1 for _ in csv.DictReader(f))

    print("[階段二：抓取內文與留言]")
    print(f"  已有 {len(posts_files)} 檔股票有輸出檔案")
    print(f"  累積已抓取文章數：{total_posts}（含使用者ID的留言共 {total_comments} 則）")

    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            p = json.load(f)
        total_done = p["already_done_before_this_run"] + p["done_this_run"]
        pct = total_done / p["total_matched"] * 100 if p["total_matched"] else 0
        print(f"  本次任務目標：{p['total_matched']} 篇（{total_done}/{p['total_matched']}，{pct:.1f}%）")
        print(f"  目前處理到：{p['current_stock_code']} {p['current_stock_name']}")
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
          f"中斷後直接重新執行同一指令即可自動接續，不會重複抓取。")
