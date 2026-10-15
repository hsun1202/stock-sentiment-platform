# -*- coding: utf-8 -*-
"""
查詢Breeze-7B批次任務的即時進度
============================================================
使用方式：
    python check_breeze_progress.py
    python check_breeze_progress.py --out-dir sentiment_output_breeze
"""

import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="sentiment_output_breeze")
    args = parser.parse_args()

    progress_path = os.path.join(args.out_dir, "breeze_progress.json")
    done_files = [f for f in os.listdir(args.out_dir) if f.endswith("_sentiment.csv")] if os.path.exists(args.out_dir) else []

    print(f"已完成股票數：{len(done_files)}")

    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            p = json.load(f)
        pct = p["comments_done"] / p["comments_total"] * 100 if p["comments_total"] else 0
        print(f"目前處理中：[{p['stock_idx']}/{p['total_stocks']}] {p['stock_code']} {p['stock_name']}")
        print(f"這檔進度：{p['comments_done']}/{p['comments_total']}（{pct:.1f}%）")
        print(f"最後更新時間：{p['updated_at']}")
        print(f"本次任務開始時間：{p['started_at']}")
    else:
        print("目前沒有正在跑的任務（progress檔不存在，可能還沒開始跑、或已經全部跑完了）")
