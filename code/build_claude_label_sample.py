# -*- coding: utf-8 -*-
"""
從全板2020-2024原始留言裡抽一批新樣本，供Claude本人標註
============================================================
目的：
    392筆人工標籤的訓練集太小，微調效果有限。這支程式從剛爬回來
    的全板留言（ptt_stock_full_2020_2024/all_comments.csv）裡
    抽一批全新、沒被標過的留言，讓Claude本人閱讀後直接標註情緒
    （positive/negative/neutral），用來擴充微調訓練集。

    這是弱監督/self-training的做法：Claude的標籤不是100%等於
    人工標籤的黃金標準（先前Kappa分析顯示人類-Claude一致性大約
    只有中等），但用來擴充訓練集規模、讓模型多看不同案例，仍然
    有幫助，而且比完全沒有額外資料好。

使用方式：
    python build_claude_label_sample.py --total 300
"""

import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments-csv", type=str, default="ptt_stock_full_2020_2024/all_comments.csv")
    parser.add_argument("--existing-labels", type=str, default="all_human_labels.csv")
    parser.add_argument("--out", type=str, default="claude_label_batch.csv")
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("讀取既有標籤（排除重複內容用）...")
    existing = pd.read_csv(args.existing_labels)
    used_content = set(existing["content"].astype(str).str.strip().tolist())
    print(f"既有標籤：{len(used_content)} 則")

    print(f"讀取全板留言：{args.comments_csv}（檔案較大，需要一點時間）...")
    df = pd.read_csv(args.comments_csv, usecols=["link", "resolved_date", "content"])
    print(f"原始留言總數：{len(df)}")

    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"].str.len().between(args.min_len, args.max_len)]
    df = df[~df["content"].isin(used_content)]
    df = df.drop_duplicates(subset=["content"])
    print(f"篩選長度({args.min_len}-{args.max_len}字)+排除重複後：{len(df)}")

    picked = df.sample(n=min(args.total, len(df)), random_state=args.seed).reset_index(drop=True)
    picked.insert(0, "id", range(1, len(picked) + 1))
    picked["claude_label"] = ""

    out_cols = ["id", "link", "resolved_date", "content", "claude_label"]
    picked[out_cols].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n已抽出 {len(picked)} 則 -> {args.out}（claude_label欄位待填）")
