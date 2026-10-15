# -*- coding: utf-8 -*-
"""
把PTT全板重爬（斷句修復版）資料比對到267檔股票，輸出格式與 match_ptt_articles_267.py
一致（{code}_{name}_comments.csv，欄位：stock_code,stock_name,article_id,resolved_date,content），
可直接餵給 breeze_infer_267.py。標題比對規則沿用 match_ptt_articles_267.is_valid_match。

python match_ptt_full_267.py --start-date 2022-01-01 --end-date 2024-12-31 --out-dir ptt_output_267_stitched
"""
import os
import argparse
import pandas as pd

from stock_list_267 import STOCK_LIST_267
from match_ptt_articles_267 import find_matches

SRC_DIR = "ptt_stock_full_2020_2024"


def article_id_from_link(link):
    return link.rsplit("/", 1)[-1].replace(".html", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2022-01-01")
    ap.add_argument("--end-date", default="2024-12-31")
    ap.add_argument("--comments", default=os.path.join(SRC_DIR, "all_comments_stitched.csv"))
    ap.add_argument("--articles", default=os.path.join(SRC_DIR, "all_articles.csv"))
    ap.add_argument("--out-dir", default="ptt_output_267_stitched")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    arts = pd.read_csv(args.articles)
    arts = arts[(arts["resolved_date"] >= args.start_date) & (arts["resolved_date"] <= args.end_date)]
    link_to_stocks = {}
    for title, link in zip(arts["title"].fillna(""), arts["link"]):
        hits = find_matches(title)
        if hits:
            link_to_stocks[link] = [(c, n) for c, n, _ in hits]
    print(f"範圍內文章 {len(arts)} 篇，標題比對到267檔其中之一的有 {len(link_to_stocks)} 篇")

    rows = {c: [] for c, _, _ in STOCK_LIST_267}
    name_of = {c: n for c, n, _ in STOCK_LIST_267}
    total_in, total_kept = 0, 0
    for chunk in pd.read_csv(args.comments, chunksize=500_000, usecols=["link", "resolved_date", "content"]):
        total_in += len(chunk)
        chunk = chunk[chunk["link"].isin(link_to_stocks)]
        chunk = chunk[chunk["content"].notna() & (chunk["content"].astype(str).str.strip() != "")]
        for link, date, content in zip(chunk["link"], chunk["resolved_date"], chunk["content"]):
            aid = article_id_from_link(link)
            for code, name in link_to_stocks[link]:
                rows[code].append((code, name, aid, date, content))
                total_kept += 1
        print(f"  已掃 {total_in:,} 則留言，命中 {total_kept:,}")

    written = 0
    for code, data in rows.items():
        if not data:
            continue
        df = pd.DataFrame(data, columns=["stock_code", "stock_name", "article_id", "resolved_date", "content"])
        df.to_csv(os.path.join(args.out_dir, f"{code}_{name_of[code]}_comments.csv"), index=False, encoding="utf-8-sig")
        written += 1
    print(f"\n[完成] 輸出 {written} 檔股票，共 {total_kept:,} 則留言 -> {args.out_dir}")


if __name__ == "__main__":
    main()
