# -*- coding: utf-8 -*-
"""
把組員提供的10個batch zip（每個裡面裝~2.5萬篇PTT文章json，來源pttweb.cc mirror）
比對到267檔股票清單（stock_list_267.py），只用標題比對（比照batch_crawler.py的
is_valid_match邏輯），輸出跟batch_output/同樣格式的 {code}_{name}_comments.csv /
_posts.csv，但因為原始資料沒有使用者ID、也沒有推噓類型區分，comments.csv不含
push_type欄位，情緒完全交給後續的Breeze-7B模型判斷（不算推噓比）。
"""
import os
import csv
import json
import zipfile
import argparse
from datetime import datetime

from stock_list_267 import STOCK_LIST_267

# 自動偵測出的前綴碰撞（比照原本batch_crawler.py的PREFIX_COLLISIONS概念）
PREFIX_COLLISIONS = [
    ("統一", "統一超"), ("台塑", "台塑化"), ("東和", "東和鋼鐵"),
    ("中鋼", "中鋼構"), ("三商", "三商電"),
]


def is_valid_match(code, name, title):
    if code in title:
        return True
    if name not in title:
        return False
    for prefix_name, suffix_name in PREFIX_COLLISIONS:
        if name == prefix_name and suffix_name in title:
            title_without_suffix = title.replace(suffix_name, "")
            if prefix_name not in title_without_suffix:
                return False
    return True


def find_matches(title):
    hits = []
    for code, name, industry in STOCK_LIST_267:
        if is_valid_match(code, name, title):
            hits.append((code, name, industry))
    return hits


def resolved_date_from_ts(ts):
    try:
        return datetime.fromtimestamp(int(ts)).date().isoformat()
    except (ValueError, OSError, TypeError):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptt-zip", default="ptt raw-20260903T133945Z-1-001.zip")
    parser.add_argument("--out-dir", default="ptt_output_267")
    parser.add_argument("--limit-batches", type=int, default=None, help="測試用，只處理前N個batch")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    outer = zipfile.ZipFile(args.ptt_zip)
    batch_names = sorted(n for n in outer.namelist() if n.endswith(".zip"))
    if args.limit_batches:
        batch_names = batch_names[: args.limit_batches]

    # 用字典累積每檔股票的 posts / comments，最後統一寫檔（避免開267支檔案handle）
    posts_rows = {}     # code -> list of dict
    comments_rows = {}  # code -> list of dict
    name_of = {c: n for c, n, i in STOCK_LIST_267}

    total_articles = 0
    total_matched_article_hits = 0
    total_pushes = 0
    no_match_count = 0

    for bi, bname in enumerate(batch_names, 1):
        print(f"=== 處理 {bname} ({bi}/{len(batch_names)}) ===")
        data = outer.read(bname)
        import io
        inner = zipfile.ZipFile(io.BytesIO(data))
        json_names = [n for n in inner.namelist() if n.endswith(".json")]
        for jn in json_names:
            total_articles += 1
            try:
                d = json.loads(inner.read(jn))
            except Exception:
                continue
            title = d.get("title", "") or ""
            body = d.get("body", "") or ""
            pushes = d.get("pushes", []) or []
            ts = d.get("timestamp", "")
            article_id = d.get("article_id", "")
            url = d.get("url", "")

            hits = find_matches(title)
            if not hits:
                no_match_count += 1
                continue

            rdate = resolved_date_from_ts(ts)
            for code, name, industry in hits:
                total_matched_article_hits += 1
                posts_rows.setdefault(code, []).append({
                    "stock_code": code, "stock_name": name, "industry": industry,
                    "article_id": article_id, "title": title, "resolved_date": rdate,
                    "url": url, "total_comments": len(pushes),
                })
                for p in pushes:
                    comments_rows.setdefault(code, []).append({
                        "stock_code": code, "stock_name": name,
                        "article_id": article_id, "resolved_date": rdate,
                        "content": p,
                    })
                    total_pushes += 1

        print(f"  累積：已處理文章 {total_articles}，命中(含多股)次數 {total_matched_article_hits}，"
              f"未命中 {no_match_count}，留言累積 {total_pushes}")

    # 寫檔
    for code, rows in posts_rows.items():
        name = name_of[code]
        path = os.path.join(args.out_dir, f"{code}_{name}_posts.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for code, rows in comments_rows.items():
        name = name_of[code]
        path = os.path.join(args.out_dir, f"{code}_{name}_comments.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print()
    print(f"[完成] 總文章數 {total_articles}，未命中任何股票 {no_match_count} "
          f"({no_match_count/total_articles*100:.1f}%)")
    print(f"命中股票數(至少1篇): {len(posts_rows)} / 267")
    print(f"總留言數（已匹配文章底下）: {total_pushes}")
    zero_stocks = [c for c, n, i in STOCK_LIST_267 if c not in posts_rows]
    print(f"完全沒配到文章的股票數: {len(zero_stocks)}")
    if zero_stocks:
        print("  ", [f"{c}{name_of[c]}" for c in zero_stocks[:20]], "..." if len(zero_stocks) > 20 else "")


if __name__ == "__main__":
    main()
