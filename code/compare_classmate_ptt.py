# -*- coding: utf-8 -*-
"""
比對同學的 PTT 資料（ptt_classmate_2019_2024/stock_YYYY.jsonl）與自己的 2020-2024 全板資料
（ptt_stock_full_2020_2024/all_posts.csv + all_comments.csv）。

比對項目：
    1. 文章：兩邊各有幾篇、重疊多少、只在其中一邊的有哪些（依月份）
    2. 留言：同一篇文章兩邊的留言數是否相同、內容序列（推噓/使用者ID/內容）是否完全相同
輸出：compare_classmate_report.txt、compare_classmate_diff_articles.csv
"""
import os
import csv
import json
import hashlib
from collections import Counter, defaultdict

import pandas as pd

BASE = r"D:\專題claude"
MINE_POSTS = os.path.join(BASE, "ptt_stock_full_2020_2024", "all_posts.csv")
MINE_COMMENTS = os.path.join(BASE, "ptt_stock_full_2020_2024", "all_comments.csv")
CLASSMATE = os.path.join(BASE, "ptt_classmate_2019_2024", "stock_{}.jsonl")
OUT_REPORT = os.path.join(BASE, "compare_classmate_report.txt")
OUT_DIFF = os.path.join(BASE, "compare_classmate_diff_articles.csv")


def aid_of(link):
    return link.rstrip("/").split("/")[-1].replace(".html", "")


def norm(s):
    return " ".join(str(s).split())


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # ---- 自己的資料 ----
    posts = pd.read_csv(MINE_POSTS, usecols=["link", "resolved_date", "title", "total_comments"])
    posts["aid"] = posts["link"].map(aid_of)
    mine_title = dict(zip(posts["aid"], posts["title"]))
    mine_date = dict(zip(posts["aid"], posts["resolved_date"]))

    mine_n = Counter()
    mine_hash = {}
    mine_hash_content = {}
    cur, h, hc = None, None, None
    for chunk in pd.read_csv(MINE_COMMENTS, chunksize=1_000_000, dtype=str, keep_default_na=False,
                             usecols=["link", "push_type", "user_id", "content"]):
        for link, t, u, c in zip(chunk["link"], chunk["push_type"], chunk["user_id"], chunk["content"]):
            a = aid_of(link)
            if a != cur:
                if cur is not None:
                    mine_hash[cur] = h.hexdigest()
                    mine_hash_content[cur] = hc.hexdigest()
                cur = a
                h = hashlib.md5(mine_hash.get(a, "").encode())  # 同一篇若不連續出現，接續累加
                hc = hashlib.md5(mine_hash_content.get(a, "").encode())
            mine_n[a] += 1
            h.update(f"{t.strip()}|{u.strip()}|{norm(c)}\n".encode())
            hc.update(f"{norm(c)}\n".encode())
    if cur is not None:
        mine_hash[cur] = h.hexdigest()
        mine_hash_content[cur] = hc.hexdigest()
    log(f"[自己] 文章 {len(posts):,} 篇，留言 {sum(mine_n.values()):,} 則")

    # ---- 同學的資料 ----
    cm_date, cm_title, cm_n, cm_hash, cm_hash_content, cm_has_time = {}, {}, {}, {}, {}, Counter()
    for y in range(2020, 2025):
        with open(CLASSMATE.format(y), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                a = d["article_id"]
                cm_date[a] = (d.get("date") or "")[:10]
                cm_title[a] = d.get("title", "")
                cs = d.get("comments") or []
                cm_n[a] = len(cs)
                h = hashlib.md5()
                hc = hashlib.md5()
                for c in cs:
                    h.update(f"{c.get('tag','').strip()}|{c.get('user_id','').strip()}|{norm(c.get('content',''))}\n".encode())
                    hc.update(f"{norm(c.get('content',''))}\n".encode())
                    if c.get("time"):
                        cm_has_time["有"] += 1
                    else:
                        cm_has_time["無"] += 1
                cm_hash[a] = h.hexdigest()
                cm_hash_content[a] = hc.hexdigest()
    log(f"[同學] 2020-2024 文章 {len(cm_date):,} 篇，留言 {sum(cm_n.values()):,} 則，留言時間戳 {dict(cm_has_time)}")

    mine_set, cm_set = set(mine_title), set(cm_date)
    both = mine_set & cm_set
    only_mine, only_cm = mine_set - cm_set, cm_set - mine_set
    log(f"\n[文章] 兩邊都有 {len(both):,}｜只在自己 {len(only_mine):,}｜只在同學 {len(only_cm):,}")

    # 文章中有沒有同一篇但日期年份超出2020-2024（同學的檔案依年份切）
    by_month = defaultdict(lambda: [0, 0, 0])
    for a in both:
        by_month[str(mine_date[a])[:7]][0] += 1
    for a in only_mine:
        by_month[str(mine_date[a])[:7]][1] += 1
    for a in only_cm:
        by_month[cm_date[a][:7]][2] += 1
    log("月份 | 兩邊都有 | 只在自己 | 只在同學")
    for m in sorted(by_month):
        b, om, oc = by_month[m]
        if om or oc:
            log(f"  {m} | {b} | {om} | {oc}")
    log("（只列出有差異的月份）")

    same_n = sum(1 for a in both if mine_n.get(a, 0) == cm_n[a])
    same_full = sum(1 for a in both if mine_hash.get(a, hashlib.md5().hexdigest()) == cm_hash[a])
    same_content = sum(1 for a in both if mine_hash_content.get(a, hashlib.md5().hexdigest()) == cm_hash_content[a])
    tot_mine = sum(mine_n.get(a, 0) for a in both)
    tot_cm = sum(cm_n[a] for a in both)
    log(f"\n[留言] 重疊文章中：留言數相同 {same_n:,}/{len(both):,}（{same_n/len(both):.2%}）")
    log(f"       內容序列(推噓+ID+內容)完全相同 {same_full:,}（{same_full/len(both):.2%}）；只比內容 {same_content:,}（{same_content/len(both):.2%}）")
    log(f"       重疊文章留言總數：自己 {tot_mine:,}｜同學 {tot_cm:,}（差 {tot_cm-tot_mine:+,}）")

    diff_n = [(a, mine_n.get(a, 0), cm_n[a]) for a in both if mine_n.get(a, 0) != cm_n[a]]
    delta = Counter()
    for a, m, c in diff_n:
        delta["同學較多" if c > m else "自己較多"] += 1
    log(f"       留言數不同的 {len(diff_n):,} 篇：{dict(delta)}")

    with open(OUT_DIFF, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "kind", "date", "title", "mine_n_comments", "classmate_n_comments", "same_content_seq"])
        for a in sorted(only_mine):
            w.writerow([a, "only_mine", mine_date[a], mine_title[a], mine_n.get(a, 0), "", ""])
        for a in sorted(only_cm):
            w.writerow([a, "only_classmate", cm_date[a], cm_title[a], "", cm_n[a], ""])
        for a in sorted(both):
            if mine_hash.get(a) != cm_hash[a]:
                w.writerow([a, "comments_differ", mine_date[a], mine_title[a], mine_n.get(a, 0), cm_n[a],
                            mine_hash_content.get(a) == cm_hash_content[a]])

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n→ {OUT_REPORT}\n→ {OUT_DIFF}")


if __name__ == "__main__":
    main()
