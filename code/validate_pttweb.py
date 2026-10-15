# -*- coding: utf-8 -*-
"""
驗證 pttweb 鏡像站資料的品質，兩種對照：
    A. PTT本站殘留的2015-2018文章（ptt_stock_full_2015_2018/，自己的爬蟲v2抓的）vs 鏡像站同一篇
    B. 2019-03（鏡像站 vs 同學的PTT本站資料）：文章覆蓋率（鏡像站多出多少後來被刪的文章）、同一篇留言是否一致
比對留言時看 (推噓, 使用者ID, 內容, MM/DD HH:MM) 的序列。
輸出：validate_pttweb_report.txt
"""
import os
import sys
import csv
import json
from collections import Counter
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ptt_2015_2024 as B  # noqa: E402

BASE = B.BASE
OUT = os.path.join(BASE, "validate_pttweb_report.txt")


def norm(s):
    return " ".join(str(s or "").split())


def load_pttweb(lo, hi):
    arts = {}
    with open(B.PTTWEB_RAW, encoding="utf-8") as f:
        for line in f:
            try:
                a = json.loads(line)["article"]
            except (ValueError, KeyError):
                continue
            dt = B.aid_datetime(a["article_aid"])
            if not (lo <= dt.date() <= hi):
                continue
            cs = []
            for r in sorted(a.get("recommend_data") or [], key=lambda r: r.get("position", 0)):
                t = r.get("text", "")
                t = t[1:] if t.startswith(":") else t
                if norm(t):
                    cs.append((B.TAG_MAP_PTTWEB.get(r.get("type"), "→"), r.get("author", "").strip(), norm(t),
                               B.norm_time_raw(r.get("ip_datetime", ""))))
            arts[a["article_aid"]] = {"title": a.get("title", ""), "comments": cs,
                                      "deleted": bool(a.get("is_deleted") or a.get("is_purged"))}
    return arts


def compare(ref, web, log, label):
    both = set(ref) & set(web)
    same, same_nt, diff_n = 0, 0, Counter()
    examples = []
    for a in both:
        r, w = ref[a]["comments"], web[a]["comments"]
        if r == w:
            same += 1
        elif [x[:3] for x in r] == [x[:3] for x in w]:
            same_nt += 1
        else:
            diff_n["鏡像站較多" if len(w) > len(r) else ("鏡像站較少" if len(w) < len(r) else "數量同內容不同")] += 1
            if len(examples) < 5:
                rs, ws = Counter(r), Counter(w)
                examples.append((a, ref[a]["title"][:30], len(r), len(w),
                                 list((rs - ws).elements())[:2], list((ws - rs).elements())[:2]))
    log(f"[{label}] 兩邊都有 {len(both):,} 篇；只在參考資料 {len(set(ref)-set(web)):,}；只在鏡像站 {len(set(web)-set(ref)):,}")
    if both:
        log(f"    留言完全相同(含時間) {same:,}（{same/len(both):.2%}）｜只差時間 {same_nt:,}｜不同 {dict(diff_n)}")
        for e in examples:
            log(f"    例：{e[0]} {e[1]} 參考{e[2]}則/鏡像{e[3]}則；參考才有{e[4]}；鏡像才有{e[5]}")


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # ---- A. PTT本站殘留 2015-2018 ----
    d = os.path.join(BASE, "ptt_stock_full_2015_2018")
    posts = pd.read_csv(os.path.join(d, "all_posts.csv"), dtype=str, keep_default_na=False)
    coms = pd.read_csv(os.path.join(d, "all_comments.csv"), dtype=str, keep_default_na=False)
    ref = {}
    for link, title in zip(posts["link"], posts["title"]):
        ref[link.split("/")[-1].replace(".html", "")] = {"title": title, "comments": []}
    for link, t, u, c, tm in zip(coms["link"], coms["push_type"], coms["user_id"], coms["content"], coms["push_time"]):
        a = link.split("/")[-1].replace(".html", "")
        if norm(c):
            ref[a]["comments"].append((t.strip(), u.strip(), norm(c), B.norm_time_raw(tm)))
    web = load_pttweb(date(2015, 1, 1), date(2018, 12, 31))
    log(f"鏡像站目前已抓 2015-2018：{len(web):,} 篇（其中PTT上已刪 {sum(v['deleted'] for v in web.values()):,}）")
    maxd = max(B.aid_datetime(x).date() for x in web) if web else date(2014, 12, 31)
    ref_in_range = {a: v for a, v in ref.items() if B.aid_datetime(a).date() <= maxd}
    compare(ref_in_range, web, log, "A. PTT本站殘留文章 vs 鏡像站（只比鏡像站已抓到的期間）")

    # ---- B. 2019-03 鏡像站 vs 同學 ----
    web19 = load_pttweb(date(2019, 3, 1), date(2019, 3, 31))
    if web19:
        cm = {}
        with open(B.CLASSMATE.format(2019), encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                dt = B.aid_datetime(x["article_id"])
                if not (date(2019, 3, 1) <= dt.date() <= date(2019, 3, 31)):
                    continue
                cm[x["article_id"]] = {"title": x.get("title", ""), "comments": [
                    (c.get("tag", "").strip(), c.get("user_id", "").strip(), norm(c.get("content")),
                     B.norm_time_raw(c.get("time_raw", ""))) for c in (x.get("comments") or []) if norm(c.get("content"))]}
        compare(cm, web19, log, "B. 2019-03 同學(PTT本站) vs 鏡像站")
        only_web = set(web19) - set(cm)
        log(f"    鏡像站多出的 {len(only_web):,} 篇中，標記為PTT上已刪：{sum(web19[a]['deleted'] for a in only_web):,}"
            f"；鏡像站文章數/PTT本站 = {len(web19)/max(1,len(cm)):.3f}")
        cats = Counter((web19[a]["title"].split("]")[0] + "]")[:8] for a in only_web)
        log(f"    多出文章的分類：{cats.most_common(8)}")
    else:
        log("B. 鏡像站尚未抓到2019-03，之後再跑")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
