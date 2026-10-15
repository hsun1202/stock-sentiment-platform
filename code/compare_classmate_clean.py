# -*- coding: utf-8 -*-
"""比對同學的清理版（ptt_classmate_clean/）與同學原始版（ptt_classmate_2019_2024/）：刪了哪些文章/留言、有沒有改內容。"""
import json, re, random
from collections import Counter
random.seed(0)
cat_re = re.compile(r"^(?:Re:\s*|Fw:\s*)?\[\s*([^\]］]{1,6})\s*[\]］]")
tot = Counter(); cats_drop = Counter(); cats_keep = Counter(); ex_drop = {}; ex_cdrop = []; ex_mod = []
keys_seen = set()
for y in range(2019, 2025):
    clean = {}
    for line in open(f"ptt_classmate_clean/stock_{y}_clean.jsonl", encoding="utf-8"):
        d = json.loads(line); clean[d["article_id"]] = d
        if y == 2019 and not keys_seen: keys_seen = set(d.keys())
    for line in open(f"ptt_classmate_2019_2024/stock_{y}.jsonl", encoding="utf-8"):
        d = json.loads(line); aid = d["article_id"]
        m = cat_re.match(d.get("title", "")); cat = m.group(1).strip() if m else "(無分類)"
        nc = len(d.get("comments") or [])
        tot["raw_articles"] += 1; tot["raw_comments"] += nc
        if aid not in clean:
            cats_drop[cat] += 1; tot["drop_art_comments"] += nc
            ex_drop.setdefault(cat, []).append(d.get("title", "")[:40])
            continue
        cats_keep[cat] += 1
        c = clean[aid]; rc = d.get("comments") or []; cc = c.get("comments") or []
        tot["kept_comments_raw"] += nc; tot["kept_comments_clean"] += len(cc)
        rset = Counter((x["user_id"], x["content"]) for x in rc); cset = Counter((x["user_id"], x["content"]) for x in cc)
        for k in (rset - cset).elements():
            if len(ex_cdrop) < 4000: ex_cdrop.append(k)
        for k in (cset - rset).elements():
            tot["comments_changed_or_new"] += 1
            if len(ex_mod) < 30: ex_mod.append(k)
        if c.get("content") != d.get("content"): tot["body_changed"] += 1
        if c.get("title") != d.get("title"): tot["title_changed"] += 1
print("清理版欄位：", sorted(keys_seen))
print(dict(tot))
print("\n被整篇刪除的文章分類（篇數）：", cats_drop.most_common(25))
print("保留的文章分類（前15）：", cats_keep.most_common(15))
for cat, _ in cats_drop.most_common(8):
    print(f"  [{cat}] 例：", random.sample(ex_drop[cat], min(5, len(ex_drop[cat]))))
print("\n被刪的單則留言（隨機40則）：", random.sample(ex_cdrop, min(40, len(ex_cdrop))))
print("\n清理版才有/內容被改的留言（前30）：", ex_mod)
