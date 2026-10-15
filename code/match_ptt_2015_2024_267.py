# -*- coding: utf-8 -*-
"""
把 PTT 2015-2024（已拼接＋內容清理，ptt_2015_2024/comments_clean.csv）比對到267檔股票。
標題比對：股票名稱規則沿用 match_ptt_articles_267.is_valid_match；股票代號改成較嚴格的 find_matches_v2：
    舊規則是「標題含代號這串數字就算」，2015年資料加入後出問題——「[閒聊] 2015/07/27 盤中閒聊」被配到豐興(2015)、
    「2014年營收」配到中鴻(2014)、帳號 kkkk123123 配到 2312。新規則：代號前後不能緊鄰其他數字；
    長得像年份的代號(1990~2030)另外不能是日期的一部分（前面是 / - .，或後面接 年、/數字、-數字、.數字）。
    全部255,001篇標題實測：舊規則比對到35,194篇、新規則34,198篇，少掉的996筆逐一抽查皆為誤配。

輸出格式與 match_ptt_full_267.py 相容（{code}_{name}_comments.csv，可直接餵 breeze_infer_267_fast.py），
另外多保留 comment_time（留言自己的時間，可做無洩漏的落後分析）、user_id、push_type、time_anomaly。

python match_ptt_2015_2024_267.py
"""
import os
import argparse
import pandas as pd

from stock_list_267 import STOCK_LIST_267
import re
from match_ptt_articles_267 import is_valid_match

SRC = "ptt_2015_2024"
NO_CODE = "￿"  # 不可能出現在標題的字元：讓 is_valid_match 只做名稱判斷
COLS = ["stock_code", "stock_name", "article_id", "resolved_date", "comment_time", "time_anomaly",
        "push_type", "user_id", "content"]


def code_in_title(code, title):
    yearlike = 1990 <= int(code) <= 2030
    for m in re.finditer(re.escape(code), title):
        s, e = m.start(), m.end()
        before = title[s - 1] if s > 0 else ""
        after = title[e:e + 2]
        if before.isdigit() or after[:1].isdigit():
            continue
        if yearlike and (before in "/-." or re.match(r"[/\-\.]\d|年", after)):
            continue
        return True
    return False


def find_matches_v2(title):
    hits = []
    for code, name, industry in STOCK_LIST_267:
        # 名稱規則照舊（把代號換成不可能出現的字元，只留名稱與前綴衝突的判斷）
        if code_in_title(code, title) or (name in title and is_valid_match(NO_CODE, name, title)):
            hits.append((code, name, industry))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", default=os.path.join(SRC, "comments_clean.csv"))
    ap.add_argument("--articles", default=os.path.join(SRC, "articles.csv"))
    ap.add_argument("--out-dir", default="ptt_output_267_2015_2024")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    arts = pd.read_csv(args.articles, usecols=["article_id", "title", "resolved_date"], dtype=str,
                       keep_default_na=False)
    aid_to_stocks = {}
    for title, aid in zip(arts["title"], arts["article_id"]):
        hits = find_matches_v2(title)
        if hits:
            aid_to_stocks[aid] = [(c, n) for c, n, _ in hits]
    print(f"文章 {len(arts):,} 篇，標題比對到267檔其中之一的有 {len(aid_to_stocks):,} 篇")

    rows = {c: [] for c, _, _ in STOCK_LIST_267}
    name_of = {c: n for c, n, _ in STOCK_LIST_267}
    total_in, total_kept = 0, 0
    for chunk in pd.read_csv(args.comments, chunksize=1_000_000, dtype=str, keep_default_na=False,
                             usecols=["article_id", "resolved_date", "comment_time", "time_anomaly",
                                      "push_type", "user_id", "content"]):
        total_in += len(chunk)
        chunk = chunk[chunk["article_id"].isin(aid_to_stocks)]
        for r in chunk.itertuples(index=False):
            for code, name in aid_to_stocks[r.article_id]:
                rows[code].append((code, name, r.article_id, r.resolved_date, r.comment_time, r.time_anomaly,
                                   r.push_type, r.user_id, r.content))
                total_kept += 1
        print(f"  已掃 {total_in:,} 則留言，命中 {total_kept:,}")

    written = 0
    for code, data in rows.items():
        if not data:
            continue
        pd.DataFrame(data, columns=COLS).to_csv(
            os.path.join(args.out_dir, f"{code}_{name_of[code]}_comments.csv"), index=False, encoding="utf-8-sig")
        written += 1
    print(f"\n[完成] 輸出 {written} 檔股票，共 {total_kept:,} 則留言 -> {args.out_dir}")


if __name__ == "__main__":
    main()
