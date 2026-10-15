# -*- coding: utf-8 -*-
"""
用「使用者ID」重做斷句留言合併（v2，取代舊版stitch_truncated_comments.py）
============================================================

跟舊版最大的差異：
    舊版（stitch_truncated_comments.py）完全沒有使用者ID可用，只能靠
    「長度+結尾特徵」這種純文字heuristic去猜下一則是不是同一個人接
    著打的，先前測試發現有真實的誤接風險。

    這次重爬的全板資料（crawl_ptt_stock_full.py）有記錄每則推文的
    使用者ID，多了關鍵防呆：只有「同一篇文章、緊接著的兩則留言、
    使用者ID完全相同」才會考慮合併，從根本上排除「誤接到別人留言」
    的風險。

    另外分析全部留言的長度分布，發現一個很強的訊號：不管是候選
    合併案例、還是全部留言本身，長度在20-30字（尤其23-27字）都有
    明顯異常突起，這代表PTT單行留言的真實截斷點就落在這個區間。
    用這個經驗區間再加一層篩選（只有長度落在這個窄區間內才視為
    「疑似截斷」），可以排除掉「剛好是一句很長、結尾沒加標點的
    完整話」這種假陽性。

    抽查了590+個真實案例（跨越300萬列以上、涵蓋全部14.3萬篇文章），
    確認品質可靠後才拿來套用到全部資料。

使用方式：
    # 套用到全部留言，輸出合併後的版本
    python stitch_truncated_comments_v2.py --apply

    # 只在小範圍測試、看合併案例，不寫檔
    python stitch_truncated_comments_v2.py --sample-rows 300000 --show-examples 30
"""

import argparse
import csv
import re

TERMINAL_MARKERS = set("。！？!?~…」』）)0123456789０-９ 的了啊呀喔嗎吧耶ㄏㄎXDwww笑✓✗=@^><")
MIN_LEN = 20
MAX_LEN = 30  # 經驗值：PTT留言長度分布在20-30字（尤其23-27字）有異常突起，是真實截斷點
MAX_NEXT_LEN = 12
URL_PATTERN = re.compile(r"(https?://|www\.)\S+$")


def looks_truncated(s):
    s = s.strip()
    n = len(s)
    if n < MIN_LEN or n > MAX_LEN:
        return False
    if URL_PATTERN.search(s):
        return False
    return s[-1] not in TERMINAL_MARKERS


def can_merge(prev, cur):
    return (prev["user_id"] == cur["user_id"]
            and prev["user_id"] != ""
            and looks_truncated(prev["content"])
            and len(cur["content"]) <= MAX_NEXT_LEN)


def apply_full(comments_csv, out_csv):
    with open(comments_csv, "r", encoding="utf-8-sig") as fin, \
         open(out_csv, "w", newline="", encoding="utf-8-sig") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        pending = None  # 目前手上還沒寫出去的一列（可能還會被下一列合併進來）
        total_in, total_out, total_merged = 0, 0, 0

        for row in reader:
            total_in += 1
            row["content"] = (row.get("content") or "").strip()
            row["user_id"] = (row.get("user_id") or "").strip()

            if pending is not None and pending["link"] == row["link"] and can_merge(pending, row):
                pending["content"] = pending["content"] + row["content"]
                total_merged += 1
                # 合併後不再嘗試連鎖合併（維持跟先前驗證時同一套保守策略）
                writer.writerow(pending)
                total_out += 1
                pending = None
            else:
                if pending is not None:
                    writer.writerow(pending)
                    total_out += 1
                pending = row

            if total_in % 1000000 == 0:
                print(f"    進度：已處理 {total_in:,} 列，目前合併 {total_merged:,} 處")

        if pending is not None:
            writer.writerow(pending)
            total_out += 1

    print(f"\n[完成] 留言數 {total_in:,} -> {total_out:,}（減少 {total_in-total_out:,}，"
          f"共合併 {total_merged:,} 處）")
    print(f"輸出檔案：{out_csv}")


def sample_test(comments_csv, sample_rows, show_examples):
    print(f"讀取前 {sample_rows:,} 列進行測試...")
    with open(comments_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= sample_rows:
                break
            row["user_id"] = (row.get("user_id") or "").strip()
            row["content"] = (row.get("content") or "").strip()
            rows.append(row)
    print(f"實際讀到 {len(rows):,} 列")

    total_merged, examples_shown = 0, 0
    i, n = 0, len(rows)
    while i < n:
        if i + 1 < n and rows[i]["link"] == rows[i + 1]["link"] and can_merge(rows[i], rows[i + 1]):
            total_merged += 1
            if examples_shown < show_examples:
                merged_text = rows[i]["content"] + rows[i + 1]["content"]
                print(f"    [{rows[i]['user_id']}] {rows[i]['content']!r} + {rows[i+1]['content']!r}"
                      f" -> {merged_text!r}")
                examples_shown += 1
            i += 2
        else:
            i += 1

    print(f"\n=== 統計（前 {len(rows):,} 列）===")
    print(f"合併次數：{total_merged}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments-csv", type=str, default="ptt_stock_full_2020_2024/all_comments.csv")
    parser.add_argument("--out-csv", type=str, default="ptt_stock_full_2020_2024/all_comments_stitched.csv")
    parser.add_argument("--apply", action="store_true", help="套用到全部資料並輸出新檔案")
    parser.add_argument("--sample-rows", type=int, default=300000)
    parser.add_argument("--show-examples", type=int, default=25)
    args = parser.parse_args()

    if args.apply:
        apply_full(args.comments_csv, args.out_csv)
    else:
        sample_test(args.comments_csv, args.sample_rows, args.show_examples)


if __name__ == "__main__":
    main()
