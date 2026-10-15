# -*- coding: utf-8 -*-
"""
修正股票名稱前綴誤判問題
============================================================

問題說明：
    30檔股票名單裡有兩對股票名稱互為前綴：
        - 長榮(2603) 是 長榮航(2618) 的前綴
        - 台塑(1301) 是 台塑化(6505) 的前綴

    原本的比對邏輯用「stock_name in title」判斷是否命中，
    這會導致任何提到「長榮航」的文章，也被誤判成同時命中「長榮」
    （因為「長榮」兩個字剛好包含在「長榮航」裡面），即使文章
    內容跟長榮（航運股）完全無關。這會讓「長榮」的資料集混入
    大量實際上是在講長榮航（航空股）的內容，污染分析結果。

修正邏輯：
    如果一篇文章「同時」命中了某個前綴股票（如：長榮）和它的
    後綴股票（如：長榮航），需要進一步判斷文章是否「獨立」提到
    前綴股票本身（例如標題裡有單獨的「長榮」出現，不是「長榮航」
    的一部分），如果沒有，就把這筆前綴股票的誤判記錄移除。

使用方式：
    python fix_prefix_collision.py --matched batch_output/matched_articles.csv
"""

import pandas as pd
import re
import argparse

# 定義所有「前綴股票 -> 後綴股票」的衝突關係
# 格式：(前綴股票名稱, 後綴股票名稱)
PREFIX_COLLISIONS = [
    ("長榮", "長榮航"),
    ("台塑", "台塑化"),
]


def is_standalone_mention(title, prefix_name, suffix_name):
    """
    判斷標題裡是否有「獨立」提到前綴股票（不是後綴股票的一部分）。
    做法：把標題裡所有的「後綴股票」字樣先移除，再檢查剩下的文字
    裡還有沒有前綴股票的名稱，如果有，代表文章確實有獨立提到前綴股票。
    """
    title_without_suffix = title.replace(suffix_name, "")
    return prefix_name in title_without_suffix


def fix_matched_articles(matched_path, out_path=None):
    df = pd.read_csv(matched_path)
    original_count = len(df)

    rows_to_remove = []

    for prefix_name, suffix_name in PREFIX_COLLISIONS:
        # 找出所有「前綴股票」命中的文章連結
        prefix_rows = df[df["stock_name"] == prefix_name]
        # 找出所有「後綴股票」命中的文章連結
        suffix_links = set(df[df["stock_name"] == suffix_name]["link"])

        # 檢查前綴股票命中的每一筆，是否其實只是因為後綴股票的字樣造成的誤判
        for idx, row in prefix_rows.iterrows():
            if row["link"] in suffix_links:
                # 這篇文章同時被標記成前綴股票和後綴股票，需要進一步判斷
                if not is_standalone_mention(row["title"], prefix_name, suffix_name):
                    rows_to_remove.append(idx)

    print(f"=== 前綴誤判修正報告 ===")
    for prefix_name, suffix_name in PREFIX_COLLISIONS:
        count = sum(1 for idx in rows_to_remove
                    if df.loc[idx, "stock_name"] == prefix_name)
        print(f"  {prefix_name} 誤判自 {suffix_name} 的筆數：{count}")

    df_fixed = df.drop(index=rows_to_remove)
    removed_total = original_count - len(df_fixed)
    print(f"\n總共移除 {removed_total} 筆誤判記錄")
    print(f"修正前：{original_count} 筆 -> 修正後：{len(df_fixed)} 筆")

    out_path = out_path or matched_path
    df_fixed.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[完成] 已輸出修正後的檔案至 {out_path}")

    return rows_to_remove


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修正股票名稱前綴誤判問題")
    parser.add_argument("--matched", type=str, required=True,
                         help="matched_articles.csv 的路徑")
    parser.add_argument("--out", type=str, default=None,
                         help="輸出路徑，不指定則直接覆蓋原檔案")
    args = parser.parse_args()

    fix_matched_articles(args.matched, args.out)
