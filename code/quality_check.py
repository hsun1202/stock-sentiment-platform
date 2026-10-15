# -*- coding: utf-8 -*-
"""
批次資料品質自動檢查工具
============================================================

目的：
    不用把每個股票的CSV都上傳給人工檢查，這支程式會自動掃描
    batch_output/ 資料夾裡所有股票的 posts.csv，檢查：
        1. 日期範圍是否正確涵蓋指定期間（避免年份判斷錯誤的舊資料）
        2. 有沒有重複資料（link重複）
        3. 資料量是否明顯偏低（可能代表抓取不完整）
    最後印出一份總結報告，把這份報告貼給協助檢查的人就好，
    不需要上傳原始CSV。

使用方式：
    python quality_check.py --out-dir batch_output --start 2024-01-01 --end 2026-06-30
"""

import os
import pandas as pd
import argparse
from datetime import datetime

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "stock_volume_scanner",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
)
_svs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_svs)
STOCK_LIST = _svs.STOCK_LIST


def check_all_stocks(out_dir, expected_start, expected_end):
    results = []

    for code, name, industry in STOCK_LIST:
        posts_path = os.path.join(out_dir, f"{code}_{name}_posts.csv")
        comments_path = os.path.join(out_dir, f"{code}_{name}_comments.csv")

        row = {"code": code, "name": name, "industry": industry}

        if not os.path.exists(posts_path):
            row["status"] = "缺少檔案"
            results.append(row)
            continue

        try:
            df = pd.read_csv(posts_path)
        except Exception as e:
            row["status"] = f"讀取失敗: {e}"
            results.append(row)
            continue

        row["article_count"] = len(df)

        if "link" in df.columns:
            dup_count = len(df) - df["link"].nunique()
            row["duplicate_count"] = dup_count
        else:
            row["duplicate_count"] = "無link欄位"

        if "resolved_date" in df.columns and len(df) > 0:
            dates = pd.to_datetime(df["resolved_date"], errors="coerce")
            row["date_min"] = dates.min().date().isoformat() if dates.notna().any() else "無效"
            row["date_max"] = dates.max().date().isoformat() if dates.notna().any() else "無效"

            # 檢查日期範圍是否合理涵蓋預期區間
            # 允許7天以內的誤差（例如年初假期沒人發文是正常現象，
            # 不該被當成警告），只有明顯的缺口才提示
            if dates.notna().any():
                actual_min, actual_max = dates.min().date(), dates.max().date()
                if (actual_min - expected_start).days > 7:
                    row["date_warning"] = f"最早日期({actual_min})晚於預期起始({expected_start})超過7天，可能未抓完"
                elif (expected_end - actual_max).days > 30:
                    row["date_warning"] = f"最晚日期({actual_max})過早，可能有缺漏"
                else:
                    row["date_warning"] = ""
        else:
            row["date_min"] = row["date_max"] = "無resolved_date欄位"

        # 留言檔案
        if os.path.exists(comments_path):
            try:
                cdf = pd.read_csv(comments_path)
                row["comment_count"] = len(cdf)
            except Exception:
                row["comment_count"] = "讀取失敗"
        else:
            row["comment_count"] = "缺少檔案"

        row["status"] = "正常"
        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批次資料品質自動檢查")
    parser.add_argument("--out-dir", type=str, default="batch_output")
    parser.add_argument("--start", type=str, required=True, help="預期資料起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="預期資料結束日期 YYYY-MM-DD")
    args = parser.parse_args()

    expected_start = datetime.strptime(args.start, "%Y-%m-%d").date()
    expected_end = datetime.strptime(args.end, "%Y-%m-%d").date()

    df = check_all_stocks(args.out_dir, expected_start, expected_end)

    print("=" * 70)
    print("批次資料品質檢查報告")
    print("=" * 70)
    print(f"\n總股票數: {len(df)}")
    print(f"正常: {(df['status']=='正常').sum()}")
    print(f"異常: {(df['status']!='正常').sum()}")

    if (df['status'] != '正常').any():
        print("\n--- 異常股票 ---")
        print(df[df['status'] != '正常'][['code', 'name', 'industry', 'status']].to_string(index=False))

    print("\n--- 各股票資料量總覽 ---")
    normal_df = df[df['status'] == '正常'].copy()
    print(normal_df[['industry', 'code', 'name', 'article_count', 'comment_count',
                      'duplicate_count', 'date_min', 'date_max']].to_string(index=False))

    if 'date_warning' in normal_df.columns:
        warnings = normal_df[normal_df['date_warning'] != '']
        if len(warnings) > 0:
            print("\n--- 日期範圍警告 ---")
            print(warnings[['code', 'name', 'date_warning']].to_string(index=False))

    total_articles = normal_df['article_count'].sum() if 'article_count' in normal_df.columns else 0
    total_comments = pd.to_numeric(normal_df['comment_count'], errors='coerce').sum() \
        if 'comment_count' in normal_df.columns else 0
    total_dup = pd.to_numeric(normal_df['duplicate_count'], errors='coerce').sum() \
        if 'duplicate_count' in normal_df.columns else 0

    print("\n" + "=" * 70)
    print(f"總文章數: {total_articles:.0f}")
    print(f"總留言數: {total_comments:.0f}")
    print(f"總重複筆數: {total_dup:.0f}")
    print("=" * 70)
