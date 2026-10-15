# -*- coding: utf-8 -*-
"""
merge_datasets.py 的267檔股票版本，邏輯一樣（技術指標為主軸、情緒因子
left join、無討論日期補0），但有兩個調整：
  1. aggregate_daily_sentiment 分組計數欄位從 "link" 改成 "article_id"
     （breeze_infer_267.py 輸出的欄位名稱不同）
  2. 技術指標篩選到只保留2022-2024（跟情緒分析的涵蓋範圍對齊）——
     技術指標本身有15年資料，但情緒分析只做了2022-2024，如果把
     2011-2021、2025這些「根本沒做情緒分析」的年份也併進來，
     情緒因子會被補0，變成看起來像「當天沒人討論」，但事實上是
     「這個年份還沒有情緒分析結果」，兩者意義不同，會誤導LSTM訓練。
"""

import os
import argparse
import pandas as pd

from stock_list_267 import STOCK_LIST_267


def aggregate_daily_sentiment(sentiment_df):
    daily = sentiment_df.groupby("resolved_date", as_index=False).agg(
        article_count=("article_id", "count"),
        total_comments=("total_comments", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        neutral_count=("neutral_count", "sum"),
    )
    daily["daily_sentiment_factor"] = (
        (daily["positive_count"] - daily["negative_count"]) / daily["total_comments"]
    ).where(daily["total_comments"] > 0, 0.0).round(4)
    return daily


def merge_one_stock(code, name, industry, technical_dir, sentiment_dir, start_date, end_date):
    tech_path = os.path.join(technical_dir, f"{code}_{name}_technical.csv")
    sent_path = os.path.join(sentiment_dir, f"{code}_{name}_sentiment.csv")

    if not os.path.exists(tech_path):
        return None, f"找不到技術指標檔案：{tech_path}"

    tech_df = pd.read_csv(tech_path)
    tech_df["Date"] = pd.to_datetime(tech_df["Date"]).dt.strftime("%Y-%m-%d")
    tech_df = tech_df[(tech_df["Date"] >= start_date) & (tech_df["Date"] <= end_date)].reset_index(drop=True)
    if len(tech_df) == 0:
        return None, f"篩選{start_date}~{end_date}後沒有交易日資料"

    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)
        daily_sent = aggregate_daily_sentiment(sent_df)
    else:
        daily_sent = pd.DataFrame(columns=[
            "resolved_date", "article_count", "total_comments",
            "positive_count", "negative_count", "neutral_count",
            "daily_sentiment_factor",
        ])

    merged = tech_df.merge(
        daily_sent, left_on="Date", right_on="resolved_date", how="left"
    ).drop(columns=["resolved_date"])

    fill_cols = ["article_count", "total_comments", "positive_count",
                 "negative_count", "neutral_count", "daily_sentiment_factor"]
    merged[fill_cols] = merged[fill_cols].fillna(0.0)

    merged.insert(0, "stock_code", code)
    merged.insert(1, "stock_name", name)
    merged.insert(2, "industry", industry)

    return merged, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合併267檔股票的技術指標與情緒因子，產出最終資料集")
    parser.add_argument("--technical-dir", type=str, default="technical_output_267")
    parser.add_argument("--sentiment-dir", type=str, default="sentiment_output_267_2022_2024")
    parser.add_argument("--out-dir", type=str, default="final_dataset_267")
    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    args = parser.parse_args()

    STOCK_LIST = STOCK_LIST_267
    os.makedirs(args.out_dir, exist_ok=True)

    all_frames = []
    success_count, fail_count = 0, 0

    print(f"=== 資料整合：{len(STOCK_LIST)} 檔股票，範圍{args.start_date}~{args.end_date} ===\n")

    for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
        merged, err = merge_one_stock(code, name, industry, args.technical_dir, args.sentiment_dir,
                                       args.start_date, args.end_date)
        if err:
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）失敗：{err}")
            fail_count += 1
            continue

        out_path = os.path.join(args.out_dir, f"{code}_{name}_final.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        all_frames.append(merged)
        success_count += 1

        has_discussion_days = (merged["total_comments"] > 0).sum()
        print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）"
              f"：{len(merged)} 個交易日，其中 {has_discussion_days} 天有PTT討論")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = os.path.join(args.out_dir, "all_stocks_final_dataset_267.csv")
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\n[完成] 已輸出整併總表：{combined_path}（共 {len(combined)} 列）")

    print(f"\n=== 整合完成 ===")
    print(f"成功：{success_count}，失敗：{fail_count}")
