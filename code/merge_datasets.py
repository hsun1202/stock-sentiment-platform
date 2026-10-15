# -*- coding: utf-8 -*-
"""
技術指標 + 情緒因子 資料整合工具
============================================================

目的：
    把 technical_output/（每日技術指標，含Min-Max正規化）與
    sentiment_output/（依文章彙總的情緒因子）依「股票代碼 + 交易日」
    對齊合併，產出最終要餵給LSTM模型的資料集。

做法：
    1. 以 technical_output 的「交易日」為主軸（因為模型是逐交易日
       預測股價，非交易日本來就不該出現在資料集裡）。
    2. sentiment_output 是逐篇文章的彙總，同一天可能有多篇文章，
       所以合併前先依 resolved_date 再彙總一次成「當日情緒」：
       當日留言數加總、當日正負留言數加總，並重新計算
       當日情緒因子 = (當日正留言 - 當日負留言) / 當日總留言數。
    3. 用 left join（以技術指標為主）合併，避免非交易日的情緒資料
       混入資料集；PTT沒人討論的交易日則情緒因子補0、文章數補0，
       不是資料缺漏，是「當天沒人討論」的真實狀態。

使用方式：
    python merge_datasets.py
    python merge_datasets.py --out-dir final_dataset
"""

import os
import argparse
import importlib.util
import pandas as pd


def load_stock_list():
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


def aggregate_daily_sentiment(sentiment_df):
    """把逐篇文章的情緒彙總表，再彙總成逐交易日的情緒因子。"""
    daily = sentiment_df.groupby("resolved_date", as_index=False).agg(
        article_count=("link", "count"),
        total_comments=("total_comments", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        neutral_count=("neutral_count", "sum"),
    )
    daily["daily_sentiment_factor"] = (
        (daily["positive_count"] - daily["negative_count"]) / daily["total_comments"]
    ).where(daily["total_comments"] > 0, 0.0).round(4)
    return daily


def merge_one_stock(code, name, industry, technical_dir, sentiment_dir):
    tech_path = os.path.join(technical_dir, f"{code}_{name}_technical.csv")
    sent_path = os.path.join(sentiment_dir, f"{code}_{name}_sentiment.csv")

    if not os.path.exists(tech_path):
        return None, f"找不到技術指標檔案：{tech_path}"

    tech_df = pd.read_csv(tech_path)
    tech_df["Date"] = pd.to_datetime(tech_df["Date"]).dt.strftime("%Y-%m-%d")

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
    parser = argparse.ArgumentParser(description="合併技術指標與情緒因子，產出最終資料集")
    parser.add_argument("--technical-dir", type=str, default="technical_output")
    parser.add_argument("--sentiment-dir", type=str, default="sentiment_output")
    parser.add_argument("--out-dir", type=str, default="final_dataset")
    args = parser.parse_args()

    STOCK_LIST = load_stock_list()
    os.makedirs(args.out_dir, exist_ok=True)

    all_frames = []
    success_count, fail_count = 0, 0

    print(f"=== 資料整合：{len(STOCK_LIST)} 檔股票 ===\n")

    for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
        merged, err = merge_one_stock(code, name, industry, args.technical_dir, args.sentiment_dir)
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
        combined_path = os.path.join(args.out_dir, "all_stocks_final_dataset.csv")
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\n[完成] 已輸出整併總表：{combined_path}（共 {len(combined)} 列）")

    print(f"\n=== 整合完成 ===")
    print(f"成功：{success_count}，失敗：{fail_count}")
