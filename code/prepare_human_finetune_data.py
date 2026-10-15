# -*- coding: utf-8 -*-
"""
用人工標註結果準備BERT微調資料（在既有詞典蒸餾模型基礎上，用真實標籤再微調）
============================================================

目的：
    之前的BERT（V2/V3）都是從詞典法的判斷結果蒸餾出來的，準確率
    上限被詞典鎖死。這支程式把annotation_result.csv（250筆人工
    標註結果）切成train/val，讓train_bert_sentiment.py可以在既有
    的BERT三分類模型基礎上，用真人標籤繼續微調，第一次真正超越
    詞典的天花板。

    切分時特別留一份val，且val不會被拿去訓練，才能公平比較
    微調前後的準確率有沒有真的進步（不能拿訓練用過的資料驗證）。

使用方式：
    python prepare_human_finetune_data.py
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="準備人工標註資料供BERT微調")
    parser.add_argument("--input", type=str, default="annotation_result.csv")
    parser.add_argument("--out-dir", type=str, default="bert_dataset_human")
    parser.add_argument("--val-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    before = len(df)
    df = df.dropna(subset=["human_label"]).copy()
    print(f"總筆數：{before}，可用（已標註）：{len(df)}")

    df["label"] = df["human_label"].map(LABEL_MAP)
    print(df["human_label"].value_counts())

    train_df, val_df = train_test_split(
        df, test_size=args.val_size, stratify=df["label"], random_state=args.seed
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", val_df)]:
        out_path = os.path.join(args.out_dir, f"{name}.csv")
        split_df[["content", "label"]].rename(columns={"content": "text"}).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        print(f"{name}: {len(split_df)} 筆 -> {out_path}")

    print("\n[完成]（val跟test用同一份，val在微調過程中完全沒被拿去訓練，可公平比較微調前後）")
