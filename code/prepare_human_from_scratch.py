# -*- coding: utf-8 -*-
"""
合併兩批人工標註（250+150），準備「從零訓練」BERT用的資料
============================================================

目的：
    之前兩次嘗試都是在詞典蒸餾出來的BERT基礎上繼續訓練，模型
    帶著詞典「沒中關鍵字就猜中立」的強烈慣性，小量真人標籤很難
    扭轉。這次改成用原始的 ckiplab/bert-base-chinese 從零訓練，
    完全不經過詞典這一層，直接用真人標籤（250+150批，扣掉沒標
    的共397筆）學習，避免繼承詞典的保守偏誤。

使用方式：
    python prepare_human_from_scratch.py
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合併人工標註資料供BERT從零訓練")
    parser.add_argument("--out-dir", type=str, default="bert_dataset_human_scratch")
    parser.add_argument("--val-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    a = pd.read_csv("annotation_result.csv")[["content", "human_label"]]
    b = pd.read_csv("annotation_result_fresh.csv")[["content", "human_label"]]
    df = pd.concat([a, b], ignore_index=True)

    before = len(df)
    df = df.dropna(subset=["human_label"]).copy()
    df = df.drop_duplicates(subset=["content"])
    print(f"合併後總筆數：{before}，可用（已標註+去重）：{len(df)}")

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

    print("\n[完成]（val跟test用同一份，是完全沒被拿去訓練的held-out集）")
