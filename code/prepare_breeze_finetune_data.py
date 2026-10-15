# -*- coding: utf-8 -*-
"""
準備Breeze-7B LoRA微調用的train/val切分
============================================================
用 all_human_labels.csv（392筆，已修正5筆客觀標錯的版本）切出
train/val，val完全不參與訓練，用來公平比較「微調前 vs 微調後」
的zero-shot準確率有沒有真的進步。

使用方式：
    python prepare_breeze_finetune_data.py
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="all_human_labels.csv")
    parser.add_argument("--label-col", type=str, default="human_label")
    parser.add_argument("--out-dir", type=str, default="breeze_finetune_data")
    parser.add_argument("--val-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.rename(columns={args.label_col: "human_label"})
    df = df.dropna(subset=["human_label", "content"]).copy()
    df = df.drop_duplicates(subset=["content"])
    print(f"可用筆數（去重後）：{len(df)}")
    print(df["human_label"].value_counts())

    train_df, val_df = train_test_split(
        df, test_size=args.val_size, stratify=df["human_label"], random_state=args.seed
    )

    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    train_df[["content", "human_label"]].to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df[["content", "human_label"]].to_csv(val_path, index=False, encoding="utf-8-sig")

    print(f"\ntrain: {len(train_df)} 筆 -> {train_path}")
    print(train_df["human_label"].value_counts())
    print(f"\nval: {len(val_df)} 筆 -> {val_path}（完全held-out，不參與訓練）")
    print(val_df["human_label"].value_counts())
