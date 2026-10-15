# -*- coding: utf-8 -*-
"""
BERT訓練資料準備（V3：三分類，納入neutral，修正V2的分布偏移問題）
============================================================

背景：
    V2（詞典判斷結果當標籤）只用了詞典判斷為明確positive/negative
    的留言訓練二分類模型，完全沒看過neutral留言（佔全部65萬則的
    92%）。套用到全部留言後，模型在訓練時沒看過的中性內容上
    表現不穩定、系統性地偏向positive，導致整體情緒因子從詞典版
    的接近0暴衝到0.33，跟詞典版的逐日相關係數只有0.26。

改進做法：
    改成三分類（positive/neutral/negative），把詞典判斷為neutral
    的留言也真正納入訓練，讓模型學會辨認「這則沒有明顯情緒傾向」，
    而不是被迫在正負之間硬選一個。三類別各自降採樣到跟最小類別
    （通常是positive或negative）等量，避免neutral佔絕大多數導致
    模型無論如何都學到「猜neutral最安全」。

使用方式：
    python prepare_bert_dataset_3class.py
"""

import os
import argparse
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_lexicon import classify_comment

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def load_stock_list():
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


def load_all_comments(batch_dir, stock_list):
    frames = []
    for code, name, industry in stock_list:
        path = os.path.join(batch_dir, f"{code}_{name}_comments.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["stock_code", "content"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="準備BERT三分類訓練資料（詞典法判斷結果當標籤）")
    parser.add_argument("--batch-dir", type=str, default="batch_output")
    parser.add_argument("--out-dir", type=str, default="bert_dataset_3class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    STOCK_LIST = load_stock_list()

    print("=== 讀取全部留言 ===")
    df = load_all_comments(args.batch_dir, STOCK_LIST)
    print(f"總留言數：{len(df)}")

    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"].str.len() > 0].copy()

    print("=== 用詞典法幫全部留言打標籤 ===")
    df["lexicon_label"] = df["content"].apply(classify_comment)
    print(df["lexicon_label"].value_counts())

    df["label"] = df["lexicon_label"].map(LABEL_MAP)

    # 三類別各自降採樣到跟最小類別等量，避免neutral獨大
    counts = df["label"].value_counts()
    n = int(counts.min())
    print(f"\n三類別原始筆數：{counts.to_dict()}")
    print(f"降採樣目標（取最小類別）：每類 {n} 筆")

    parts = []
    for lbl in [0, 1, 2]:
        sub = df[df["label"] == lbl]
        parts.append(sub.sample(n=n, random_state=args.seed) if len(sub) > n else sub)
    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"降採樣平衡後：三類各 {n} 筆，共 {len(balanced)} 筆")

    train_df, temp_df = train_test_split(
        balanced, test_size=0.10, stratify=balanced["label"], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=args.seed
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(args.out_dir, f"{name}.csv")
        split_df[["content", "label"]].rename(columns={"content": "text"}).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        print(f"{name}: {len(split_df)} 筆 -> {out_path}")

    print("\n[完成] BERT三分類訓練資料已準備好（0=negative, 1=neutral, 2=positive）")
