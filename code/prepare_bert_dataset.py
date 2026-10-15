# -*- coding: utf-8 -*-
"""
BERT弱監督訓練資料準備
============================================================

目的：
    PTT沒有人工標註的情緒語料庫，改用留言本身的「推文型態」
    (push_type) 當弱監督標籤：
        推 -> positive (1)
        噓 -> negative (0)
        →  -> 捨棄（語意最不明確，不當標籤用）
    這樣不用人工標註，就能用30檔股票、65萬則留言產生訓練資料。

已知限制：
    推/噓不完全等於真實情緒（可能有反串、嘴砲推文、單純+1），
    這是弱監督方法本身的局限，訓練出來的模型學到的是「PTT鄉民
    推噓習慣」，不是100%純情緒判斷，寫報告時需要說明。

類別不平衡處理：
    原始推文佔比約85%、噓文約15%，差距太大會讓模型傾向於
    全部猜推文。這裡採用「多數類別(推)降採樣到跟少數類別(噓)
    等量」的簡單作法，讓訓練集正負1:1。

使用方式：
    python prepare_bert_dataset.py
"""

import os
import argparse
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split


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
        df = pd.read_csv(path, usecols=["stock_code", "push_type", "content"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="準備BERT弱監督訓練資料（推/噓當標籤）")
    parser.add_argument("--batch-dir", type=str, default="batch_output")
    parser.add_argument("--out-dir", type=str, default="bert_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    STOCK_LIST = load_stock_list()

    print("=== 讀取全部留言 ===")
    df = load_all_comments(args.batch_dir, STOCK_LIST)
    print(f"總留言數：{len(df)}")

    # 只保留推/噓，捨棄→，並過濾空白內容
    df = df[df["push_type"].isin(["推", "噓"])].copy()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"].str.len() > 0]
    df["label"] = (df["push_type"] == "推").astype(int)

    print(f"可用（推+噓）留言數：{len(df)}")
    print(df["push_type"].value_counts())

    # 多數類別(推)降採樣到跟噓文等量，處理類別不平衡
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n = min(len(pos), len(neg))
    pos_sampled = pos.sample(n=n, random_state=args.seed)
    balanced = pd.concat([pos_sampled, neg], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"\n降採樣平衡後：正負各 {n} 筆，共 {len(balanced)} 筆")

    # train/val/test = 90/5/5，分層抽樣維持類別比例
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

    print("\n[完成] BERT訓練資料已準備好")
