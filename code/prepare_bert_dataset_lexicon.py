# -*- coding: utf-8 -*-
"""
BERT訓練資料準備（V2：改用詞典法判斷結果當標籤，而非推/噓）
============================================================

背景：
    第一版用「推/噓」當弱監督標籤訓練BERT，結果測試發現
    push_type跟留言實際情緒幾乎沒有一致對應關係（例如含有
    「不看好」的留言裡，被推的比例高達59.5%），代表用推/噓
    當標籤本質上是在教模型學「這則會不會被推」，不是教它學
    「這則講的是看多還是看空」，訓練出來的模型在「不看好」
    「跌停 塊陶」這類明確負面用語上全部誤判成正面。

改進做法（知識蒸餾）：
    改用已經修好否定詞邏輯漏洞的權重詞典（sentiment_lexicon.py）
    幫全部留言打標籤，只保留詞典判斷為明確positive/negative的
    留言（neutral不拿來訓練，跟原本推文模式一樣只用有明確極性
    的資料）。這樣訓練出來的BERT學的是詞典的判斷邏輯，且能夠
    類推到詞典沒收錄、但語意相近的講法，是詞典法的「泛化版」，
    而不是另一個跟真實情緒無關的雜訊標籤。

使用方式：
    python prepare_bert_dataset_lexicon.py
"""

import os
import argparse
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_lexicon import classify_comment


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
    parser = argparse.ArgumentParser(description="準備BERT訓練資料（詞典法判斷結果當標籤）")
    parser.add_argument("--batch-dir", type=str, default="batch_output")
    parser.add_argument("--out-dir", type=str, default="bert_dataset_lexicon")
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

    # 只保留明確positive/negative，neutral不拿來訓練
    df = df[df["lexicon_label"].isin(["positive", "negative"])].copy()
    df["label"] = (df["lexicon_label"] == "positive").astype(int)

    print(f"\n可用（positive+negative）留言數：{len(df)}")

    # 多數類別降採樣到跟少數類別等量，處理類別不平衡
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n = min(len(pos), len(neg))
    pos_sampled = pos.sample(n=n, random_state=args.seed) if len(pos) > n else pos
    neg_sampled = neg.sample(n=n, random_state=args.seed) if len(neg) > n else neg
    balanced = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"降採樣平衡後：正負各 {n} 筆，共 {len(balanced)} 筆")

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

    print("\n[完成] BERT訓練資料（詞典法標籤版）已準備好")
