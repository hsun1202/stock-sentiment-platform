# -*- coding: utf-8 -*-
"""
用微調後的BERT模型，對30檔股票的全部留言做情緒推論
============================================================

目的：
    train_bert_sentiment.py 只在推/噓的弱監督子集上訓練。
    這支程式把訓練好的模型套用到 batch_output/ 裡「全部」留言
    （不管有沒有推/噓標籤），逐則判斷positive/negative/neutral，
    再依文章、依交易日彙總成情緒因子，輸出格式跟
    sentiment_lexicon.py 的批次模式一致，方便直接替換進
    merge_datasets.py 的流程。

    正負機率介於[neutral_low, neutral_high]之間的留言判為中立，
    避免模型信心很低的情況被硬分成正面或負面。

使用方式：
    python infer_bert_sentiment.py --model-dir bert_sentiment_model/final
"""

import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_stock_list():
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


LABEL_NAMES_3CLASS = {0: "negative", 1: "neutral", 2: "positive"}


@torch.no_grad()
def predict_batch(texts, tokenizer, model, device, max_length=64, batch_size=256):
    """二分類模型用：回傳每則留言 P(positive)。"""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(positive)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


@torch.no_grad()
def predict_batch_argmax(texts, tokenizer, model, device, max_length=64, batch_size=256):
    """三分類模型用：直接取機率最高的類別，不用手動設中立門檻。"""
    all_preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


def aggregate_by_article(df, date_col):
    def compute_factor(group):
        pos = (group["text_sentiment"] == "positive").sum()
        neg = (group["text_sentiment"] == "negative").sum()
        total = len(group)
        factor = (pos - neg) / total if total > 0 else 0.0
        return pd.Series({
            "total_comments": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": total - pos - neg,
            "text_sentiment_factor": round(factor, 4),
        })

    grouped = df.groupby(["stock_code", "link", date_col], as_index=False).apply(
        compute_factor, include_groups=False
    )
    grouped = grouped.rename(columns={date_col: "resolved_date"})
    return grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用微調後的BERT對全部留言做情緒推論")
    parser.add_argument("--batch-dir", type=str, default="batch_output")
    parser.add_argument("--model-dir", type=str, default="bert_sentiment_model/final")
    parser.add_argument("--out-dir", type=str, default="sentiment_output_bert")
    parser.add_argument("--neutral-low", type=float, default=0.4)
    parser.add_argument("--neutral-high", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 使用裝置：{device} ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    num_labels = model.config.num_labels
    print(f"模型類別數：{num_labels}（{'三分類 argmax' if num_labels == 3 else '二分類 + 機率門檻'}）")

    STOCK_LIST = load_stock_list()
    os.makedirs(args.out_dir, exist_ok=True)

    for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
        comments_path = os.path.join(args.batch_dir, f"{code}_{name}_comments.csv")
        out_path = os.path.join(args.out_dir, f"{code}_{name}_sentiment.csv")

        if not os.path.exists(comments_path):
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：找不到留言檔案，跳過")
            continue

        df = pd.read_csv(comments_path)
        date_col = "resolved_date" if "resolved_date" in df.columns else "board_date"

        texts = df["content"].fillna("").astype(str).tolist()

        if num_labels == 3:
            preds = predict_batch_argmax(texts, tokenizer, model, device, batch_size=args.batch_size)
            df["text_sentiment"] = [LABEL_NAMES_3CLASS[p] for p in preds]
        else:
            probs = predict_batch(texts, tokenizer, model, device, batch_size=args.batch_size)
            labels = np.where(probs >= args.neutral_high, "positive",
                      np.where(probs <= args.neutral_low, "negative", "neutral"))
            df["text_sentiment"] = labels

        article_summary = aggregate_by_article(df, date_col)
        article_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）："
              f"{len(df)} 則留言 -> {len(article_summary)} 篇文章的情緒因子")

    print("\n[完成] 全部股票已用BERT模型重新推論情緒")
