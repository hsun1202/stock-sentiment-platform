# -*- coding: utf-8 -*-
"""
人工標註樣本抽樣工具
============================================================

目的：
    從全部30檔股票的留言中，抽出一份適合人工標註的樣本，
    用來驗證詞典法／BERT三分類的真實準確率（目前所有驗證都是
    機器方法互相比對，還沒有跟人類判斷比對過）。

抽樣策略（不是純隨機，刻意讓樣本更有資訊量）：
    A. 詞典與BERT判斷不一致的留言（最值得看，兩邊至少一個錯）
    B. 詞典與BERT都判斷為明確正/負且一致的留言（檢查兩者會不會
       同時錯在同一個地方——因為BERT是從詞典蒸餾出來的，一致
       不代表對）
    C. 兩者都判斷為neutral的留言（純隨機基準，確保neutral類別
       也有樣本，不然無法算neutral的precision/recall）

使用方式：
    python build_annotation_sample.py --total 250
"""

import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentiment_lexicon import classify_comment

BERT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def load_stock_list():
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


@torch.no_grad()
def predict_bert(texts, tokenizer, model, device, max_length=64, batch_size=256):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="建立人工標註樣本")
    parser.add_argument("--batch-dir", type=str, default="batch_output")
    parser.add_argument("--model-dir", type=str, default="bert_sentiment_model_3class/final")
    parser.add_argument("--out", type=str, default="annotation_sample.csv")
    parser.add_argument("--pool-per-stock", type=int, default=500, help="每檔股票先抽的候選池大小")
    parser.add_argument("--total", type=int, default=250, help="最終要標註的總筆數")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    STOCK_LIST = load_stock_list()
    rng = np.random.RandomState(args.seed)

    print("=== 建立候選池（每檔股票抽樣） ===")
    pool_frames = []
    for code, name, industry in STOCK_LIST:
        path = os.path.join(args.batch_dir, f"{code}_{name}_comments.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["stock_code", "link", "resolved_date", "content"])
        df["content"] = df["content"].astype(str).str.strip()
        df = df[df["content"].str.len() > 0]
        if len(df) == 0:
            continue
        n = min(args.pool_per_stock, len(df))
        sampled = df.sample(n=n, random_state=args.seed)
        sampled["industry"] = industry
        sampled["stock_name"] = name
        pool_frames.append(sampled)
    pool = pd.concat(pool_frames, ignore_index=True)
    print(f"候選池大小：{len(pool)}")

    print("\n=== 詞典法判斷 ===")
    pool["lexicon_label"] = pool["content"].apply(classify_comment)
    print(pool["lexicon_label"].value_counts())

    print("\n=== BERT三分類判斷 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    preds = predict_bert(pool["content"].tolist(), tokenizer, model, device)
    pool["bert_label"] = [BERT_LABELS[p] for p in preds]
    print(pool["bert_label"].value_counts())

    pool["agree"] = pool["lexicon_label"] == pool["bert_label"]
    disagree = pool[~pool["agree"]]
    agree_nonneutral = pool[pool["agree"] & (pool["lexicon_label"] != "neutral")]
    agree_neutral = pool[pool["agree"] & (pool["lexicon_label"] == "neutral")]

    print(f"\n不一致：{len(disagree)}　雙方一致(非中性)：{len(agree_nonneutral)}　雙方一致(中性)：{len(agree_neutral)}")

    # 配額：不一致優先，其次雙方一致非中性，最後中性基準
    n_disagree = min(len(disagree), int(args.total * 0.40))
    n_agree_nn = min(len(agree_nonneutral), int(args.total * 0.32))
    n_agree_n = args.total - n_disagree - n_agree_nn
    n_agree_n = min(len(agree_neutral), n_agree_n)

    picked = pd.concat([
        disagree.sample(n=n_disagree, random_state=args.seed),
        agree_nonneutral.sample(n=n_agree_nn, random_state=args.seed),
        agree_neutral.sample(n=n_agree_n, random_state=args.seed),
    ], ignore_index=True)

    # 補足到目標總數（若某類別池不夠大）
    shortfall = args.total - len(picked)
    if shortfall > 0:
        remaining = pool.drop(picked.index, errors="ignore")
        remaining = remaining[~remaining["content"].isin(picked["content"])]
        if len(remaining) > 0:
            extra = remaining.sample(n=min(shortfall, len(remaining)), random_state=args.seed)
            picked = pd.concat([picked, extra], ignore_index=True)

    picked = picked.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # 洗牌，標註時不暴露分組
    picked.insert(0, "id", range(1, len(picked) + 1))
    picked["human_label"] = ""  # 待標欄位

    out_cols = ["id", "stock_code", "stock_name", "industry", "resolved_date", "content",
                "lexicon_label", "bert_label", "human_label"]
    picked[out_cols].to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\n=== 完成 ===")
    print(f"實際抽出：{len(picked)} 筆（不一致 {n_disagree}、雙方一致非中性 {n_agree_nn}、雙方一致中性 {n_agree_n}）")
    print(f"已輸出至 {args.out}（human_label欄位是空的，等待標註）")
