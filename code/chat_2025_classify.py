# -*- coding: utf-8 -*-
"""2025 年閒聊串抽樣留言 → Breeze-7B 強制三選一（與 2015-2024 的 market_sentiment_classify.py 完全相同的分類器與設定）
每 5,000 則存一次（chat_labels_2025.partial.csv），中斷後重跑自動接續。
python chat_2025_classify.py"""
import os
import time

import pandas as pd

from breeze_infer_267_fast import FastClassifier, load_model

IN, OUT = "market_sentiment/chat_sample_2025.csv", "market_sentiment/chat_labels_2025.csv"
PART, CHUNK = "market_sentiment/chat_labels_2025.partial.csv", 5000
df = pd.read_csv(IN, dtype=str, keep_default_na=False)
texts = df.content.tolist()
labels = pd.read_csv(PART)["text_sentiment"].tolist() if os.path.exists(PART) else []
print(f"共 {len(texts):,} 則，已完成 {len(labels):,}", flush=True)
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=16, mode="constrained")
t0, start = time.time(), len(labels)
while len(labels) < len(texts):
    s = len(labels)
    labels += clf.classify(texts[s:s + CHUNK])
    pd.DataFrame({"text_sentiment": labels}).to_csv(PART, index=False)
    rate = (len(labels) - start) / (time.time() - t0)
    print(f"  進度 {len(labels):,}/{len(texts):,}，{rate:.0f} 則/秒，預估剩 {(len(texts) - len(labels)) / rate / 60:.0f} 分鐘", flush=True)
df["text_sentiment"] = labels
df.to_csv(OUT, index=False, encoding="utf-8-sig")
os.remove(PART)
print("完成 →", OUT, df.text_sentiment.value_counts(normalize=True).round(3).to_dict(), flush=True)
