# -*- coding: utf-8 -*-
"""大盤散戶情緒溫度計（2/3）：用與個股分析相同的 Breeze-7B 強制三選一分類器，分類閒聊串抽樣留言。
每 20,000 則存一次（market_sentiment/chat_labels.partial.csv），中斷後重跑自動接續。
python market_sentiment_classify.py"""
import os, time
import pandas as pd
from breeze_infer_267_fast import load_model, FastClassifier

IN, OUT = "market_sentiment/chat_sample.csv", "market_sentiment/chat_labels.csv"
PART, CHUNK = "market_sentiment/chat_labels.partial.csv", 20000
df = pd.read_csv(IN, dtype=str, keep_default_na=False)
texts = df.content.tolist()
labels = pd.read_csv(PART)["text_sentiment"].tolist() if os.path.exists(PART) else []
print(f"共 {len(texts):,} 則，已完成 {len(labels):,}", flush=True)
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=16, mode="constrained")  # 32 在閒聊長留言時 VRAM 接近滿載只有 33 則/秒
t0, start = time.time(), len(labels)
while len(labels) < len(texts):
    s = len(labels)
    labels += clf.classify(texts[s:s + CHUNK])
    pd.DataFrame({"text_sentiment": labels}).to_csv(PART, index=False)
    done = len(labels) - start
    rate = done / (time.time() - t0)
    print(f"  進度 {len(labels):,}/{len(texts):,}，{rate:.0f} 則/秒，預估剩 {(len(texts)-len(labels))/rate/60:.0f} 分鐘", flush=True)
df["text_sentiment"] = labels
df.to_csv(OUT, index=False, encoding="utf-8-sig")
os.remove(PART)
print("完成 →", OUT, df.text_sentiment.value_counts(normalize=True).round(3).to_dict(), flush=True)
