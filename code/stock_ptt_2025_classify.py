# -*- coding: utf-8 -*-
"""2025 年 PTT 個股留言 → Breeze-7B 強制三選一（與 2015-2024 個股分析相同的分類器與設定）
同一則留言若因文章提到多檔股票而重複，只分類一次再對應回去。每 5,000 則存一次，中斷後重跑自動接續。
可選環境變數 CAP：每個（股票, 交易日）最多抽幾則（預設不抽樣）。
python stock_ptt_2025_classify.py"""
import os
import time

import pandas as pd

from breeze_infer_267_fast import FastClassifier, load_model

IN, OUT = "ptt_2025_stock_comments.csv", "ptt_2025_stock_labels.csv"
PART, CHUNK = "ptt_2025_stock_labels.partial.csv", 5000
df = pd.read_csv(IN, dtype=str, keep_default_na=False)
cap = int(os.environ.get("CAP", "0"))
if cap:
    df = df.groupby(["stock_code", "Date"], group_keys=False).apply(lambda x: x.sample(min(len(x), cap), random_state=0))
uniq = df.content.drop_duplicates().tolist()
labels = pd.read_csv(PART)["text_sentiment"].tolist() if os.path.exists(PART) else []
print(f"留言 {len(df):,} 則（不重複內容 {len(uniq):,}），已完成 {len(labels):,}", flush=True)
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=16, mode="constrained")
t0, start = time.time(), len(labels)
while len(labels) < len(uniq):
    s = len(labels)
    labels += clf.classify(uniq[s:s + CHUNK])
    pd.DataFrame({"text_sentiment": labels}).to_csv(PART, index=False)
    rate = (len(labels) - start) / (time.time() - t0)
    print(f"  進度 {len(labels):,}/{len(uniq):,}，{rate:.0f} 則/秒，預估剩 {(len(uniq) - len(labels)) / rate / 60:.0f} 分鐘", flush=True)
mp = dict(zip(uniq, labels))
df["text_sentiment"] = df.content.map(mp)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
os.remove(PART)
print("完成 →", OUT, df.text_sentiment.value_counts(normalize=True).round(3).to_dict(), flush=True)
