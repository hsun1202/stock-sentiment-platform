# -*- coding: utf-8 -*-
"""盤中即時情緒：把「盤中/盤後閒聊」在 09:00~09:30 的全部留言分類（先前每日只抽樣約35則），降低開盤情緒的抽樣雜訊。
同一個 Breeze 強制三選一分類器、batch 16；每 20,000 則存檔可續跑 → market_sentiment/first30_labels.csv
python intraday_classify_full.py"""
import os, time, pandas as pd
from breeze_infer_267_fast import load_model, FastClassifier
OUT, PART = "market_sentiment/first30_labels.csv", "market_sentiment/first30_labels.partial.csv"
arts = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["article_id", "title"], dtype=str, keep_default_na=False)
chat = set(arts.loc[arts.title.str.contains(r"盤[中後]\s*閒聊", regex=True), "article_id"])
parts = []
for ch in pd.read_csv("ptt_2015_2024/comments_clean.csv", chunksize=2_000_000, dtype=str, keep_default_na=False,
                      usecols=["article_id", "comment_time", "time_anomaly", "content"]):
    ch = ch[ch.article_id.isin(chat) & (ch.comment_time != "") & (ch.time_anomaly != "1")]
    hm = ch.comment_time.str[11:13].astype(int) * 100 + ch.comment_time.str[14:16].astype(int)
    parts.append(ch[(hm >= 900) & (hm < 930)])
df = pd.concat(parts, ignore_index=True).sort_values("comment_time").reset_index(drop=True)
texts = df.content.tolist()
labels = pd.read_csv(PART)["text_sentiment"].tolist() if os.path.exists(PART) else []
print(f"開盤半小時留言 {len(texts):,} 則，已完成 {len(labels):,}", flush=True)
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=16, mode="constrained")
t0, st = time.time(), len(labels)
while len(labels) < len(texts):
    s = len(labels)
    labels += clf.classify(texts[s:s + 20000])
    pd.DataFrame({"text_sentiment": labels}).to_csv(PART, index=False)
    r = (len(labels) - st) / (time.time() - t0)
    print(f"  進度 {len(labels):,}/{len(texts):,}，{r:.0f} 則/秒，預估剩 {(len(texts)-len(labels))/r/60:.0f} 分鐘", flush=True)
df["text_sentiment"] = labels
df.to_csv(OUT, index=False, encoding="utf-8-sig"); os.remove(PART)
print("完成 →", OUT, flush=True)
