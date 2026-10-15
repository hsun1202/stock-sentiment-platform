# -*- coding: utf-8 -*-
"""開盤半小時留言分類（加速版）：每個交易日最多分類 150 則（intraday_classify_full.py 全量 80 萬則要約3小時）。
- 沿用全量版已完成的部分（first30_labels.partial.csv 對應依 comment_time 排序後的前 N 則，那些日子全部保留）
- 其餘日子每天隨機抽樣，讓「已完成 + 新抽樣」每天合計 ≤ 150 則（原本每日抽樣只有約35則、表態約23則，雜訊大）
輸出：market_sentiment/first30_labels.csv（欄位含 comment_time、content、text_sentiment、sampled）
python intraday_classify_capped.py"""
import os, time
import pandas as pd
from breeze_infer_267_fast import load_model, FastClassifier

CAP = 150
OUT, OLD, PART = ("market_sentiment/first30_labels.csv", "market_sentiment/first30_labels.partial.csv",
                  "market_sentiment/first30_capped.partial.csv")
arts = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["article_id", "title"], dtype=str, keep_default_na=False)
chat = set(arts.loc[arts.title.str.contains(r"盤[中後]\s*閒聊", regex=True), "article_id"])
parts = []
for ch in pd.read_csv("ptt_2015_2024/comments_clean.csv", chunksize=2_000_000, dtype=str, keep_default_na=False,
                      usecols=["article_id", "comment_time", "time_anomaly", "content"]):
    ch = ch[ch.article_id.isin(chat) & (ch.comment_time != "") & (ch.time_anomaly != "1")]
    hm = ch.comment_time.str[11:13].astype(int) * 100 + ch.comment_time.str[14:16].astype(int)
    parts.append(ch[(hm >= 900) & (hm < 930)])
df = pd.concat(parts, ignore_index=True).sort_values("comment_time").reset_index(drop=True)  # 與全量版相同排序
old = pd.read_csv(OLD)["text_sentiment"].tolist()
done = df.iloc[:len(old)].copy()
done["text_sentiment"] = old
done["sampled"] = 0
df["day"] = df.comment_time.str[:10]
done["day"] = done.comment_time.str[:10]
last_day = done.day.iloc[-1]
done = done[done.day != last_day]                      # 最後一天可能只做了一半 → 丟掉，改用抽樣
rest = df[~df.day.isin(set(done.day))]
todo = rest.groupby("day", group_keys=False).apply(lambda x: x.sample(min(len(x), CAP), random_state=0)).reset_index(drop=True)
print(f"開盤半小時留言 {len(df):,} 則；沿用已分類 {len(done):,} 則（{done.day.nunique()} 天全量）；"
      f"其餘 {rest.day.nunique()} 天每天抽≤{CAP} 則，共 {len(todo):,} 則待分類", flush=True)
labels = pd.read_csv(PART)["text_sentiment"].tolist() if os.path.exists(PART) else []
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=16, mode="constrained")
t0, st = time.time(), len(labels)
texts = todo.content.tolist()
while len(labels) < len(texts):
    s = len(labels)
    labels += clf.classify(texts[s:s + 20000])
    pd.DataFrame({"text_sentiment": labels}).to_csv(PART, index=False)
    r = (len(labels) - st) / (time.time() - t0)
    print(f"  進度 {len(labels):,}/{len(texts):,}，{r:.0f} 則/秒，預估剩 {(len(texts)-len(labels))/r/60:.0f} 分鐘", flush=True)
todo["text_sentiment"] = labels
todo["sampled"] = 1
out = pd.concat([done, todo], ignore_index=True).drop(columns="day")
out.to_csv(OUT, index=False, encoding="utf-8-sig")
os.remove(PART)
print(f"完成 → {OUT}：{len(out):,} 則；每日中位數 {out.comment_time.str[:10].value_counts().median():.0f} 則", flush=True)
