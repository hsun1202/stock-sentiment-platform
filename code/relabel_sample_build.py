# -*- coding: utf-8 -*-
"""
抽樣給 Claude 重新標註（更準的分類器檢驗）
  母體：2015-2024 個股留言（sentiment_output_267_2015_2024/*_comment_labels.csv，含 Breeze 標籤），
        依留言時間歸到交易日（同 build_dataset_v3.py：前一交易日 13:30 後 ~ 今天 13:30），只取「今天收盤前」的留言（pre+intra）
  抽樣：Breeze 有表態（正+負≥1）的股票日中隨機抽 400 個（seed=0）；每個股票日最多 20 則（隨機）
  輸出：relabel/sample.csv（id, stock_code, stock_name, Date, title, push_type, content, breeze）
        relabel/batch_XX.txt（每批 250 則，給 Claude 閱讀：id｜股票｜標題｜推噓｜留言）——不含 Breeze 標籤，避免影響判斷
python relabel_sample_build.py
"""
import glob
import os

import numpy as np
import pandas as pd

os.makedirs("relabel", exist_ok=True)
cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_csv("final_dataset_v3/all_stocks_v3.csv", usecols=["Date"]).Date.unique())))
cal = cal.union(pd.DatetimeIndex(pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"]).Date)).sort_values()
close_dt = cal.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
parts = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in glob.glob("sentiment_output_267_2015_2024/*_comment_labels.csv")]
cm = pd.concat(parts, ignore_index=True)
cm = cm[(cm.comment_time != "") & (cm.time_anomaly != "1")].copy()
ct = pd.to_datetime(cm.comment_time).values
i = np.searchsorted(close_dt, ct, side="left")
ok = i < len(cal)
cm = cm[ok].copy()
cm["Date"] = cal[i[ok]]
cm = cm[(cm.Date >= "2015-01-05") & (cm.Date <= "2024-12-31")]
cm["expr"] = cm.text_sentiment.isin(["positive", "negative"]).astype(int)
days = cm.groupby(["stock_code", "Date"]).agg(n=("expr", "size"), n_expr=("expr", "sum")).reset_index()
days = days[days.n_expr >= 1]
print(f"有表態的股票日 {len(days):,}（留言中位數 {days.n.median():.0f}）")
pick = days.sample(400, random_state=0)[["stock_code", "Date"]]
s = cm.merge(pick, on=["stock_code", "Date"])
s = s.groupby(["stock_code", "Date"], group_keys=False).apply(lambda x: x.sample(min(len(x), 20), random_state=0))
arts = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["article_id", "title"], dtype=str, keep_default_na=False).drop_duplicates("article_id")
s = s.merge(arts, on="article_id", how="left")
s = s.sample(frac=1, random_state=1).reset_index(drop=True)      # 打散順序，避免同一股票日連在一起影響判斷
s["id"] = np.arange(1, len(s) + 1)
s = s.rename(columns={"text_sentiment": "breeze"})
s[["id", "stock_code", "stock_name", "Date", "article_id", "title", "push_type", "content", "breeze"]].to_csv("relabel/sample.csv", index=False, encoding="utf-8-sig")
B = 250
for b in range(0, len(s), B):
    with open(f"relabel/batch_{b // B + 1:02d}.txt", "w", encoding="utf-8") as f:
        for r in s.iloc[b:b + B].itertuples():
            f.write(f"{r.id}｜{r.stock_name}｜{str(r.title)[:40]}｜{r.push_type}｜{str(r.content)[:120]}\n")
print(f"抽樣 {len(s):,} 則、{pick.shape[0]} 個股票日、{(len(s) - 1) // B + 1} 批；Breeze 分布 {s.breeze.value_counts(normalize=True).round(3).to_dict()}")
