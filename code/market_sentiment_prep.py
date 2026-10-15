# -*- coding: utf-8 -*-
"""
大盤散戶情緒溫度計（1/3）：準備盤中/盤後閒聊串的留言
============================================================
  - 閒聊串：標題含「盤中閒聊／盤後閒聊」的每日討論串（2015-2024 共 5,095 篇、623 萬則留言），散戶即時心情最集中的地方
  - 留言依「自己的時間」分到交易日 t 的時間窗 (t-1 收盤 13:30, t 收盤 13:30]（盤後閒聊的留言會算到隔一個交易日）
    → 交易日 t 的指數只用 t 收盤前已發的留言，預測 t 收盤之後的大盤不會偷看答案
  - 每個交易日：全部留言數（關注度，不抽樣）；隨機抽最多 300 則做情緒分類（每天估計誤差約 ±3%）
  - 使用清理後的留言（comments_clean.csv；已拼接、去廢話），排除 time_anomaly
交易日曆：證交所官方 FMTQIK（chip_raw/calendar/*.json）
輸出：market_sentiment/chat_daily_volume.csv、market_sentiment/chat_sample.csv
python market_sentiment_prep.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd

OUT = "market_sentiment"
PER_DAY = 300
os.makedirs(OUT, exist_ok=True)


def roc(s):
    y, m, d = s.strip().split("/")
    return pd.Timestamp(int(y) + 1911, int(m), int(d))


rows = []
for f in sorted(glob.glob("chip_raw/calendar/*.json")):
    j = json.load(open(f, encoding="utf-8"))
    for r in j.get("data") or []:
        rows.append({"Date": roc(r[0]), "taiex": float(r[4].replace(",", "")),
                     "turnover": float(r[2].replace(",", ""))})
cal = pd.DataFrame(rows).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
cal.to_csv(os.path.join(OUT, "taiex_official.csv"), index=False, encoding="utf-8-sig")
print(f"官方大盤 {cal.Date.min().date()}~{cal.Date.max().date()}，{len(cal)} 個交易日")
CAL = cal.Date.values
close_dt = CAL + np.timedelta64(13 * 3600 + 30 * 60, "s")

arts = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["article_id", "title"], dtype=str, keep_default_na=False)
chat = set(arts.loc[arts.title.str.contains(r"盤[中後]\s*閒聊|盤中聊天|盤後聊天", regex=True), "article_id"])
print(f"閒聊串 {len(chat):,} 篇")
parts = []
for ch in pd.read_csv("ptt_2015_2024/comments_clean.csv", chunksize=2_000_000, dtype=str, keep_default_na=False,
                      usecols=["article_id", "comment_time", "time_anomaly", "content"]):
    ch = ch[ch.article_id.isin(chat) & (ch.comment_time != "") & (ch.time_anomaly != "1")]
    parts.append(ch)
cm = pd.concat(parts, ignore_index=True)
i = np.searchsorted(close_dt, pd.to_datetime(cm.comment_time).values, side="left")
cm = cm[i < len(CAL)].copy()
cm["Date"] = CAL[i[i < len(CAL)]]
vol = cm.groupby("Date").size().rename("chat_n").reset_index()
vol.to_csv(os.path.join(OUT, "chat_daily_volume.csv"), index=False, encoding="utf-8-sig")
print(f"閒聊留言（清理後、有時間）{len(cm):,} 則 → {vol.Date.nunique()} 個交易日，每日中位數 {vol.chat_n.median():.0f} 則")
samp = cm.groupby("Date", group_keys=False).apply(lambda x: x.sample(min(len(x), PER_DAY), random_state=0))
samp[["Date", "article_id", "comment_time", "content"]].to_csv(os.path.join(OUT, "chat_sample.csv"), index=False,
                                                                encoding="utf-8-sig")
print(f"抽樣 {len(samp):,} 則（每日最多 {PER_DAY}）；不重複內容 {samp.content.str[:200].nunique():,} 種")
