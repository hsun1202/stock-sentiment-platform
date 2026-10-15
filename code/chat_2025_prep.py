# -*- coding: utf-8 -*-
"""
2025 年閒聊串（盤中/盤後閒聊）留言 → 交易日 → 抽樣（給 Breeze 分類），做「市場層級 PTT 情緒」的 2025 樣本外測試
  來源：pttweb_stock_2025/raw_articles.jsonl（crawl_pttweb_api.py，只抓 500 篇閒聊串）
  留言時間「MM/DD HH:MM」補年份：用文章發文年份；若月份早於發文月份（跨年）則 +1 年
  交易日歸屬：同 market_sentiment_prep.py——(前一交易日 13:30, 今天 13:30] 歸今天
  清理（簡化版；2015-2024 用的是 comments_clean.csv 的完整清理）：去掉開頭冒號、空白、純網址、純符號；
  每個交易日隨機抽 ≤300 則（random_state=0，與 2015-2024 相同）
輸出：market_sentiment/chat_2025_comments.csv、market_sentiment/chat_sample_2025.csv、chat_daily_volume_2025.csv
python chat_2025_prep.py
"""
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

rows = []
with open("pttweb_stock_2025/raw_articles.jsonl", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        a = j.get("article") or j
        title = a.get("title", "")
        if not re.search(r"盤[中後]\s*閒聊|盤中聊天|盤後聊天", title):
            continue
        post = datetime.fromtimestamp(int(a["aid_time"]) / 1e9)
        for r in a.get("recommend_data") or []:
            raw = (r.get("ip_datetime") or "").strip()
            m = re.search(r"(\d{2})/(\d{2})\s+(\d{2}):(\d{2})", raw)
            if not m:
                continue
            mo, d, hh, mm = map(int, m.groups())
            yr = post.year + (1 if mo < post.month - 6 else 0)
            try:
                t = datetime(yr, mo, d, hh, mm)
            except ValueError:
                continue
            text = re.sub(r"^:\s*", "", r.get("text", "")).strip()
            rows.append({"article_id": a.get("article_aid"), "title": title, "comment_time": t, "content": text, "type": r.get("type")})
cm = pd.DataFrame(rows)
print(f"閒聊串留言（原始）{len(cm):,} 則，{cm.article_id.nunique()} 篇")
cm = cm[cm.content.str.len() > 0]
cm = cm[~cm.content.str.fullmatch(r"(https?://\S+\s*)+")]
cm = cm[cm.content.str.contains(r"[一-鿿A-Za-z]", regex=True)]
print(f"簡化清理後 {len(cm):,} 則")

cal = pd.to_datetime(pd.read_csv("chip_raw/trading_days_2025.csv", dtype=str)["date"], format="%Y%m%d")
cal = pd.DatetimeIndex(sorted(set(cal) | {pd.Timestamp("2024-12-31")}))
close_dt = cal.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
i = np.searchsorted(close_dt, cm.comment_time.values, side="left")
cm = cm[i < len(cal)].copy()
cm["Date"] = cal[i[i < len(cal)]]
cm = cm[cm.Date >= "2025-01-02"]
cm.to_csv("market_sentiment/chat_2025_comments.csv", index=False, encoding="utf-8-sig")
vol = cm.groupby("Date").size().rename("chat_n").reset_index()
vol.to_csv("market_sentiment/chat_daily_volume_2025.csv", index=False, encoding="utf-8-sig")
print(f"歸到 {vol.Date.nunique()} 個交易日，每日中位數 {vol.chat_n.median():.0f} 則（2015-2024 參考：見 chat_daily_volume.csv）")
samp = cm.groupby("Date", group_keys=False).apply(lambda x: x.sample(min(len(x), 300), random_state=0))
samp[["Date", "article_id", "comment_time", "content"]].to_csv("market_sentiment/chat_sample_2025.csv", index=False, encoding="utf-8-sig")
print(f"抽樣 {len(samp):,} 則（每日最多 300）→ market_sentiment/chat_sample_2025.csv")
