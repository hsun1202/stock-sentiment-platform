# -*- coding: utf-8 -*-
"""
2025 年 PTT 個股留言（給「只用 PTT 情緒」的 2025 樣本外檢驗）
  來源：pttweb_stock_2025_stocks/raw_articles.jsonl（只抓標題比對到 267 檔的 3,667 篇；比對規則＝match_ptt_2015_2024_267.find_matches_v2）
  留言時間「MM/DD HH:MM」補年份（用發文年份，跨年 +1）；留言早於發文超過 1 分鐘視為時間異常、排除
  交易日：(前一交易日 13:30, 今天 13:30] 歸今天；今天 09:00 前＝pre，09:00~13:30＝intra（同 build_dataset_v3.py）
  清理：簡化版（去開頭冒號、空白、純網址、無文字）；2015-2024 用的是完整清理版
輸出：ptt_2025_stock_comments.csv（stock_code, Date, part, comment_time, content）
python stock_ptt_2025_prep.py
"""
import importlib.util
import json
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("m", "match_ptt_2015_2024_267.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

rows = []
with open("pttweb_stock_2025_stocks/raw_articles.jsonl", encoding="utf-8") as f:
    for line in f:
        a = json.loads(line)
        a = a.get("article") or a
        codes = m.find_matches_v2(a.get("title", ""))
        if not codes:
            continue
        codes = [c[0] if isinstance(c, (tuple, list)) else c for c in codes]
        post = datetime.fromtimestamp(int(a["aid_time"]) / 1e9)
        for r in a.get("recommend_data") or []:
            mm = re.search(r"(\d{2})/(\d{2})\s+(\d{2}):(\d{2})", (r.get("ip_datetime") or ""))
            if not mm:
                continue
            mo, d, hh, mi = map(int, mm.groups())
            yr = post.year + (1 if mo < post.month - 6 else 0)
            try:
                t = datetime(yr, mo, d, hh, mi)
            except ValueError:
                continue
            if t < post - timedelta(minutes=1):
                continue
            text = re.sub(r"^:\s*", "", r.get("text", "")).strip()
            if not text or re.fullmatch(r"(https?://\S+\s*)+", text) or not re.search(r"[一-鿿A-Za-z]", text):
                continue
            for c in codes:
                rows.append({"stock_code": str(c), "article_id": a.get("article_aid"), "comment_time": t, "content": text})
cm = pd.DataFrame(rows)
print(f"留言（清理後，依股票展開）{len(cm):,} 則；{cm.article_id.nunique():,} 篇；{cm.stock_code.nunique()} 檔")
cal = pd.to_datetime(pd.read_csv("chip_raw/trading_days_2025.csv", dtype=str)["date"], format="%Y%m%d")
cal = pd.DatetimeIndex(sorted(set(cal) | {pd.Timestamp("2024-12-31")}))
close_dt = cal.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
open_dt = cal.values + np.timedelta64(9 * 3600, "s")
i = np.searchsorted(close_dt, cm.comment_time.values, side="left")
keep = i < len(cal)
cm, i = cm[keep].copy(), i[keep]
cm["Date"] = cal[i]
cm["part"] = np.where(cm.comment_time.values < open_dt[i], "pre", "intra")
cm = cm[cm.Date >= "2025-01-02"]
cm.to_csv("ptt_2025_stock_comments.csv", index=False, encoding="utf-8-sig")
print(f"歸到 2025 交易日：{len(cm):,} 則；盤前 {(cm.part == 'pre').mean():.1%}；每股票日中位數 {cm.groupby(['stock_code', 'Date']).size().median():.0f} 則")
