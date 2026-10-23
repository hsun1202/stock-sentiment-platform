# -*- coding: utf-8 -*-
"""
H 節：抓 2025-09-01～2026-10-22 證交所公開資料（每日行情 MI_INDEX、除權息 TWT49U），沿用 papertrade/tw_data.py 的快取（papertrade/raw/）。
輸出 ai_strategies/twse_2026_prices.csv、twse_2026_exright.csv。
cd D:/專題claude && PYTHONIOENCODING=utf-8 python ai_strategies/fetch_twse_2026.py
"""
import sys

import pandas as pd

sys.path.insert(0, "papertrade")
import tw_data  # noqa: E402

START, END = "2025-09-01", "2026-10-22"
days = tw_data.trading_days(START, END)
print(f"交易日 {len(days)} 天：{days[0]}～{days[-1]}", flush=True)
frames, miss = [], []
for i, d in enumerate(days):
    q = tw_data.daily_quotes(d)
    if q is None:
        miss.append(d)
    else:
        frames.append(q)
    if (i + 1) % 20 == 0:
        print(f"  行情 {i + 1}/{len(days)}（缺 {len(miss)}）", flush=True)
px = pd.concat(frames, ignore_index=True)
px.to_csv("ai_strategies/twse_2026_prices.csv", index=False, encoding="utf-8-sig")
ex = []
for p in pd.period_range(START[:7], END[:7], freq="M"):
    a = max(p.start_time.strftime("%Y-%m-%d"), START)
    b = min(p.end_time.strftime("%Y-%m-%d"), END)
    ex.append(tw_data.exright(a, b))
ex = pd.concat(ex, ignore_index=True).drop_duplicates(["date", "code"])
ex.to_csv("ai_strategies/twse_2026_exright.csv", index=False, encoding="utf-8-sig")
print(f"完成：行情 {px.date.nunique()} 天、{px.code.nunique()} 個代號；缺 {miss}；除權息 {len(ex)} 筆", flush=True)
