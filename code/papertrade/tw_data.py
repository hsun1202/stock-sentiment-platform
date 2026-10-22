# -*- coding: utf-8 -*-
"""證交所公開資料：交易日曆（FMTQIK）、每日收盤行情（MI_INDEX）、除權息計算結果（TWT49U）。原始 JSON 快取在 raw/。"""
import json
import os
import time

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 2.5          # 證交所每次請求間隔（秒）；太密會被暫時封鎖


def _get(url, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, headers=H, timeout=30)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (k + 1))
    return None


def _cached(kind, key, url, refresh=False):
    """抓一次就存檔；refresh=True 時重抓（用在當月日曆、最近的除權息）。回傳 dict 或 None。"""
    d = os.path.join(RAW, kind)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{key}.json")
    if os.path.exists(path) and not refresh:
        return json.load(open(path, encoding="utf-8"))
    j = _get(url)
    time.sleep(DELAY)
    if j is not None and j.get("stat") == "OK":
        json.dump(j, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        return j
    return j


def roc_to_date(s):
    """'115/10/21' 或 '115年10月21日' → '2026-10-21'"""
    s = s.strip().replace("年", "/").replace("月", "/").replace("日", "")
    y, m, d = s.split("/")
    return f"{int(y) + 1911}-{int(m):02d}-{int(d):02d}"


def num(x):
    x = str(x).replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return float("nan")      # '--'（當天沒成交）


def trading_days(start, end):
    """證交所官方交易日（含 end 當月；當月每次重抓，才會包含最新的日子）。"""
    days = []
    this_month = pd.Timestamp.today().strftime("%Y%m")
    for p in pd.period_range(start[:7], end[:7], freq="M"):
        key = p.strftime("%Y%m")
        j = _cached("calendar", key, f"https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date={key}01&response=json",
                    refresh=(key >= this_month))
        if j and j.get("stat") == "OK":
            days += [roc_to_date(r[0]) for r in j["data"]]
    return sorted(d for d in set(days) if start <= d <= end)


def daily_quotes(day):
    """某交易日全部上市證券的收盤行情 → DataFrame；資料還沒公布時回傳 None。"""
    key = day.replace("-", "")
    j = _cached("mi_index", key, f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={key}&type=ALLBUT0999&response=json")
    if not j or j.get("stat") != "OK":
        path = os.path.join(RAW, "mi_index", f"{key}.json")
        if os.path.exists(path):
            os.remove(path)
        return None
    for t in j.get("tables", []):
        f = t.get("fields") or []
        if "證券代號" in f and "開盤價" in f:
            df = pd.DataFrame(t["data"], columns=f)
            out = pd.DataFrame({"code": df["證券代號"].str.strip(), "name": df["證券名稱"].str.strip(), "date": day,
                                "volume": df["成交股數"].map(num), "value": df["成交金額"].map(num),
                                "open": df["開盤價"].map(num), "high": df["最高價"].map(num),
                                "low": df["最低價"].map(num), "close": df["收盤價"].map(num)})
            return out
    return None


def exright(start, end, refresh=False):
    """除權息計算結果表 → DataFrame（date, code, prev_close, ref_price, kind）。"""
    a, b = start.replace("-", ""), end.replace("-", "")
    j = _cached("twt49u", f"{a}_{b}", f"https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={a}&endDate={b}&response=json",
                refresh=refresh)
    if not j or j.get("stat") != "OK" or not j.get("data"):
        return pd.DataFrame(columns=["date", "code", "prev_close", "ref_price", "kind"])
    df = pd.DataFrame(j["data"], columns=j["fields"])
    return pd.DataFrame({"date": df["資料日期"].map(roc_to_date), "code": df["股票代號"].str.strip(),
                         "prev_close": df["除權息前收盤價"].map(num), "ref_price": df["除權息參考價"].map(num),
                         "kind": df["權/息"].str.strip()})
