# -*- coding: utf-8 -*-
"""
公開資訊觀測站「上市公司綜合損益表彙總表」（t163sb04，依年度／季別查全部已申報公司）每日快照。
某公司某季第一次出現在快照的日期＝公布日；數值保留第一次看到的（年初累計數，earnings_signals.py 再換成單季）。
證交所 OpenAPI（t187ap06_L_*）只放最近一個「申報期限已過」的季度，申報期間公布的公司要等期限後才出現，
所以這裡改用觀測站彙總表。累計轉單季、各產業版本欄位名稱不同的處理，參考組員 jameschen108/tw-stock-quant 的 public_data.py。
cd D:/專題claude && python platform_feed/mops_earnings.py
"""
import datetime as dt
import os
import re
import time

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw", "t163sb04")
DB = os.path.join(BASE, "data", "earnings_public.csv")
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
URL = "https://mopsov.twse.com.tw/mops/web/ajax_t163sb04"
NI = ["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主", "本期淨利（淨損）", "本期稅後淨利（淨損）"]   # 沒有子公司的公司歸屬母公司欄是「--」，改用本期淨利
EPS = ["基本每股盈餘（元）"]


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return float("nan")


def parse(html):
    rows = []
    for tb in re.findall(r"<table[^>]*>(.*?)</table>", html, flags=re.S):
        hdr = None
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tb, flags=re.S):
            ths = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<th[^>]*>(.*?)</th>", tr, flags=re.S)]
            if "公司代號" in ths:
                hdr = ths
                continue
            tds = [re.sub(r"<[^>]+>", "", x).replace("&nbsp;", "").strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
            if hdr and tds and re.fullmatch(r"\d{4}", tds[0]) and len(tds) == len(hdr):
                d = dict(zip(hdr, tds))
                ni = next((v for v in (_num(d.get(k)) for k in NI if k in d) if v == v), float("nan"))
                eps = next((v for v in (_num(d.get(k)) for k in EPS if k in d) if v == v), float("nan"))
                rows.append((tds[0], tds[1], ni, eps))
    return pd.DataFrame(rows, columns=["code", "name", "ni_ytd", "eps_ytd"]).drop_duplicates("code")


def fetch(year, season, today):
    data = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": "sii",
            "year": year - 1911, "season": f"{season:02d}"}
    for tries in range(3):
        try:
            r = requests.post(URL, data=data, headers=H, timeout=90)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(10 * (tries + 1))
    else:
        return pd.DataFrame(columns=["code", "name", "ni_ytd", "eps_ytd"])
    time.sleep(4)
    html = r.content.decode("utf-8", errors="replace")
    os.makedirs(RAW, exist_ok=True)
    open(os.path.join(RAW, f"{year}Q{season}_{today:%Y%m%d}.html"), "w", encoding="utf-8").write(html)
    return parse(html)


def update(today=None, log=print):
    today = today or dt.date.today()
    q0 = pd.Period(today, "Q") - 1                    # 最近一個已結束的季度＋前一季（晚申報的）
    db = pd.read_csv(DB, dtype={"code": str}) if os.path.exists(DB) else pd.DataFrame(
        columns=["code", "name", "year", "season", "ni_ytd", "eps_ytd", "first_seen"])
    seen = set(zip(db.code, db.year.astype(int), db.season.astype(int))) if len(db) else set()
    boot = db.empty                                   # 第一次跑：之前已公布的不知道哪天，first_seen 留空（改用 TEJ 發布日）
    new = []
    for q in (q0, q0 - 1):
        snap = fetch(q.year, q.quarter, today)
        add = snap[[(c, q.year, q.quarter) not in seen for c in snap.code]].copy()
        add["year"], add["season"], add["first_seen"] = q.year, q.quarter, (None if boot else today.isoformat())
        new.append(add)
        log(f"[季報快照 {today}] {q}：目前 {len(snap)} 家，新出現 {len(add)} 家")
    db = pd.concat([db] + new, ignore_index=True)
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    db.to_csv(DB, index=False, encoding="utf-8-sig")
    return db


if __name__ == "__main__":
    update()
