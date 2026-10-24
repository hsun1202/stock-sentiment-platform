# -*- coding: utf-8 -*-
"""
公開資訊觀測站「上市公司營業收入彙總表」（t21sc03，每天重新出表）每日快照。
某公司某月營收「第一次出現在快照」的日期＝公布日（快照在收盤後抓，當天晚上可用 → 隔天開盤可買）。
營收值保留第一次看到的數字（之後的更正不覆蓋），避免回頭用到更正後的數字。
cd D:/專題claude && python platform_feed/mops_revenue.py
"""
import datetime as dt
import os
import re
import time

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw", "t21sc03")
DB = os.path.join(BASE, "data", "revenue_public.csv")
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
URL = "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{y}_{m}_{k}.html"      # k=0 國內公司、1 外國公司（KY）


def parse(html):
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S):
        tds = [re.sub(r"<[^>]+>", "", x).replace("&nbsp;", "").strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        if len(tds) >= 5 and re.fullmatch(r"\d{4}", tds[0]):
            rows.append((tds[0], tds[1], tds[2].replace(",", ""), tds[4].replace(",", "")))
    df = pd.DataFrame(rows, columns=["code", "name", "R", "R_ly"])
    df["R"] = pd.to_numeric(df.R, errors="coerce")
    df["R_ly"] = pd.to_numeric(df.R_ly, errors="coerce")
    return df.dropna(subset=["R"])


def fetch(per, today):
    """per：pandas Period（營收月份）。回傳該月目前已公布的公司；原始頁面存檔。"""
    out = []
    for k in (0, 1):
        url = URL.format(y=per.year - 1911, m=per.month, k=k)
        for tries in range(3):
            try:
                r = requests.get(url, headers=H, timeout=60)
                break
            except requests.RequestException:
                time.sleep(10 * (tries + 1))
        else:
            continue
        time.sleep(4)
        if r.status_code != 200:
            continue
        html = r.content.decode("cp950", errors="replace")
        os.makedirs(RAW, exist_ok=True)
        open(os.path.join(RAW, f"{per.strftime('%Y%m')}_{k}_{today:%Y%m%d}.html"), "w", encoding="utf-8").write(html)
        out.append(parse(html))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["code", "name", "R", "R_ly"])


def update(today=None, log=print):
    today = today or dt.date.today()
    cur = pd.Period(today, "M")
    db = pd.read_csv(DB, dtype={"code": str}) if os.path.exists(DB) else pd.DataFrame(columns=["code", "name", "per", "R", "R_ly", "first_seen"])
    seen = set(zip(db.code, db.per))
    boot = db.empty                                    # 第一次跑：之前已公布的不知道哪天公布，first_seen 留空（改用 TEJ 發布日）
    new = []
    for per in (cur - 1, cur - 2):                     # 上個月（正常申報）＋上上個月（晚報的）
        snap = fetch(per, today)
        p = per.strftime("%Y-%m")
        add = snap[[(c, p) not in seen for c in snap.code]].copy()
        add["per"], add["first_seen"] = p, (None if boot else today.isoformat())
        new.append(add)
        log(f"[月營收快照 {today}] {p}：目前 {len(snap)} 家，新出現 {len(add)} 家")
    db = pd.concat([db] + new, ignore_index=True)
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    db.to_csv(DB, index=False, encoding="utf-8-sig")
    return db


if __name__ == "__main__":
    update()
