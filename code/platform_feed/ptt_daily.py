# -*- coding: utf-8 -*-
"""
PTT Stock 板每日爬蟲（www.ptt.cc，公開頁面）。每次往回掃到「最近 N 天」發的文章，每篇重抓一次（閒聊串的留言會一直增加），
留言以（文章 id, 第幾則）去重後存到 platform_feed/data/ptt/comments.csv。
  欄位：aid, title, post_time, idx, tag（推／噓／→）, user, time（留言時間）, content
  留言時間只有「MM/DD HH:MM」：年份用發文年份，月份比發文月份小就 +1 年。
cd D:/專題claude && python platform_feed/ptt_daily.py --days 3        # 平常：最近 3 天
                    python platform_feed/ptt_daily.py --days 45       # 第一次：往回補 45 天
"""
import argparse
import html
import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "data", "ptt")
DB = os.path.join(DIR, "comments.csv")
ART = os.path.join(DIR, "articles.csv")
URL = "https://www.ptt.cc"
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
S.cookies.set("over18", "1")


def get(path, delay):
    for k in range(4):
        try:
            r = S.get(URL + path, timeout=30)
            time.sleep(delay)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    return None


def scan(since: datetime, delay: float, log=print):
    """從最新的列表頁往回，收集發文時間 ≥ since 的文章（id 裡的 M.<unix 秒> 就是發文時間）"""
    path, arts, pages = "/bbs/Stock/index.html", [], 0
    while path:
        t = get(path, delay)
        pages += 1
        if t is None:
            break
        found_old = False
        body = t.split('<div class="r-list-sep"></div>')[0]               # 第一頁底下的置底文（很舊）不算
        for e in re.findall(r'<div class="r-ent">(.*?)<div class="mark">', body, flags=re.S):
            a = re.search(r'<a href="(/bbs/Stock/(M\.(\d+)\.A\.[0-9A-F]+)\.html)">(.*?)</a>', e)
            if not a:
                continue
            pt = datetime.fromtimestamp(int(a.group(3)))
            if pt >= since:
                arts.append({"path": a.group(1), "aid": a.group(2), "post_time": pt, "title": html.unescape(a.group(4)).strip()})
            else:
                found_old = True
        prev = re.search(r'href="(/bbs/Stock/index\d+\.html)">&lsaquo;', t)
        path = None if found_old or not prev else prev.group(1)
    arts = {a["aid"]: a for a in arts}                                 # 置底文會在每頁出現
    log(f"[PTT 列表] 掃 {pages} 頁，{len(arts)} 篇（{since:%m-%d %H:%M} 之後發文）")
    return list(arts.values())


def parse(t, art):
    rows = []
    y0, m0 = art["post_time"].year, art["post_time"].month
    for i, p in enumerate(re.findall(r'<div class="push">(.*?)</div>', t, flags=re.S)):
        tag = re.search(r'push-tag">([^<]*)<', p)
        user = re.search(r'push-userid">([^<]*)<', p)
        cont = re.search(r'push-content">(.*?)</span>', p, flags=re.S)
        when = re.search(r'push-ipdatetime">([^<]*)<', p)
        if not (tag and cont and when):
            continue
        m = re.search(r"(\d{2})/(\d{2})\s+(\d{2}):(\d{2})", when.group(1))
        if not m:
            continue
        mo, d, hh, mm = map(int, m.groups())
        try:
            ts = datetime(y0 + (1 if mo < m0 else 0), mo, d, hh, mm)
        except ValueError:
            continue
        txt = html.unescape(re.sub(r"<[^>]+>", "", cont.group(1))).lstrip(":").strip()
        rows.append({"aid": art["aid"], "title": art["title"], "post_time": art["post_time"].isoformat(timespec="seconds"), "idx": i,
                     "tag": tag.group(1).strip(), "user": user.group(1).strip() if user else "", "time": ts.isoformat(timespec="minutes"),
                     "content": txt})
    return rows


def update(days=3, delay=0.4, log=print):
    os.makedirs(DIR, exist_ok=True)
    arts = scan(datetime.now() - timedelta(days=days), delay, log)
    new, t0 = [], time.time()
    for k, a in enumerate(arts):
        t = get(a["path"], delay)
        if t:
            new += parse(t, a)
        if (k + 1) % 200 == 0:
            log(f"  [PTT 文章] {k + 1}/{len(arts)}，留言 {len(new):,}，{(time.time() - t0) / 60:.0f} 分鐘")
    new = pd.DataFrame(new, columns=["aid", "title", "post_time", "idx", "tag", "user", "time", "content"])
    old = pd.read_csv(DB, dtype=str, keep_default_na=False) if os.path.exists(DB) else new.iloc[:0].astype(str)
    allc = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(["aid", "idx"], keep="last")
    allc.to_csv(DB, index=False, encoding="utf-8-sig")
    pd.DataFrame(arts).drop(columns="path").assign(post_time=lambda d: d.post_time.astype(str)).pipe(
        lambda d: pd.concat([pd.read_csv(ART, dtype=str) if os.path.exists(ART) else d.iloc[:0], d.astype(str)]).drop_duplicates("aid", keep="last")
    ).to_csv(ART, index=False, encoding="utf-8-sig")
    log(f"[PTT] 本次 {len(arts)} 篇、{len(new):,} 則留言；資料庫共 {len(allc):,} 則（{allc.time.min()[:10]}～{allc.time.max()[:10]}）")
    return allc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3)
    ap.add_argument("--delay", type=float, default=0.4)
    update(ap.parse_args().days, ap.parse_args().delay)
