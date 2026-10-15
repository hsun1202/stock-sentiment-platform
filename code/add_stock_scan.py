# -*- coding: utf-8 -*-
"""
單一股票補掃工具
============================================================

情境：
    30檔股票名單裡有股票異動（例如原本的股票下市，換成新的股票），
    但整體階段一已經標記完成（scan_done.marker存在），導致重新執行
    batch_crawler.py時，階段一會被跳過，新加入的股票永遠不會被
    掃描比對到。

    這支工具讓你「只針對新加入的股票」單獨掃描一次PTT列表頁，
    把命中結果加進既有的 matched_articles.csv，不需要重新掃描
    整個30檔（那會很浪費時間）。

使用方式：
    python add_stock_scan.py --code 6873 --name 泓德能源 --industry 油電燃氣 \
        --start 2024-01-01 --end 2026-06-30 --out-dir batch_output
"""

import time
import os
import csv
import re
import argparse
from datetime import datetime, date, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

PTT_DOMAIN = "https://www.ptt.cc"
BOARD = "Stock"
COOKIES = {"over18": "1"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


def build_session():
    session = requests.Session()
    session.cookies.update(COOKIES)
    session.headers.update(HEADERS)
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = build_session()


def fetch_html(url, retries=3, sleep_base=2.0):
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            print(f"    [警告] 抓取失敗 ({attempt+1}/{retries}): {e}")
            time.sleep(sleep_base * (attempt + 1))
    return None


def get_board_index_url(page=None):
    if page is None:
        return f"{PTT_DOMAIN}/bbs/{BOARD}/index.html"
    return f"{PTT_DOMAIN}/bbs/{BOARD}/index{page}.html"


def get_latest_page_number():
    html = fetch_html(get_board_index_url())
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for btn in soup.select("a.btn.wide"):
        if "上頁" in btn.text:
            match = re.search(r"index(\d+)\.html", btn.get("href", ""))
            if match:
                return int(match.group(1)) + 1
    return None


def parse_article_list(html):
    soup = BeautifulSoup(html, "html.parser")
    articles = []
    for entry in soup.select("div.r-ent"):
        title_tag = entry.select_one("div.title a")
        if not title_tag:
            continue
        date_tag = entry.select_one("div.date")
        articles.append({
            "title": title_tag.text.strip(),
            "link": PTT_DOMAIN + title_tag["href"],
            "date": date_tag.text.strip() if date_tag else "",
        })
    return articles


def extract_date_from_link(link):
    match = re.search(r"M\.(\d+)\.A", link)
    if not match:
        return None
    try:
        return datetime.fromtimestamp(int(match.group(1))).date()
    except (ValueError, OSError):
        return None


def scan_for_multiple_stocks(stock_list, until_date, end_date, delay, max_pages=40000):
    """
    單獨掃描一次列表頁，同時比對多檔股票（stock_list為[(code,name,industry),...]），
    只需要掃描一次列表頁，效率比對每檔股票各自掃描快很多。
    """
    latest_page = get_latest_page_number()
    if latest_page is None:
        raise RuntimeError("無法取得最新頁碼")

    names = "、".join(f"{c} {n}" for c, n, i in stock_list)
    print(f"[補掃] 同時掃描 {len(stock_list)} 檔股票：{names}")
    print(f"[補掃] 最新頁碼：{latest_page}")

    matches = []
    page_num = latest_page
    pages_done = 0

    while page_num >= 1 and pages_done < max_pages:
        html = fetch_html(get_board_index_url(page_num))
        if not html:
            page_num -= 1
            pages_done += 1
            continue
        articles = parse_article_list(html)
        for a in articles:
            a["_resolved_date"] = extract_date_from_link(a["link"])
        valid_dates = [a["_resolved_date"] for a in articles if a["_resolved_date"] is not None]

        if until_date is not None and valid_dates and all(d < until_date for d in valid_dates):
            print(f"    [停止] 第{page_num}頁日期已早於{until_date}，補掃完成")
            break

        for a in articles:
            art_date = a["_resolved_date"]
            if until_date is not None and art_date is not None and art_date < until_date:
                continue
            if end_date is not None and art_date is not None and art_date > end_date:
                continue
            for code, name, industry in stock_list:
                if code in a["title"] or name in a["title"]:
                    matches.append({
                        "stock_code": code, "stock_name": name, "industry": industry,
                        "title": a["title"], "link": a["link"],
                        "resolved_date": art_date.isoformat() if art_date else "",
                    })

        if pages_done % 100 == 0 and pages_done > 0:
            latest_date_str = valid_dates[0].isoformat() if valid_dates else "未知"
            print(f"    ...已掃描 {pages_done} 頁，累積命中 {len(matches)} 篇，目前爬到日期：{latest_date_str}")

        time.sleep(delay)
        page_num -= 1
        pages_done += 1

    return matches


def scan_for_one_stock(code, name, industry, until_date, end_date, delay, max_pages=40000):
    """單獨掃描一次列表頁，只比對這一檔股票，回傳命中清單。"""
    latest_page = get_latest_page_number()
    if latest_page is None:
        raise RuntimeError("無法取得最新頁碼")

    print(f"[補掃] {code} {name}（{industry}），最新頁碼：{latest_page}")
    matches = []
    page_num = latest_page
    pages_done = 0

    while page_num >= 1 and pages_done < max_pages:
        html = fetch_html(get_board_index_url(page_num))
        if not html:
            page_num -= 1
            pages_done += 1
            continue
        articles = parse_article_list(html)
        for a in articles:
            a["_resolved_date"] = extract_date_from_link(a["link"])
        valid_dates = [a["_resolved_date"] for a in articles if a["_resolved_date"] is not None]

        if until_date is not None and valid_dates and all(d < until_date for d in valid_dates):
            print(f"    [停止] 第{page_num}頁日期已早於{until_date}，補掃完成")
            break

        for a in articles:
            art_date = a["_resolved_date"]
            if until_date is not None and art_date is not None and art_date < until_date:
                continue
            if end_date is not None and art_date is not None and art_date > end_date:
                continue
            if code in a["title"] or name in a["title"]:
                matches.append({
                    "stock_code": code, "stock_name": name, "industry": industry,
                    "title": a["title"], "link": a["link"],
                    "resolved_date": art_date.isoformat() if art_date else "",
                })

        if pages_done % 100 == 0 and pages_done > 0:
            latest_date_str = valid_dates[0].isoformat() if valid_dates else "未知"
            print(f"    ...已掃描 {pages_done} 頁，累積命中 {len(matches)} 篇，目前爬到日期：{latest_date_str}")

        time.sleep(delay)
        page_num -= 1
        pages_done += 1

    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="單一或多檔股票補掃工具")
    parser.add_argument("--code", type=str, required=True,
                         help="股票代碼，多檔用逗號分隔，例如：8341,8422,3708")
    parser.add_argument("--name", type=str, required=True,
                         help="股票名稱，多檔用逗號分隔，順序需與--code對應，例如：日友,可寧衛,上緯投控")
    parser.add_argument("--industry", type=str, required=True,
                         help="產業別，多檔用逗號分隔（也可只輸入一個套用到全部），例如：綠能環保")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD，預設今天")
    parser.add_argument("--out-dir", type=str, default="batch_output")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    codes = [c.strip() for c in args.code.split(",")]
    names = [n.strip() for n in args.name.split(",")]
    industries_raw = [i.strip() for i in args.industry.split(",")]

    if len(codes) != len(names):
        raise ValueError(f"--code（{len(codes)}個）跟 --name（{len(names)}個）數量不一致")

    if len(industries_raw) == 1:
        industries = industries_raw * len(codes)
    elif len(industries_raw) == len(codes):
        industries = industries_raw
    else:
        raise ValueError(f"--industry 數量必須是1個（套用全部）或跟股票數量一致")

    stock_list = list(zip(codes, names, industries))

    until_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    matches = scan_for_multiple_stocks(stock_list, until_date, end_date, args.delay)

    print(f"\n[完成] 共命中 {len(matches)} 篇文章（{len(stock_list)}檔股票合計）")
    for code, name, industry in stock_list:
        count = sum(1 for m in matches if m["stock_code"] == code)
        print(f"    {code} {name}：{count}篇")

    matched_path = os.path.join(args.out_dir, "matched_articles.csv")
    fieldnames = ["stock_code", "stock_name", "industry", "title", "link", "resolved_date"]

    # 檢查是否已經存在這些股票的記錄，避免重複加入
    existing_keys = set()
    if os.path.exists(matched_path):
        with open(matched_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row.get("stock_code"), row.get("link")))

    new_matches = [m for m in matches if (m["stock_code"], m["link"]) not in existing_keys]
    print(f"\n[資訊] 其中 {len(matches) - len(new_matches)} 篇已存在，新增 {len(new_matches)} 篇")

    file_exists = os.path.exists(matched_path)
    with open(matched_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_matches)

    print(f"[完成] 已將 {len(new_matches)} 筆新記錄加入 {matched_path}")
    print(f"[下一步] 執行 batch_crawler.py --phase fetch 即可自動抓取這些新增文章的內文與留言")
