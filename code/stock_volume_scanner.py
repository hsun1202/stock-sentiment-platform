# -*- coding: utf-8 -*-
"""
30檔股票 PTT 討論量快篩工具
============================================================

目的：
    在正式對30檔股票跑大規模爬蟲（2.5年份資料）之前，
    先快速掃描每檔股票「近期」在PTT Stock板的討論量，
    篩掉討論度太低、抓不到足夠資料的股票，避免大規模爬蟲
    抓了老半天卻拿到稀疏資料。

做法：
    只統計「文章標題數量」，不逐篇抓取內文和留言（那是正式
    爬蟲才要做的事），所以速度比完整爬蟲快很多，可以在短時間
    內掃完30檔股票。

使用方式：
    python stock_volume_scanner.py --pages 150 --until-date 2026-06-01
    （--pages 是安全上限，--until-date 是實際要掃到的日期，
      邏輯跟 專題爬蟲.py 的 --until-date 一致）

輸出：
    volume_scan_result.csv，包含每檔股票的文章命中數，
    依數量排序，方便判斷哪些股票討論度不足需要替換。
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import time
import re
import csv
import argparse
from datetime import datetime, date

PTT_DOMAIN = "https://www.ptt.cc"
BOARD = "Stock"
COOKIES = {"over18": "1"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

# ============================================================
# 產業平衡版 30 檔股票名單
# ============================================================
STOCK_LIST = [
    # (股票代碼, 股票名稱, 產業)
    # ============================================================
    # 整合三輪快篩結果後的最終名單，維持六大產業架構（比照原論文精神）
    # 每個產業選出「已實測驗證討論量最高」的5檔股票
    # ============================================================

    # 金融保險：實測表現由高到低
    ("2882", "國泰金", "金融保險"),
    ("2884", "玉山金", "金融保險"),
    ("2890", "永豐金", "金融保險"),
    ("2887", "台新金", "金融保險"),
    ("2891", "中信金", "金融保險"),

    # 半導體/電子（對應原論文Manufacturing，台股電子權重最高）
    ("2330", "台積電", "半導體/電子"),
    ("2454", "聯發科", "半導體/電子"),
    ("2308", "台達電", "半導體/電子"),
    ("2317", "鴻海", "半導體/電子"),
    ("3711", "日月光投控", "半導體/電子"),

    # 傳產原物料（對應原論文Mining，鋼鐵+石化集團股，較常見於新聞版面）
    ("2002", "中鋼", "傳產原物料"),
    ("2027", "大成鋼", "傳產原物料"),
    ("1301", "台塑", "傳產原物料"),
    ("1303", "南亞", "傳產原物料"),
    ("2023", "燁輝", "傳產原物料"),

    # 航運業
    ("2603", "長榮", "航運業"),
    ("2609", "陽明", "航運業"),
    ("2615", "萬海", "航運業"),
    ("2618", "長榮航", "航運業"),
    ("2610", "華航", "航運業"),

    # 綠能環保（比照原論文Energy類別，改用官方2025年獨立分類，
    # 取代原本錯誤歸類的「油電燃氣」；已用完整2.5年資料驗證
    # 雲豹能源、泓德能源，其餘3檔依市值規模與媒體露出度選定）
    ("6869", "雲豹能源", "綠能環保"),
    ("6873", "泓德能源", "綠能環保"),
    ("8341", "日友", "綠能環保"),
    ("8422", "可寧衛", "綠能環保"),
    ("3708", "上緯投控", "綠能環保"),

    # 建材營造（同樣為討論量結構性偏弱的產業）
    ("2545", "皇翔", "建材營造"),
    ("2542", "興富發", "建材營造"),
    ("9945", "潤泰新", "建材營造"),
    ("5522", "遠雄", "建材營造"),
    ("2548", "華固", "建材營造"),
]


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


def fetch_html(url, retries=3, sleep_base=1.5):
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
            "date": date_tag.text.strip() if date_tag else "",
        })
    return articles


def parse_board_date(date_str, ref_year=None):
    if ref_year is None:
        ref_year = date.today().year
    try:
        month, day = date_str.strip().split("/")
        month, day = int(month), int(day)
        this_month = date.today().month
        year = ref_year
        if month - this_month > 6:
            year -= 1
        return date(year, month, day)
    except (ValueError, AttributeError):
        return None


def scan_all_pages(max_pages, until_date, delay=0.5):
    """
    掃描看板列表頁，收集所有文章標題與日期，直到觸及目標日期為止。
    這一步只做一次，掃到的所有文章標題會拿去比對30檔股票，
    比逐一對每檔股票各掃一次快30倍。
    """
    latest_page = get_latest_page_number()
    if latest_page is None:
        print("[錯誤] 無法取得最新頁碼")
        return []

    print(f"[資訊] 最新頁碼：{latest_page}，開始往回掃描...")
    all_articles = []

    for offset in range(max_pages):
        page_num = latest_page - offset
        if page_num < 1:
            break
        html = fetch_html(get_board_index_url(page_num))
        if not html:
            continue
        articles = parse_article_list(html)
        all_articles.extend(articles)

        if offset % 20 == 0:
            print(f"  已掃描 {offset+1} 頁，累積 {len(all_articles)} 篇文章標題...")

        if until_date is not None and articles:
            page_dates = [parse_board_date(a["date"]) for a in articles]
            page_dates = [d for d in page_dates if d is not None]
            if page_dates and all(d < until_date for d in page_dates):
                print(f"[停止] 第 {page_num} 頁日期全部早於 {until_date}，停止掃描")
                break

        time.sleep(delay)

    print(f"[完成] 共掃描 {len(all_articles)} 篇文章標題")
    return all_articles


def count_matches(all_articles, stock_list):
    """對每檔股票，統計標題中包含代碼或名稱的文章數量。"""
    results = []
    for code, name, industry in stock_list:
        count = sum(1 for a in all_articles if code in a["title"] or name in a["title"])
        results.append({
            "industry": industry, "code": code, "name": name, "article_count": count,
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="30檔股票PTT討論量快篩工具")
    parser.add_argument("--pages", type=int, default=200, help="最多掃描幾頁（安全上限）")
    parser.add_argument("--until-date", type=str, default=None,
                         help="掃到這個日期為止，格式YYYY-MM-DD，不指定則用--pages控制")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="volume_scan_result.csv")
    args = parser.parse_args()

    until_date_obj = None
    if args.until_date:
        until_date_obj = datetime.strptime(args.until_date, "%Y-%m-%d").date()

    print(f"=== 開始快篩 {len(STOCK_LIST)} 檔股票的PTT討論量 ===\n")
    all_articles = scan_all_pages(args.pages, until_date_obj, args.delay)

    print("\n[分析] 比對30檔股票標題命中數...")
    results = count_matches(all_articles, STOCK_LIST)
    results.sort(key=lambda r: r["article_count"], reverse=True)

    print("\n=== 各股票討論量排名 ===")
    print(f"{'產業':<10}{'代碼':<8}{'名稱':<10}{'文章數':<8}")
    for r in results:
        flag = " ⚠️討論量偏低" if r["article_count"] < 5 else ""
        print(f"{r['industry']:<10}{r['code']:<8}{r['name']:<10}{r['article_count']:<8}{flag}")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["industry", "code", "name", "article_count"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[完成] 結果已輸出至 {args.out}")

    low_volume = [r for r in results if r["article_count"] < 5]
    if low_volume:
        print(f"\n[提醒] 有 {len(low_volume)} 檔股票討論量偏低（<5篇），建議考慮替換：")
        for r in low_volume:
            print(f"  {r['industry']} - {r['code']} {r['name']}：{r['article_count']}篇")
