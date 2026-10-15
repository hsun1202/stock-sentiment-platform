# -*- coding: utf-8 -*-
"""
267檔股票批次爬蟲（兩階段架構，2020-2024年份資料，含推文使用者ID）
============================================================

跟原本 batch_crawler.py（30檔、2.5年）的差異：
    1. 股票清單換成267檔（stock_list_267.py 的 STOCK_LIST_267），
       前綴碰撞規則也換成267檔版本的5組（統一/統一超、台塑/台塑化、
       東和/東和鋼鐵、中鋼/中鋼構、三商/三商電）。
    2. 預設資料範圍改成 2020-01-01 ~ 2024-12-31（原本30檔版本
       是2024-2026）。
    3. 留言除了推噓類型、內容、時間之外，新增記錄「使用者ID」
       （PTT推文原始HTML裡的 span.push-userid，兩個既有資料集
       —— 自己爬的30檔版本、組員提供的267檔pttweb.cc鏡像 ——
       都沒有存這個欄位，導致留言斷句問題事後無法修補。這次
       重爬的主要目的就是補上這個欄位，讓「同一篇文章、同一個
       使用者ID、相鄰時間」的斷句留言可以嚴謹地重新接回去）。

架構（跟原版一樣的兩階段＋中斷續傳設計，理由不重複說明，
細節請對照 batch_crawler.py 內的註解）：
    階段一（scan）：只掃描一次列表頁，同時比對267檔股票，
                    命中結果存成 matched_articles.csv。
    階段二（fetch）：逐篇抓取內文與留言（含使用者ID），
                    依股票分別輸出。

使用方式：
    python batch_crawler_267.py --start 2020-01-01 --end 2024-12-31 --delay 1.0
    python batch_crawler_267.py --start 2020-01-01 --end 2024-12-31 --phase scan
    python batch_crawler_267.py --start 2020-01-01 --end 2024-12-31 --phase fetch
"""

import time
import os
import csv
import re
import json
import argparse
from datetime import datetime, date, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "stock_list_267",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_list_267.py")
)
_sl267 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sl267)
STOCK_LIST = _sl267.STOCK_LIST_267

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
    last_err = None
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            last_err = e
            wait = sleep_base * (attempt + 1)
            print(f"    [警告] 抓取失敗 ({attempt+1}/{retries}): {e}")
            time.sleep(wait)
    print(f"    [失敗] 已達最大重試次數，放棄：{url}（{last_err}）")
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


def parse_article_content(html):
    soup = BeautifulSoup(html, "html.parser")
    main_content = soup.select_one("#main-content")
    if not main_content:
        return None
    meta_values = main_content.select("span.article-meta-value")
    post_time = meta_values[3].text.strip() if len(meta_values) >= 4 else ""
    pushes = []
    for push in main_content.select("div.push"):
        tag_span = push.select_one("span.push-tag")
        userid_span = push.select_one("span.push-userid")
        content_span = push.select_one("span.push-content")
        time_span = push.select_one("span.push-ipdatetime")
        if tag_span and content_span:
            pushes.append({
                "type": tag_span.text.strip(),
                "user_id": userid_span.text.strip() if userid_span else "",
                "content": content_span.text.lstrip(": ").strip(),
                "time": time_span.text.strip() if time_span else "",
            })
    for push in main_content.select("div.push"):
        push.extract()
    for meta in main_content.select("div.article-metaline, div.article-metaline-right"):
        meta.extract()
    body_text = main_content.get_text().strip()
    category_match = re.search(r"分類[：:]\s*([多空討論心得]+)", body_text)
    category = category_match.group(1) if category_match else "未分類"
    return {"post_time": post_time, "category": category, "body": body_text, "pushes": pushes}


def calculate_sentiment_factor(pushes):
    push_count = sum(1 for p in pushes if p["type"] == "推")
    boo_count = sum(1 for p in pushes if p["type"] == "噓")
    total = len(pushes)
    if total == 0:
        return 0.0, push_count, boo_count, total
    return (push_count - boo_count) / total, push_count, boo_count, total


def extract_date_from_link(link):
    """直接從PTT文章網址解析出精確發文時間（Unix時間戳），100%準確不需猜測年份。"""
    match = re.search(r"M\.(\d+)\.A", link)
    if not match:
        return None
    try:
        return datetime.fromtimestamp(int(match.group(1))).date()
    except (ValueError, OSError):
        return None


# 267檔版本的前綴碰撞規則（比照 match_ptt_articles_267.py，5組）
PREFIX_COLLISIONS = [
    ("統一", "統一超"), ("台塑", "台塑化"), ("東和", "東和鋼鐵"),
    ("中鋼", "中鋼構"), ("三商", "三商電"),
]


def is_valid_match(code, name, title):
    """
    判斷文章標題是否真的命中這檔股票。
    如果命中的是股票代碼，直接視為有效；如果是靠股票名稱命中，
    需要額外排除「名稱只是因為是另一檔股票名稱的前綴」造成的誤判。
    """
    if code in title:
        return True
    if name not in title:
        return False
    for prefix_name, suffix_name in PREFIX_COLLISIONS:
        if name == prefix_name and suffix_name in title:
            title_without_suffix = title.replace(suffix_name, "")
            if prefix_name not in title_without_suffix:
                return False  # 標題只提到後綴股票，前綴股票是誤判
    return True


# ============================================================
# 階段一：單次掃描，同時比對所有股票
# ============================================================

def dedupe_csv_by_link(path, link_col="link"):
    """讀取CSV，依link欄位去除重複列（適用於posts.csv，每篇文章的連結唯一）。"""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = []
            for row in reader:
                if fieldnames and None not in row and len(row) == len(fieldnames):
                    rows.append(row)
    except (OSError, csv.Error):
        return 0
    if not rows:
        return 0
    seen = set()
    deduped = []
    for row in rows:
        key = row.get(link_col, "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    removed = len(rows) - len(deduped)
    if removed > 0:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deduped)
    return removed


def dedupe_csv_by_composite_key(path, key_cols):
    """讀取CSV，依多個欄位組成的複合鍵去除重複列（適用於matched_articles.csv）。"""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = []
            for row in reader:
                if fieldnames and None not in row and len(row) == len(fieldnames):
                    rows.append(row)
    except (OSError, csv.Error):
        return 0
    if not rows:
        return 0
    seen = set()
    deduped = []
    for row in rows:
        key = tuple(row.get(col, "") for col in key_cols)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    removed = len(rows) - len(deduped)
    if removed > 0:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deduped)
    return removed


def dedupe_csv_full_row(path):
    """讀取CSV，只有當所有欄位都完全相同才視為重複並移除（適用於comments.csv）。"""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = []
            for row in reader:
                if fieldnames and None not in row and len(row) == len(fieldnames):
                    rows.append(row)
    except (OSError, csv.Error):
        return 0
    if not rows:
        return 0
    seen = set()
    deduped = []
    for row in rows:
        key = tuple(row.get(col, "") for col in fieldnames)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    removed = len(rows) - len(deduped)
    if removed > 0:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deduped)
    return removed


def phase_scan(out_dir, until_date, end_date, max_pages, delay):
    """
    只掃描一次PTT列表頁，掃描過程中同時檢查每篇文章標題是否命中
    267檔股票中的任何一檔，把命中結果存成統一的 matched_articles.csv。
    支援中斷續傳（設計細節同 batch_crawler.py）。
    """
    matched_path = os.path.join(out_dir, "matched_articles.csv")
    checkpoint_path = os.path.join(out_dir, "scan.checkpoint")
    done_marker_path = os.path.join(out_dir, "scan_done.marker")

    removed = dedupe_csv_by_composite_key(matched_path, key_cols=["stock_code", "link"])
    if removed > 0:
        print(f"[階段一-自我修復] 偵測並移除了 {removed} 筆重複資料（可能是中斷造成的）")

    if os.path.exists(done_marker_path):
        print("[階段一] 先前已完整掃描完成（偵測到 scan_done.marker），跳過階段一")
        return True

    resume_page = None
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                resume_page = int(f.read().strip())
            print(f"[階段一-續傳] 偵測到中斷紀錄，從第 {resume_page} 頁接續")
        except (ValueError, OSError):
            print("[階段一] checkpoint內容不完整，改從最新頁重新掃描（重複資料會自動清除）")
            resume_page = None

    latest_page = get_latest_page_number()
    if latest_page is None:
        raise RuntimeError("無法取得最新頁碼，可能是網路問題")

    start_from_page = resume_page if resume_page is not None else latest_page
    print(f"[階段一] 最新頁碼：{latest_page}，本次從第 {start_from_page} 頁開始掃描")

    fieldnames = ["stock_code", "stock_name", "industry", "title", "link", "resolved_date"]
    file_is_new = not os.path.exists(matched_path)
    f = open(matched_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if file_is_new:
        writer.writeheader()

    total_matched = 0
    finished_normally = False

    try:
        page_num = start_from_page
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

            if pages_done == 0:
                print(f"    [除錯] 本頁 {len(articles)} 篇，日期範例："
                      f"{[a['_resolved_date'].isoformat() if a['_resolved_date'] else None for a in articles[:3]]}")

            if until_date is not None and valid_dates and all(d < until_date for d in valid_dates):
                print(f"    [停止] 第{page_num}頁日期已早於{until_date}，掃描完成")
                finished_normally = True
                break

            for a in articles:
                art_date = a["_resolved_date"]
                if until_date is not None and art_date is not None and art_date < until_date:
                    continue
                if end_date is not None and art_date is not None and art_date > end_date:
                    continue
                for code, name, industry in STOCK_LIST:
                    if is_valid_match(code, name, a["title"]):
                        writer.writerow({
                            "stock_code": code, "stock_name": name, "industry": industry,
                            "title": a["title"], "link": a["link"],
                            "resolved_date": art_date.isoformat() if art_date else "",
                        })
                        total_matched += 1
                f.flush()

            with open(checkpoint_path, "w", encoding="utf-8") as ckpt:
                ckpt.write(str(page_num - 1))

            if pages_done % 50 == 0 and pages_done > 0:
                latest_date_str = valid_dates[0].isoformat() if valid_dates else "未知"
                print(f"    [階段一進度] 已掃描 {pages_done} 頁（本次執行），"
                      f"累積命中 {total_matched} 篇，目前爬到日期：{latest_date_str}")

            time.sleep(delay)
            page_num -= 1
            pages_done += 1
        else:
            if page_num < 1:
                finished_normally = True
            else:
                print(f"    [警告] 階段一達到 max_pages={max_pages} 安全上限，尚未掃到 {until_date}")
                print(f"    [提示] checkpoint已保留，重新執行會自動接續")
    finally:
        f.close()

    if finished_normally and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if finished_normally:
        with open(done_marker_path, "w", encoding="utf-8") as marker:
            marker.write(f"completed at {datetime.now().isoformat()}\n")

    print(f"[階段一{'完成' if finished_normally else '部分完成，需接續'}] "
          f"本次新增命中 {total_matched} 篇文章")
    return finished_normally


# ============================================================
# 階段二：讀取命中清單，逐篇抓取內文與留言（含使用者ID）
# ============================================================

def phase_fetch(out_dir, delay):
    """
    讀取階段一產生的 matched_articles.csv，逐篇抓取內文與留言，
    依股票分別輸出成 {代碼}_{名稱}_posts.csv / _comments.csv。
    留言額外記錄 user_id 欄位（跟舊版最大的差異）。
    斷點續傳設計同 batch_crawler.py：以 posts.csv 裡實際存在的
    link 值當作「已完成」的唯一真相來源。
    """
    matched_path = os.path.join(out_dir, "matched_articles.csv")
    if not os.path.exists(matched_path):
        print("[階段二] 找不到 matched_articles.csv，請先執行階段一（scan）")
        return

    with open(matched_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_matched = list(reader)

    fetched_links = set()
    for code, name, industry in STOCK_LIST:
        posts_path = os.path.join(out_dir, f"{code}_{name}_posts.csv")
        comments_path = os.path.join(out_dir, f"{code}_{name}_comments.csv")
        removed_p = dedupe_csv_by_link(posts_path)
        removed_c = dedupe_csv_full_row(comments_path)
        if removed_p > 0 or removed_c > 0:
            print(f"[階段二-自我修復] {code} {name}：清除 {removed_p} 筆重複文章、"
                  f"{removed_c} 筆重複留言")
        if os.path.exists(posts_path):
            with open(posts_path, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("link"):
                        fetched_links.add(row["link"])

    print(f"[階段二-續傳] 已確認 {len(fetched_links)} 篇文章的資料已存在，會自動跳過")

    to_fetch = [row for row in all_matched if row["link"] not in fetched_links]
    print(f"[階段二] 命中清單共 {len(all_matched)} 篇，尚待抓取 {len(to_fetch)} 篇")

    progress_path = os.path.join(out_dir, "fetch_progress.json")
    started_at = datetime.now().isoformat()

    def write_progress(done_this_run, current_stock_code="", current_stock_name=""):
        payload = {
            "total_matched": len(all_matched),
            "already_done_before_this_run": len(fetched_links),
            "done_this_run": done_this_run,
            "remaining": len(to_fetch) - done_this_run,
            "current_stock_code": current_stock_code,
            "current_stock_name": current_stock_name,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
        }
        with open(progress_path, "w", encoding="utf-8") as pf:
            json.dump(payload, pf, ensure_ascii=False, indent=2)

    posts_fieldnames = ["stock_code", "title", "link", "resolved_date", "post_time",
                         "category", "push_count", "boo_count", "arrow_count",
                         "total_comments", "sentiment_factor"]
    comments_fieldnames = ["stock_code", "link", "resolved_date", "push_type", "user_id", "content"]

    posts_files, comments_files = {}, {}

    def get_writers(code, name):
        if code not in posts_files:
            posts_path = os.path.join(out_dir, f"{code}_{name}_posts.csv")
            comments_path = os.path.join(out_dir, f"{code}_{name}_comments.csv")
            p_is_new = not os.path.exists(posts_path)
            c_is_new = not os.path.exists(comments_path)
            pf = open(posts_path, "a", newline="", encoding="utf-8-sig")
            cf = open(comments_path, "a", newline="", encoding="utf-8-sig")
            pw = csv.DictWriter(pf, fieldnames=posts_fieldnames)
            cw = csv.DictWriter(cf, fieldnames=comments_fieldnames)
            if p_is_new:
                pw.writeheader()
            if c_is_new:
                cw.writeheader()
            posts_files[code] = (pf, pw)
            comments_files[code] = (cf, cw)
        return posts_files[code][1], comments_files[code][1]

    try:
        for i, row in enumerate(to_fetch, 1):
            art_html = fetch_html(row["link"])
            time.sleep(delay)
            if not art_html:
                continue
            parsed = parse_article_content(art_html)
            if not parsed:
                continue
            sentiment, push_c, boo_c, total = calculate_sentiment_factor(parsed["pushes"])

            get_writers(row["stock_code"], row["stock_name"])
            pw = posts_files[row["stock_code"]][1]
            cw = comments_files[row["stock_code"]][1]

            pw.writerow({
                "stock_code": row["stock_code"], "title": row["title"], "link": row["link"],
                "resolved_date": row["resolved_date"], "post_time": parsed["post_time"],
                "category": parsed["category"], "push_count": push_c, "boo_count": boo_c,
                "arrow_count": total - push_c - boo_c, "total_comments": total,
                "sentiment_factor": round(sentiment, 4),
            })
            for p in parsed["pushes"]:
                cw.writerow({
                    "stock_code": row["stock_code"], "link": row["link"],
                    "resolved_date": row["resolved_date"],
                    "push_type": p["type"], "user_id": p["user_id"], "content": p["content"],
                })
            posts_files[row["stock_code"]][0].flush()
            comments_files[row["stock_code"]][0].flush()

            if i % 20 == 0 or i == len(to_fetch):
                write_progress(i, row["stock_code"], row["stock_name"])

            if i % 50 == 0:
                print(f"    [階段二進度] 已抓取 {i}/{len(to_fetch)} 篇"
                      f"（{row['stock_code']} {row['stock_name']}）")
    finally:
        for pf, _ in posts_files.values():
            pf.close()
        for cf, _ in comments_files.values():
            cf.close()

    print(f"[階段二完成] 本次共抓取 {len(to_fetch)} 篇文章的內文與留言")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="267檔股票批次爬蟲（兩階段架構，含使用者ID）")
    parser.add_argument("--start", type=str, default="2020-01-01", help="資料起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-12-31", help="資料結束日期 YYYY-MM-DD")
    parser.add_argument("--out-dir", type=str, default="ptt_267_userid", help="輸出資料夾")
    parser.add_argument("--pages", type=int, default=60000, help="階段一掃描的安全上限頁數")
    parser.add_argument("--delay", type=float, default=1.0, help="每次請求間隔秒數")
    parser.add_argument("--phase", type=str, default="both", choices=["scan", "fetch", "both"],
                         help="只跑階段一(scan)、只跑階段二(fetch)、或依序跑完兩階段(both，預設)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    until_date = datetime.strptime(args.start, "%Y-%m-%d").date()

    print(f"=== 267檔批次爬蟲開始（兩階段架構，含使用者ID）===")
    print(f"股票總數：{len(STOCK_LIST)} 檔")
    print(f"資料範圍：{until_date} ~ {end_date}")
    print(f"輸出資料夾：{args.out_dir}/\n")

    total_start = time.time()

    if args.phase in ("scan", "both"):
        print("\n----- 階段一：單次掃描比對所有股票 -----")
        scan_done = phase_scan(args.out_dir, until_date, end_date, args.pages, args.delay)
        if not scan_done:
            print("\n[提示] 階段一尚未完成，請重新執行同一指令接續，"
                  "完成後才會自動進入階段二")
            if args.phase == "both":
                import sys
                sys.exit(0)

    if args.phase in ("fetch", "both"):
        print("\n----- 階段二：逐篇抓取內文與留言 -----")
        phase_fetch(args.out_dir, args.delay)

    total_elapsed = time.time() - total_start
    print(f"\n=== 本次執行結束，總耗時 {total_elapsed/3600:.2f} 小時 ===")
