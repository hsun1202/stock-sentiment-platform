# -*- coding: utf-8 -*-
"""
CMoney股市爆料同學會爬蟲（需要登入token，抓267檔股票2024年的討論）
============================================================

背景：
    CMoney個股討論列表的API（Article/Stocks/{code}/AllLatest）需要登入
    才能存取，公開瀏覽會被擋（401 Unauthorized）。這支程式假設你已經
    在瀏覽器裡用自己的帳號登入過，並且把登入後的access_token存到
    cmoney_token.json（由Claude從瀏覽器localStorage的oidc.user物件
    裡取出，不經手密碼本身）。

    access_token有效期大約1天左右，過期後這支程式會偵測到401、
    保留進度後乾淨地停下來，這時候需要：
        1. 在瀏覽器裡重新登入
        2. 請Claude重新取出新的token、更新cmoney_token.json
        3. 重新執行同一指令，會自動從中斷點接續（不重抓已完成的部分）

API重點（從CMoney前端原始碼逆向工程確認，不是用猜的）：
    GET /api/mach/api/Article/Stocks/{code}/AllLatest?limit=20&cursor={cursor}
    headers: X-Version: 3.0, Authorization: Bearer {access_token}
    回應：{articles: [...], hasNext: bool, nextCursor: number}
    是按時間新到舊排序，用nextCursor可以往回翻頁（回溯歷史）。
    limit上限是20（超過會回400）。

    每篇文章的 content.commodityTags 裡有 bullOrBear 欄位
    （1=作者自己標的看多、0=看空），是CMoney平台本身的自報情緒標籤，
    不是我們自己判斷的。

使用方式（先用少數股票驗證速度，再決定要不要全部267檔跑）：
    python crawl_cmoney.py --codes 2330 --start 2024-01-01 --end 2024-12-31
    python crawl_cmoney.py --start 2024-01-01 --end 2024-12-31   # 全部267檔
"""

import os
import csv
import json
import time
import argparse
from datetime import datetime, date

import re
import requests

# 少數留言內容裡混雜了NUL字元或其他控制字元（例如貼上PDF/特殊排版來源的文字），
# Python的csv模組遇到這種字元會直接丟例外中斷整支程式（曾經在跑到第92檔時
# 讓爬蟲整個當掉），寫入前一律清掉，才不會讓少數幾則留言拖垮整個爬蟲。
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(text):
    return _CONTROL_CHARS_RE.sub("", text or "")

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "stock_list_267",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_list_267.py")
)
_sl267 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sl267)
STOCK_LIST_267 = _sl267.STOCK_LIST_267

BASE = "https://www.cmoney.tw/api/mach/api/Article/Stocks"
TOKEN_FILE = "cmoney_token.json"


def load_token():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["access_token"], data.get("expires_at")


def build_session():
    s = requests.Session()
    s.headers.update({
        "X-Version": "3.0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    return s


# 記錄目前用的token跟它的檔案修改時間，避免每次請求都重新讀檔+印訊息
_token_state = {"token": None, "mtime": None}


def refresh_token_if_changed(session):
    """
    每次請求前呼叫：檢查 cmoney_token.json 有沒有被更新過（用檔案修改時間判斷），
    如果你重新登入、Claude幫你把新token寫進這個檔案，這裡會自動偵測到並套用，
    不需要重新啟動整支爬蟲程式。這就是「定期登入」的關鍵——只要在token還沒
    過期前更新檔案，爬蟲就不會中斷。
    """
    try:
        mtime = os.path.getmtime(TOKEN_FILE)
    except OSError:
        return
    if mtime != _token_state["mtime"]:
        token, expires_at = load_token()
        session.headers["Authorization"] = f"Bearer {token}"
        _token_state["token"] = token
        _token_state["mtime"] = mtime
        remaining = (expires_at - time.time()) / 3600 if expires_at else None
        print(f"    [token] 偵測到token檔案更新，已套用新token"
              + (f"（剩餘約{remaining:.1f}小時）" if remaining else ""))


class TokenExpired(Exception):
    pass


class FetchFailed(Exception):
    """
    重試多次後仍然失敗（網路問題、伺服器暫時錯誤等），跟「這檔股票真的
    沒有更多文章了」是完全不同的兩件事，絕對不能混為一談——否則會像
    這支程式最早的版本一樣，把暫時性的網路斷線誤判成「已經爬到底」，
    錯誤地標記完成、漏掉後面本來還沒抓到的資料。遇到這個例外時，
    checkpoint已經保留在最後一次成功翻頁的位置，重新執行同一指令
    就會自動從那裡接續，不會漏抓也不會誤判完成。
    """
    pass


def fetch_page(session, code, cursor, retries=5):
    refresh_token_if_changed(session)
    url = f"{BASE}/{code}/AllLatest"
    params = {"limit": 20}
    if cursor is not None:
        params["cursor"] = cursor
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            wait = min(3 * (attempt + 1), 30)
            print(f"    [警告] 請求失敗（{attempt+1}/{retries}），{wait}秒後重試：{e}")
            time.sleep(wait)
            continue
        if resp.status_code == 401:
            # 401不再直接放棄：等待使用者重新登入更新token檔案，定期輪詢檢查
            print("    [Token過期] 等待重新登入...（請在瀏覽器重新登入CMoney，"
                  "Claude取得新token寫入cmoney_token.json後會自動偵測接續）")
            waited = 0
            old_mtime = _token_state["mtime"]
            while waited < 3600:  # 最多等1小時，避免無限卡死
                time.sleep(10)
                waited += 10
                try:
                    new_mtime = os.path.getmtime(TOKEN_FILE)
                except OSError:
                    continue
                if new_mtime != old_mtime:
                    refresh_token_if_changed(session)
                    break
            else:
                raise TokenExpired()
            continue
        if resp.status_code != 200:
            wait = min(3 * (attempt + 1), 30)
            print(f"    [警告] 非預期狀態碼 {resp.status_code}，{wait}秒後重試：{resp.text[:200]}")
            time.sleep(wait)
            continue
        return resp.json()
    raise FetchFailed(f"重試{retries}次後仍失敗：{code} cursor={cursor}")


def crawl_stock(session, code, name, start_date, end_date, out_dir, delay):
    """
    往回翻頁直到碰到 start_date 之前，把落在 [start_date, end_date] 區間
    的文章存下來。支援中斷續傳：checkpoint記錄目前翻到哪個cursor、
    已經進到「範圍內」還是還在「範圍後（比end_date新，跳過中)」。
    """
    out_path = os.path.join(out_dir, f"{code}_{name}.csv")
    done_marker = os.path.join(out_dir, f"{code}_{name}.done")
    checkpoint_path = os.path.join(out_dir, f"{code}_{name}.checkpoint")

    if os.path.exists(done_marker):
        print(f"  [{code} {name}] 已完成，跳過")
        return 0

    cursor = None
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            cursor = json.load(f).get("cursor")
        print(f"  [{code} {name}] 偵測到中斷紀錄，從cursor={cursor}接續")

    fieldnames = ["article_id", "creatorId", "createTime", "createDate", "title",
                  "text", "commentCount", "bullOrBear", "commodityTags"]
    file_is_new = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if file_is_new:
        writer.writeheader()

    total_saved = 0
    total_pages = 0
    t0 = time.time()
    try:
        while True:
            try:
                data = fetch_page(session, code, cursor)
            except FetchFailed as e:
                # 重試多次仍失敗：不是「爬到底了」，是暫時性問題。
                # checkpoint留在最後一次成功翻頁的位置，不寫done marker，
                # 重新執行同一指令會自動從這裡接續，不會漏抓也不會誤判完成。
                print(f"  [{code} {name}] {e}，保留進度，稍後可重新執行同一指令接續")
                return total_saved
            articles = data.get("articles", [])
            if not articles:
                # API正常回應、但這頁真的沒有文章了，才是真正「爬到底」
                break
            for a in articles:
                ct = a.get("createTime", 0)
                art_date = datetime.fromtimestamp(ct / 1000).date()
                if art_date > end_date:
                    continue  # 還沒進到範圍，繼續往回翻
                if art_date < start_date:
                    # 已經翻過頭了，範圍內的都存完了
                    f.flush()
                    with open(done_marker, "w", encoding="utf-8") as marker:
                        marker.write(f"completed at {datetime.now().isoformat()}\n")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    elapsed = time.time() - t0
                    print(f"  [{code} {name}] 完成，共存 {total_saved} 篇，"
                          f"翻了 {total_pages} 頁，耗時 {elapsed/60:.1f} 分鐘")
                    return total_saved

                content = a.get("content", {})
                tags = content.get("commodityTags", [])
                my_tag = next((t for t in tags if t.get("key") == code), None)
                bull_or_bear = my_tag.get("bullOrBear") if my_tag else ""
                writer.writerow({
                    "article_id": a.get("id"),
                    "creatorId": a.get("creatorId"),
                    "createTime": ct,
                    "createDate": art_date.isoformat(),
                    "title": sanitize(content.get("title", "")),
                    "text": sanitize((content.get("text") or "").replace("\n", " ")),
                    "commentCount": a.get("commentCount", 0),
                    "bullOrBear": bull_or_bear,
                    "commodityTags": json.dumps(tags, ensure_ascii=False),
                })
                total_saved += 1
            f.flush()

            total_pages += 1
            cursor = data.get("nextCursor")
            has_next = data.get("hasNext", False)

            with open(checkpoint_path, "w", encoding="utf-8") as ckpt:
                json.dump({"cursor": cursor}, ckpt)

            if total_pages % 100 == 0:
                oldest = articles[-1].get("createTime", 0)
                oldest_date = datetime.fromtimestamp(oldest / 1000).date()
                elapsed = time.time() - t0
                print(f"    [{code} {name}] 已翻 {total_pages} 頁，存 {total_saved} 篇，"
                      f"目前翻到 {oldest_date}，耗時 {elapsed/60:.1f} 分鐘")

            if not has_next:
                break
            time.sleep(delay)
    finally:
        f.close()

    # 正常翻完（沒有更多頁了，可能這檔股票資料本來就不多）
    with open(done_marker, "w", encoding="utf-8") as marker:
        marker.write(f"completed at {datetime.now().isoformat()}\n")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    elapsed = time.time() - t0
    print(f"  [{code} {name}] 完成（翻到底了），共存 {total_saved} 篇，"
          f"翻了 {total_pages} 頁，耗時 {elapsed/60:.1f} 分鐘")
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMoney股市爆料同學會爬蟲")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--out-dir", type=str, default="cmoney_267")
    parser.add_argument("--codes", type=str, default=None,
                         help="只抓指定股票代碼（逗號分隔），用於先驗證速度")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    stock_list = STOCK_LIST_267
    if args.codes:
        wanted = set(args.codes.split(","))
        stock_list = [s for s in stock_list if s[0] in wanted]

    print(f"=== CMoney爬蟲開始 ===")
    print(f"股票數：{len(stock_list)}，範圍：{start_date} ~ {end_date}")

    session = build_session()
    refresh_token_if_changed(session)  # 啟動時先套用一次目前的token

    total_start = time.time()
    grand_total = 0
    try:
        for i, (code, name, industry) in enumerate(stock_list, 1):
            print(f"\n[{i}/{len(stock_list)}] {code} {name}（{industry}）")
            grand_total += crawl_stock(session, code, name, start_date, end_date, args.out_dir, args.delay)
    except TokenExpired:
        print("\n" + "=" * 50)
        print("[Token過期] 請重新在瀏覽器登入CMoney，取得新token更新到 cmoney_token.json，")
        print("然後重新執行同一指令，會自動從中斷點接續，不會重複抓取。")
        print("=" * 50)
        raise SystemExit(1)

    elapsed = time.time() - total_start
    print(f"\n=== 全部完成，共 {grand_total} 篇，耗時 {elapsed/3600:.2f} 小時 ===")
