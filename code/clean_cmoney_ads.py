# -*- coding: utf-8 -*-
"""
清掉CMoney文章/留言裡看起來像廣告、推銷、機器人洗版的內容
============================================================

判斷規則（用真實資料抽樣驗證過，不是憑感覺猜的）：
    規則A：內文出現「飆漲追蹤器」——這是CMoney自動化機器人「營收快訊」
           貼文的固定樣板（"公司營收一公布，你1秒就收到！立即下載
           營收飆漲追蹤器👉連結"），確認是機器人自動發文，不是真人留言。
    規則B：內文同時符合「有CMoney自家推廣連結」（cmy.tw短網址、
           *.page.link深連結、cmoney.tw/r/或/app/導購路徑）
           +「有推銷字眼」（訂閱/會員/課程/早鳥/直播/限時等）——
           抽查過真的都是課程訂閱、APP推銷、直播導流這類廣告文。
    規則C：同時出現「免責聲明」+「APP」——通常是收費APP/社團的
           制式免責聲明段落，會跟著整篇推銷文一起出現。

    這三個規則衡量起來是「寧可漏抓、不要錯殺」的保守設計：文章大約
    抓到1.7%、留言大約0.007%被標記，抽查過準確率高，但也代表可能
    還有一些沒有連結、純文字推銷的廣告沒被抓到（例如個人品牌型的
    分析文最後幾句話置入業配），這類case比較難用規則區分，先不處理。

使用方式：
    python clean_cmoney_ads.py
輸出：
    cmoney_267_clean/{原檔名}.csv  （文章，逐檔輸出，過濾掉被標記的列）
    cmoney_267_comments/all_comments_clean.csv （留言）
    cmoney_ads_removed_articles.csv / cmoney_ads_removed_comments.csv
        （被移除的內容存檔，供人工抽查有沒有誤殺）
"""

import os
import re
import glob
import pandas as pd

AD_LINK_RE = re.compile(r"cmy\.tw/|\.page\.link/|cmoney\.tw/r/\d|cmoney\.tw/app/|appsite\.cmoney\.tw")
PROMO_WORD_RE = re.compile(r"訂閱|會員|課程|早鳥|直播|加碼贈|限時|營收飆漲追蹤器|飆漲追蹤器")


def is_ad(text):
    text = text or ""
    if "飆漲追蹤器" in text:
        return True
    if AD_LINK_RE.search(text) and PROMO_WORD_RE.search(text):
        return True
    if "免責聲明" in text and "APP" in text:
        return True
    return False


def clean_articles(in_dir, out_dir, removed_path):
    os.makedirs(out_dir, exist_ok=True)
    total_before, total_after = 0, 0
    removed_rows = []
    for path in glob.glob(os.path.join(in_dir, "*.csv")):
        df = pd.read_csv(path, dtype=str)
        total_before += len(df)
        mask = df["text"].fillna("").apply(is_ad)
        removed_rows.append(df[mask])
        clean_df = df[~mask]
        total_after += len(clean_df)
        out_path = os.path.join(out_dir, os.path.basename(path))
        clean_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    removed_df = pd.concat(removed_rows, ignore_index=True) if removed_rows else pd.DataFrame()
    removed_df.to_csv(removed_path, index=False, encoding="utf-8-sig")
    print(f"[文章] 清理前 {total_before} 筆 -> 清理後 {total_after} 筆"
          f"（移除 {total_before - total_after} 筆，{(total_before-total_after)/total_before*100:.2f}%）")
    print(f"[文章] 被移除的內容存到 {removed_path}（供人工抽查有沒有誤殺）")


def clean_comments(in_path, out_path, removed_path):
    df = pd.read_csv(in_path, dtype=str)
    mask = df["text"].fillna("").apply(is_ad)
    removed_df = df[mask]
    clean_df = df[~mask]
    clean_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    removed_df.to_csv(removed_path, index=False, encoding="utf-8-sig")
    print(f"[留言] 清理前 {len(df)} 筆 -> 清理後 {len(clean_df)} 筆"
          f"（移除 {len(removed_df)} 筆，{len(removed_df)/len(df)*100:.4f}%）")
    print(f"[留言] 被移除的內容存到 {removed_path}（供人工抽查有沒有誤殺）")


if __name__ == "__main__":
    print("=== 清理文章 ===")
    clean_articles("cmoney_267", "cmoney_267_clean", "cmoney_ads_removed_articles.csv")
    print("\n=== 清理留言 ===")
    clean_comments("cmoney_267_comments/all_comments.csv",
                    "cmoney_267_comments/all_comments_clean.csv",
                    "cmoney_ads_removed_comments.csv")
