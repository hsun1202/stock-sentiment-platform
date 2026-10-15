# -*- coding: utf-8 -*-
"""
PTT 留言內容清理：去掉「沒有參考價值的廢話」，保留有情緒/看法的留言
============================================================

輸入：ptt_2015_2024/comments_stitched.csv（build_ptt_2015_2024.py 產出，已拼接）、articles.csv（取文章分類）
輸出：ptt_2015_2024/comments_clean.csv   保留下來的留言（欄位同輸入）
      ptt_2015_2024/noise_report.txt     各規則刪除數量、各年保留比例、用Claude標註驗證的結果
      ptt_2015_2024/noise_samples.csv    每條規則隨機抽50則被刪的留言，方便人工檢查

設計原則：PTT股板很多「很短但有情緒」的留言（噴、崩、完了、丸子、咕嚕咕嚕、穩了、塊陶、賺爛了），
所以不用「字數太少就刪」，而是只刪「確定沒有方向」的內容。規則（依序判斷，命中第一條就刪）：
    A. article_admin   [公告]文章底下的留言（處置公告引用的違規推文、水桶、樂透開獎、板規討論，4,661篇）；
                       以及標題是遊戲類的活動文（搶答、樂透、抽獎、猜收盤、名人堂、大地遊戲）。
                       其他[活動]文（「叫我賠錢王」、操作心得徵文）底下是真的在聊股票，保留。
    A2. chant          同一篇文章裡 ≥20 個不同帳號貼完全相同的一句話、且不含情緒關鍵字 → 抽獎/活動口令
                       （例：「加入口袋，共創小資新時代」1,447人各貼一次；「分點籌碼要看處置王！」997人）
    B. url_only        只有網址/圖片連結（沒有文字可以判斷情緒）
    C. no_text         沒有任何中文或英文字：純數字、標點、符號、ASCII圖（例：995、992885、？、…、┘└┘）
    D. laugh           只有笑聲/語助詞/驚嘆：哈哈、XD、笑死、ㄏㄏ、QQ、喔、欸、傻眼
    E. meta            沒有方向的社交/板務用語：推、卡、低調、朝聖、感謝分享、早、簽到、+1、反觀、你會贏
    F. flood           同一人在同一篇文章重複貼完全相同的內容 → 只留第一則
    G. catchphrase     同一人在 ≥30 篇不同文章貼完全相同的一句話，而且這句話不含任何情緒關鍵字
                       （個人簽名/洗版，例：某帳號在2,306篇貼「無腦多睏霸數錢」）
情緒關鍵字（SENTIMENT_KEEP）命中的留言，A2/G 規則不會刪（第一版漏了衝阿、殺丫、救命、直上五百等，已補）；
D/E 是完全比對整句，也不會誤刪「噴噴」「完了」這類。
"""
import os
import re
import csv
import random
import argparse
from collections import Counter, defaultdict

import pandas as pd

BASE = r"D:\專題claude"
D = os.path.join(BASE, "ptt_2015_2024")

ADMIN_CATEGORIES = {"公告"}
GAME_TITLE = re.compile(r"搶答|樂透|抽獎|猜.{0,12}(?:收盤|指數|點數|價)|名人堂|大地遊戲|未讀取")  # 只套用在[活動]分類
CHANT_MIN_USERS = 20
URL_ONLY = re.compile(r"^(?:(?:https?://|www\.)\S+\s*)+$")
HAS_TEXT = re.compile(r"[\u4e00-\u9fffA-Za-z]")
LAUGH_TOKENS = ["笑死人", "笑死", "笑爛", "好笑", "傻眼", "哇靠", "挖靠", "XD", "xD", "Xd", "xd", "QAQ", "QQ", "qq",
                "lol", "LOL", "wow", "Wow", "OMG", "哈", "呵", "嘿", "嘻", "ㄏ", "ㄎ", "ㄟ", "囧", "喔", "噢", "哦", "欸",
                "誒", "嗯", "恩", "啊", "阿", "耶", "呀", "喲", "唷", "哇", "蛤", "D", "w", "W"]
LAUGH = re.compile(r"^(?:" + "|".join(map(re.escape, LAUGH_TOKENS)) + r"|[\s\?\？!！~～\.。…,，:：\-_=^><@])+$")
META_PHRASES = {
    "推", "推推", "推推推", "推一個", "推一下", "先推", "推文", "給推", "必推", "推好文", "補推", "補血",
    "卡", "卡位", "卡卡", "前排", "頭香", "搶頭香", "低調", "朝聖", "簽到", "打卡", "早", "早安", "午安", "晚安", "安安",
    "感謝分享", "謝謝分享", "感恩分享", "感謝", "謝謝", "感恩", "謝啦", "謝囉", "謝", "多謝",
    "好文", "優文", "認真文", "廢文", "有料", "專業", "厲害", "長知識", "學到了", "筆記", "收藏", "加油",
    "好了啦", "好", "好喔", "喔喔", "是喔", "真的假的", "真假", "然後呢", "所以呢", "所以", "什麼", "三小",
    "看不懂", "不懂", "了解", "嗯嗯", "同意", "同上", "樓上", "樓下", "我也是", "+1", "1@", "反觀", "你會贏", "你會贏的",
    "已檢舉", "檢舉", "標題", "內文", "來源", "原po", "原PO", "原文", "讚", "讚讚", "水",
}
META_STRIP = re.compile(r"[\s\?\？!！~～\.。…,，]+$")
SENTIMENT_KEEP = re.compile(
    r"噴|崩|漲|跌|多|空|丸|完了|完蛋|咕嚕|穩|陶|逃|賺|賠|虧|套|韭菜|歐印|梭哈|抄底|接刀|停損|停利|出貨|割|慘|爽|"
    r"恭喜|畢業|起飛|上車|下車|利多|利空|反彈|漲停|跌停|爆量|天價|糕|死|GG|燒|破|買|賣|all ?in|"
    r"衝|殺|救命|不妙|嘎|直上|狂|下去|拉|飛|月球|哭|怕|委屈|凹|嚇|抱|99|久久|九九|加碼|減碼|滿倉|空手|進場|出場|"
    r"逃命|牛|熊|泡沫|危險|機會|便宜|貴|高點|低點|底|頂|香|臭|爛|垃圾|神|糟|妙|哀|痛|苦|開心|難過|恐慌|絕望|希望|"
    r"倒|撐|守|輸|贏|虧|血|慘|樂觀|悲觀|看好|看壞|危|安心|擔心|猛|急|沖|救|發財|預期|要v|上看|下看", re.I)
CATCHPHRASE_MIN_ARTICLES = 30


def content_rule(c):
    """只看內容就能判斷的規則（B~E）。回傳規則名稱或 None。"""
    if URL_ONLY.match(c):
        return "url_only"
    if not HAS_TEXT.search(c):
        return "no_text"
    if LAUGH.match(c):
        return "laugh"
    if META_STRIP.sub("", c) in META_PHRASES:
        return "meta"
    return None


def find_catchphrases(path):
    """回傳 (口頭禪, 口令)：
    口頭禪：同一人在 ≥30 篇不同文章貼完全相同一句話（且不含情緒關鍵字）→ 個人簽名/洗版。
    口令：同一篇文章裡 ≥20 個不同帳號貼完全相同一句話（且不含情緒關鍵字）→ 抽獎/活動口令。"""
    parts = []
    for ch in pd.read_csv(path, chunksize=3_000_000, usecols=["article_id", "user_id", "content"],
                          dtype=str, keep_default_na=False):
        ch["content"] = ch["content"].str.strip()
        parts.append(ch.drop_duplicates())
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    n = df.groupby(["user_id", "content"]).size()
    n = n[n >= CATCHPHRASE_MIN_ARTICLES]
    keep = {(u, c): k for (u, c), k in n.items() if u and not SENTIMENT_KEEP.search(c)}
    print(f"[口頭禪] 符合條件的(使用者,句子)組合 {len(keep):,} 組")
    m = df.groupby(["article_id", "content"])["user_id"].nunique()
    m = m[m >= CHANT_MIN_USERS]
    chant = {(a, c): k for (a, c), k in m.items() if not SENTIMENT_KEEP.search(c)}
    print(f"[口令] 符合條件的(文章,句子)組合 {len(chant):,} 組")
    return keep, chant


def validate_with_claude_labels(log):
    p = os.path.join(BASE, "claude_all_labels.csv")
    lab = pd.read_csv(p, dtype=str, keep_default_na=False)
    lab["rule"] = lab["content"].str.strip().map(content_rule)
    tab = pd.crosstab(lab["rule"].fillna("保留"), lab["claude_label"], margins=True)
    log("\n[驗證] Claude逐則標註的1,100則留言，套用內容規則(B~E)後：")
    log(tab.to_string())
    removed = lab[lab["rule"].notna()]
    if len(removed):
        log(f"    被刪的 {len(removed)} 則中 neutral 佔 {(removed['claude_label']=='neutral').mean():.1%}；"
            f"全體 neutral 佔 {(lab['claude_label']=='neutral').mean():.1%}")
        for r in removed[removed["claude_label"] != "neutral"].itertuples():
            log(f"    被刪但Claude標為{r.claude_label}：{r.content!r}（{r.rule}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default=os.path.join(D, "comments_stitched.csv"))
    ap.add_argument("--out-csv", default=os.path.join(D, "comments_clean.csv"))
    args = ap.parse_args()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    arts = pd.read_csv(os.path.join(D, "articles.csv"), usecols=["article_id", "category", "title"],
                       dtype=str, keep_default_na=False)
    is_admin = arts["category"].isin(ADMIN_CATEGORIES)
    is_game = (arts["category"] == "活動") & arts["title"].str.contains(GAME_TITLE)  # 一般新聞也會用「樂透」，不能只看標題
    admin_articles = set(arts.loc[is_admin | is_game, "article_id"])
    log(f"[公告] {is_admin.sum():,} 篇、[遊戲類活動] {(is_game & ~is_admin).sum():,} 篇，底下留言整篇排除")
    log("    遊戲類標題例：" + "；".join(arts.loc[is_game & ~is_admin, "title"].head(8)))

    catch, chant = find_catchphrases(args.in_csv)
    top_chant = sorted(chant.items(), key=lambda kv: -kv[1])[:10]
    log("[口令] 人數最多的前10組：" + "；".join(f"{c[:18]}({k}人)" for (a, c), k in top_chant))
    top_catch = sorted(catch.items(), key=lambda kv: -kv[1])[:15]
    log("[口頭禪] 篇數最多的前15組：" + "；".join(f"{u}:{c[:15]}({k}篇)" for (u, c), k in top_catch))

    counts = Counter()
    per_year = defaultdict(Counter)
    samples = defaultdict(list)
    seen_rule = Counter()
    random.seed(0)
    with open(args.in_csv, encoding="utf-8-sig") as fin, \
         open(args.out_csv, "w", newline="", encoding="utf-8-sig") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        cur_art, posted = None, set()
        for i, row in enumerate(r, 1):
            c = (row["content"] or "").strip()
            u = row["user_id"]
            aid = row["article_id"]
            if aid != cur_art:
                cur_art, posted = aid, set()
            if aid in admin_articles:
                rule = "article_admin"
            else:
                rule = content_rule(c)
                if rule is None and (aid, c) in chant:
                    rule = "chant"
                if rule is None and (u, c) in posted:
                    rule = "flood"
                if rule is None and (u, c) in catch:
                    rule = "catchphrase"
            posted.add((u, c))
            y = row["resolved_date"][:4]
            per_year[y]["total"] += 1
            if rule:
                counts[rule] += 1
                seen_rule[rule] += 1
                # 蓄水池抽樣：每條規則均勻抽50則
                if len(samples[rule]) < 50:
                    samples[rule].append((rule, y, u, c))
                elif random.random() < 50 / seen_rule[rule]:
                    samples[rule][random.randrange(50)] = (rule, y, u, c)
            else:
                per_year[y]["kept"] += 1
                w.writerow(row)
            if i % 5_000_000 == 0:
                print(f"    已處理 {i:,} 則")

    total = sum(v["total"] for v in per_year.values())
    kept = sum(v["kept"] for v in per_year.values())
    log(f"\n[總計] {total:,} 則 → 保留 {kept:,} 則（{kept/total:.1%}），刪除 {total-kept:,} 則")
    for k, v in counts.most_common():
        log(f"    {k:14s} {v:>12,}（{v/total:.2%}）")
    log("\n年份 | 原本 | 保留 | 保留比例")
    for y in sorted(per_year):
        t, k = per_year[y]["total"], per_year[y]["kept"]
        log(f"  {y} | {t:,} | {k:,} | {k/t:.1%}")

    validate_with_claude_labels(log)

    with open(os.path.join(D, "noise_samples.csv"), "w", newline="", encoding="utf-8-sig") as f:
        sw = csv.writer(f)
        sw.writerow(["rule", "year", "user_id", "content"])
        for rule in samples:
            sw.writerows(samples[rule])
    with open(os.path.join(D, "noise_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n→ {args.out_csv}\n→ {os.path.join(D, 'noise_report.txt')}\n→ {os.path.join(D, 'noise_samples.csv')}")


if __name__ == "__main__":
    main()
