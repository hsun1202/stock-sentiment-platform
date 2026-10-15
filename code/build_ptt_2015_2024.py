# -*- coding: utf-8 -*-
"""
PTT Stock板 2015-01-01 ~ 2024-12-31 合併＋清理＋留言拼接
============================================================

資料來源（2026-09-26 起：10年都含「後來在PTT上被刪的文章」，收錄標準一致；deleted_on_ptt 欄位可篩掉）：
    2019-03-01以後另外加入鏡像站的已刪文（pttweb_stock_2015_2019 的2019-03 ＋ pttweb_stock_2019_2024_deleted）。
    原因：鏡像站保留已刪文、PTT本站沒有；2019-03重疊月實測鏡像站多19%且全是已刪文，不補會在來源交界出現文章量斷層。

    2015-01-01 ~ 2019-02-28：pttweb.cc 鏡像站（crawl_pttweb_api.py → pttweb_stock_2015_2019/raw_articles.jsonl）
        PTT本站這段期間的一般文章已被看板上限清掉，只能用鏡像站。
    2019-03-01 ~ 2024-12-31：同學爬的 PTT 本站資料（ptt_classmate_2019_2024/stock_YYYY.jsonl）
        與自己的2020-2024爬蟲逐篇比對過：文章完全相同、留言只多了自己爬蟲誤刪的重複留言（compare_classmate_ptt.py）。

處理步驟（每篇文章各自處理）：
    1. 統一格式：文章(ID、日期、標題、作者、內文)、留言(推/噓/→、使用者ID、內容、時間)
    2. 留言時間補年份：PTT留言只有「MM/DD HH:MM」，以文章發文時間推算年份（規則見 infer_years；
       早於發文日31天內的標記 time_anomaly，例如板主公告內引用的違規留言）
       兩個來源都從原始的 MM/DD HH:MM 重新推算，確保規則一致。
    3. 清理：去掉內容空白的留言；同一篇文章只保留一份
    4. 留言拼接：沿用 stitch_truncated_comments_v2.py 的規則（同使用者、緊接著、前一則20-30字且結尾不像句尾、
       後一則≤12字），另外加一條「兩則留言時間相差≤1分鐘」（被截斷的長留言是一次送出的）。

輸出（ptt_2015_2024/）：
    articles.csv            文章（含內文body）
    comments.csv            留言（未拼接）
    comments_stitched.csv   留言（拼接後，後續情緒分類用這份）
    build_report.txt        各年份統計、異常計數
"""
import os
import re
import csv
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta

BASE = r"D:\專題claude"
PTTWEB_RAWS = [os.path.join(BASE, "pttweb_stock_2015_2019", "raw_articles.jsonl"),          # 2015-01 ~ 2019-03
               os.path.join(BASE, "pttweb_stock_2019_2024_deleted", "raw_articles.jsonl")]  # 2019-04 ~ 2024 已刪文
PTTWEB_RAW = PTTWEB_RAWS[0]  # validate_pttweb.py 用
CLASSMATE = os.path.join(BASE, "ptt_classmate_2019_2024", "stock_{}.jsonl")
SPLIT_DATE = date(2019, 3, 1)
START, END = date(2015, 1, 1), date(2024, 12, 31)

TAG_MAP_PTTWEB = {"+": "推", "-": "噓", "=": "→"}
AID_RE = re.compile(r"M\.(\d+)\.A\.[0-9A-F]{3}")
TIME_RE = re.compile(r"(\d{1,2})/(\d{1,2})(?:\s+(\d{1,2}):(\d{2}))?")

# ---- 拼接規則（同 stitch_truncated_comments_v2.py）----
TERMINAL_MARKERS = set("。！？!?~…」』）)0123456789０-９ 的了啊呀喔嗎吧耶ㄏㄎXDwww笑✓✗=@^><")
MIN_LEN, MAX_LEN, MAX_NEXT_LEN = 20, 30, 12
URL_PATTERN = re.compile(r"(https?://|www\.)\S+$")


def looks_truncated(s):
    n = len(s)
    if n < MIN_LEN or n > MAX_LEN or URL_PATTERN.search(s):
        return False
    return s[-1] not in TERMINAL_MARKERS


def can_merge(prev, cur, require_same_minute=True):
    if not (prev["user_id"] and prev["user_id"] == cur["user_id"]
            and looks_truncated(prev["content"]) and len(cur["content"]) <= MAX_NEXT_LEN):
        return False
    if require_same_minute:
        # 被截斷的長留言是一次送出的：2020年實測v2規則的候選中，相差0分鐘93.3%、1分鐘6.2%（跨分鐘邊界），
        # 相差≥2分鐘只佔0.5%且多半是同一人分開發的兩句話（甚至隔兩天），所以要求相差≤1分鐘
        a, b = prev["_dt"], cur["_dt"]
        return a is not None and b is not None and timedelta(0) <= b - a <= timedelta(minutes=1)
    return True


def aid_datetime(aid):
    m = AID_RE.search(aid)
    return datetime.fromtimestamp(int(m.group(1))) if m else None


def norm_time_raw(raw):
    m = TIME_RE.search(raw or "")
    if not m:
        return ""
    mo, d, h, mi = m.groups()
    return f"{int(mo):02d}/{int(d):02d} {int(h):02d}:{mi}" if h else f"{int(mo):02d}/{int(d):02d}"


def make_dt(year, raw_norm):
    try:
        if len(raw_norm) > 5:
            return datetime.strptime(f"{year}/{raw_norm}", "%Y/%m/%d %H:%M")
        return datetime.strptime(f"{year}/{raw_norm}", "%Y/%m/%d")
    except ValueError:  # 例如非閏年的 02/29
        return None


def infer_years(post_dt, raws):
    """推算每則留言的完整時間。回傳 [(datetime or None, anomaly_flag)]。

    PTT留言只有「MM/DD HH:MM」。每則留言各自判斷（不依序推算）：
        月日在發文時間之後（容許2天誤差）→ 發文同一年
        早於發文日、但在31天內          → 同一年，標記 time_anomaly（板主公告引用的違規留言、作者貼進內文的舊推文）
        其他                            → 下一年（跨年留言，例如12月的文章在1月被推）
    不依序推算的原因：實測有些文章的留言本來就不是照時間排（作者編輯時把整段舊推文搬到前面），
    依序推算會讓一則異常留言把後面整篇都推錯年（試過兩版，與同學原本的時間欄位仍有0.2~0.3%不一致）。
    代價：隔兩年以上才推的留言年份會少算，極少見。
    """
    tol = timedelta(days=2)
    out = []
    for raw in raws:
        if not raw:
            out.append((None, False))
            continue
        dt = make_dt(post_dt.year, raw)
        if dt is not None and dt >= post_dt - tol:
            out.append((dt, False))
        elif dt is not None and post_dt - dt <= timedelta(days=31):
            out.append((dt, True))
        else:
            dt2 = make_dt(post_dt.year + 1, raw)
            out.append((dt2, False) if dt2 is not None else (dt, True))
    return out


# ============================================================
# 讀兩個來源，轉成統一格式
# ============================================================

def load_ptt_ids():
    """同學PTT本站資料裡所有文章ID（2019-03以後，這些以PTT本站版本為準）。"""
    ids = set()
    for y in range(SPLIT_DATE.year, END.year + 1):
        with open(CLASSMATE.format(y), encoding="utf-8") as f:
            for line in f:
                ids.add(json.loads(line)["article_id"])
    return ids


def iter_pttweb(stats):
    """2019-03以前：鏡像站全部文章。2019-03以後：只取PTT本站資料沒有的（=後來在PTT上被刪的文章）。"""
    ptt_ids = load_ptt_ids()
    for path in PTTWEB_RAWS:
        if not os.path.exists(path):
            stats[f"pttweb_missing_file_{os.path.basename(os.path.dirname(path))}"] += 1
            continue
        yield from _iter_pttweb_file(path, ptt_ids, stats)


def _iter_pttweb_file(path, ptt_ids, stats):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                a = json.loads(line)["article"]
            except (ValueError, KeyError):
                stats["pttweb_bad_line"] += 1
                continue
            aid = a["article_aid"]
            post_dt = aid_datetime(aid)
            if post_dt is None or not (START <= post_dt.date() <= END):
                continue
            if post_dt.date() >= SPLIT_DATE and aid in ptt_ids:
                continue  # PTT本站有這篇 → 用同學的PTT本站版本
            if post_dt.date() >= SPLIT_DATE and not (a.get("is_deleted") or a.get("is_purged")):
                stats["pttweb_after_split_not_in_ptt_but_not_deleted"] += 1
            segs = sorted(a.get("main_text_data") or [], key=lambda s: s.get("position", 0))
            body = "".join(s.get("text", "") for s in segs).strip("\n")
            comments = []
            for r in sorted(a.get("recommend_data") or [], key=lambda r: r.get("position", 0)):
                text = r.get("text", "")
                if text.startswith(":"):
                    text = text[1:]
                t = r.get("type", "")
                if t not in TAG_MAP_PTTWEB:
                    stats[f"pttweb_unknown_tag_{t}"] += 1
                comments.append({"tag": TAG_MAP_PTTWEB.get(t, "→"), "user_id": r.get("author", ""),
                                 "content": text, "time_raw": r.get("ip_datetime", "")})
            yield {"article_id": aid, "post_dt": post_dt, "title": a.get("title", ""),
                   "author": a.get("author", ""), "body": body, "source": "pttweb",
                   "deleted_on_ptt": bool(a.get("is_deleted") or a.get("is_purged")),
                   "is_marked": bool(a.get("is_marked")), "comments": comments}


def iter_classmate(stats):
    for y in range(SPLIT_DATE.year, END.year + 1):
        with open(CLASSMATE.format(y), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                aid = d["article_id"]
                post_dt = aid_datetime(aid)
                if post_dt is None or not (SPLIT_DATE <= post_dt.date() <= END):
                    continue
                comments = [{"tag": c.get("tag", ""), "user_id": c.get("user_id", ""),
                             "content": c.get("content", ""), "time_raw": c.get("time_raw", ""),
                             "classmate_time": c.get("time", "")}
                            for c in (d.get("comments") or [])]
                yield {"article_id": aid, "post_dt": post_dt, "title": d.get("title", ""),
                       "author": d.get("author_id", ""), "body": d.get("content", ""), "source": "ptt",
                       "deleted_on_ptt": False, "is_marked": None, "comments": comments}


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(BASE, "ptt_2015_2024"))
    ap.add_argument("--no-same-minute", action="store_true", help="拼接時不要求時間相近（對照用）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    art_fields = ["article_id", "link", "resolved_date", "post_time", "title", "author", "category",
                  "n_push", "n_boo", "n_arrow", "n_comments", "source", "deleted_on_ptt", "is_marked", "body"]
    com_fields = ["article_id", "link", "resolved_date", "idx", "push_type", "user_id", "content",
                  "time_raw", "comment_time", "time_anomaly", "source"]
    fa = open(os.path.join(args.out_dir, "articles.csv"), "w", newline="", encoding="utf-8-sig")
    fc = open(os.path.join(args.out_dir, "comments.csv"), "w", newline="", encoding="utf-8-sig")
    fs = open(os.path.join(args.out_dir, "comments_stitched.csv"), "w", newline="", encoding="utf-8-sig")
    wa, wc, ws = (csv.DictWriter(fa, art_fields), csv.DictWriter(fc, com_fields),
                  csv.DictWriter(fs, com_fields + ["stitched"]))
    for w in (wa, wc, ws):
        w.writeheader()

    stats = Counter()
    per_year = defaultdict(Counter)
    seen = set()
    cm_time_check = Counter()

    for src in (iter_pttweb(stats), iter_classmate(stats)):
        for art in src:
            aid = art["article_id"]
            if aid in seen:
                stats["dup_article"] += 1
                continue
            seen.add(aid)
            post_dt = art["post_dt"]
            day = post_dt.date().isoformat()
            yr = per_year[post_dt.year]
            link = f"https://www.ptt.cc/bbs/Stock/{aid}.html"

            # 清理留言：去頭尾空白、去掉空內容
            coms = []
            for c in art["comments"]:
                c["content"] = (c["content"] or "").strip()
                c["user_id"] = (c["user_id"] or "").strip()
                c["tag"] = (c["tag"] or "").strip()
                if not c["content"]:
                    stats["drop_empty_comment"] += 1
                    continue
                c["time_raw_norm"] = norm_time_raw(c["time_raw"])
                if not c["time_raw_norm"]:
                    stats["comment_no_time"] += 1
                coms.append(c)

            times = infer_years(post_dt, [c["time_raw_norm"] for c in coms])
            rows = []
            for i, (c, (dt, anomaly)) in enumerate(zip(coms, times)):
                if anomaly:
                    stats["time_anomaly"] += 1
                if "classmate_time" in c and c["classmate_time"] and dt is not None:
                    cm_time_check["same" if c["classmate_time"][:16] == dt.isoformat()[:16] else "diff"] += 1
                rows.append({"article_id": aid, "link": link, "resolved_date": day, "idx": i + 1,
                             "push_type": c["tag"], "user_id": c["user_id"], "content": c["content"],
                             "time_raw": c["time_raw_norm"],
                             "comment_time": dt.isoformat(timespec="minutes") if dt else "",
                             "time_anomaly": int(anomaly), "source": art["source"],
                             "time_raw_norm": c["time_raw_norm"], "_dt": dt})

            # 拼接
            stitched, i = [], 0
            while i < len(rows):
                if i + 1 < len(rows) and can_merge(rows[i], rows[i + 1], not args.no_same_minute):
                    m = dict(rows[i])
                    m["content"] = rows[i]["content"] + rows[i + 1]["content"]
                    m["stitched"] = 1
                    stitched.append(m)
                    i += 2  # 合併後不連鎖（與v2相同的保守策略）
                else:
                    m = dict(rows[i])
                    m["stitched"] = 0
                    stitched.append(m)
                    i += 1
            for r in rows:
                r.pop("time_raw_norm")
                r.pop("_dt")
                wc.writerow(r)
            for r in stitched:
                r.pop("time_raw_norm")
                r.pop("_dt")
                ws.writerow(r)

            tags = Counter(c["tag"] for c in coms)
            cat = re.match(r"^(?:Re:\s*|Fw:\s*)?\[([^\]]{1,4})\]", art["title"] or "")
            wa.writerow({"article_id": aid, "link": link, "resolved_date": day,
                         "post_time": post_dt.isoformat(timespec="seconds"), "title": art["title"],
                         "author": art["author"], "category": cat.group(1) if cat else "",
                         "n_push": tags["推"], "n_boo": tags["噓"], "n_arrow": tags["→"],
                         "n_comments": len(coms), "source": art["source"],
                         "deleted_on_ptt": int(art["deleted_on_ptt"]),
                         "is_marked": "" if art["is_marked"] is None else int(art["is_marked"]),
                         "body": art["body"]})
            yr["articles"] += 1
            yr["comments"] += len(rows)
            yr["comments_stitched"] += len(stitched)
            yr[f"src_{art['source']}"] += 1
            if art["deleted_on_ptt"]:
                yr["deleted_on_ptt"] += 1
            stats["articles"] += 1
            if stats["articles"] % 20000 == 0:
                print(f"    已處理 {stats['articles']:,} 篇（{day}）")

    for fh in (fa, fc, fs):
        fh.close()

    lines = [f"PTT Stock 2015-2024 合併報告（{datetime.now():%Y-%m-%d %H:%M}）",
             f"拼接規則：{'v2規則' if args.no_same_minute else 'v2規則＋時間相差≤1分鐘'}", "",
             "年份 | 文章 | 來源 | 其中PTT上已刪 | 留言 | 拼接後留言 | 合併處"]
    for y in sorted(per_year):
        c = per_year[y]
        src = "/".join(f"{k[4:]}:{v}" for k, v in c.items() if k.startswith("src_"))
        lines.append(f"{y} | {c['articles']:,} | {src} | {c['deleted_on_ptt']:,} | {c['comments']:,} | "
                     f"{c['comments_stitched']:,} | {c['comments']-c['comments_stitched']:,}")
    lines += ["", "其他統計：" + ", ".join(f"{k}={v:,}" for k, v in sorted(stats.items())),
              f"留言年份推算 vs 同學原本的time欄位（到分鐘）：{dict(cm_time_check)}"]
    rep = "\n".join(lines)
    print(rep)
    with open(os.path.join(args.out_dir, "build_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")


if __name__ == "__main__":
    main()
