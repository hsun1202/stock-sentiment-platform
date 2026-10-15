# -*- coding: utf-8 -*-
"""
嘗試把PTT因為單行字數上限被截斷的留言，跟後面緊接著的留言接回去。

限制（誠實講在前面）：
    沒有使用者ID，沒辦法100%確定下一則留言是不是同一個人接著打的，
    這是「盡量做、不是嚴謹解法」的heuristic修補，不是真正的斷點續傳
    重建。策略故意保守：
    1. 只在「長度夠長（很像真的撞到PTT單行上限）」+「結尾不像句子
       結束」兩個條件都成立時才嘗試接
    2. 只往「緊接著的下一則」接，最多連續接2次（避免誤觸發連鎖反應）
    3. 如果接完之後看起來還是不完整，才繼續接下一則；已經完整就停止
    這樣寧可漏掉一些「中間被別人插話」的斷句案例，也不要誤接兩個
    本來就各自完整、只是沒加標點的獨立留言。
"""
import argparse
import glob
import os
import pandas as pd

# 結尾出現這些字元/詞，視為「像是講完了」，不嘗試接下一則
TERMINAL_MARKERS = set("。！？!?~…」』）)0123456789０-９ 的了啊呀喔嗎吧耶ㄏㄎXDwww笑✓✗")
MIN_LEN_TO_TRY_STITCH = 20  # 低於這個長度，PTT本來就不太可能因為上限被切（原始留言中位數約13~14字）
MAX_NEXT_LEN = 12  # 關鍵防呆：真正的斷句延續通常很短（PTT使用者馬上補打剩下的字），
                    # 如果「下一則」本身很長，代表它是獨立留言、不是延續，不接


def looks_truncated(s):
    s = s.strip()
    if len(s) < MIN_LEN_TO_TRY_STITCH:
        return False
    return s[-1] not in TERMINAL_MARKERS


def stitch_group(contents):
    """對同一篇文章底下、按原始順序排列的留言list，嘗試接回被截斷的句子。
    只接一次（不連鎖），且只有下一則夠短時才接，降低誤接風險。"""
    result = []
    i = 0
    n = len(contents)
    merged_count = 0
    while i < n:
        cur = contents[i]
        if looks_truncated(cur) and i + 1 < n and len(contents[i + 1].strip()) <= MAX_NEXT_LEN:
            cur = cur + contents[i + 1]
            i += 1
            merged_count += 1
        result.append(cur)
        i += 1
    return result, merged_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="ptt_output_267")
    parser.add_argument("--out-dir", default="ptt_output_267_stitched")
    parser.add_argument("--sample-only", type=str, default=None,
                         help="測試用，只印出這個代號的前後對照，不寫檔")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.in_dir, "*_comments.csv"))

    if args.sample_only:
        f = [x for x in files if os.path.basename(x).startswith(args.sample_only + "_")][0]
        df = pd.read_csv(f)
        total_merged = 0
        for aid, g in df.groupby("article_id", sort=False):
            contents = g["content"].fillna("").astype(str).tolist()
            stitched, merged = stitch_group(contents)
            if merged > 0:
                total_merged += merged
                print(f"--- 文章 {aid}（本篇接了{merged}處）---")
                orig_idx = 0
                for s in stitched:
                    print("  ", repr(s))
        print(f"\n總共接了 {total_merged} 處")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    total_before, total_after, total_merged_all = 0, 0, 0
    for f in files:
        df = pd.read_csv(f)
        total_before += len(df)
        new_rows = []
        for (aid, code, name), g in df.groupby(["article_id", "stock_code", "stock_name"], sort=False):
            contents = g["content"].fillna("").astype(str).tolist()
            dates = g["resolved_date"].tolist()
            stitched, merged = stitch_group(contents)
            total_merged_all += merged
            for s in stitched:
                new_rows.append({"stock_code": code, "stock_name": name,
                                  "article_id": aid, "resolved_date": dates[0], "content": s})
        new_df = pd.DataFrame(new_rows)
        total_after += len(new_df)
        out_path = os.path.join(args.out_dir, os.path.basename(f))
        new_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[完成] 留言數 {total_before:,} -> {total_after:,}（減少 {total_before-total_after:,}，"
          f"共接了 {total_merged_all:,} 處）")


if __name__ == "__main__":
    main()
