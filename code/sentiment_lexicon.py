# -*- coding: utf-8 -*-
"""
台股 PTT 留言文字情緒分析 (V6：權重詞典 + 成語覆寫機制)
============================================================

版本說明：
    V6：用250筆人工標註樣本驗證詞典法準確率（62.1%），並依實測
    結果移除兩個誤判率偏高的詞：
        - 「委屈了」（原BULLISH權重2）：人工標註顯示這個詞語意
          太模糊，常常只是單純哀怨、不隱含「會漲」的方向性，
          甚至偶爾出現在看空語境（如「委屈了 下去休息吧」）。
        - 「出貨」（原BEARISH權重2）：除了先前發現的航運股物流
          語境歧義，人工標註顯示連「出貨文」這種股版黑話用法，
          真人也常判斷為中立而非明確負面，訊號不夠可靠。看空的
          「倒貨」語意已由獨立詞條保留，不受影響。

    這版本是 V4（權重詞典，含 NTUSD 通用財經詞彙）與先前 V3
    （成語覆寫機制）的合併版。

    V4 的權重分級設計（PTT黑話3分 > 通用財經詞1分）是好的進步，
    但少了「利多出盡」「利空出盡」這類成語覆寫，導致這兩句
    極常見的股版用語被誤判（字面上的「利多」「利空」跟整句
    真實語意相反）。這版本把成語覆寫機制加回來，且優先於
    一般權重計分執行。

使用方式：
    python sentiment_lexicon.py --input 2330_ptt_posts_20260730_comments.csv --out 2330_sentiment_result.csv
"""

import pandas as pd
import argparse
import re

# ============================================================
# 台股 PTT 鄉民專用情緒詞典 (V4 權重版，維持不變)
# ============================================================

BULLISH_TERMS = {
    # ── PTT 專屬黑話與強烈情緒 ──
    "歐印": 3, "梭哈": 3, "all in": 3, "財富自由": 3, "賣房買": 3,
    "賺爛": 3, "嘎爆空軍": 3, "噴出天際": 3, "送分題": 3, "漲停": 3,
    "買爆": 3, "紅燈": 3,
    "噴出": 2, "起飛": 2, "井噴": 2, "上車": 2,
    "數錢": 2, "加碼": 2, "抄底": 2, "多軍": 2, "嘎空": 2,
    "做多": 2, "填息": 2, "外資買爆": 2, "滿面紅光": 2,
    "舒服": 1, "穩了": 1, "有戲": 1, "向上": 1, "撿便宜": 1,
    "肉很多": 1, "反彈": 1,

    # ── 來自 NTUSD 的通用正向詞 (賦予權重 1，處理正經的分析文) ──
    "看好": 1, "樂觀": 1, "受惠": 1, "復甦": 1, "帶動": 1,
    "成長": 1, "增加": 1, "優秀": 1, "良好": 1, "優於預期": 1,
    "雙增": 1, "利多": 1
}

BEARISH_TERMS = {
    # ── PTT 專屬黑話與強烈情緒 ──
    "塊陶": 3, "畢業": 3, "睡公園": 3, "跳樓": 3, "血流成河": 3,
    "跌停": 3, "法會": 3, "斷頭": 3, "大屠殺": 3, "關燈": 3,
    "丸子": 2, "完了": 2, "已反應": 2, "已反映": 2,
    "倒貨": 2, "割韭菜": 2, "咕嚕": 2, "吃筍": 2, "吃損": 2,
    "空軍": 2, "電梯向下": 2, "抓交替": 2, "探底": 2, "破底": 2,
    "綠油油": 2, "做空": 2,
    "套牢": 1, "住套房": 1, "酸": 1, "涼了": 1, "幫QQ": 1,
    "下車": 1, "賣飛": 1, "吃麵": 1,

    # ── 來自 NTUSD 的通用負向詞 (賦予權重 1，處理正經的分析文) ──
    "看壞": 1, "悲觀": 1, "衰退": 1, "萎縮": 1, "惡化": 1,
    "拖累": 1, "減少": 1, "虧損": 1, "疑慮": 1, "不如預期": 1,
    "保守": 1, "利空": 1, "失望": 1
}

NEGATION_WORDS = ["不", "沒", "別", "勿", "非", "無"]  # 簡易否定詞判斷

# ============================================================
# 慣用語覆寫表（Idiom Overrides）—— 這次合併的重點
# ============================================================
# 「利多出盡」字面含有看多的「利多」，但整句意思其實是
# 「好消息已反映完畢，要開始賣了」= 看空訊號。
# 這類成語必須「優先」於一般詞彙比對處理，並把命中的成語
# 從文字中移除，避免成語內的單字被權重計分重複、甚至反向計分。
#
# 權重設為 3，比照 PTT 黑話等級的強烈訊號（這類成語在股版
# 幾乎是明確的表態，訊號強度不該只算 1 分）。
IDIOM_OVERRIDES = {
    "利多出盡": -3,
    "利空出盡": 3,
    "追高殺低": -2,
    "高處不勝寒": -2,
}
IDIOM_PATTERN = re.compile("|".join(re.escape(k) for k in IDIOM_OVERRIDES.keys()))


def classify_comment(text):
    """
    對單則留言文字做情緒判斷（權重總分制 + 成語覆寫）。
    回傳: "positive" / "negative" / "neutral"

    判斷順序：
        1. 先攔截「慣用語覆寫表」，並把命中的成語從文字中移除，
           避免成語內的單字（如「利多」）被一般權重詞彙重複計分。
        2. 用一般權重詞彙比對剩餘文字，計算看多/看空總分；
           不論看多詞或看空詞，只要前面出現否定詞，都反向計分
           （修正舊版只對看空詞做否定修正，導致「不看好」「不樂觀」
           這類常見用語被誤判成正面的不對稱漏洞）。
        3. 加總比較，分數高的一方勝出；打平則視為中立。
    """
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    bull_score = 0
    bear_score = 0

    # 1. 慣用語覆寫（優先處理，並從文字中移除避免重複計分）
    remaining_text = text
    for match in IDIOM_PATTERN.finditer(text):
        idiom = match.group()
        score = IDIOM_OVERRIDES[idiom]
        if score > 0:
            bull_score += score
        else:
            bear_score += abs(score)
        remaining_text = remaining_text.replace(idiom, "")

    # 2. 一般權重詞彙比對（用移除成語後的剩餘文字），看多/看空詞
    #    都套用同一套否定詞修正邏輯，避免只修正單一方向的不對稱漏洞
    for term, weight in BULLISH_TERMS.items():
        idx = remaining_text.find(term)
        while idx != -1:
            prefix = remaining_text[max(0, idx - 2):idx]
            if any(neg in prefix for neg in NEGATION_WORDS):
                bear_score += weight
            else:
                bull_score += weight
            idx = remaining_text.find(term, idx + len(term))

    for term, weight in BEARISH_TERMS.items():
        idx = remaining_text.find(term)
        while idx != -1:
            prefix = remaining_text[max(0, idx - 2):idx]
            if any(neg in prefix for neg in NEGATION_WORDS):
                bull_score += weight
            else:
                bear_score += weight
            idx = remaining_text.find(term, idx + len(term))

    # 4. 判定最終情緒
    if bull_score > bear_score:
        return "positive"
    elif bear_score > bull_score:
        return "negative"
    else:
        return "neutral"


def analyze_comments_file(input_path):
    """讀取爬蟲輸出的留言 CSV，逐則判斷情緒，並依文章/日期彙整。"""
    df = pd.read_csv(input_path)
    if "content" not in df.columns:
        raise ValueError("輸入檔案缺少 'content' 欄位，請確認是否為 *_comments.csv 檔案")

    df["text_sentiment"] = df["content"].apply(classify_comment)

    print("=== 全體留言情緒分布（文字判斷法）===")
    print(df["text_sentiment"].value_counts())
    print()

    if "push_type" in df.columns:
        print("=== 對照：推噓型態 vs 文字判斷情緒 交叉表 ===")
        print(pd.crosstab(df["push_type"], df["text_sentiment"]))
        print()

    return df


def aggregate_by_article(df):
    """依文章（link）彙總，計算「文字情緒因子」。"""
    # 相容新舊兩種爬蟲輸出格式：新版batch_crawler.py用resolved_date，
    # 舊版專題爬蟲.py用board_date
    date_col = "resolved_date" if "resolved_date" in df.columns else "board_date"

    def compute_factor(group):
        pos = (group["text_sentiment"] == "positive").sum()
        neg = (group["text_sentiment"] == "negative").sum()
        total = len(group)
        factor = (pos - neg) / total if total > 0 else 0.0
        return pd.Series({
            "total_comments": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": total - pos - neg,
            "text_sentiment_factor": round(factor, 4),
        })

    grouped = df.groupby(["stock_code", "link", date_col], as_index=False).apply(
        compute_factor, include_groups=False
    )
    grouped = grouped.rename(columns={date_col: "resolved_date"})
    return grouped


import os
import importlib.util


def load_stock_list():
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTT 留言文字情緒分析（權重詞典 + 成語覆寫版）")
    parser.add_argument("--input", type=str, default=None,
                         help="單檔模式：爬蟲輸出的 *_comments.csv 檔案")
    parser.add_argument("--out", type=str, default="sentiment_result.csv",
                         help="單檔模式：輸出結果 CSV 檔名")
    parser.add_argument("--batch-dir", type=str, default=None,
                         help="批次模式：PTT資料所在資料夾（例如 batch_output），"
                              "會自動處理股票名單裡的全部股票")
    parser.add_argument("--out-dir", type=str, default="sentiment_output",
                         help="批次模式：輸出資料夾")
    args = parser.parse_args()

    if args.batch_dir:
        # ---- 批次模式：處理30檔股票 ----
        STOCK_LIST = load_stock_list()
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"=== 批次情緒分析：{len(STOCK_LIST)} 檔股票 ===\n")

        success_count, skip_count, fail_count = 0, 0, 0
        for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
            comments_path = os.path.join(args.batch_dir, f"{code}_{name}_comments.csv")
            out_path = os.path.join(args.out_dir, f"{code}_{name}_sentiment.csv")

            if os.path.exists(out_path):
                print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）已存在，跳過")
                skip_count += 1
                continue

            if not os.path.exists(comments_path):
                print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）找不到留言檔案，跳過")
                fail_count += 1
                continue

            print(f"[{i}/{len(STOCK_LIST)}] 分析 {code} {name}（{industry}）...")
            try:
                df = analyze_comments_file(comments_path)
                article_summary = aggregate_by_article(df)
                article_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"    [完成] {len(article_summary)} 篇文章的情緒因子")
                success_count += 1
            except Exception as e:
                print(f"    [錯誤] {e}")
                fail_count += 1

        print(f"\n=== 批次完成 ===")
        print(f"成功：{success_count}，跳過（已完成）：{skip_count}，失敗：{fail_count}")

    elif args.input:
        # ---- 單檔模式（沿用原本邏輯）----
        df = analyze_comments_file(args.input)
        article_summary = aggregate_by_article(df)

        print("=== 各文章「文字情緒因子」彙總（前 10 筆）===")
        print(article_summary.head(10).to_string(index=False))

        article_summary.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[完成] 已輸出彙總結果至 {args.out}")
    else:
        print("[錯誤] 請指定 --input（單檔模式）或 --batch-dir（批次模式）")
