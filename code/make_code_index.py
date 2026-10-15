# -*- coding: utf-8 -*-
"""
產生 GitHub 專案用的程式索引 code/INDEX.md：每支程式的用途（取程式開頭說明的第一段）、對應的結果檔，依主題分組。
由 sync_github.sh 在每次同步前自動執行，所以新程式會自動出現在索引裡。
python make_code_index.py <匯出資料夾>
"""
import ast
import io
import os
import re
import sys

DST = sys.argv[1] if len(sys.argv) > 1 else "D:/專題claude_github"
CODE, RES = os.path.join(DST, "code"), os.path.join(DST, "results")

GROUPS = [
    ("資料蒐集：爬蟲與下載", r"^(crawl_|batch_crawler|download_|validate_pttweb|check_crawl|check_crawler|watchdog_|after_download|add_stock_scan|stock_volume_scanner|cmoney_probe)"),
    ("資料清理與資料集建構", r"^(build_|merge_|clean_|cmoney_clean|cmoney_daily_build|cmoney_date_fix|stitch_|rebuild_|fix_|match_ptt|stock_list|compute_technical|stock_indicators|data_audit|audit_|quality_check|tej_build|cnyes_build|make_dataset|prepare_lstm|daytrade_build|chat_2025_prep|stock_ptt_2025_prep|market_sentiment_prep)"),
    ("情緒分類：Breeze、BERT、微調、重標", r"^(breeze_|finetune_|eval_|infer_|train_bert|prepare_bert|prepare_breeze|prepare_human|test_bert|test_breeze|test_constrained|test_fast|test_logit|test_padded|test_taiwan|benchmark_|debug_|relevance_|merge_finetune|relabel_|classifier_noise|scratch_claude|build_claude|build_annotation|build_fresh|sentiment_lexicon|market_sentiment_classify|news_sentiment_classify|intraday_classify|chat_2025_classify|stock_ptt_2025_classify|check_breeze|audit_breeze|ft_|after_breeze|after_market|emotion_classify)"),
    ("預測檢驗：PTT／CMoney 情緒 → 漲跌與報酬", r"^(direction_|retail_|ptt_|cmoney_direction|cmoney_daily_test|sentiment_factor|lag_|granger|spike_|longshort|return_regression|sign_strategy|split5|size_group|learning_curve|holdout_|stock_ptt_2025_test|market_ptt|taiex_|logreg_|lstm_|train_lstm|train_xgboost|summarize_|compare_lstm|run_lstm|schedule_lstm|volatility_test|vol_volume|extract_predictions|after_lstm|daytrade_test|margin_return|backtest_v4|news_sentiment_analysis|news_drift|emotion_market|market_sentiment|intraday_sentiment|afterhours_|compare_classmate)"),
    ("共用工具", r"^(features_v4|run_with_ft|analyze_v2)"),
    ("TEJ 全市場與散戶交易行為", r"^(tej_retail|tej_fullmarket|tej_oos2025)"),
    ("事件：公告、新聞、大波動日", r"^(bigmove_|mops_|cnyes_bigmove|tej_event|buyback_|newsmove_|negnews_|bigup_|attention_disposition|retail_chase)"),
    ("交易頻率：月頻、週頻選股", r"^(monthly_|weekly_|overnight_sentiment)"),
    ("情緒當開關", r"^(sentiment_regime)"),
    ("專案維護", r"^(sync_github|make_code_index)"),
]


def first_doc(path):
    try:
        src = io.open(path, encoding="utf-8").read()
    except Exception:
        return ""
    if path.endswith(".py"):
        try:
            d = ast.get_docstring(ast.parse(src)) or ""
        except Exception:
            d = ""
        if not d:
            m = re.search(r'"""(.*?)"""', src, re.S)
            d = m.group(1) if m else ""
    else:
        d = "\n".join(l[1:].strip() for l in src.splitlines()[1:8] if l.startswith("#"))
    d = d.strip().split("\n\n")[0]
    d = " ".join(x.strip() for x in d.splitlines() if x.strip() and not set(x.strip()) <= set("=-") and not x.strip().startswith(("python ", "bash ", "用法")))
    return (d[:220] + "…") if len(d) > 220 else d


files = sorted(f for f in os.listdir(CODE) if f.endswith((".py", ".sh")))
res = set(os.listdir(RES)) if os.path.isdir(RES) else set()
used, out = set(), ["# 程式索引", "", "自動產生（`make_code_index.py`）。說明取自每支程式開頭的註解；「結果」是 `results/` 裡同名的輸出檔。", ""]
for title, pat in GROUPS:
    rows = [f for f in files if re.search(pat, f) and f not in used]
    if not rows:
        continue
    out += [f"## {title}", "", "| 程式 | 用途 | 結果 |", "|---|---|---|"]
    for f in rows:
        used.add(f)
        stem = f.rsplit(".", 1)[0]
        r = sorted(x for x in res if x.startswith(stem + ".") or x.startswith(stem + "_"))[:4]
        desc = first_doc(os.path.join(CODE, f)).replace("|", "／") or "—"
        out.append(f"| `{f}` | {desc} | {'、'.join(f'`{x}`' for x in r) or '—'} |")
    out.append("")
rest = [f for f in files if f not in used]
if rest:
    out += ["## 其他", "", "| 程式 | 用途 |", "|---|---|"]
    out += [f"| `{f}` | {(first_doc(os.path.join(CODE, f)).replace('|', '／') or '—')} |" for f in rest]
io.open(os.path.join(CODE, "INDEX.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"INDEX.md：{len(files)} 支程式，未分類 {len(rest)} 支")
