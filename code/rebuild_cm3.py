# -*- coding: utf-8 -*-
"""
把補抓的開盤前貼文（cmoney_daily_pre/）併進 CMoney 資料 → final_dataset_v3_cm3/（cm2 不動，舊結果可重現）。
流程同 cmoney_daily_build.py / cmoney_daily_build2.py：去重（與原本已抓的比對）→ 廣告/空白 → 清理 flag（與舊資料合併計算，
舊文章維持原本的保留與否）→ 分類（同 build2 的最終模型：PTT 金標＋CMoney 金標、3 seeds 3000–3002；只分類 cmoney_daily_lab2.pkl 沒有的內文）
→ 分到交易日（pre／intra，規則同 build）→ 資料集。
python cmoney_full/rebuild_cm3.py
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from clean_cmoney_ads import is_ad  # noqa: E402
from cmoney_clean import flag  # noqa: E402
from finetune_gold_cv import L, L2I, load_gold, predict_proba, train_model  # noqa: E402

LOG, t0 = [], time.time()


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


old = pd.read_csv("cmoney_daily_labels.csv", dtype={"stock_code": str, "id": str}, escapechar="\\")       # 原本全部（清理前）
clean = pd.read_csv("cmoney_daily_labels2.csv", dtype={"stock_code": str, "id": str}, escapechar="\\")    # 原本清理後＋sent2
parts = []
for f in glob.glob("cmoney_daily_pre/*.jsonl"):
    if os.path.getsize(f) == 0:
        continue
    d = pd.read_json(f, lines=True, dtype={"id": str})
    d["stock_code"] = os.path.basename(f)[:-6]
    parts.append(d)
new = pd.concat(parts, ignore_index=True).drop_duplicates(["stock_code", "id"])
n0 = len(new)
raw_ids = set()
for f in glob.glob("cmoney_daily/*.jsonl"):
    if os.path.getsize(f) == 0:                 # 沒有貼文的股票是空檔
        continue
    code = os.path.basename(f)[:-6]
    raw_ids.update(code + "_" + pd.read_json(f, lines=True, dtype={"id": str}).id.astype(str))
new = new[~(new.stock_code + "_" + new.id.astype(str)).isin(raw_ids)].copy()
say(f"補抓 {n0:,} 篇 → 扣掉原本已抓過的 {len(new):,} 篇（{new.stock_code.nunique()} 檔）")
new["content"] = (new.title.fillna("").astype(str).str.strip() + " " + new.text.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()).str.strip()
ad, empty = new.content.map(is_ad), new.content.str.len() == 0
new = new[~ad & ~empty].copy()
new["content"] = new.content.map(lambda z: "".join(ch if ch >= " " and ch != chr(127) else " " for ch in z)).str[:150]
say(f"廣告/機器人 {int(ad.sum()):,}、空白 {int(empty.sum()):,} → {len(new):,} 篇")

# 分到交易日（同 cmoney_daily_build.py）
cal = pd.read_csv("market_sentiment/taiex_official.csv", usecols=["Date"]).Date
CAL = pd.DatetimeIndex(pd.to_datetime(cal[(cal >= "2019-12-01") & (cal <= "2024-12-31")]))
close_dt = CAL.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
open_dt = CAL.values + np.timedelta64(9 * 3600, "s")
ct = (pd.to_datetime(new.t, unit="ms") + pd.Timedelta(hours=8)).values
i = np.searchsorted(close_dt, ct, side="left")
ok = i < len(CAL)
new, ct, i = new[ok].copy(), ct[ok], i[ok]
new["Date"] = CAL.strftime("%Y-%m-%d")[i]
new["part"] = np.where(ct < open_dt[i], "pre", "intra")
new["post_time"] = pd.to_datetime(ct).strftime("%Y-%m-%dT%H:%M")
new = new[new.Date >= "2020-01-01"]
say(f"分到交易日 {len(new):,} 篇；pre {(new.part == 'pre').mean():.1%}、intra {(new.part == 'intra').mean():.1%}（補的是開盤前，應幾乎全是 pre）")

# 清理：與舊資料一起算 flag（重複內容、壞帳號是全體統計），只套用到新文章
cols = ["stock_code", "id", "creator", "content"]
both = pd.concat([old[cols].assign(src="old"), new[cols].assign(src="new")], ignore_index=True)
both["content"] = both.content.fillna("").astype(str)
both["flag"] = flag(both)
fl = both[both.src == "new"].flag.values
say(f"新文章清理移除 {(fl != '').mean():.1%}")
new = new[fl == ""].copy()

# 分類
lab2 = pd.read_pickle("cmoney_daily_lab2.pkl")
known = dict(zip(lab2.content, lab2.sent2))
uniq = [c for c in new.content.drop_duplicates() if c not in known]
say(f"需分類的新內文 {len(uniq):,}（其餘 {new.content.nunique() - len(uniq):,} 個已有分類）")
if uniq:
    MODEL = "ckiplab/bert-base-chinese"
    tok = AutoTokenizer.from_pretrained(MODEL)
    s, oldg = load_gold()
    ptt_b = s.content.tolist() + oldg.content.tolist()
    ptt_y = [L2I[x] for x in s.gold] + [L2I[x] for x in oldg.gold]
    G = pd.read_csv("cmoney_gold_all.csv", dtype={"stock_code": str, "id": str}, keep_default_na=False)
    G = G[G.flag == ""].reset_index(drop=True)
    P = np.zeros((len(uniq), 3), dtype=np.float32)
    for seed in range(3):
        model = train_model(MODEL, tok, None, ptt_b + G.content.tolist(), ptt_y + [L2I[x] for x in G.gold], 4, 3e-5, 32, 64, 3000 + seed)
        P += predict_proba(model, tok, None, uniq, 128, bs=128)
        del model
        torch.cuda.empty_cache()
        say(f"seed {seed} 完成（累計 {time.time() - t0:.0f} 秒）")
    known.update(dict(zip(uniq, [L[k] for k in P.argmax(1)])))
new["sent2"] = new.content.map(known)
say(f"新文章分布（正/負/中）{new.sent2.value_counts(normalize=True).reindex(L).round(3).tolist()}｜原本 {clean.sent2.value_counts(normalize=True).reindex(L).round(3).tolist()}")
new[["stock_code", "id", "Date", "part", "post_time", "creator", "comments", "bb", "sent2", "content"]].to_csv(
    "cmoney_full/cmoney_preopen_labels.csv", index=False, encoding="utf-8-sig", escapechar=chr(92))

# 資料集
allc = pd.concat([clean[["stock_code", "Date", "part", "sent2"]], new[["stock_code", "Date", "part", "sent2"]]], ignore_index=True)
base = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
PTT = [c for c in base.columns if c.startswith("ptt_")]


def agg(x, prefix):
    a = x.assign(pos=(x.sent2 == "positive").astype(int), neg=(x.sent2 == "negative").astype(int)).groupby(["stock_code", "Date"]).agg(
        n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    return a.rename(columns={"n": f"{prefix}_n", "pos": f"{prefix}_pos", "neg": f"{prefix}_neg"})


ptt = agg(allc[allc.part == "pre"], "ptt_pre").merge(agg(allc[allc.part == "intra"], "ptt_intra"), on=["stock_code", "Date"], how="outer").merge(
    agg(allc, "ptt_art"), on=["stock_code", "Date"], how="outer")
out = base.drop(columns=PTT).merge(ptt, on=["stock_code", "Date"], how="left")
assert len(out) == len(base)
out[PTT] = out[PTT].fillna(0).astype(int)
os.makedirs("final_dataset_v3_cm3", exist_ok=True)
out[base.columns].to_csv("final_dataset_v3_cm3/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
cm2 = pd.read_csv("final_dataset_v3_cm2/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["ptt_pre_n"])
say(f"final_dataset_v3_cm3 完成：開盤前則數合計 {int(cm2.ptt_pre_n.sum()):,} → {int(out.ptt_pre_n.sum()):,}；有開盤前貼文的股票日 {int((cm2.ptt_pre_n > 0).sum()):,} → {int((out.ptt_pre_n > 0).sum()):,}")
say(f"共 {time.time() - t0:.0f} 秒")
open("cmoney_full/rebuild_cm3.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
