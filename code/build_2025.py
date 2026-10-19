# -*- coding: utf-8 -*-
"""
2025 年 CMoney：收盤前抽樣（cmoney_daily_2025/）＋熱門股開盤前補頁（cmoney_daily_pre_2025/）→ 清理 → 分類 → 股票日則數。
流程與 2020–2024 相同（cmoney_daily_build.py、cmoney_daily_build2.py、rebuild_cm3.py）：廣告/空白 → 清理 flag（與舊資料合併計算全體統計）
→ 分類（build2 最終模型的訓練設定：PTT 金標＋CMoney 金標、3 seeds 3000–3002；已分類過的內文沿用）→ 分到交易日（pre／intra）。
輸出：final_dataset_v3_cm3x/all_stocks_v3.csv（只有 stock_code、Date、ptt_* 欄；2015–2024＝cm3，2025＝本次），給 oos_2025.py 用。
python cmoney_full/build_2025.py
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


def read_dir(d):
    parts = []
    for f in glob.glob(f"{d}/*.jsonl"):
        if os.path.getsize(f) == 0:
            continue
        x = pd.read_json(f, lines=True, dtype={"id": str})
        x["stock_code"] = os.path.basename(f)[:-6]
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


a, b = read_dir("cmoney_daily_2025"), read_dir("cmoney_daily_pre_2025")
cm = pd.concat([a, b], ignore_index=True)
n0 = len(cm)
cm["id"] = cm.id.astype(str)
cm = cm.drop_duplicates(["stock_code", "id"])
say(f"2025：收盤前抽樣 {len(a):,} 篇＋開盤前補頁 {len(b):,} 篇 → 去重 {len(cm):,}（{cm.stock_code.nunique()} 檔）")
cm["content"] = (cm.title.fillna("").astype(str).str.strip() + " " + cm.text.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()).str.strip()
ad, empty = cm.content.map(is_ad), cm.content.str.len() == 0
cm = cm[~ad & ~empty].copy()
cm["content"] = cm.content.map(lambda z: "".join(ch if ch >= " " and ch != chr(127) else " " for ch in z)).str[:150]
say(f"廣告/機器人 {int(ad.sum()):,}、空白 {int(empty.sum()):,} → {len(cm):,} 篇")

cal = sorted(set(pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", usecols=["Date"]).Date))
CAL = pd.DatetimeIndex(pd.to_datetime([d for d in cal if "2024-12-01" <= d <= "2025-12-31"]))
close_dt = CAL.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
open_dt = CAL.values + np.timedelta64(9 * 3600, "s")
ct = (pd.to_datetime(cm.t, unit="ms") + pd.Timedelta(hours=8)).values
i = np.searchsorted(close_dt, ct, side="left")
ok = i < len(CAL)
cm, ct, i = cm[ok].copy(), ct[ok], i[ok]
cm["Date"] = CAL.strftime("%Y-%m-%d")[i]
cm["part"] = np.where(ct < open_dt[i], "pre", "intra")
cm = cm[cm.Date >= "2025-01-01"]
say(f"分到 2025 交易日 {len(cm):,} 篇；pre {(cm.part == 'pre').mean():.1%}、intra {(cm.part == 'intra').mean():.1%}（2020–2024 為 pre 39%／intra 61%，補頁前）")

old = pd.read_csv("cmoney_daily_labels.csv", dtype={"stock_code": str, "id": str}, escapechar="\\")
cols = ["stock_code", "id", "creator", "content"]
both = pd.concat([old[cols].assign(src="old"), cm[cols].assign(src="new")], ignore_index=True)
both["content"] = both.content.fillna("").astype(str)
both["flag"] = flag(both)
fl = both[both.src == "new"].flag.values
say(f"清理移除 {(fl != '').mean():.1%}（2020–2024 原本 11.4%）")
cm = cm[fl == ""].copy()

lab2 = pd.read_pickle("cmoney_daily_lab2.pkl")
known = dict(zip(lab2.content, lab2.sent2))
if os.path.exists("cmoney_full/cmoney_preopen_labels.csv"):
    pl = pd.read_csv("cmoney_full/cmoney_preopen_labels.csv", escapechar="\\", usecols=["content", "sent2"]).dropna()
    known.update(dict(zip(pl.content, pl.sent2)))
uniq = [c for c in cm.content.drop_duplicates() if c not in known]
say(f"需分類的內文 {len(uniq):,}（其餘 {cm.content.nunique() - len(uniq):,} 個已有分類）")
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
cm["sent2"] = cm.content.map(known)
say(f"2025 分布（正/負/中）{cm.sent2.value_counts(normalize=True).reindex(L).round(3).tolist()}（2020–2024：0.251/0.252/0.497）")
cm[["stock_code", "id", "Date", "part", "creator", "comments", "bb", "sent2", "content"]].to_csv(
    "cmoney_full/cmoney_2025_labels.csv", index=False, encoding="utf-8-sig", escapechar=chr(92))


def agg(x, prefix):
    a_ = x.assign(pos=(x.sent2 == "positive").astype(int), neg=(x.sent2 == "negative").astype(int)).groupby(["stock_code", "Date"]).agg(
        n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    return a_.rename(columns={"n": f"{prefix}_n", "pos": f"{prefix}_pos", "neg": f"{prefix}_neg"})


d25 = agg(cm[cm.part == "pre"], "ptt_pre").merge(agg(cm[cm.part == "intra"], "ptt_intra"), on=["stock_code", "Date"], how="outer")
SC = [f"ptt_{p}_{k}" for p in ("pre", "intra") for k in ("n", "pos", "neg")]
grid = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "in265"])
grid = grid[(grid.in265 == 1) & (grid.Date >= "2025-01-01")][["stock_code", "Date"]]
d25 = grid.merge(d25, on=["stock_code", "Date"], how="left")
d25[SC] = d25[SC].fillna(0).astype(int)
old3 = pd.read_csv("final_dataset_v3_cm3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date"] + SC)
out = pd.concat([old3, d25[["stock_code", "Date"] + SC]], ignore_index=True)
os.makedirs("final_dataset_v3_cm3x", exist_ok=True)
out.to_csv("final_dataset_v3_cm3x/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
x = d25.ptt_pre_n + d25.ptt_intra_n
say(f"final_dataset_v3_cm3x 完成：2025 股票日 {len(d25):,}、有貼文 {(x > 0).mean():.0%}；2024 同指標 "
    f"{((old3.Date >= '2024-01-01') & ((old3.ptt_pre_n + old3.ptt_intra_n) > 0)).sum() / (old3.Date >= '2024-01-01').sum():.0%}")
say(f"共 {time.time() - t0:.0f} 秒")
open("cmoney_full/build_2025.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
