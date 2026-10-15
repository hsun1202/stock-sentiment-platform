# -*- coding: utf-8 -*-
"""
CMoney 同學會（cmoney_daily/*.jsonl，2020-2024 收盤前抽樣）→ 情緒分類 → 股票日資料集
  1) 去重、濾廣告/機器人貼文（沿用 clean_cmoney_ads.is_ad）
  2) 用 Claude 金標微調的 BERT（只看內文版，3 seeds 平均）分類；另保留作者自標多空 bb
  3) 依發文時間分到交易日：前一交易日收盤(13:30)後~當日開盤(09:00)前＝pre；09:00~13:30＝intra（與 build_dataset_v3 相同）
  4) 產生兩個對照用資料集（其餘欄位與 final_dataset_v3 完全相同，只換 ptt_* 欄位）：
       final_dataset_v3_cm/   ：ptt_* ＝ CMoney（2020 以前為 0）
       final_dataset_v3_p20/  ：ptt_* ＝ 原 PTT(Breeze)，但 2020 以前清成 0（同期間對照）
輸出：cmoney_daily_labels.csv、cmoney_daily_build.log
python cmoney_daily_build.py
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from clean_cmoney_ads import is_ad
from finetune_gold_cv import L, L2I, load_gold, predict_proba, train_model

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


t0 = time.time()
parts = []
for f in glob.glob("cmoney_daily/*.jsonl"):
    if os.path.getsize(f) == 0:
        continue
    d = pd.read_json(f, lines=True, dtype={"id": str})
    d["stock_code"] = os.path.basename(f)[:-6]
    parts.append(d)
cm = pd.concat(parts, ignore_index=True)
n0 = len(cm)
cm = cm.drop_duplicates(["stock_code", "id"])
cm["title"] = cm.title.fillna("").astype(str)
cm["text"] = cm.text.fillna("").astype(str)
cm["content"] = (cm.title.str.strip() + " " + cm.text.str.replace(r"\s+", " ", regex=True).str.strip()).str.strip()
ad = cm.content.map(is_ad)
empty = cm.content.str.len() == 0
say(f"文章 {n0:,} → 去重 {len(cm):,}；廣告/機器人 {ad.sum():,}（{ad.mean():.1%}）、空白 {empty.sum():,}")
cm = cm[~ad & ~empty].copy()
cm["content"] = cm.content.map(lambda z: "".join(ch if ch >= " " and ch != chr(127) else " " for ch in z)).str[:150]
say(f"可用 {len(cm):,} 篇、{cm.stock_code.nunique()} 檔；內文長度中位數 {cm.content.str.len().median():.0f} 字；作者自標 多 {(cm.bb == 1).mean():.1%}／空 {(cm.bb == -1).mean():.1%}")

# ---------- 分類 ----------
MODEL = "ckiplab/bert-base-chinese"
s, old = load_gold()
tok = AutoTokenizer.from_pretrained(MODEL)
tr_b = s.content.tolist() + old.content.tolist()
y = [L2I[x] for x in s.gold] + [L2I[x] for x in old.gold]
uniq = cm.content.drop_duplicates().tolist()
say(f"不重複內文 {len(uniq):,}")
if os.path.exists("cmoney_daily_lab.pkl"):
    lab = pd.read_pickle("cmoney_daily_lab.pkl")
    say("沿用已存的分類結果 cmoney_daily_lab.pkl")
else:
    P = np.zeros((len(uniq), 3), dtype=np.float32)
    for seed in range(3):
        model = train_model(MODEL, tok, None, tr_b, y, 4, 3e-5, 32, 64, 2000 + seed)
        P += predict_proba(model, tok, None, uniq, 128, bs=128)
        del model
        torch.cuda.empty_cache()
        say(f"seed {seed} 完成（累計 {time.time() - t0:.0f} 秒）")
    P /= 3
    lab = pd.DataFrame({"content": uniq, "text_sentiment": [L[i] for i in P.argmax(1)], "conf": P.max(1).round(3),
                        "p_pos": P[:, 0].round(4), "p_neg": P[:, 1].round(4)})
    lab.to_pickle("cmoney_daily_lab.pkl")
cm = cm.merge(lab, on="content", how="left")
say(f"分類分布（正/負/中）：{cm.text_sentiment.value_counts(normalize=True).reindex(L).round(3).tolist()}")
x = cm[cm.bb.isin([1, -1])]
say(f"與作者自標比較（{len(x):,} 篇有自標）：自標看多的 → 模型 正 {(x[x.bb == 1].text_sentiment == 'positive').mean():.0%}／負 {(x[x.bb == 1].text_sentiment == 'negative').mean():.0%}；"
    f"自標看空的 → 模型 正 {(x[x.bb == -1].text_sentiment == 'positive').mean():.0%}／負 {(x[x.bb == -1].text_sentiment == 'negative').mean():.0%}")

# ---------- 分到交易日 ----------
cal = pd.read_csv("market_sentiment/taiex_official.csv", usecols=["Date"]).Date
CAL = pd.DatetimeIndex(pd.to_datetime(cal[(cal >= "2019-12-01") & (cal <= "2024-12-31")]))
CAL_S = CAL.strftime("%Y-%m-%d")
close_dt = CAL.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
open_dt = CAL.values + np.timedelta64(9 * 3600, "s")
ct = (pd.to_datetime(cm.t, unit="ms") + pd.Timedelta(hours=8)).values
i = np.searchsorted(close_dt, ct, side="left")
ok = i < len(CAL)
cm, ct, i = cm[ok].copy(), ct[ok], i[ok]
cm["Date"] = CAL_S[i]
cm["part"] = np.where(ct < open_dt[i], "pre", "intra")
cm["post_time"] = pd.to_datetime(ct).strftime("%Y-%m-%dT%H:%M")
cm = cm[cm.Date >= "2020-01-01"]
say(f"分到交易日後 {len(cm):,} 篇；盤前（前一日收盤後~開盤前）{(cm.part == 'pre').mean():.1%}、盤中 {(cm.part == 'intra').mean():.1%}")
cm[["stock_code", "id", "Date", "part", "post_time", "creator", "comments", "bb", "text_sentiment", "conf", "content"]].to_csv(
    "cmoney_daily_labels.csv", index=False, encoding="utf-8-sig", escapechar=chr(92))

# ---------- 資料集 ----------
base = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
PTT = [c for c in base.columns if c.startswith("ptt_")]


def agg(x, prefix, pos, neg):
    a = x.assign(pos=pos.astype(int), neg=neg.astype(int)).groupby(["stock_code", "Date"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    return a.rename(columns={"n": f"{prefix}_n", "pos": f"{prefix}_pos", "neg": f"{prefix}_neg"})


def make(tag, pos, neg):
    pre, intra = cm.part == "pre", cm.part == "intra"
    ptt = agg(cm[pre], "ptt_pre", pos[pre], neg[pre]).merge(agg(cm[intra], "ptt_intra", pos[intra], neg[intra]), on=["stock_code", "Date"], how="outer").merge(
        agg(cm, "ptt_art", pos, neg), on=["stock_code", "Date"], how="outer")
    new = base.drop(columns=PTT).merge(ptt, on=["stock_code", "Date"], how="left")
    assert len(new) == len(base)
    new[PTT] = new[PTT].fillna(0).astype(int)
    os.makedirs(f"final_dataset_v3_{tag}", exist_ok=True)
    new[base.columns].to_csv(f"final_dataset_v3_{tag}/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
    ex = (new.ptt_pre_pos + new.ptt_pre_neg + new.ptt_intra_pos + new.ptt_intra_neg) >= 1
    say(f"final_dataset_v3_{tag}：有表態的股票日 {int(ex.sum()):,}（{new[ex].stock_code.nunique()} 檔）；有文章的股票日 {int(((new.ptt_pre_n + new.ptt_intra_n) > 0).sum()):,}")


make("cm", cm.text_sentiment == "positive", cm.text_sentiment == "negative")          # CMoney × 微調 BERT
make("cmbb", cm.bb == 1, cm.bb == -1)                                                  # CMoney × 作者自標多空
p20 = base.copy()
p20.loc[p20.Date < "2020-01-01", PTT] = 0
os.makedirs("final_dataset_v3_p20", exist_ok=True)
p20.to_csv("final_dataset_v3_p20/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
ex = (p20[["ptt_pre_pos", "ptt_pre_neg", "ptt_intra_pos", "ptt_intra_neg"]].fillna(0).sum(axis=1)) >= 1
say(f"final_dataset_v3_p20（PTT，只留 2020-2024）：有表態的股票日 {int(ex.sum()):,}")
say(f"完成，共 {time.time() - t0:.0f} 秒")
open("cmoney_daily_build.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
