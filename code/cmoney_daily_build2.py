# -*- coding: utf-8 -*-
"""
CMoney 第二版：先清理（cmoney_clean.flag）→ 用「PTT 金標 + CMoney 金標」訓練 → 重新分類 → 重建資料集
  CMoney 金標：cmoney_gold_all.csv（Claude 親自標 1,764 篇：160 個股票日 1,564 篇 + 隨機 200 篇）
  1) 交叉驗證（依股票日分組 5 折，只在「清理後保留」的 CMoney 金標上評估）：
       只用 PTT 金標訓練的舊模型（cmoney_daily_labels.csv 的 text_sentiment） vs 加入 CMoney 金標訓練的新模型（折外預測）
  2) 用全部金標訓練最終模型（3 seeds），重新分類清理後的文章
  3) 資料集：final_dataset_v3_cmc/ （清理後 × 舊模型）→ 只看「清理」的影響
             final_dataset_v3_cm2/ （清理後 × 新模型）→ 清理 + CMoney 自己的標註
             final_dataset_v3_cmbb2/（清理後 × 作者自標多空）
輸出：cmoney_daily_build2.log、cmoney_daily_labels2.csv、cmoney_gold_oof.csv
python cmoney_daily_build2.py
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

import finetune_gold_cv as F
from cmoney_clean import flag
from finetune_gold_cv import L, L2I, load_gold, predict_proba, report, train_model

LOG = F.LOG


def say(s=""):
    F.say(s)


t0 = time.time()
MODEL = "ckiplab/bert-base-chinese"
tok = AutoTokenizer.from_pretrained(MODEL)
cm = pd.read_csv("cmoney_daily_labels.csv", dtype={"stock_code": str, "id": str}, escapechar="\\")
cm["content"] = cm.content.fillna("").astype(str)
cm["flag"] = flag(cm)
say(f"CMoney 文章 {len(cm):,}；清理移除 {(cm.flag != '').mean():.1%}：" + "、".join(f"{k} {v:.2%}" for k, v in cm.flag[cm.flag != ""].value_counts(normalize=False).div(len(cm)).items()))
clean = cm[cm.flag == ""].copy()
say(f"保留 {len(clean):,} 篇、{clean.stock_code.nunique()} 檔")

s, old = load_gold()
ptt_b = s.content.tolist() + old.content.tolist()
ptt_y = [L2I[x] for x in s.gold] + [L2I[x] for x in old.gold]
G = pd.read_csv("cmoney_gold_all.csv", dtype={"stock_code": str, "id": str}, keep_default_na=False)
G = G[G.flag == ""].reset_index(drop=True)
G["grp"] = G.stock_code + "_" + G.Date
say(f"\nCMoney 金標（清理後保留）{len(G):,} 篇、{G.grp.nunique()} 個股票日；分布（正/負/中）{G.gold.value_counts(normalize=True).reindex(L).round(3).tolist()}")

# ---------- 1) 交叉驗證 ----------
say("\n在 CMoney 金標上的表現：")
G["old_model"] = G.text_sentiment
report(G, "old_model", "舊模型（只用PTT標註）")
P = np.zeros((len(G), 3), dtype=np.float32)
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(G, groups=G.grp)):
    model = train_model(MODEL, tok, None, ptt_b + G.content.iloc[tr].tolist(), ptt_y + [L2I[x] for x in G.gold.iloc[tr]], 4, 3e-5, 32, 64, 300 + k)
    P[te] = predict_proba(model, tok, None, G.content.iloc[te].tolist(), 128)
    del model
    torch.cuda.empty_cache()
G["new_model"] = [L[i] for i in P.argmax(1)]
report(G, "new_model", "新模型（PTT+CMoney，折外）")
bb = G[G.bb.astype(str).isin(["1", "-1", "1.0", "-1.0"])]
say(f"  作者自標（{len(bb)} 篇有標）：與金標同向 {int(((bb.bb.astype(float) == 1) & (bb.gold == 'positive') | (bb.bb.astype(float) == -1) & (bb.gold == 'negative')).sum())}、"
    f"相反 {int(((bb.bb.astype(float) == 1) & (bb.gold == 'negative') | (bb.bb.astype(float) == -1) & (bb.gold == 'positive')).sum())}、金標中立 {int((bb.gold == 'neutral').sum())}")
G.to_csv("cmoney_gold_oof.csv", index=False, encoding="utf-8-sig")

# ---------- 2) 最終模型 ----------
uniq = clean.content.drop_duplicates().tolist()
say(f"\n重新分類：不重複內文 {len(uniq):,}")
P = np.zeros((len(uniq), 3), dtype=np.float32)
for seed in range(3):
    model = train_model(MODEL, tok, None, ptt_b + G.content.tolist(), ptt_y + [L2I[x] for x in G.gold], 4, 3e-5, 32, 64, 3000 + seed)
    P += predict_proba(model, tok, None, uniq, 128, bs=128)
    del model
    torch.cuda.empty_cache()
    say(f"seed {seed} 完成（累計 {time.time() - t0:.0f} 秒）")
P /= 3
lab = pd.DataFrame({"content": uniq, "sent2": [L[i] for i in P.argmax(1)], "conf2": P.max(1).round(3)})
lab.to_pickle("cmoney_daily_lab2.pkl")
clean = clean.merge(lab, on="content", how="left")
say(f"分布（正/負/中）：新模型 {clean.sent2.value_counts(normalize=True).reindex(L).round(3).tolist()}｜舊模型 {clean.text_sentiment.value_counts(normalize=True).reindex(L).round(3).tolist()}｜新舊一致 {(clean.sent2 == clean.text_sentiment).mean():.1%}")
clean.to_csv("cmoney_daily_labels2.csv", index=False, encoding="utf-8-sig", escapechar=chr(92))

# ---------- 3) 資料集 ----------
base = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
PTT = [c for c in base.columns if c.startswith("ptt_")]


def agg(x, prefix, pos, neg):
    a = x.assign(pos=pos.astype(int), neg=neg.astype(int)).groupby(["stock_code", "Date"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    return a.rename(columns={"n": f"{prefix}_n", "pos": f"{prefix}_pos", "neg": f"{prefix}_neg"})


def make(tag, pos, neg):
    pre, intra = clean.part == "pre", clean.part == "intra"
    ptt = agg(clean[pre], "ptt_pre", pos[pre], neg[pre]).merge(agg(clean[intra], "ptt_intra", pos[intra], neg[intra]), on=["stock_code", "Date"], how="outer").merge(
        agg(clean, "ptt_art", pos, neg), on=["stock_code", "Date"], how="outer")
    new = base.drop(columns=PTT).merge(ptt, on=["stock_code", "Date"], how="left")
    assert len(new) == len(base)
    new[PTT] = new[PTT].fillna(0).astype(int)
    os.makedirs(f"final_dataset_v3_{tag}", exist_ok=True)
    new[base.columns].to_csv(f"final_dataset_v3_{tag}/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
    ex = (new.ptt_pre_pos + new.ptt_pre_neg + new.ptt_intra_pos + new.ptt_intra_neg) >= 1
    say(f"final_dataset_v3_{tag}：有表態的股票日 {int(ex.sum()):,}")


make("cmc", clean.text_sentiment == "positive", clean.text_sentiment == "negative")
make("cm2", clean.sent2 == "positive", clean.sent2 == "negative")
make("cmbb2", clean.bb == 1, clean.bb == -1)
say(f"完成，共 {time.time() - t0:.0f} 秒")
open("cmoney_daily_build2.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
