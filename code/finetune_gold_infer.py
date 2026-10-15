# -*- coding: utf-8 -*-
"""
用全部 Claude 金標（重標 5,194 則 + 舊金標 1,100 則）訓練最終的微調 BERT（多個 seed 平均），
重新分類 2015-2024 全部個股留言，並重建資料集的 PTT 欄位。
  輸入：sentiment_output_267_2015_2024/*_comment_labels.csv（原 Breeze 標籤）、ptt_2015_2024/articles.csv（標題）
  輸出：sentiment_output_267_ft/*_comment_labels.csv（text_sentiment＝微調模型；另留 breeze 欄）
        final_dataset_v3_ft/all_stocks_v3.csv（只有 ptt_* 欄位換成新標籤，其餘與 final_dataset_v3 完全相同）
        finetune_gold_infer.log
python finetune_gold_infer.py --variant context --seeds 3
"""
import argparse
import glob
import io
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from finetune_gold_cv import L, L2I, ctx, load_gold, predict_proba, train_model

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ckiplab/bert-base-chinese")
ap.add_argument("--tokenizer", default=None)
ap.add_argument("--variant", default="context")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out-dir", default="sentiment_output_267_ft")
ap.add_argument("--ds-dir", default="final_dataset_v3_ft")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.ds_dir, exist_ok=True)
t0 = time.time()

# ---------- 讀全部留言 ----------
files = sorted(glob.glob("sentiment_output_267_2015_2024/*_comment_labels.csv"))
parts = []
for f in files:
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    d["_file"] = os.path.basename(f)
    parts.append(d)
cm = pd.concat(parts, ignore_index=True)
arts = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["article_id", "title"], dtype=str, keep_default_na=False).drop_duplicates("article_id")
cm = cm.merge(arts, on="article_id", how="left")
cm["title"] = cm.title.fillna("")
say(f"留言 {len(cm):,} 則（{len(files)} 檔股票）；找不到標題 {(cm.title == '').mean():.2%}")
if args.variant == "context":
    cm["_a"] = [ctx(n, t, p) for n, t, p in zip(cm.stock_name, cm.title, cm.push_type)]
    max_len = 128
else:
    cm["_a"] = ""
    max_len = 64
uniq = cm[["_a", "content"]].drop_duplicates().reset_index(drop=True)
say(f"不重複的（上下文, 留言）{len(uniq):,} 組")

# ---------- 訓練最終模型並推論 ----------
s, old = load_gold()
tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
tr_b = s.content.tolist() + old.content.tolist()
y = [L2I[x] for x in s.gold] + [L2I[x] for x in old.gold]
tr_a = None
if args.variant == "context":
    tr_a = [ctx(n, t, p) for n, t, p in zip(s.stock_name, s.title, s.push_type)] + [""] * len(old)
ua = uniq._a.tolist() if args.variant == "context" else None
ub = uniq.content.tolist()
P = np.zeros((len(uniq), 3), dtype=np.float32)
for seed in range(args.seeds):
    model = train_model(args.model, tok, tr_a, tr_b, y, args.epochs, args.lr, 32, max_len, 1000 + seed)
    P += predict_proba(model, tok, ua, ub, max_len, bs=128)
    np.save(os.path.join(args.out_dir, f"_proba_sum_after_seed{seed}.npy"), P)
    del model
    torch.cuda.empty_cache()
    say(f"seed {seed} 完成（累計 {time.time() - t0:.0f} 秒）")
P /= args.seeds
uniq["ft"] = [L[i] for i in P.argmax(1)]
uniq["ft_conf"] = P.max(1).round(3)
cm = cm.merge(uniq, on=["_a", "content"], how="left")
assert cm.ft.notna().all()
cm = cm.rename(columns={"text_sentiment": "breeze"}).rename(columns={"ft": "text_sentiment"})
say(f"分布（正/負/中）：微調 {cm.text_sentiment.value_counts(normalize=True).reindex(L).round(3).tolist()}｜Breeze {cm.breeze.value_counts(normalize=True).reindex(L).round(3).tolist()}")
say(f"與 Breeze 一致率 {(cm.text_sentiment == cm.breeze).mean():.1%}")
chk = s.merge(cm[["stock_code", "article_id", "content", "push_type", "text_sentiment"]].drop_duplicates(["stock_code", "article_id", "content", "push_type"]),
              on=["stock_code", "article_id", "content", "push_type"], how="left")
say(f"（檢查）金標樣本上最終模型與金標一致率 {(chk.text_sentiment == chk.gold).mean():.1%}——這是訓練內的數字，不能當準確率，只確認流程接對")
cols = [c for c in cm.columns if c not in ("_file", "_a", "title")]
for f, d in cm.groupby("_file"):
    d[cols].to_csv(os.path.join(args.out_dir, f), index=False, encoding="utf-8-sig")

# ---------- 重建 PTT 欄位（邏輯與 build_dataset_v3.py 相同） ----------
cal1 = pd.to_datetime(pd.read_csv("chip_raw/trading_days_2014_2019.csv", dtype=str)["date"], format="%Y%m%d")
pcr = pd.concat([pd.read_csv(io.StringIO(open(f, encoding="utf-8").read()), index_col=False).iloc[:, :7]
                 for f in sorted(glob.glob("chip_raw/pcr/*.csv"))])
pcr.columns = ["Date", "put_vol", "call_vol", "pc_vol_ratio", "put_oi", "call_oi", "pc_oi_ratio"]
pcr["Date"] = pd.to_datetime(pcr.Date)
cal2 = pcr.drop_duplicates("Date").loc[lambda x: x.Date >= "2020-01-01", "Date"]
CAL = pd.DatetimeIndex(sorted(set(cal1) | set(cal2)))
CAL_S = CAL.strftime("%Y-%m-%d")
close_dt = CAL.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
open_dt = CAL.values + np.timedelta64(9 * 3600, "s")


def build_ptt(c, col):
    c = c[(c.comment_time != "") & (c.time_anomaly != "1")].copy()
    c["pos"] = (c[col] == "positive").astype(int)
    c["neg"] = (c[col] == "negative").astype(int)
    ct = pd.to_datetime(c.comment_time).values
    i = np.searchsorted(close_dt, ct, side="left")
    ok = i < len(CAL)
    c, ct, i = c[ok].copy(), ct[ok], i[ok]
    c["Date"] = CAL_S[i]
    c["part"] = np.where(ct < open_dt[i], "pre", "intra")
    j = np.searchsorted(CAL.values, pd.to_datetime(c.resolved_date).values, side="left")
    c["Date_art"] = np.where(j < len(CAL), CAL_S[np.minimum(j, len(CAL) - 1)], "")

    def agg(x, key, prefix):
        a = x.groupby(["stock_code", key]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
        return a.rename(columns={key: "Date", "n": f"{prefix}_n", "pos": f"{prefix}_pos", "neg": f"{prefix}_neg"})
    return agg(c[c.part == "pre"], "Date", "ptt_pre").merge(agg(c[c.part == "intra"], "Date", "ptt_intra"), on=["stock_code", "Date"], how="outer").merge(
        agg(c[c.Date_art != ""], "Date_art", "ptt_art"), on=["stock_code", "Date"], how="outer")


base = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
PTT_COLS = [c for c in base.columns if c.startswith("ptt_")]
# 先用 Breeze 標籤重算一次，確認與原資料集完全相同（流程驗證），再換成微調標籤
for col, check in [("breeze", True), ("text_sentiment", False)]:
    ptt = build_ptt(cm, col)
    new = base.drop(columns=PTT_COLS).merge(ptt, on=["stock_code", "Date"], how="left")
    new[PTT_COLS] = new[PTT_COLS].fillna(0).astype(int)
    new = new[base.columns]
    if check:
        same = all((new[c].values == base[c].fillna(0).astype(int).values).all() for c in PTT_COLS)
        say(f"（檢查）用 Breeze 標籤重算的 PTT 欄位與原資料集完全相同：{same}")
        assert same
new.to_csv(os.path.join(args.ds_dir, "all_stocks_v3.csv"), index=False, encoding="utf-8-sig")
for c in ("pre", "intra"):
    say(f"ptt_{c}：正 {int(base[f'ptt_{c}_pos'].sum()):,} → {int(new[f'ptt_{c}_pos'].sum()):,}｜負 {int(base[f'ptt_{c}_neg'].sum()):,} → {int(new[f'ptt_{c}_neg'].sum()):,}｜則數 {int(new[f'ptt_{c}_n'].sum()):,}")
say(f"完成，共 {time.time() - t0:.0f} 秒")
open("finetune_gold_infer.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
