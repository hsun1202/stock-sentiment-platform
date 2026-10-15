# -*- coding: utf-8 -*-
"""
快速驗證：CMoney的breeze_infer_cmoney.py把留言全部套用「文章日期」，忽略留言自己的
createTime，17.1%的留言其實是文章發布後1天以上才發的。這支腳本只挑留言量最大的幾檔
股票，用留言自己的真實日期重新分類，比較「修正日期後」跟「舊版(文章日期)」的
同日/落後情緒對隔日漲跌的AUC，看同日訊號是不是真的來自日期錯置。

python cmoney_date_fix_quicktest.py
"""
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from breeze_infer_267_fast import FastClassifier, load_model

N_STOCKS = 10
cov = pd.read_csv("cmoney_coverage_windows.csv").sort_values("n", ascending=False).head(N_STOCKS)
print("挑選股票:", cov.code.tolist(), flush=True)

# --- 組出「文章own-date」+「留言own-date」的完整文字清單 ---
rows = []
for code in cov.code.astype(str):
    afiles = glob.glob(f"cmoney_267_clean/{code}_*.csv")
    if not afiles:
        continue
    art = pd.read_csv(afiles[0], dtype=str)
    art["createTime"] = pd.to_numeric(art["createTime"])
    art["own_date"] = pd.to_datetime(art["createTime"], unit="ms").dt.strftime("%Y-%m-%d")
    art_date_of = dict(zip(art.article_id, art.own_date))
    for aid, txt, d in zip(art.article_id, art.text.fillna(""), art.own_date):
        if str(txt).strip():
            rows.append((code, aid, d, d, str(txt)[:200]))  # (stock, article_id, old_date=own_date, new_date=own_date, text) -- 文章本身沒有錯置問題
    ids = set(art.article_id)
    cm = pd.read_csv("cmoney_267_comments/all_comments_clean.csv", dtype=str)
    cm = cm[cm.article_id.isin(ids)]
    cm["createTime"] = pd.to_numeric(cm["createTime"])
    cm["own_date"] = pd.to_datetime(cm["createTime"], unit="ms").dt.strftime("%Y-%m-%d")
    cm["old_date"] = cm.article_id.map(art_date_of)
    for aid, txt, od, nd in zip(cm.article_id, cm.text.fillna(""), cm.old_date, cm.own_date):
        if str(txt).strip():
            rows.append((code, aid, od, nd, str(txt)[:200]))
    print(f"  {code}: 文章{len(art)}篇 留言{len(cm)}則", flush=True)

df = pd.DataFrame(rows, columns=["stock_code", "article_id", "old_date", "new_date", "text"])
print(f"\n總文字量: {len(df)}  日期錯置(new!=old)比例: {(df.old_date != df.new_date).mean()*100:.1f}%", flush=True)

print("載入模型分類...", flush=True)
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=32, mode="constrained")
texts = df.text.tolist()
labels = []
for i in range(0, len(texts), 4000):
    labels.extend(clf.classify(texts[i:i + 4000]))
    print(f"  進度 {min(i+4000,len(texts))}/{len(texts)}", flush=True)
df["label"] = labels
df.to_csv("cmoney_date_fix_labels.csv", index=False, encoding="utf-8-sig")


def daily_factor(d, date_col):
    g = d.groupby(["stock_code", date_col], as_index=False).agg(
        total=("label", "count"),
        pos=("label", lambda s: (s == "positive").sum()),
        neg=("label", lambda s: (s == "negative").sum()),
    )
    g["factor"] = ((g.pos - g.neg) / g.total).round(4)
    return g.rename(columns={date_col: "date"})


old = daily_factor(df, "old_date")   # 舊版：套用文章日期(有錯置)
new = daily_factor(df, "new_date")   # 新版：套用各自真實日期(修正後)
old.to_csv("cmoney_daily_old.csv", index=False)
new.to_csv("cmoney_daily_new.csv", index=False)
print("\n已存 cmoney_daily_old.csv / cmoney_daily_new.csv，完成分類部分", flush=True)
