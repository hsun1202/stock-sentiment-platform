# -*- coding: utf-8 -*-
"""
只針對「留言」(不含文章全文)驗證CMoney日期錯置對同日訊號的影響——留言短、
長度分佈接近PTT，用已驗證過的classify()方法(精確長度分桶)速度應該接近PTT水準，
不會像混進長文章全文那樣退化。

比較：
  old_date = 文章日期(breeze_infer_cmoney.py目前的錯誤做法)
  new_date = 留言自己的createTime換算日期(正確做法)
只在兩者不同的留言上，看修正前後同日情緒對隔日漲跌的預測力有沒有差異。

python cmoney_date_fix_comments_only.py
"""
import glob
import numpy as np
import pandas as pd

from breeze_infer_267_fast import FastClassifier, load_model

N_STOCKS = 10
SAMPLE_N = 30000
cov = pd.read_csv("cmoney_coverage_windows.csv").sort_values("n", ascending=False).head(N_STOCKS)
codes = cov.code.astype(str).tolist()
print("股票:", codes, flush=True)

art_date_of = {}
for code in codes:
    afiles = glob.glob(f"cmoney_267_clean/{code}_*.csv")
    if not afiles:
        continue
    art = pd.read_csv(afiles[0], dtype=str, usecols=["article_id", "createTime"])
    art["createTime"] = pd.to_numeric(art["createTime"])
    art["own_date"] = pd.to_datetime(art["createTime"], unit="ms").dt.strftime("%Y-%m-%d")
    art_date_of.update(dict(zip((code + "_" + art.article_id), art.own_date)))

cm = pd.read_csv("cmoney_267_comments/all_comments_clean.csv", dtype=str)
cm["createTime"] = pd.to_numeric(cm["createTime"])
cm["own_date"] = pd.to_datetime(cm["createTime"], unit="ms").dt.strftime("%Y-%m-%d")

# 把留言配對回股票代碼(透過article_id屬於哪支股票的文章清單)
aid_to_code = {}
for code in codes:
    afiles = glob.glob(f"cmoney_267_clean/{code}_*.csv")
    if not afiles:
        continue
    ids = pd.read_csv(afiles[0], dtype=str, usecols=["article_id"]).article_id
    for a in ids:
        aid_to_code[a] = code

cm["stock_code"] = cm.article_id.map(aid_to_code)
cm = cm.dropna(subset=["stock_code"])
cm["old_date"] = (cm.stock_code + "_" + cm.article_id).map(art_date_of)
cm = cm.dropna(subset=["old_date"])
cm["text"] = cm["text"].astype(str)
cm = cm[cm.text.str.strip() != ""]
print(f"留言總數(10檔): {len(cm)}  日期錯置比例: {(cm.old_date != cm.own_date).mean()*100:.1f}%", flush=True)

sample = cm.sample(min(SAMPLE_N, len(cm)), random_state=0)
print(f"抽樣 {len(sample)} 則進行分類", flush=True)

tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=32, mode="constrained")
texts = sample.text.str[:200].tolist()
labels = []
import time
t0 = time.time()
for i in range(0, len(texts), 2000):
    labels.extend(clf.classify(texts[i:i + 2000]))
    dt = time.time() - t0
    done = min(i + 2000, len(texts))
    print(f"  進度 {done}/{len(texts)}  ({done/dt:.1f}則/秒)", flush=True)
sample = sample.copy()
sample["label"] = labels
sample.to_csv("cmoney_comments_datefix_labels.csv", index=False, encoding="utf-8-sig")
print("完成分類，已存 cmoney_comments_datefix_labels.csv", flush=True)
