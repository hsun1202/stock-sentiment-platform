# -*- coding: utf-8 -*-
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from breeze_infer_267_fast import FastClassifier, load_model

df = pd.read_csv("claude_all_labels.csv")
texts = df["content"].astype(str).tolist()
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=32, mode="constrained")
clf.classify(texts[:64])
clf.label_cache.clear()
t = time.time()
pred = clf.classify(texts)
dt = time.time() - t
L = ["positive", "negative", "neutral"]
print(f"[正式版 constrained] {len(texts)}則 {dt:.1f}秒 -> {len(texts) / dt:.1f}則/秒")
print(f"Accuracy {accuracy_score(df.claude_label, pred) * 100:.1f}%  Macro-F1 {f1_score(df.claude_label, pred, average='macro', labels=L):.3f}")
print("預測分佈:", pd.Series(pred).value_counts().to_dict())
print(confusion_matrix(df.claude_label, pred, labels=L))
print("驗證完成")
