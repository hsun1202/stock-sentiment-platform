# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, itertools
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

d = pd.read_csv("eval_constrained_v2_scores.csv")
LABELS = ["positive", "negative", "neutral"]
y = d.claude_label.values
def col(p, names): return np.stack([d[f"{p}_{n}"].values for n in names if f"{p}_{n}" in d], axis=1)

EN = {"positive": ["pos", "▁pos", "▁posit", "▁positive"], "negative": ["neg", "▁neg", "negative", "▁negative"],
      "neutral": ["neut", "▁neu", "▁neut", "▁neutr", "▁neutral"]}
def en(l): return logsumexp(col("a", EN[l]), axis=1)
A = np.stack([en(l) for l in LABELS], axis=1)                      # config A：只看英文標籤

kan = d["a_看"].values
pos_b = np.logaddexp.reduce([logsumexp(col("a", ["正", "正面", "看好"]), axis=1)] + [kan + d["b_多"].values, kan + d["b_好"].values] + [en("positive")], axis=0)
neg_b = np.logaddexp.reduce([logsumexp(col("a", ["負", "負面"]), axis=1)] + [kan + d["b_空"].values, kan + d["b_壞"].values] + [en("negative")], axis=0)
neu_b = np.logaddexp.reduce([logsumexp(col("a", ["中立", "中性"]), axis=1), en("neutral")], axis=0)
B = np.stack([pos_b, neg_b, neu_b], axis=1)                          # config B：加上中文答案，排除「看不懂」

def ev(S, b=(0, 0), name=""):
    pred = np.array(LABELS)[(S + np.array([b[0], b[1], 0])).argmax(1)]
    print(f"{name:34s} Acc {accuracy_score(y, pred)*100:5.1f}%  MacroF1 {f1_score(y, pred, average='macro', labels=LABELS):.3f}  分佈 {pd.Series(pred).value_counts().to_dict()}")
    return pred

print("現行流程(自由生成解析)         Acc  52.4%  MacroF1 0.375")
ev(A, name="A 只看英文標籤(即上次)")
pB = ev(B, name="B 加中文答案、排除看不懂")

grid = np.arange(-4, 4.01, 0.5)
def best_bias(S, idx):
    best = (-1, (0, 0))
    for b in itertools.product(grid, grid):
        pred = np.array(LABELS)[(S[idx] + np.array([b[0], b[1], 0])).argmax(1)]
        f = f1_score(y[idx], pred, average="macro", labels=LABELS)
        if f > best[0]: best = (f, b)
    return best
for name, S in (("A", A), ("B", B)):
    fs, preds = [], np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(S, y):
        f, b = best_bias(S, tr)
        preds[te] = np.array(LABELS)[(S[te] + np.array([b[0], b[1], 0])).argmax(1)]
    print(f"{name} + 偏差校正(5折交叉驗證)        Acc {accuracy_score(y, preds)*100:5.1f}%  MacroF1 {f1_score(y, preds, average='macro', labels=LABELS):.3f}")
    if name == "B":
        full = best_bias(B, np.arange(len(y))); print("   全資料最佳偏差(pos,neg相對neutral):", full[1], " (僅供參考，會略樂觀)")
        predB = preds
print("\nB+校正 混淆矩陣 (列=Claude, 欄=預測; pos,neg,neu):"); print(confusion_matrix(y, predB, labels=LABELS))
