# -*- coding: utf-8 -*-
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from breeze_infer_267 import build_prompt, parse_label
from breeze_infer_267_fast import load_model

df = pd.read_csv("claude_all_labels.csv")
texts = df["content"].astype(str).tolist()
tok, model = load_model()
tok.padding_side = "left"

replies = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        chunk = texts[i:i + 16]
        enc = tok([build_prompt(tok, t) for t in chunk], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            replies.append(tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // 16) % 10 == 0:
            print(f"進度 {i + len(chunk)}/{len(texts)}", flush=True)


def has_label(r):
    r = r.strip().lower()
    return any(k in r for k in ("positive", "negative", "neutral", "看多", "看好", "看空", "看壞", "中立", "中性", "正面", "負面"))


df["reply"] = replies
df["breeze"] = [parse_label(r) for r in replies]
df["reply_has_label"] = [has_label(r) for r in replies]
df.to_csv("audit_breeze_replies.csv", index=False, encoding="utf-8-sig")

n = len(df)
print(f"\n=== 有回出標籤字的比例：{df.reply_has_label.mean() * 100:.1f}%  (沒回標籤而被預設neutral：{(~df.reply_has_label).sum()}則) ===")
labs = ["positive", "negative", "neutral"]
print("Breeze輸出分佈:", df.breeze.value_counts().to_dict(), " Claude分佈:", df.claude_label.value_counts().to_dict())
print(f"整體 Accuracy {accuracy_score(df.claude_label, df.breeze) * 100:.1f}%  Macro-F1 {f1_score(df.claude_label, df.breeze, average='macro', labels=labs):.3f}")
sub = df[df.reply_has_label]
print(f"只看有回標籤的 {len(sub)}則：Accuracy {accuracy_score(sub.claude_label, sub.breeze) * 100:.1f}%  Macro-F1 {f1_score(sub.claude_label, sub.breeze, average='macro', labels=labs):.3f}")
sub2 = df[~df.reply_has_label]
print(f"沒回標籤(被預設neutral) {len(sub2)}則 的 Claude標籤分佈:", sub2.claude_label.value_counts().to_dict())
print("混淆矩陣 (列=Claude, 欄=Breeze; 順序 pos,neg,neu):")
print(confusion_matrix(df.claude_label, df.breeze, labels=labs))
print("\n沒回標籤的回覆範例:")
print(sub2[["content", "reply", "claude_label"]].head(12).to_string())
