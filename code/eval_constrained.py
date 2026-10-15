# -*- coding: utf-8 -*-
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from breeze_infer_267 import build_prompt
from breeze_infer_267_fast import load_model

LABELS = ["positive", "negative", "neutral"]
df = pd.read_csv("claude_all_labels.csv")
texts = df["content"].astype(str).tolist()
tok, model = load_model()
tok.padding_side = "left"

unk = tok.unk_token_id
groups = {}
for w in LABELS:
    ids = set()
    for s in (w, "▁" + w, w[:3], "▁" + w[:3]):
        i = tok.convert_tokens_to_ids(s)
        if i is not None and i != unk:
            ids.add(i)
    groups[w] = sorted(ids)
print("候選token:", {w: [(i, tok.convert_ids_to_tokens(i)) for i in ids] for w, ids in groups.items()}, flush=True)

rows = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        chunk = texts[i:i + 16]
        enc = tok([build_prompt(tok, t) for t in chunk], return_tensors="pt", padding=True).to(model.device)
        lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1, :].float(), dim=-1)
        top = lp.argmax(-1).tolist()
        for j in range(len(chunk)):
            sc = {w: torch.logsumexp(lp[j, groups[w]], dim=0).item() for w in LABELS}
            rows.append({**{f"s_{w}": sc[w] for w in LABELS}, "top_token": tok.convert_ids_to_tokens(top[j])})
        if (i // 16) % 10 == 0:
            print(f"進度 {i + len(chunk)}/{len(texts)}", flush=True)

sc = pd.DataFrame(rows)
out = pd.concat([df, sc], axis=1)
out["pred"] = out[[f"s_{w}" for w in LABELS]].idxmax(axis=1).str[2:]
out.to_csv("eval_constrained_scores.csv", index=False, encoding="utf-8-sig")

print(f"\n=== 強制三選一（無校正）：Accuracy {accuracy_score(out.claude_label, out.pred) * 100:.1f}%  "
      f"Macro-F1 {f1_score(out.claude_label, out.pred, average='macro', labels=LABELS):.3f} ===")
print("預測分佈:", out.pred.value_counts().to_dict(), " Claude分佈:", out.claude_label.value_counts().to_dict())
print("混淆矩陣 (列=Claude, 欄=預測; 順序 pos,neg,neu):")
print(confusion_matrix(out.claude_label, out.pred, labels=LABELS))
print("模型第一個想輸出的token(前8):", out.top_token.value_counts().head(8).to_dict())
