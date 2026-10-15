# -*- coding: utf-8 -*-
import torch
import pandas as pd

from breeze_infer_267 import build_prompt
from breeze_infer_267_fast import load_model

df = pd.read_csv("claude_all_labels.csv")
texts = df["content"].astype(str).tolist()
tok, model = load_model()
tok.padding_side = "left"
unk = tok.unk_token_id
vocab = tok.get_vocab()

KAN = vocab["看"]
first_names = set()
for w in ("positive", "negative", "neutral"):
    for k in range(3, len(w) + 1):
        for s in (w[:k], "▁" + w[:k]):
            if s in vocab:
                first_names.add(s)
first_names |= {s for s in ("正", "負", "正面", "負面", "中立", "中性", "看", "無法", "不", "多", "空", "看多", "看空", "看好", "看壞", "偏多", "偏空") if s in vocab}
second_names = [s for s in ("多", "好", "空", "壞", "不", "漲", "跌", "衰", "強", "弱", "偏", "中", "正", "負", "無", "懂") if s in vocab]
print("第一步候選:", sorted(first_names), "\n第二步候選(接在『看』後):", second_names, flush=True)

f_ids = [vocab[s] for s in sorted(first_names)]
s_ids = [vocab[s] for s in second_names]
rows = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        chunk = texts[i:i + 16]
        enc = tok([build_prompt(tok, t) for t in chunk], return_tensors="pt", padding=True).to(model.device)
        ids = torch.cat([enc["input_ids"], torch.full((len(chunk), 1), KAN, device=model.device)], dim=1)
        mask = torch.cat([enc["attention_mask"], torch.ones((len(chunk), 1), dtype=enc["attention_mask"].dtype, device=model.device)], dim=1)
        lg = model(input_ids=ids, attention_mask=mask, logits_to_keep=2).logits.float()
        lp1 = torch.log_softmax(lg[:, 0, :], dim=-1)   # 第一個答案token的分佈
        lp2 = torch.log_softmax(lg[:, 1, :], dim=-1)   # 已輸出「看」之後，第二個token的分佈
        t1, t2 = lp1.argmax(-1).tolist(), lp2.argmax(-1).tolist()
        for j in range(len(chunk)):
            r = {f"a_{tok.convert_ids_to_tokens(t)}": lp1[j, t].item() for t in f_ids}
            r.update({f"b_{tok.convert_ids_to_tokens(t)}": lp2[j, t].item() for t in s_ids})
            r["top1"] = tok.convert_ids_to_tokens(t1[j])
            r["top2_after_kan"] = tok.convert_ids_to_tokens(t2[j])
            rows.append(r)
        if (i // 16) % 10 == 0:
            print(f"進度 {i + len(chunk)}/{len(texts)}", flush=True)

out = pd.concat([df, pd.DataFrame(rows)], axis=1)
out.to_csv("eval_constrained_v2_scores.csv", index=False, encoding="utf-8-sig")
print("完成", out.shape, flush=True)
print("『看』之後模型最想接的字(前10):", out.top2_after_kan.value_counts().head(10).to_dict())
