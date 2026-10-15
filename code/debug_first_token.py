# -*- coding: utf-8 -*-
import torch
from breeze_infer_267 import build_prompt
from breeze_infer_267_fast import load_model

tok, model = load_model()
print("VRAM after load (GB):", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
texts = ["歐印", "丸子", "笑死", "今天天氣不錯", "利多出盡", "所以是做多？"]
tok.padding_side = "left"
enc = tok([build_prompt(tok, t) for t in texts], return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
for j, t in enumerate(texts):
    ids = out[j][enc["input_ids"].shape[1]:].tolist()
    print(repr(t), ids, [tok.convert_ids_to_tokens(i) for i in ids], flush=True)
for w in ["positive", "negative", "neutral", " positive", "▁positive", "▁negative", "▁neutral"]:
    print(repr(w), tok.convert_tokens_to_ids(w) if w.startswith("▁") else tok(w, add_special_tokens=False).input_ids)
print("peak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
