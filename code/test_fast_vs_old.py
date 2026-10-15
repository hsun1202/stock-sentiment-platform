# -*- coding: utf-8 -*-
import glob
import time
import pandas as pd

from breeze_infer_267 import classify_batch
from breeze_infer_267_fast import FastClassifier, load_model

N_AGREE, N_SPEED = 800, 2400

frames = [pd.read_csv(f, usecols=["content"]) for f in glob.glob("ptt_output_267_stitched/*_comments.csv")]
all_texts = pd.concat(frames)["content"].dropna().astype(str)
sample = all_texts.sample(N_AGREE + N_SPEED, random_state=0).tolist()
agree_texts, speed_texts = sample[:N_AGREE], sample[N_AGREE:]

print("載入模型...", flush=True)
tok, model = load_model()

t = time.time()
old = classify_batch(agree_texts, tok, model, batch_size=16)
t_old = time.time() - t
print(f"[舊版] {N_AGREE}則 {t_old:.1f}秒 -> {N_AGREE / t_old:.1f}則/秒", flush=True)

clf = FastClassifier(tok, model, batch_size=32)
t = time.time()
new = clf.classify(agree_texts)
t_new = time.time() - t
print(f"[新版 含去重] {N_AGREE}則 {t_new:.1f}秒 -> {N_AGREE / t_new:.1f}則/秒 (含建立KV cache)", flush=True)

same = sum(a == b for a, b in zip(old, new))
print(f"\n=== 標籤一致率：{same}/{N_AGREE} = {same / N_AGREE * 100:.2f}% ===")
diff = [(agree_texts[i][:40], old[i], new[i]) for i in range(N_AGREE) if old[i] != new[i]]
for d in diff[:15]:
    print("  不一致:", d)
print("舊版分佈:", pd.Series(old).value_counts().to_dict())
print("新版分佈:", pd.Series(new).value_counts().to_dict(), flush=True)

for bs in (16, 32, 64):
    c = FastClassifier(tok, model, batch_size=bs)
    c.classify(agree_texts[:64])  # 暖機，建立prefix cache
    c.label_cache.clear()
    t = time.time()
    c.classify(speed_texts)
    dt = time.time() - t
    uniq = c.stats["unique_classified"] - 64 if c.stats["unique_classified"] > 64 else c.stats["unique_classified"]
    print(f"[新版 batch={bs}] {N_SPEED}則(去重前) 實際分類約{len(set(t[:200] for t in speed_texts))}則 {dt:.1f}秒", flush=True)
