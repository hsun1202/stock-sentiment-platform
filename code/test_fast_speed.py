# -*- coding: utf-8 -*-
import glob
import time
import pandas as pd

from breeze_infer_267_fast import FastClassifier, load_model

N = 400
frames = [pd.read_csv(f, usecols=["content"]) for f in glob.glob("ptt_output_267_stitched/*_comments.csv")]
texts = pd.concat(frames)["content"].dropna().astype(str).sample(N + 64, random_state=1).tolist()
warm, test = texts[:64], texts[64:]

print("載入模型...", flush=True)
tok, model = load_model()
for bs in (16, 32, 64):
    c = FastClassifier(tok, model, batch_size=bs)
    c.classify(warm)
    c.label_cache.clear()
    t = time.time()
    c.classify(test)
    dt = time.time() - t
    print(f"[batch={bs}] {N}則 {dt:.1f}秒 -> {N / dt:.1f}則/秒", flush=True)
