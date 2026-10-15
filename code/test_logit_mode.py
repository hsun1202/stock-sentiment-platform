# -*- coding: utf-8 -*-
import glob
import time
import pandas as pd

from breeze_infer_267 import classify_batch
from breeze_infer_267_fast import FastClassifier, load_model

frames = [pd.read_csv(f, usecols=["content"]) for f in glob.glob("ptt_output_267_stitched/*_comments.csv")]
pool = pd.concat(frames)["content"].dropna().astype(str)
texts = pool.sample(864, random_state=0).tolist()
warm, test = texts[:64], texts[64:]          # 800則測試
hard = pd.read_csv(glob.glob("ptt_output_267_stitched/8454_*_comments.csv")[0])["content"].dropna().astype(str).tolist()[:800]

print("載入模型...", flush=True)
tok, model = load_model()


def run(mode, data):
    c = FastClassifier(tok, model, batch_size=32, mode=mode)
    c.classify(warm)
    c.label_cache.clear()
    t = time.time()
    labs = c.classify(data)
    dt = time.time() - t
    return labs, dt, c.stats


gen, t_gen, _ = run("generate", test)
print(f"[generate模式] 800則 {t_gen:.1f}秒 -> {800 / t_gen:.1f}則/秒", flush=True)
lg, t_lg, st = run("logit", test)
print(f"[logit模式]    800則 {t_lg:.1f}秒 -> {800 / t_lg:.1f}則/秒  退回生成 {st['logit_fallback']} 則", flush=True)
same = sum(a == b for a, b in zip(gen, lg))
print(f"logit vs generate 一致率：{same}/800 = {same / 8:.2f}%", flush=True)

old = classify_batch(test[:400], tok, model, batch_size=16)
s1 = sum(a == b for a, b in zip(old, lg[:400]))
s2 = sum(a == b for a, b in zip(old, gen[:400]))
print(f"vs 原始舊版(400則)：logit {s1}/400 = {s1 / 4:.2f}%   generate {s2}/400 = {s2 / 4:.2f}%", flush=True)

_, t_h, st = run("logit", hard)
print(f"[logit模式 長留言股票8454] 800則 {t_h:.1f}秒 -> {800 / t_h:.1f}則/秒", flush=True)
