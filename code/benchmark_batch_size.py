# -*- coding: utf-8 -*-
"""測試不同batch size的吞吐量跟VRAM使用量，找出這張8.5GB顯卡撐得住的最快設定"""
import time
import glob
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from breeze_infer_267 import SYSTEM, FEWSHOT, build_prompt, classify_batch

MODEL = "MediaTek-Research/Breeze-7B-Instruct-v1_0"

print("=== 載入模型 ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
model.eval()

# 抓500則真實留言當測試樣本
files = glob.glob("ptt_output_267/*_comments.csv")
texts = []
for f in files[:5]:
    df = pd.read_csv(f)
    texts.extend(df["content"].fillna("").astype(str).tolist())
texts = texts[:500]
print(f"測試樣本數: {len(texts)}")

for bs in [16, 32, 48, 64, 96]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        _ = classify_batch(texts, tok, model, batch_size=bs)
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        rate = len(texts) / elapsed
        print(f"batch_size={bs:3d}  總耗時={elapsed:6.1f}s  速度={rate:6.1f}則/秒  峰值VRAM={peak_mem:.2f}GB")
    except torch.cuda.OutOfMemoryError:
        print(f"batch_size={bs:3d}  OOM（記憶體不足），停止測試更大的batch size")
        break
