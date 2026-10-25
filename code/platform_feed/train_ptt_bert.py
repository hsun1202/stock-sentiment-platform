# -*- coding: utf-8 -*-
"""
平台每日情緒用的分類器：中文 BERT（ckiplab/bert-base-chinese）用全部 PTT 金標（5,194＋1,100 則）微調，context 版
（輸入「股票名｜標題｜推噓」＋留言），設定同 finetune_gold_cv.py。交叉驗證：一致率 69.2%、Kappa 0.44（Breeze 62.3%、0.37）。
存到 platform_feed/models/ptt_bert_context/。
cd D:/專題claude && python platform_feed/train_ptt_bert.py
"""
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import finetune_gold_cv as F  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ptt_bert_context")
MODEL = "ckiplab/bert-base-chinese"
os.chdir(os.path.dirname(OUT).rsplit(os.sep, 2)[0])          # 專題資料夾（讀 relabel/、claude_all_labels.csv）
s, old = F.load_gold()
tok = AutoTokenizer.from_pretrained(MODEL)
a = [F.ctx(n, t, p) for n, t, p in zip(s.stock_name, s.title, s.push_type)] + [""] * len(old)
b = s.content.tolist() + old.content.tolist()
y = [F.L2I[g] for g in s.gold.tolist() + old.gold.tolist()]
print(f"金標 {len(b):,} 則，開始訓練", flush=True)
model = F.train_model(MODEL, tok, a, b, y, epochs=4, lr=3e-5, bs=32, max_len=128, seed=0)
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
open(os.path.join(OUT, "labels.txt"), "w", encoding="utf-8").write("\n".join(F.L))
print("已存", OUT, flush=True)
