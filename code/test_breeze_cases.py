# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "MediaTek-Research/Breeze-7B-Instruct-v1_0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
model.eval()

SYSTEM = (
    "你是一個判斷PTT股票板留言情緒的助手。判斷這則留言對「該檔股票的股價方向」"
    "表達的是看多(positive)、看空(negative)、還是沒有明顯傾向(neutral)。"
    "看多：認為會漲、看好後市（含反諷成語如「利空出盡」）。"
    "看空：認為會跌、看壞後市（含反諷成語如「利多出盡」）。"
    "中立：純聊天、問問題、無關閒聊、看不出方向、或語意不足以判斷。"
    "只回答一個詞：positive、negative、或neutral，不要其他任何文字。"
)

tests = [
    ("我覺得不看好這檔", "negative"),
    ("大家都不樂觀", "negative"),
    ("這檔我很看好", "positive"),
    ("跌停 塊陶", "negative"),
    ("有聽說電動車多花幾千鎂一櫃要趕出貨", "爭議"),
    ("上一波已經出貨了 ETF現在撿太扯了吧", "negative"),
    ("利多出盡", "negative"),
    ("利空出盡", "positive"),
    ("歐印", "positive"),
    ("這波應該要獲利了結了吧", "negative"),
    ("感覺會繼續噴出", "positive"),
    ("業績優於預期 但是股價卻不漲", "negative(轉折)"),
    ("外資一直倒貨 完全不想上車", "negative"),
    ("今天量縮 觀望中", "neutral"),
    ("明天記得開盤", "neutral"),
    ("有人知道除息日嗎", "neutral"),
]

out_path = "breeze_case_test_result.txt"
with open(out_path, "w", encoding="utf-8") as f:
    with torch.no_grad():
        for t, truth in tests:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": t},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
            reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            line = f"{t!r:35s} 應為={truth:14s} -> {reply}"
            print(line)
            f.write(line + "\n")
print(f"\n[done] written to {out_path}")
