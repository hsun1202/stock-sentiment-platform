# -*- coding: utf-8 -*-
"""
用392筆人工標註資料，對Breeze-7B做QLoRA微調（情緒分類任務）
============================================================

背景：
    Breeze-7B在392筆人工標籤上zero-shot準確率約68%，Macro-F1約
    0.55，是目前情緒因子雜訊的主要來源之一（原論文用的CNN-BiLSTM
    模型準確率有90%以上，但那是英文金融文本；中文+PTT黑話+反諷
    用詞的任務難度高很多）。這支程式嘗試用同樣這392筆標籤（該有
    的5筆客觀標錯已經修正過）反過來微調Breeze-7B本身，看能不能
    把準確率再往上推。

方法：
    QLoRA（4-bit量化基座 + LoRA低秩adapter），只訓練adapter參數，
    8GB顯卡跑得動。訓練格式跟breeze_infer_267.py的推論格式完全
    一致（同一套system prompt、同一組few-shot範例），確保微調後
    的模型接的是同一個「輸入介面」，不會因為格式不一致而白訓練。

    Loss只算在「答案」那幾個token上（prompt部分的token標籤設為
    -100，不計入loss），這是標準的instruction-tuning作法，避免
    模型學著去生成system prompt或使用者輸入本身。

使用方式：
    python finetune_breeze_lora.py
    python finetune_breeze_lora.py --epochs 5 --lr 2e-4
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "breeze_infer_267",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "breeze_infer_267.py")
)
_bi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bi)
MODEL = _bi.MODEL
SYSTEM = _bi.SYSTEM
FEWSHOT = _bi.FEWSHOT


def build_messages(text):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex_text, ex_label in FEWSHOT:
        messages.append({"role": "user", "content": ex_text})
        messages.append({"role": "assistant", "content": ex_label})
    messages.append({"role": "user", "content": text[:200]})
    return messages


class SentimentSFTDataset(Dataset):
    """
    每一筆訓練資料 = (prompt部分token, 完整token含答案)。
    prompt部分的label設為-100（不算loss），只有答案token
    （例如"positive"）+ eos會被拿來算loss。
    """

    def __init__(self, df, tok, max_len=512):
        self.examples = []
        for _, row in df.iterrows():
            text, label = row["content"], row["human_label"]
            prompt_messages = build_messages(text)
            prompt_str = tok.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_str = prompt_str + label + tok.eos_token

            prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
            full_ids = tok(full_str, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[:max_len]

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            labels = labels[:len(full_ids)]

            self.examples.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA微調Breeze-7B做PTT情緒分類")
    parser.add_argument("--train-csv", type=str, default="breeze_finetune_data/train.csv")
    parser.add_argument("--out-dir", type=str, default="breeze_lora_adapter")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== 載入 Breeze-7B（4-bit量化，QLoRA微調基座）===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
    # 關掉gradient checkpointing：實測開著的話每個optimizer step要8-10分鐘
    # （反覆重算activation跟4-bit dequantize疊在一起效率很差），關掉後
    # 同樣的一步只要2秒左右，代價是峰值顯存變高，所以搭配batch_size=1。
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("=== 準備訓練資料 ===")
    train_df = pd.read_csv(args.train_csv)
    train_ds = SentimentSFTDataset(train_df, tok)
    print(f"訓練樣本數：{len(train_ds)}")

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, data_collator=collator,
    )

    print("=== 開始微調 ===")
    trainer.train()

    adapter_path = os.path.join(args.out_dir, "final")
    model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"\n[完成] LoRA adapter已存到 {adapter_path}")
