# -*- coding: utf-8 -*-
"""
PTT 留言細分情緒分類（8類）：多(看多) 空(看空) 貪(貪婪/興奮) 慌(恐慌) 追(FOMO/怕錯過) 絕(絕望/認賠) 諷(反串/嘲諷) 中(中立/無關)
============================================================
強制八選一：比較模型回答這 8 個字的機率（都是單一 token），另存各類機率（之後可算「恐慌程度」等連續指數）
驗證：python emotion_classify.py --eval [--model breeze|qwen]   → 對 emotion/gold_300.csv（Claude 人工標註）的準確率、各類 F1
套用：python emotion_classify.py --run IN.csv OUT.csv [--model ...]（IN 需有 content 欄）
few-shot 範例為自寫，不取自驗證集。
"""
import os
import sys
import copy
import time
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

LABELS = list("多空貪慌追絕諷中")
CODES = list("ABCDEFGH")  # 模型回答代號：Qwen 會把「貪」「諷」拆成兩個 token，改用單一 token 的 A~H（第一版因此 Qwen 永遠不會選這兩類）
MODELS = {"breeze": "MediaTek-Research/Breeze-7B-Instruct-v1_0", "qwen": "Qwen/Qwen2.5-7B-Instruct"}
SYSTEM_CODE = ("你是台股 PTT 股票板留言的情緒分析專家。把一則留言分成以下八類之一，只回答代號（一個英文字母）：\n"
          "A＝看多：預期會漲、建議買進（情緒平穩）\nB＝看空：預期會跌、建議賣出或放空\n"
          "C＝貪婪：興奮、炫耀賺錢、過度自信（如：噴爆、賺爛、無腦多）\nD＝恐慌：害怕、擔心崩盤（如：好可怕、要崩了嗎、快逃）\n"
          "E＝怕錯過（FOMO）：後悔沒買、急著想追（如：早知道就買、還能上車嗎）\nF＝絕望：套牢痛苦、認賠、放棄（如：再也不碰股票、只能吃土）\n"
          "G＝反串嘲諷：說反話、嘲諷別人（如：讚讚又破底、空軍還好嗎）\nH＝中立：資訊、提問、閒聊或與行情無關")
FEWSHOT_CODE = [("拉回就加碼，年底前看高一線", "A"), ("量縮價跌 月線下彎 先出場觀望", "B"), ("今天又賺一台車 無腦多就對了", "C"),
           ("盤中直接崩 好恐怖 會不會跌到萬五", "D"), ("早知道上週就該上車 現在追還來得及嗎", "E"),
           ("套在山頂三年 認賠殺出 這輩子不玩了", "F"), ("讚喔 融資戶又被洗出去了 真感人", "G"),
           ("今天成交量多少？", "H"), ("午餐吃什麼好", "H"), ("明天除息 記得看一下", "H")]
SYSTEM_CHAR = ("你是台股 PTT 股票板留言的情緒分析專家。把一則留言分成以下八類之一，只回答一個字：\n"
               "多＝看多、預期會漲、建議買進（情緒平穩）\n空＝看空、預期會跌、建議賣出或放空\n"
               "貪＝貪婪、興奮、炫耀賺錢、過度自信（如：噴爆、賺爛、無腦多）\n慌＝恐慌、害怕、擔心崩盤（如：好可怕、要崩了嗎、快逃）\n"
               "追＝怕錯過（FOMO）、後悔沒買、急著想追（如：早知道就買、還能上車嗎）\n絕＝絕望、套牢痛苦、認賠、放棄（如：再也不碰股票、只能吃土）\n"
               "諷＝反串、說反話、嘲諷別人（如：讚讚又破底、空軍還好嗎）\n中＝中立、資訊、提問、閒聊或與行情無關")
FEWSHOT_CHAR = [(t, LABELS[CODES.index(a)]) for t, a in FEWSHOT_CODE]
# 驗證（emotion/gold_300.csv，Claude 人工標註）Macro-F1：Breeze 中文字 0.285 / 代號 0.192；Qwen 中文字 無效（貪、諷是兩個token）/ 代號 0.265
USE_CODES = {"breeze": False, "qwen": True}
SYSTEM, FEWSHOT, ANSWERS = None, None, None


def configure(name):
    global SYSTEM, FEWSHOT, ANSWERS
    if USE_CODES[name]:
        SYSTEM, FEWSHOT, ANSWERS = SYSTEM_CODE, FEWSHOT_CODE, CODES
    else:
        SYSTEM, FEWSHOT, ANSWERS = SYSTEM_CHAR, FEWSHOT_CHAR, LABELS


def load(name):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODELS[name], quantization_config=q, device_map="auto")
    m.eval()
    return tok, m


def prompt(tok, text):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, a in FEWSHOT:
        msgs += [{"role": "user", "content": t}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": text[:150]})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Scorer:
    def __init__(self, tok, model, batch=16):
        self.tok, self.model, self.batch = tok, model, batch
        v = tok.get_vocab()
        self.ids = []
        for w in ANSWERS:
            if w in v:  # 詞表裡有這個字就直接用（Breeze 單獨斷字會多一個「▁」token）
                self.ids.append(v[w])
                continue
            ids = tok(w, add_special_tokens=False).input_ids
            assert len(ids) == 1, (w, ids)  # 例：Qwen 的「貪」「諷」是兩個 token → 要用代號版
            self.ids.append(ids[0])
        self.prefix = None
        self.cache_map = {}

    @torch.no_grad()
    def probs(self, texts):
        ids = self.tok([prompt(self.tok, t) for t in texts]).input_ids
        if self.prefix is None:
            f, n = ids[0], len(ids[0])
            for s in ids[1:256]:
                i = 0
                while i < min(n, len(s)) and f[i] == s[i]:
                    i += 1
                n = i
            self.prefix = f[:min(n, min(len(s) for s in ids[:256]) - 1)]
            self.cache = self.model(input_ids=torch.tensor([self.prefix], device=self.model.device), use_cache=True).past_key_values
        P = len(self.prefix)
        out = [None] * len(ids)
        buckets = {}
        for k, s in enumerate(ids):
            buckets.setdefault(len(s) if s[:P] == self.prefix else -1, []).append(k)
        for L, idx in buckets.items():
            for st in range(0, len(idx), self.batch):
                part = idx[st:st + self.batch]
                if L == -1:  # 前綴不一致（極少）→ 單筆完整算
                    for k in part:
                        lg = self.model(input_ids=torch.tensor([ids[k]], device=self.model.device)).logits[0, -1].float()
                        out[k] = torch.softmax(lg[self.ids], -1).tolist()
                    continue
                suf = torch.tensor([ids[k][P:] for k in part], device=self.model.device)
                c = copy.deepcopy(self.cache)
                c.batch_repeat_interleave(len(part))
                lg = self.model(input_ids=suf, past_key_values=c, use_cache=True, logits_to_keep=1).logits[:, -1, :].float()
                pr = torch.softmax(lg[:, self.ids], -1).tolist()
                for k, p in zip(part, pr):
                    out[k] = p
        return np.array(out)


def evaluate(sc):
    g = pd.read_csv("emotion/gold_300.csv", dtype=str, keep_default_na=False)
    p = sc.probs(g.content.tolist())
    g["pred"] = [LABELS[i] for i in p.argmax(1)]
    acc = (g.pred == g.gold).mean()
    print(f"八類準確率 {acc:.1%}（全猜「中」的基準 {(g.gold == '中').mean():.1%}）")
    f1s = []
    for lab in LABELS:
        tp = ((g.pred == lab) & (g.gold == lab)).sum()
        pr = tp / max(1, (g.pred == lab).sum())
        rc = tp / max(1, (g.gold == lab).sum())
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        f1s.append(f1)
        print(f"  {lab}: 標準答案 {(g.gold == lab).sum():3d} 則、預測 {(g.pred == lab).sum():3d} 則、精確 {pr:.0%}、召回 {rc:.0%}、F1 {f1:.2f}")
    print(f"  Macro-F1 {np.mean(f1s):.3f}")
    # 粗分：方向（多+貪+追 vs 空+慌+絕）與「諷」
    grp = {"多": "多方", "貪": "多方", "追": "多方", "空": "空方", "慌": "空方", "絕": "空方", "諷": "諷", "中": "中"}
    print(f"  合併成 多方/空方/諷/中 四類的準確率 {(g.pred.map(grp) == g.gold.map(grp)).mean():.1%}")
    print(pd.crosstab(g.gold, g.pred).reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--run", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--model", default="breeze", choices=list(MODELS))
    a = ap.parse_args()
    configure(a.model)
    tok, m = load(a.model)
    sc = Scorer(tok, m)
    if a.eval:
        g = evaluate(sc)
        g.to_csv(f"emotion/gold_300_pred_{a.model}.csv", index=False, encoding="utf-8-sig")
    if a.run:
        df = pd.read_csv(a.run[0], dtype=str, keep_default_na=False)
        part = a.run[1] + ".partial.npy"
        done = np.load(part) if os.path.exists(part) else np.zeros((0, 8))
        t0 = time.time()
        texts = df.content.tolist()
        while len(done) < len(texts):
            s = len(done)
            done = np.vstack([done, sc.probs(texts[s:s + 10000])])
            np.save(part, done)
            r = (len(done) - s) / (time.time() - t0); t0 = time.time()
            print(f"  進度 {len(done):,}/{len(texts):,}，{r:.0f} 則/秒", flush=True)
        for i, lab in enumerate(LABELS):
            df[f"p_{lab}"] = done[:, i]
        df["emotion"] = [LABELS[i] for i in done.argmax(1)]
        df.to_csv(a.run[1], index=False, encoding="utf-8-sig")
        os.remove(part)
        print("完成 →", a.run[1], df.emotion.value_counts(normalize=True).round(3).to_dict(), flush=True)
