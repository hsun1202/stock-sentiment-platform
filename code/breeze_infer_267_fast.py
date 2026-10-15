# -*- coding: utf-8 -*-
"""
breeze_infer_267.py 的加速版，提示詞/模型/解析規則完全沿用，輸出格式與斷點續傳也相同。
加速方式（不改變輸入token，結果與舊版應幾乎一致）：
  1. 系統提示+few-shot（每則留言都相同的開頭）只跑一次forward，存成KV cache重複使用
  2. 依token長度分桶，同長度的留言一批，不需要padding
  3. 相同留言只分類一次（跨chunk、跨股票共用快取）

python breeze_infer_267_fast.py --batch-dir ptt_output_267_stitched --out-dir sentiment_output_267_stitched \
    --start-date 2022-01-01 --end-date 2024-12-31
"""
import os
import copy
import time
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache

from stock_list_267 import STOCK_LIST_267
from breeze_infer_267 import (MODEL, build_prompt, parse_label, aggregate_by_article,
                              write_progress, PROGRESS_FILE, classify_batch)


class FastClassifier:
    def __init__(self, tok, model, batch_size=32, max_new_tokens=6, mode="constrained"):
        self.tok, self.model = tok, model
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.mode = mode
        self.label_token = {}
        for w in ("positive", "negative", "neutral"):
            ids = tok(w, add_special_tokens=False).input_ids
            assert len(ids) == 1, f"{w} 不是單一token：{ids}"
            self.label_token[ids[0]] = w
        vocab = tok.get_vocab()

        def ids_of(names):
            return [vocab[n] for n in names if n in vocab]
        self.kan = vocab["看"]
        self.first = {
            "positive": ids_of(["pos", "▁pos", "▁posit", "▁positive", "正", "正面", "看好"]),
            "negative": ids_of(["neg", "▁neg", "negative", "▁negative", "負", "負面"]),
            "neutral": ids_of(["neut", "▁neu", "▁neut", "▁neutr", "▁neutral", "中立", "中性"]),
        }
        self.second = {"positive": ids_of(["多", "好"]), "negative": ids_of(["空", "壞"])}
        self.label_cache = {}
        self.prefix_ids = None
        self.prefix_cache = None
        self.stats = {"unique_classified": 0, "cache_hits": 0, "slow_path": 0, "logit_fallback": 0}

    def _init_prefix(self, id_lists):
        sample = id_lists[:256]
        first, n = sample[0], len(sample[0])
        for s in sample[1:]:
            i, m = 0, min(n, len(s))
            while i < m and first[i] == s[i]:
                i += 1
            n = i
        n = min(n, min(len(s) for s in sample) - 1)
        self.prefix_ids = first[:n]
        ids = torch.tensor([self.prefix_ids], device=self.model.device)
        with torch.no_grad():
            self.prefix_cache = self.model(input_ids=ids, use_cache=True).past_key_values

    @torch.no_grad()
    def _generate_group(self, id_lists):
        B, L = len(id_lists), len(id_lists[0])
        input_ids = torch.tensor(id_lists, device=self.model.device)
        cache = copy.deepcopy(self.prefix_cache)
        cache.batch_repeat_interleave(B)
        out = self.model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            past_key_values=cache, max_new_tokens=self.max_new_tokens,
            do_sample=False, pad_token_id=self.tok.eos_token_id,
        )
        return [parse_label(self.tok.decode(out[j][L:], skip_special_tokens=True)) for j in range(B)]

    @torch.no_grad()
    def _constrained_group(self, id_lists):
        P = len(self.prefix_ids)
        B = len(id_lists)
        suffix = torch.tensor([ids[P:] + [self.kan] for ids in id_lists], device=self.model.device)
        cache = copy.deepcopy(self.prefix_cache)
        cache.batch_repeat_interleave(B)
        lg = self.model(input_ids=suffix, past_key_values=cache, use_cache=True, logits_to_keep=2).logits.float()
        lp1 = torch.log_softmax(lg[:, 0, :], dim=-1)
        lp2 = torch.log_softmax(lg[:, 1, :], dim=-1)
        score = {}
        for lab in ("positive", "negative", "neutral"):
            s = torch.logsumexp(lp1[:, self.first[lab]], dim=-1)
            if lab in self.second:
                s = torch.logaddexp(s, lp1[:, self.kan] + torch.logsumexp(lp2[:, self.second[lab]], dim=-1))
            score[lab] = s
        order = ("positive", "negative", "neutral")
        pick = torch.stack([score[l] for l in order], dim=1).argmax(dim=1).tolist()
        return [order[k] for k in pick]

    @torch.no_grad()
    def _score_group(self, id_lists):
        P = len(self.prefix_ids)
        B = len(id_lists)
        suffix = torch.tensor([ids[P:] for ids in id_lists], device=self.model.device)
        cache = copy.deepcopy(self.prefix_cache)
        cache.batch_repeat_interleave(B)
        logits = self.model(input_ids=suffix, past_key_values=cache, use_cache=True,
                            logits_to_keep=1).logits[:, -1, :]
        top = logits.argmax(dim=-1).tolist()
        labels, redo = [None] * B, []
        for j, tid in enumerate(top):
            if tid in self.label_token:
                labels[j] = self.label_token[tid]
            else:
                redo.append(j)
        if redo:
            self.stats["logit_fallback"] += len(redo)
            for j, lab in zip(redo, self._generate_group([id_lists[j] for j in redo])):
                labels[j] = lab
        return labels

    @torch.no_grad()
    def _constrained_group_padded(self, id_lists):
        """跟_constrained_group邏輯相同，但允許同一批內長度不同：右邊補pad即可，
        causal attention下pad token不會影響真實token的計算，只要事後在各自真實
        長度的位置取值，不需要額外處理attention_mask/position_ids。"""
        P = len(self.prefix_ids)
        B = len(id_lists)
        suffixes = [ids[P:] + [self.kan] for ids in id_lists]
        lens = torch.tensor([len(s) for s in suffixes], device=self.model.device)
        maxlen = int(lens.max())
        pad_id = self.tok.pad_token_id
        padded = torch.full((B, maxlen), pad_id, dtype=torch.long, device=self.model.device)
        for i, s in enumerate(suffixes):
            padded[i, :len(s)] = torch.tensor(s, device=self.model.device)
        cache = copy.deepcopy(self.prefix_cache)
        cache.batch_repeat_interleave(B)
        logits = self.model(input_ids=padded, past_key_values=cache, use_cache=True).logits.float()
        idx = torch.arange(B, device=self.model.device)
        lp1 = torch.log_softmax(logits[idx, lens - 2, :], dim=-1)
        lp2 = torch.log_softmax(logits[idx, lens - 1, :], dim=-1)
        score = {}
        for lab in ("positive", "negative", "neutral"):
            s = torch.logsumexp(lp1[:, self.first[lab]], dim=-1)
            if lab in self.second:
                s = torch.logaddexp(s, lp1[:, self.kan] + torch.logsumexp(lp2[:, self.second[lab]], dim=-1))
            score[lab] = s
        order = ("positive", "negative", "neutral")
        pick = torch.stack([score[l] for l in order], dim=1).argmax(dim=1).tolist()
        return [order[k] for k in pick]

    def classify_sorted(self, texts):
        """跟classify()相同的去重+快取邏輯，但長度分批用「排序後固定batch_size切段+padding」，
        取代「精確長度分桶」——留言/文章長度差異很大的語料(如CMoney)用這個才不會退化成
        一堆極小批次。結果應與classify()在數學上等價(labels相同)，只是分批方式不同。"""
        keys = [t[:200] for t in texts]
        todo = [k for k in dict.fromkeys(keys) if k not in self.label_cache]
        self.stats["cache_hits"] += len(keys) - len(todo)
        if todo:
            prompts = [build_prompt(self.tok, k) for k in todo]
            id_lists = self.tok(prompts).input_ids
            if self.prefix_cache is None:
                self._init_prefix(id_lists)
            P = len(self.prefix_ids)
            fast, slow = [], []
            for k, ids in zip(todo, id_lists):
                if len(ids) > P and ids[:P] == self.prefix_ids:
                    fast.append((k, ids))
                else:
                    slow.append(k)
            fast.sort(key=lambda kv: len(kv[1]))
            for s in range(0, len(fast), self.batch_size):
                part = fast[s:s + self.batch_size]
                labels = self._constrained_group_padded([ids for _, ids in part])
                for (k, _), lab in zip(part, labels):
                    self.label_cache[k] = lab
            if slow:
                self.stats["slow_path"] += len(slow)
                for k, lab in zip(slow, classify_batch(slow, self.tok, self.model, batch_size=16)):
                    self.label_cache[k] = lab
            self.stats["unique_classified"] += len(todo)
        return [self.label_cache[k] for k in keys]

    def classify(self, texts):
        keys = [t[:200] for t in texts]
        todo = [k for k in dict.fromkeys(keys) if k not in self.label_cache]
        self.stats["cache_hits"] += len(keys) - len(todo)
        if todo:
            prompts = [build_prompt(self.tok, k) for k in todo]
            id_lists = self.tok(prompts).input_ids
            if self.prefix_cache is None:
                self._init_prefix(id_lists)
            P = len(self.prefix_ids)
            fast, slow = {}, []
            for k, ids in zip(todo, id_lists):
                if len(ids) > P and ids[:P] == self.prefix_ids:
                    fast.setdefault(len(ids), []).append((k, ids))
                else:
                    slow.append(k)
            for L, items in fast.items():
                for s in range(0, len(items), self.batch_size):
                    part = items[s:s + self.batch_size]
                    group = [ids for _, ids in part]
                    if self.mode == "constrained":
                        labels = self._constrained_group(group)
                    elif self.mode == "logit":
                        labels = self._score_group(group)
                    else:
                        labels = self._generate_group(group)
                    for (k, _), lab in zip(part, labels):
                        self.label_cache[k] = lab
            if slow:
                self.stats["slow_path"] += len(slow)
                for k, lab in zip(slow, classify_batch(slow, self.tok, self.model, batch_size=16)):
                    self.label_cache[k] = lab
            self.stats["unique_classified"] += len(todo)
        return [self.label_cache[k] for k in keys]


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
    model.eval()
    return tok, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breeze-7B加速版情緒推論（含斷點續傳）")
    parser.add_argument("--batch-dir", type=str, default="ptt_output_267_stitched")
    parser.add_argument("--out-dir", type=str, default="sentiment_output_267_stitched")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="constrained", choices=["constrained", "generate"])
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--only-codes", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    STOCK_LIST = STOCK_LIST_267
    if args.only_codes:
        wanted = set(args.only_codes.split(","))
        STOCK_LIST = [s for s in STOCK_LIST if s[0] in wanted]

    print("=== 載入 Breeze-7B（4-bit量化）===")
    tok, model = load_model()
    clf = FastClassifier(tok, model, batch_size=args.batch_size, mode=args.mode)
    print("=== 模型載入完成，開始批次處理 ===\n")
    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
        comments_path = os.path.join(args.batch_dir, f"{code}_{name}_comments.csv")
        out_path = os.path.join(args.out_dir, f"{code}_{name}_sentiment.csv")
        partial_path = os.path.join(args.out_dir, f"{code}_{name}.partial.csv")

        if os.path.exists(out_path):
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）已存在，跳過")
            continue
        if not os.path.exists(comments_path):
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：找不到留言檔案，跳過")
            continue

        df = pd.read_csv(comments_path)
        date_col = "resolved_date"
        if args.start_date or args.end_date:
            dates = pd.to_datetime(df[date_col])
            mask = pd.Series(True, index=df.index)
            if args.start_date:
                mask &= dates >= pd.Timestamp(args.start_date)
            if args.end_date:
                mask &= dates <= pd.Timestamp(args.end_date)
            df = df[mask].reset_index(drop=True)
        if len(df) == 0:
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：篩選日期範圍後沒有留言，跳過")
            continue
        texts = df["content"].fillna("").astype(str).tolist()

        start_idx, labels = 0, []
        if os.path.exists(partial_path):
            labels = pd.read_csv(partial_path)["text_sentiment"].tolist()
            start_idx = len(labels)
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：接續上次進度，已完成 {start_idx}/{len(texts)}")
        else:
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）：{len(texts)} 則留言，開始分類...")

        while start_idx < len(texts):
            end_idx = min(start_idx + args.chunk_size, len(texts))
            labels.extend(clf.classify(texts[start_idx:end_idx]))
            pd.DataFrame({"text_sentiment": labels}).to_csv(partial_path, index=False)
            start_idx = end_idx
            write_progress(args.out_dir, i, len(STOCK_LIST), code, name, start_idx, len(texts), run_started_at)
            print(f"    進度 {start_idx}/{len(texts)}  (快取命中 {clf.stats['cache_hits']}, 實際分類 {clf.stats['unique_classified']})")

        df["text_sentiment"] = labels
        if "comment_time" in df.columns:
            # 2015-2024版輸入有每則留言自己的時間 → 另存逐則標籤，之後可依留言時間做無洩漏的落後分析
            df.to_csv(os.path.join(args.out_dir, f"{code}_{name}_comment_labels.csv"), index=False, encoding="utf-8-sig")
        article_summary = aggregate_by_article(df, date_col)
        article_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
        os.remove(partial_path)
        print(f"    [完成] {len(article_summary)} 篇文章的情緒因子 -> {out_path}")

    progress_path = os.path.join(args.out_dir, PROGRESS_FILE)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print("\n[完成] 267檔股票已用Breeze-7B加速版完成情緒推論")
