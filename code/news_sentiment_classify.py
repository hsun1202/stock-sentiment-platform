# -*- coding: utf-8 -*-
"""
新聞情緒：用 LLM 判斷 PTT [新聞] 標題對相關公司/台股是 利多、利空 還是 無關（參考 Lopez-Lira & Tang 2023）
============================================================
  - 對象：PTT 股板 [新聞] 原發文（不含 Re:/Fw:）2015-2024 共約 8.3 萬篇；其中 1.38 萬篇標題比對得到 267 檔股票
  - 模型：Breeze-7B（與留言情緒同一個模型），強制三選一：比較模型回答「好」「壞」「無」的機率
      news_score = log P(好) − log P(壞)（連續分數，越大越利多）；news_label = 三者機率最大者
  - 只看標題（與 Lopez-Lira & Tang 相同），發文時間 post_time 保留，之後依時間窗對齊（不偷看）
  - 共用開頭 KV cache＋同長度分桶（同 breeze_infer_267_fast 的加速方式）；每 5,000 則存檔，可續跑
輸出：news_sentiment/news_scores.csv（article_id、post_time、title、stock_codes、news_score、p_good、p_bad、p_none、news_label）
python news_sentiment_classify.py
"""
import os
import copy
import time
import torch
import numpy as np
import pandas as pd

from breeze_infer_267_fast import load_model
from match_ptt_2015_2024_267 import find_matches_v2

OUT = "news_sentiment"
os.makedirs(OUT, exist_ok=True)
SYSTEM = ("你是台股分析師。判斷一則財經新聞標題，對標題提到的公司股價（沒有提到公司時，對台股大盤）是利多、利空還是無關。"
          "利多回答「好」，利空回答「壞」，無關或看不出影響回答「無」。只回答一個字。")
FEWSHOT = [("台積電3月營收創同期新高 年增逾3成", "好"), ("長榮遭外資降評 目標價砍半", "壞"),
           ("華航新任總經理今天上任", "無"), ("美國升息2碼 通膨壓力未解", "壞"),
           ("大立光法說會 下季出貨看增", "好"), ("中鋼舉辦員工運動會", "無"),
           ("國巨大股東申報轉讓 持股大減", "壞"), ("台股成交量創新高 外資連買10日", "好")]


def build_prompt(tok, title):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, a in FEWSHOT:
        msgs += [{"role": "user", "content": t}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": title[:120]})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Scorer:
    def __init__(self, tok, model, batch=16):
        self.tok, self.model, self.batch = tok, model, batch
        v = tok.get_vocab()
        self.ids = [v["好"], v["壞"], v["無"]]
        self.prefix, self.cache = None, None

    @torch.no_grad()
    def score(self, prompts):
        ids = self.tok(prompts).input_ids
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
            assert s[:P] == self.prefix
            buckets.setdefault(len(s), []).append(k)
        for L, idx in buckets.items():
            for st in range(0, len(idx), self.batch):
                part = idx[st:st + self.batch]
                suf = torch.tensor([ids[k][P:] for k in part], device=self.model.device)
                c = copy.deepcopy(self.cache)
                c.batch_repeat_interleave(len(part))
                lg = self.model(input_ids=suf, past_key_values=c, use_cache=True, logits_to_keep=1).logits[:, -1, :].float()
                lp = torch.log_softmax(lg, -1)[:, self.ids]
                pr = torch.softmax(lp, -1)
                for k, row_lp, row_p in zip(part, lp.tolist(), pr.tolist()):
                    out[k] = (row_lp[0] - row_lp[1], *row_p)
        return out


if __name__ == "__main__":
    a = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["article_id", "title", "category", "post_time"],
                    dtype=str, keep_default_na=False)
    n = a[(a.category == "新聞") & ~a.title.str.match(r"^\s*(Re|Fw)", case=False)].copy()
    n["clean_title"] = n.title.str.replace(r"^\s*\[新聞\]\s*", "", regex=True).str.strip()
    n["stock_codes"] = n.title.map(lambda t: ",".join(c for c, _, _ in find_matches_v2(t)))
    part_path = os.path.join(OUT, "news_scores.partial.csv")
    done = pd.read_csv(part_path, dtype=str) if os.path.exists(part_path) else pd.DataFrame(columns=["article_id"])
    todo = n[~n.article_id.isin(set(done.article_id))]
    print(f"新聞 {len(n):,} 篇（比對到股票 {(n.stock_codes != '').sum():,}），已完成 {len(done):,}，待做 {len(todo):,}", flush=True)
    tok, model = load_model()
    sc = Scorer(tok, model)
    t0 = time.time()
    for s in range(0, len(todo), 5000):
        ch = todo.iloc[s:s + 5000]
        r = sc.score([build_prompt(tok, t) for t in ch.clean_title])
        o = ch[["article_id", "post_time", "title", "stock_codes"]].copy()
        o["news_score"], o["p_good"], o["p_bad"], o["p_none"] = zip(*r)
        o.to_csv(part_path, mode="a", header=not os.path.exists(part_path), index=False, encoding="utf-8-sig")
        k = s + len(ch)
        print(f"  進度 {k:,}/{len(todo):,}，{k/(time.time()-t0):.0f} 則/秒", flush=True)
    res = pd.read_csv(part_path, dtype={"article_id": str, "stock_codes": str}, keep_default_na=False)
    res["news_label"] = np.array(["好", "壞", "無"])[res[["p_good", "p_bad", "p_none"]].astype(float).values.argmax(1)]
    res.to_csv(os.path.join(OUT, "news_scores.csv"), index=False, encoding="utf-8-sig")
    os.remove(part_path)
    print("完成：", res.news_label.value_counts(normalize=True).round(3).to_dict(), flush=True)
