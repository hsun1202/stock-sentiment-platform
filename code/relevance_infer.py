# -*- coding: utf-8 -*-
"""
留言「是否與股市/投資相關」的第二道過濾（Breeze-7B，強制二選一：比較模型回答「是」與「否」的機率）
============================================================

為什麼要有這一步：比對到267檔股票的留言隨機抽400則，人工（Claude）判斷約24%與股票/投資無關
（政治謾罵、能源政策、員工薪水、玩笑、岔題），其中政治謾罵很容易被情緒分類器判成negative，污染個股情緒。
設計：不改動已驗證過的三分類情緒模型，另外算一個相關性分數 rel_score = log P(是) − log P(否)。
    只需要對「情緒被判成 positive/negative」的留言算——情緒指標用 (正−負)/(正+負)，neutral 本來就不影響分數。

用法：
    # 1) 驗證：在人工標註的兩組各200則上比較兩種提示（只看留言 / 標題+留言），A組選門檻、B組驗證
    python relevance_infer.py --eval
    # 2) 套用到情緒分類結果（sentiment_output_267_2015_2024/*_comment_labels.csv 中的正/負留言）
    python relevance_infer.py --run --with-title   # 或不加 --with-title，依 --eval 結果決定
輸出（--run）：relevance_scores_2015_2024.csv（key=標題+留言 或 留言 → rel_score），可續傳
"""
import os
import glob
import copy
import time
import argparse

import torch
import pandas as pd

from breeze_infer_267_fast import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = ("你是台股PTT股票板的版務助理。判斷一則留言是否在討論股票、投資、股價、市場行情、總體經濟或公司的營運與獲利。"
          "只要跟投資或公司經營有關就回答「是」；政治謾罵、與投資無關的閒聊或玩笑、人身攻擊、與股市無關的社會議題回答「否」。"
          "只回答一個字：是 或 否。")
FEWSHOT = [
    ("明天開高走低 先跑為妙", "是"),
    ("綠共又在護航 選民都是白痴", "否"),
    ("營收這麼好還跌 外資一直倒貨", "是"),
    ("這主播好正 求名字", "否"),
    ("電價一漲成本就上去 毛利一定會降", "是"),
    ("你的頭像好好笑", "否"),
    ("丸子 套在山頂了", "是"),
    ("美國人就是懶 不想加班", "否"),
]


def build_prompt(tok, text, title=None):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, lab in FEWSHOT:
        msgs.append({"role": "user", "content": t})
        msgs.append({"role": "assistant", "content": lab})
    user = f"文章標題：{title[:60]}\n留言：{text[:200]}" if title else text[:200]
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class RelevanceScorer:
    """共用開頭KV cache＋同長度分桶（同 breeze_infer_267_fast.FastClassifier 的加速方式）。"""

    def __init__(self, tok, model, batch_size=32):
        self.tok, self.model, self.batch_size = tok, model, batch_size
        vocab = tok.get_vocab()
        self.yes = [vocab[w] for w in ("是", "▁是") if w in vocab]
        self.no = [vocab[w] for w in ("否", "▁否", "不") if w in vocab]
        assert self.yes and self.no, (self.yes, self.no)
        self.prefix_ids, self.prefix_cache = None, None

    def _init_prefix(self, id_lists):
        first, n = id_lists[0], len(id_lists[0])
        for s in id_lists[1:256]:
            i, m = 0, min(n, len(s))
            while i < m and first[i] == s[i]:
                i += 1
            n = i
        n = min(n, min(len(s) for s in id_lists[:256]) - 1)
        self.prefix_ids = first[:n]
        with torch.no_grad():
            self.prefix_cache = self.model(input_ids=torch.tensor([self.prefix_ids], device=self.model.device),
                                           use_cache=True).past_key_values

    @torch.no_grad()
    def _group(self, id_lists):
        P, B = len(self.prefix_ids), len(id_lists)
        suffix = torch.tensor([ids[P:] for ids in id_lists], device=self.model.device)
        cache = copy.deepcopy(self.prefix_cache)
        cache.batch_repeat_interleave(B)
        lg = self.model(input_ids=suffix, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :]
        lp = torch.log_softmax(lg.float(), dim=-1)
        return (torch.logsumexp(lp[:, self.yes], -1) - torch.logsumexp(lp[:, self.no], -1)).tolist()

    def score(self, prompts):
        id_lists = self.tok(prompts).input_ids
        if self.prefix_cache is None:
            self._init_prefix(id_lists)
        P = len(self.prefix_ids)
        out = [None] * len(prompts)
        buckets = {}
        for i, ids in enumerate(id_lists):
            assert ids[:P] == self.prefix_ids
            buckets.setdefault(len(ids), []).append(i)
        for L, idx in buckets.items():
            for s in range(0, len(idx), self.batch_size):
                part = idx[s:s + self.batch_size]
                for i, v in zip(part, self._group([id_lists[i] for i in part])):
                    out[i] = v
        return out


def auc(y, s):
    pos = [v for v, t in zip(s, y) if t == 0]  # 把「無關」當正類：分數越低越可能無關
    neg = [v for v, t in zip(s, y) if t == 1]
    wins = sum((p < n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(tok, model):
    a = pd.read_csv(os.path.join(BASE, "offtopic_sample_200_labeled.csv"), dtype=str, keep_default_na=False)
    b = pd.read_csv(os.path.join(BASE, "offtopic_sample_200b_labeled.csv"), dtype=str, keep_default_na=False)
    lines = []
    for with_title in (False, True):
        scorer = RelevanceScorer(tok, model)
        res = {}
        for name, d in (("A", a), ("B", b)):
            prompts = [build_prompt(tok, c, t if with_title else None) for c, t in zip(d["content"], d["title"])]
            d = d.copy()
            d["score"] = scorer.score(prompts)
            d["y"] = d["relevant_claude"].astype(int)
            res[name] = d
        # A 組選門檻：最大化 Youden（抓到的無關比例 − 誤刪的相關比例）
        da = res["A"]
        best = max(sorted(set(da["score"])), key=lambda th: ((da.score < th) & (da.y == 0)).sum() / (da.y == 0).sum()
                   - ((da.score < th) & (da.y == 1)).sum() / (da.y == 1).sum())
        tag = "標題+留言" if with_title else "只看留言"
        for name, d in res.items():
            drop = d.score < best
            caught = (drop & (d.y == 0)).sum() / (d.y == 0).sum()
            wrong = (drop & (d.y == 1)).sum() / (d.y == 1).sum()
            remain_off = ((~drop) & (d.y == 0)).sum() / max(1, (~drop).sum())
            lines.append(f"[{tag}] {name}組 AUC={auc(d.y.tolist(), d.score.tolist()):.3f}  門檻={best:.2f}（A組選）"
                         f"  抓到無關 {caught:.0%}｜誤刪相關 {wrong:.0%}｜過濾後剩餘無關比例 {remain_off:.0%}"
                         f"（原本 {(d.y == 0).mean():.0%}）")
            d.to_csv(os.path.join(BASE, f"relevance_eval_{'title' if with_title else 'text'}_{name}.csv"),
                     index=False, encoding="utf-8-sig")
    rep = "\n".join(lines)
    print(rep)
    with open(os.path.join(BASE, "relevance_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")


def run(tok, model, with_title, chunk=4000):
    arts = pd.read_csv(os.path.join(BASE, "ptt_2015_2024", "articles.csv"), usecols=["article_id", "title"],
                       dtype=str, keep_default_na=False)
    title_of = dict(zip(arts["article_id"], arts["title"]))
    keys = {}
    for f in glob.glob(os.path.join(BASE, "sentiment_output_267_2015_2024", "*_comment_labels.csv")):
        d = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["article_id", "content", "text_sentiment"])
        d = d[d["text_sentiment"].isin(["positive", "negative"])]
        for aid, c in zip(d["article_id"], d["content"]):
            t = title_of.get(aid, "") if with_title else ""
            keys[(t, c[:200])] = None
    keys = list(keys)
    out_path = os.path.join(BASE, "relevance_scores_2015_2024.csv")
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        done = set(zip(prev["title"], prev["content"]))
    todo = [k for k in keys if k not in done]
    print(f"[相關性] 需要評分的(標題,留言) {len(keys):,} 組，已完成 {len(done):,}，待做 {len(todo):,}")
    scorer = RelevanceScorer(tok, model)
    t0 = time.time()
    first = not os.path.exists(out_path)
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        sc = scorer.score([build_prompt(tok, c, t or None) for t, c in part])
        pd.DataFrame({"title": [t for t, _ in part], "content": [c for _, c in part], "rel_score": sc}).to_csv(
            out_path, mode="a", header=first, index=False, encoding="utf-8-sig")
        first = False
        n = s + len(part)
        rate = n / (time.time() - t0)
        print(f"    進度 {n:,}/{len(todo):,}，{rate:.1f}則/秒，預估剩 {(len(todo)-n)/rate/3600:.1f} 小時")
    print("[相關性完成]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--with-title", action="store_true")
    args = ap.parse_args()
    tok, model = load_model()
    if args.eval:
        evaluate(tok, model)
    if args.run:
        run(tok, model, args.with_title)
