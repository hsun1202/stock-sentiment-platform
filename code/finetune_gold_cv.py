# -*- coding: utf-8 -*-
"""
用 Claude 金標微調中文 BERT 做三分類（正/負/中），並用「依股票日分組的 5 折交叉驗證」公平評估。
  金標：relabel/sample.csv + relabel/labels_XX.txt（5,194 則、400 個股票日，有股票名與標題）
        claude_all_labels.csv（1,100 則，只有留言文字；只放訓練、不參與評估）
  輸入兩種版本：content＝只看留言；context＝「股票名｜標題｜推噓」+ 留言
  評估（全部用折外預測）：與金標的一致率、Kappa、Macro-F1、方向相反率、
        股票日淨情緒的相關與衰減係數 b（金標淨情緒對模型淨情緒的斜率），並與 Breeze 並排。
輸出：finetune_gold_cv.log、finetune_gold_oof.csv
python finetune_gold_cv.py [--model ckiplab/bert-base-chinese] [--epochs 4] [--seeds 1]
"""
import argparse
import glob
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
L = ["positive", "negative", "neutral"]
L2I = {l: i for i, l in enumerate(L)}
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def load_gold():
    m = {"p": "positive", "n": "negative", "u": "neutral"}
    lab = {}
    for f in sorted(glob.glob("relabel/labels_*.txt")):
        for x in open(f, encoding="utf-8").read().split():
            i, l = x.split(":")
            lab[int(i)] = m[l]
    s = pd.read_csv("relabel/sample.csv", dtype={"stock_code": str}, keep_default_na=False)
    s["gold"] = s.id.map(lab)
    assert s.gold.notna().all()
    s["grp"] = s.stock_code + "_" + s.Date.astype(str)
    old = pd.read_csv("claude_all_labels.csv", dtype=str, keep_default_na=False).rename(columns={"claude_label": "gold"})
    old = old[old.gold.isin(L)].reset_index(drop=True)
    return s, old


def ctx(name, title, push):
    return f"{name}｜{title}｜{push}"


def encode(tok, a, b, max_len):
    if a is None:
        return tok(b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return tok(a, b, truncation="only_first", max_length=max_len, padding=True, return_tensors="pt")


def train_model(model_name, tok, tr_a, tr_b, y, epochs, lr, bs, max_len, seed, weights=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(tr_b)
    steps = epochs * ((n + bs - 1) // bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    w = None if weights is None else torch.tensor(weights, dtype=torch.float).cuda()
    y = torch.tensor(y)
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            enc = encode(tok, None if tr_a is None else [tr_a[i] for i in idx], [tr_b[i] for i in idx], max_len)
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y[idx].cuda(), weight=w, label_smoothing=0.05)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
    return model


@torch.no_grad()
def predict_proba(model, tok, a, b, max_len, bs=256):
    model.eval()
    out = []
    order = np.argsort([len(x) + (0 if a is None else len(a[i])) for i, x in enumerate(b)])
    for s in range(0, len(b), bs):
        idx = order[s:s + bs]
        enc = encode(tok, None if a is None else [a[i] for i in idx], [b[i] for i in idx], max_len)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
        out.append((idx, p))
        if len(b) > 100000 and (s // bs) % 500 == 0:
            print(f"      推論 {s:,}/{len(b):,}", flush=True)
    res = np.zeros((len(b), 3), dtype=np.float32)
    for idx, p in out:
        res[idx] = p
    return res


def net(g, col):
    p, n = (g[col] == "positive").sum(), (g[col] == "negative").sum()
    return (p - n) / (p + n) if p + n else np.nan


def report(d, col, name):
    acc, kap = accuracy_score(d.gold, d[col]), cohen_kappa_score(d.gold, d[col])
    f1 = f1_score(d.gold, d[col], labels=L, average="macro")
    both = d[d.gold.isin(L[:2]) & d[col].isin(L[:2])]
    opp = (both.gold != both[col]).mean()
    pp = (d[d[col] == "positive"].gold == "positive").mean()
    pn = (d[d[col] == "negative"].gold == "negative").mean()
    G = d.groupby("grp")
    day = pd.DataFrame({"c": G.apply(lambda g: net(g, "gold")), "m": G.apply(lambda g: net(g, col))}).dropna()
    b = np.cov(day.c, day.m)[0, 1] / day.m.var()
    r = np.corrcoef(day.c, day.m)[0, 1]
    se = np.sqrt((1 - r ** 2) * day.c.var() / day.m.var() / (len(day) - 2))
    dist = d[col].value_counts(normalize=True).reindex(L).fillna(0).round(2).tolist()
    say(f"  {name:<22s} 一致率 {acc:.1%}｜Kappa {kap:.2f}｜Macro-F1 {f1:.3f}｜方向相反 {opp:.1%}｜判正的真為正 {pp:.0%}｜判負的真為負 {pn:.0%}"
        f"｜分布(正/負/中) {dist}｜股票日 n={len(day)} 相關 {r:.2f}、b={b:.2f}±{se:.2f}")
    cm = confusion_matrix(d.gold, d[col], labels=L)
    say(f"  {'':<22s} 混淆矩陣（列＝金標、欄＝模型；正/負/中）：{cm.tolist()}")
    return acc, kap, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckiplab/bert-base-chinese")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--variants", default="content,context")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    s, old = load_gold()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    say(f"模型 {args.model}｜epochs {args.epochs}｜lr {args.lr}｜seeds {args.seeds}")
    say(f"金標：重標樣本 {len(s):,} 則（{s.grp.nunique()} 個股票日）＋舊金標 {len(old):,} 則（只進訓練）")
    say(f"金標分布（正/負/中）：{s.gold.value_counts(normalize=True).reindex(L).round(3).tolist()}")
    say("\n基準：")
    report(s, "breeze", "Breeze（現行）")

    folds = list(GroupKFold(n_splits=5).split(s, groups=s.grp))
    for variant in args.variants.split(","):
        P = np.zeros((len(s), 3), dtype=np.float32)
        for seed in range(args.seeds):
            for k, (tr, te) in enumerate(folds):
                tr_b = s.content.iloc[tr].tolist() + old.content.tolist()
                y = [L2I[x] for x in s.gold.iloc[tr]] + [L2I[x] for x in old.gold]
                if variant == "context":
                    tr_a = [ctx(n, t, p) for n, t, p in zip(s.stock_name.iloc[tr], s.title.iloc[tr], s.push_type.iloc[tr])] + [""] * len(old)
                    te_a = [ctx(n, t, p) for n, t, p in zip(s.stock_name.iloc[te], s.title.iloc[te], s.push_type.iloc[te])]
                    max_len = 128
                else:
                    tr_a = te_a = None
                    max_len = 64
                model = train_model(args.model, tok, tr_a, tr_b, y, args.epochs, args.lr, args.bs, max_len, seed * 100 + k)
                P[te] += predict_proba(model, tok, te_a, s.content.iloc[te].tolist(), max_len)
                del model
                torch.cuda.empty_cache()
                print(f"    {variant} seed {seed} fold {k} done", flush=True)
        P /= args.seeds
        s[f"bert_{variant}"] = [L[i] for i in P.argmax(1)]
        for j, l in enumerate(L):
            s[f"p_{variant}_{l}"] = P[:, j]
        say(f"\n微調 BERT（{variant}，折外預測）：")
        report(s, f"bert_{variant}", f"BERT-{variant}")
    s.to_csv(f"finetune_gold_oof{args.tag}.csv", index=False, encoding="utf-8-sig")
    open(f"finetune_gold_cv{args.tag}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    say("DONE")
