# -*- coding: utf-8 -*-
"""
分類器不夠準，最多會把 PTT 情緒的效果壓低多少？（測量誤差／衰減分析）
  1) 用目前的 Breeze（constrained 三選一）重新分類 1,100 則 Claude 標註留言與 392 則人工標註留言 → 混淆矩陣、方向錯誤率
  2) 模擬：以實際的「每個股票日留言數」分布，產生真實標籤 → 依混淆矩陣產生分類器標籤，
     比較「真實淨情緒」與「分類器淨情緒」；衰減係數 b＝真實淨情緒對分類器淨情緒的迴歸斜率。
     觀察到的迴歸係數 ≈ 真實係數 × b ⇒ 真實係數 ≈ 觀察值 ÷ b（古典測量誤差）
  3) 代入樣本內觀察值，推估「分類器完美時」效果的上限，並與交易成本比較
輸出：classifier_noise_test.log、classifier_gold_preds.csv
python classifier_noise_test.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from breeze_infer_267_fast import FastClassifier, load_model

warnings.filterwarnings("ignore")
LAB = ["positive", "negative", "neutral"]
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


gold = pd.read_csv("claude_all_labels.csv", dtype=str, keep_default_na=False).rename(columns={"claude_label": "gold"})
hum = pd.read_csv("all_human_labels.csv", dtype=str, keep_default_na=False).rename(columns={"human_label": "gold"})
tok, model = load_model()
clf = FastClassifier(tok, model, batch_size=16, mode="constrained")
gold["pred"] = clf.classify(gold.content.tolist())
hum["pred"] = clf.classify(hum.content.tolist())
gold.to_csv("classifier_gold_preds.csv", index=False, encoding="utf-8-sig")

for name, d in [("Claude 標註 1,100 則", gold), ("人工標註", hum)]:
    d = d[d.gold.isin(LAB)]
    cm = confusion_matrix(d.gold, d.pred, labels=LAB)
    say(f"\n=== {name}（{len(d)} 則）：準確率 {accuracy_score(d.gold, d.pred):.1%}｜Macro-F1 {f1_score(d.gold, d.pred, labels=LAB, average='macro'):.3f}"
        f"｜Kappa {cohen_kappa_score(d.gold, d.pred):.2f} ===")
    say("  列＝真實、欄＝分類器（正/負/中）：")
    for i, l in enumerate(LAB):
        say(f"    真實{l:<9s} {cm[i].tolist()}  → 判對 {cm[i, i] / cm[i].sum():.0%}")
    both = d[d.gold.isin(LAB[:2]) & d.pred.isin(LAB[:2])]
    say(f"  真實與分類器都有表態（正或負）的 {len(both)} 則中，方向相反的占 {(both.gold != both.pred).mean():.1%}")
    say(f"  分類器判「正」的留言，真實為正 {(d[d.pred == 'positive'].gold == 'positive').mean():.0%}、為負 {(d[d.pred == 'positive'].gold == 'negative').mean():.0%}；"
        f"判「負」的，真實為負 {(d[d.pred == 'negative'].gold == 'negative').mean():.0%}、為正 {(d[d.pred == 'negative'].gold == 'positive').mean():.0%}")
m = gold.merge(hum[["content", "gold"]].rename(columns={"gold": "human"}), on="content")
if len(m):
    say(f"\n人工 vs Claude 標註（重疊 {len(m)} 則）：一致 {(m.gold == m.human).mean():.1%}、Kappa {cohen_kappa_score(m.gold, m.human):.2f} → 同一則留言，人與人/模型本來就常看法不同")

# ---- 2) 衰減模擬 ----
g = gold[gold.gold.isin(LAB)]
C = confusion_matrix(g.gold, g.pred, labels=LAB).astype(float)
C = C / C.sum(axis=1, keepdims=True)                    # P(分類器 | 真實)
pi = g.gold.value_counts(normalize=True).reindex(LAB).values
df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", usecols=["ptt_pre_n", "ptt_intra_n", "ptt_pre_pos", "ptt_pre_neg", "ptt_intra_pos", "ptt_intra_neg"]).fillna(0)
n_all = (df.ptt_pre_n + df.ptt_intra_n).astype(int)
ex = df.ptt_pre_pos + df.ptt_pre_neg + df.ptt_intra_pos + df.ptt_intra_neg
net_obs = ((df.ptt_pre_pos + df.ptt_intra_pos - df.ptt_pre_neg - df.ptt_intra_neg) / ex.replace(0, np.nan))[ex >= 1]
n_pool = n_all[ex >= 1].values
say(f"\n實際資料：有表態的股票日 {len(net_obs):,}，每日留言數中位數 {np.median(n_pool):.0f}；分類器淨情緒的標準差 {net_obs.std():.3f}")
rng = np.random.default_rng(0)


def simulate(kappa, N=60000):
    n = rng.choice(n_pool, N)
    p = rng.dirichlet(pi * kappa, N)                      # 每個股票日自己的 正/負/中 比例
    true_net, pred_net = np.full(N, np.nan), np.full(N, np.nan)
    for i in range(N):
        t = rng.multinomial(n[i], p[i])                   # 真實 正/負/中 則數
        pr = sum(rng.multinomial(t[k], C[k]) for k in range(3))
        if t[0] + t[1] > 0:
            true_net[i] = (t[0] - t[1]) / (t[0] + t[1])
        if pr[0] + pr[1] > 0:
            pred_net[i] = (pr[0] - pr[1]) / (pr[0] + pr[1])
    ok = ~np.isnan(true_net) & ~np.isnan(pred_net)
    b = np.cov(true_net[ok], pred_net[ok])[0, 1] / np.var(pred_net[ok])
    return np.nanstd(pred_net), b, np.corrcoef(true_net[ok], pred_net[ok])[0, 1]


say("\n衰減模擬（調整各股票日情緒差異的大小 kappa，使模擬的分類器淨情緒標準差接近實際值）")
best = None
for kappa in [1, 2, 4, 8, 16, 32]:
    sd, b, r = simulate(kappa)
    say(f"  kappa={kappa:<3d} 模擬標準差 {sd:.3f}｜衰減係數 b={b:.2f}｜真實與分類器淨情緒相關 {r:.2f}")
    if best is None or abs(sd - net_obs.std()) < abs(best[1] - net_obs.std()):
        best = (kappa, sd, b, r)
k, sd, b, r = best
say(f"\n最接近實際的設定：kappa={k}（標準差 {sd:.3f} vs 實際 {net_obs.std():.3f}）→ 衰減係數 b≈{b:.2f}、相關≈{r:.2f}")
say("推估「分類器完美」時的效果（真實係數 ≈ 觀察值 ÷ b）：")
for lab, beta, se in [("隔夜（控制今天/漲跌停/留言量）", 4.49, 4.49 / 3.78), ("隔夜（再控制過去報酬）", 2.56, 2.56 / 2.18), ("隔天收盤→收盤", 4.16, 4.16 / 1.84)]:
    hi = beta + 1.96 * se
    say(f"  {lab:<18s} 觀察 {beta:+.1f} bp（95% 上界 {hi:+.1f}）→ 完美分類器約 {beta / b:+.1f} bp（上界 {hi / b:+.1f}）；係數是「淨情緒每 1 單位」，全負→全正＝2 倍")
say("  對照：一般交易來回成本 47~59 bp、當沖 32~44 bp。")
open("classifier_noise_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
