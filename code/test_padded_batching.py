# -*- coding: utf-8 -*-
import time
import glob
import pandas as pd

from breeze_infer_267_fast import FastClassifier, load_model

# 混合短(PTT留言)+長(CMoney文章/留言)的異質長度樣本，模擬真實情況
ptt = pd.concat([pd.read_csv(f, usecols=["content"]) for f in glob.glob("ptt_output_267_stitched_2020_2024/*_comments.csv")[:5]])
ptt_texts = ptt["content"].dropna().astype(str).sample(400, random_state=0).tolist()

cm_files = glob.glob("cmoney_267_clean/2330_*.csv")
cm = pd.read_csv(cm_files[0], usecols=["text"])
cm_texts = cm["text"].dropna().astype(str).sample(400, random_state=0).tolist()

mix = ptt_texts + cm_texts
print(f"樣本數 {len(mix)}，長度: min={min(len(t) for t in mix)} max={min(200,max(len(t) for t in mix))} "
      f"median={sorted(len(t) for t in mix)[len(mix)//2]}", flush=True)

tok, model = load_model()

clf1 = FastClassifier(tok, model, batch_size=32, mode="constrained")
t = time.time()
labels1 = clf1.classify(mix)
t1 = time.time() - t
print(f"[舊版 精確長度分桶] {t1:.1f}秒 -> {len(mix)/t1:.1f}則/秒", flush=True)

clf2 = FastClassifier(tok, model, batch_size=32, mode="constrained")
t = time.time()
labels2 = clf2.classify_sorted(mix)
t2 = time.time() - t
print(f"[新版 排序+padding分批] {t2:.1f}秒 -> {len(mix)/t2:.1f}則/秒", flush=True)

same = sum(a == b for a, b in zip(labels1, labels2))
print(f"\n兩版標籤一致率: {same}/{len(mix)} = {same/len(mix)*100:.2f}%")
diff = [(mix[i][:30], labels1[i], labels2[i]) for i in range(len(mix)) if labels1[i] != labels2[i]]
for d in diff[:10]:
    print("  不一致:", d)
print("完成")
