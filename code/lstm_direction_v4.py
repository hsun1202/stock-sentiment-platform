# -*- coding: utf-8 -*-
"""
LSTM 正式版：十年資料（2015-2024）逐年滾動，預測個股「隔天開盤→收盤」漲跌（可交易），
比較 技術面 / +PTT / +籌碼 / +PTT+籌碼，並與 XGBoost 版（direction_test_v4.py）對照。
============================================================
為什麼不照原論文預測股價/看MAPE：舊版 LSTM（per-stock 價格迴歸）的 MAPE 被「價格水準外推誤差」主導
（直接用昨天收盤價 MAPE 1.34%，LSTM 11.5%），偵測不到情緒這種小效果。改成漲跌方向＋AUC，
並與 XGBoost 用完全相同的特徵（features_v4.py）、切分、評估方式。

設計：
  - 輸入：每筆 = 同一檔股票過去 20 個交易日的特徵序列（t-19..t），預測 t+1 開盤→收盤漲跌
  - 所有股票一起訓練一個模型；特徵用訓練期的平均/標準差標準化、截在 ±5、缺值補 0
  - 模型：LSTM(hidden 64, 1層) → Dropout 0.2 → Linear；Adam lr 1e-3、batch 1024、最多 15 epochs，
    驗證集 AUC 連續 3 個 epoch 沒進步就停（取最佳 epoch）
  - 測試年 Y = 2018~2024：訓練 2015 ~ (Y-1)-06、驗證 (Y-1) 下半年、測試 Y 全年（同 XGBoost 版）
  - 每組 10 個種子平均機率；ΔAUC 以週 block bootstrap 估 95%CI；7 年彙總
  - 每個 (年, 組合, 種子) 的測試集機率存 lstm_v4/，中斷後重跑自動跳過
python lstm_direction_v4.py
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from features_v4 import build

SEEDS = tuple(range(10))  # 一次訓練不到1分鐘，用10個種子平均降低LSTM的種子雜訊（3種子版結果另存 *_3seeds）
WIN, BATCH, EPOCHS, PATIENCE, BOOT = 20, 1024, 15, 3, 300
TEST_YEARS = range(2018, 2025)
RETAIL = "--retail" in __import__("sys").argv  # 散戶情緒版：+PTT / +融資融券 / 兩者（不放法人）
OUT = "lstm_v4_retail" if RETAIL else "lstm_v4"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feat, meta, ARMS = build(retail=RETAIL)
code = meta["stock_code"].values
# 視窗起點：t-19 必須是同一檔股票
has_hist = np.zeros(len(meta), dtype=bool)
has_hist[WIN - 1:] = code[WIN - 1:] == code[:len(code) - WIN + 1]
base_ok = meta["valid"].values & has_hist
date, end_date, y_all = meta["date"], meta["end_date"], meta["y"].values.astype(np.float32)
offsets = torch.arange(-(WIN - 1), 1, device=dev)


class Net(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.lstm = nn.LSTM(nf, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def predict(model, X, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 8192):
            b = idx[s:s + 8192]
            out.append(torch.sigmoid(model(X[b[:, None] + offsets])).float().cpu())
    return torch.cat(out).numpy()


def fit_one(X, tr, va, te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Y = torch.tensor(y_all, device=dev)
    tr_t = torch.tensor(tr, device=dev)
    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss = lossf(model(X[b[:, None] + offsets]), Y[b])
            loss.backward()
            opt.step()
        auc = roc_auc_score(y_all[va], predict(model, X, torch.tensor(va, device=dev)))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return predict(model, X, torch.tensor(te, device=dev)), best, ep + 1


results = {}
t0 = time.time()
for Y in TEST_YEARS:
    tr = np.where(base_ok & (end_date <= f"{Y-1}-06-30").values)[0]
    va = np.where(base_ok & (date >= f"{Y-1}-07-01").values & (end_date <= f"{Y-1}-12-31").values)[0]
    te = np.where(base_ok & (date >= f"{Y}-01-01").values & (date <= f"{Y}-12-31").values
                  & (end_date <= f"{Y+1}-01-31").values)[0]
    probs = {}
    for arm, cols in ARMS.items():
        A = feat[cols].values.astype(np.float32)
        mu, sd = np.nanmean(A[tr], axis=0), np.nanstd(A[tr], axis=0) + 1e-6
        A = np.nan_to_num(np.clip((A - mu) / sd, -5, 5), nan=0.0)
        X = torch.tensor(A, device=dev)
        ps = []
        for s in SEEDS:
            f = os.path.join(OUT, f"{Y}_{arm}_{s}.npy")
            if os.path.exists(f):
                ps.append(np.load(f))
                continue
            p, vauc, ne = fit_one(X, tr, va, te, s)
            np.save(f, p)
            ps.append(p)
            print(f"    {Y} {arm} seed{s}: 驗證AUC {vauc:.4f}（{ne} epochs）測試AUC {roc_auc_score(y_all[te], p):.4f}"
                  f"  累計 {(time.time()-t0)/60:.0f} 分鐘", flush=True)
        probs[arm] = np.mean(ps, axis=0)
        del X
        torch.cuda.empty_cache()
    yte = y_all[te]
    wk = date.values[te].astype("datetime64[W]")
    u = np.unique(wk)
    widx = {w: np.where(wk == w)[0] for w in u}
    rng = np.random.default_rng(Y)
    boots = [np.concatenate([widx[w] for w in rng.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
    base = roc_auc_score(yte, probs["技術面"])
    r = {"n_train": len(tr), "n_test": len(te), "base_auc": base, "arms": {}}
    line = f"  {Y}: train {len(tr):,} test {len(te):,} 技術面AUC {base:.4f}"
    for arm in ARMS:
        if arm == "技術面":
            continue
        d = roc_auc_score(yte, probs[arm]) - base
        bd = np.array([roc_auc_score(yte[b], probs[arm][b]) - roc_auc_score(yte[b], probs["技術面"][b]) for b in boots])
        ci = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        r["arms"][arm] = {"d_auc": d, "ci95": ci}
        line += f"｜{arm} {d:+.4f}{'✔' if ci[0] > 0 else ''}"
    print(line, flush=True)
    results[Y] = r

print("\n--- LSTM 7年彙總（ΔAUC 平均、幾年CI>0、幾年為正）---")
summ = {}
for arm in ARMS:
    if arm == "技術面":
        continue
    ds = np.array([results[Y]["arms"][arm]["d_auc"] for Y in TEST_YEARS])
    sig = sum(results[Y]["arms"][arm]["ci95"][0] > 0 for Y in TEST_YEARS)
    t = float(ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds))))
    summ[arm] = {"mean_d_auc": float(ds.mean()), "t_stat": t, "years_sig": int(sig), "years_pos": int((ds > 0).sum())}
    print(f"  {arm:20s} 平均ΔAUC {ds.mean():+.4f}（t={t:.2f}）｜顯著年數 {sig}/7｜為正年數 {(ds > 0).sum()}/7")
print("  技術面 AUC 各年：" + "、".join(f"{Y}:{results[Y]['base_auc']:.4f}" for Y in TEST_YEARS))
json.dump({"by_year": {str(k): v for k, v in results.items()}, "summary": summ},
          open("lstm_direction_v4_retail_results.json" if RETAIL else "lstm_direction_v4_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n完成 → " + ("lstm_direction_v4_retail_results.json" if RETAIL else "lstm_direction_v4_results.json"))
