# -*- coding: utf-8 -*-
"""
原論文作法（Jing, Wu & Wang 2021）用十年資料重跑：每檔股票各自訓練一個 LSTM，預測「隔天收盤價」，看 MAPE
============================================================
與舊版 train_lstm_per_stock.py 相同：per-stock、窗口10天、依時間切 train 75% / val 10% / test 15%、
    論文 Table 6 超參數（5層、20 hidden、lr 0.01、batch 10、最多100 epochs、early stopping patience 15）
改進：
  1. 資料：final_dataset_v3（2015-2024 十年、還原股價、修補壞資料）
  2. 特徵正規化只用「訓練期」的 min/max（舊版 *_scaled 用全期間，對測試期有資訊洩漏）
  3. 散戶情緒只用「隔天開盤前可得」：PTT 依留言時間（t 收盤前＋t 收盤後~t+1 開盤前），融資融券用 t 盤後公布的
  4. 四組比較：技術面 / +PTT（散戶說的）/ +融資融券（散戶做的）/ +PTT+融資融券
  5. 同時算兩個對照：天真預測（拿 t 收盤價當 t+1 預測）的 MAPE、以及預測價格換算成漲跌方向的準確率
用法：python lstm_price_v4.py --arm 技術面 --seed 0 --out-dir lstm_price_v4/技術面_seed0
      （整批由 run_lstm_price_v4.sh 平行跑；summarize_lstm_price_v4.py 彙總）
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TECH = ["MA5", "MA30", "MA60", "EMA5", "EMA30", "EMA60", "MACD_6_15_6", "MACD_12_26_9", "MACD_30_60_30",
        "RSI14", "WILLR14", "MOM14", "CMO14", "ULTOSC_7_14_28", "OBV", "ADOSC_3_10"]  # 與舊版相同的16項
PTT = ["ptt_fac", "ptt_att"]
MARGIN = ["margin_chg", "short_chg", "short_margin_ratio"]  # 散戶「做」的情緒：融資融券（研究主軸是散戶情緒，法人不放）
ARMS = {"技術面": TECH, "+PTT": TECH + PTT, "+融資融券": TECH + MARGIN, "+PTT+融資融券": TECH + PTT + MARGIN}


class LSTMRegressor(nn.Module):
    def __init__(self, nf, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(nf, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def load(path="final_dataset_v3/all_stocks_v3.csv"):
    df = pd.read_csv(path, dtype={"stock_code": str})
    df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
    nxt = lambda c: df.groupby("stock_code")[c].shift(-1).fillna(0)  # noqa: E731
    n = df.ptt_pre_n + df.ptt_intra_n + nxt("ptt_pre_n")
    p = df.ptt_pre_pos + df.ptt_intra_pos + nxt("ptt_pre_pos")
    q = df.ptt_pre_neg + df.ptt_intra_neg + nxt("ptt_pre_neg")
    df["ptt_fac"] = np.where(p + q > 0, (p - q) / (p + q).replace(0, 1), 0.0)
    df["ptt_att"] = np.log1p(n)
    for c in ("foreign_ratio", "trust_ratio", "dealer_ratio"):
        df[c] = df[c].clip(-1, 1)
    df["margin_chg"] = df["margin_chg"].clip(-0.5, 0.5).fillna(0)
    df["short_chg"] = df["short_chg"].clip(-1, 3).fillna(0)
    df["short_margin_ratio"] = df["short_margin_ratio"].clip(0, 1).fillna(0)
    return df


def windows(sub, cols, win, train_frac, val_frac):
    sub = sub.dropna(subset=TECH).reset_index(drop=True)
    n_seq = len(sub) - win
    if n_seq < 100:
        return None
    tr_end = int(n_seq * train_frac)
    va_end = tr_end + int(n_seq * val_frac)
    F = sub[cols].values.astype(np.float64)
    tr_rows = slice(0, win + tr_end)  # 訓練窗口涵蓋的列
    fmin, fmax = F[tr_rows].min(axis=0), F[tr_rows].max(axis=0)
    F = (F - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
    close = sub["Close"].values
    cmin, cmax = close[win:win + tr_end].min(), close[win:win + tr_end].max()
    crng = cmax - cmin if cmax > cmin else 1.0
    X = np.stack([F[t:t + win] for t in range(n_seq)]).astype(np.float32)
    y = ((close[win:] - cmin) / crng).astype(np.float32)
    split = np.array(["train"] * tr_end + ["val"] * (va_end - tr_end) + ["test"] * (n_seq - va_end))
    prev = close[win - 1:-1]  # 每個目標日的前一天收盤（天真預測）
    return X, y, split, close[win:], prev, cmin, crng, sub["Date"].values[win:]


def train_one(X, y, split, a, dev):
    m = LSTMRegressor(X.shape[-1], a.hidden, a.layers, a.dropout).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(X[split == "train"], device=dev), torch.tensor(y[split == "train"], device=dev)
    Xv, yv = torch.tensor(X[split == "val"], device=dev), torch.tensor(y[split == "val"], device=dev)
    best, state, left = float("inf"), None, a.patience
    for _ in range(a.epochs):
        m.train()
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            lossf(m(Xt[b]), yt[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = lossf(m(Xv), yv).item()
        if v < best:
            best, state, left = v, {k: t.clone() for k, t in m.state_dict().items()}, a.patience
        else:
            left -= 1
            if left <= 0:
                break
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(X[split == "test"], device=dev)).cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--hidden", type=int, default=20)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--only-codes", default=None, help="測試用：只跑這些股票（逗號分隔）")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = load()
    if a.only_codes:
        df = df[df.stock_code.isin(a.only_codes.split(","))]
    rows = []
    for code, sub in df.groupby("stock_code"):
        w = windows(sub, ARMS[a.arm], a.win, a.train_frac, a.val_frac)
        if w is None:
            continue
        X, y, split, actual, prev, cmin, crng, dates = w
        te = split == "test"
        pred = train_one(X, y, split, a, dev) * crng + cmin
        act, pv = actual[te], prev[te]
        rows.append({"stock_code": code, "n_test": int(te.sum()), "test_start": str(dates[te][0]),
                     "mape": float(np.mean(np.abs((act - pred) / act))),
                     "naive_mape": float(np.mean(np.abs((act - pv) / act))),
                     "dir_acc": float(np.mean(np.sign(pred - pv) == np.sign(act - pv))),
                     "up_rate": float(np.mean(act > pv))})
        print(f"{code}: MAPE {rows[-1]['mape']:.2%}（天真 {rows[-1]['naive_mape']:.2%}）方向準確率 {rows[-1]['dir_acc']:.1%}",
              flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(a.out_dir, "per_stock.csv"), index=False, encoding="utf-8-sig")
    json.dump({"arm": a.arm, "seed": a.seed, "n_stocks": len(r), "mape": float(r.mape.mean()),
               "naive_mape": float(r.naive_mape.mean()), "dir_acc": float(r.dir_acc.mean())},
              open(os.path.join(a.out_dir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[完成] {a.arm} seed{a.seed}：{len(r)} 檔，MAPE {r.mape.mean():.2%}、天真 {r.naive_mape.mean():.2%}、方向 {r.dir_acc.mean():.1%}")
