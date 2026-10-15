# -*- coding: utf-8 -*-
"""
LSTM股價預測模型訓練（技術指標 + 情緒因子 -> 下一日收盤價）
============================================================

目的：
    讀取prepare_lstm_dataset.py產出的池化序列資料，訓練LSTM
    預測下一日收盤價，在test set上算MAPE（比照Jing et al. 2021
    論文的評估方式）。

    用 --no-sentiment 可以跑「只用技術指標」的基準對照組，
    跟預設的「技術指標+情緒因子」版本比較MAPE，驗證加入情緒
    因子是否真的能改善預測（這是整個專題要驗證的核心問題）。

使用方式：
    python train_lstm.py --out-dir lstm_model_with_sentiment
    python train_lstm.py --no-sentiment --out-dir lstm_model_without_sentiment
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # 取序列最後一個時間步的hidden state
        return self.head(last).squeeze(-1)


def compute_mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練LSTM股價預測模型")
    parser.add_argument("--data-dir", type=str, default="lstm_data")
    parser.add_argument("--out-dir", type=str, default="lstm_model")
    parser.add_argument("--no-sentiment", action="store_true", help="消融實驗：不用情緒因子，只用技術指標")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="early stopping耐心值")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 使用裝置：{device} ===")

    d = np.load(os.path.join(args.data_dir, "lstm_dataset.npz"), allow_pickle=True)
    X_key = "X_without" if args.no_sentiment else "X_with"
    X = d[X_key]
    y = d["y_scaled"]
    y_raw = d["y_raw"]
    split = d["split"]
    stock_code = d["stock_code"]
    industry = d["industry"]

    print(f"特徵版本：{'不含情緒因子（基準對照組）' if args.no_sentiment else '含情緒因子'}，特徵維度={X.shape[-1]}")

    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    train_ds = SeqDataset(X[train_mask], y[train_mask])
    val_ds = SeqDataset(X[val_mask], y[val_mask])
    test_ds = SeqDataset(X[test_mask], y[test_mask])
    print(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4, shuffle=False)

    model = LSTMRegressor(X.shape[-1], args.hidden_size, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(yb)
        val_loss /= len(val_ds)

        print(f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}（val loss連續{args.patience}輪沒進步）")
                break

    model.load_state_dict(best_state)

    # 在test set上算MAPE（反正規化回真實股價）
    scalers = pd.read_csv(os.path.join(args.data_dir, "close_scalers.csv")).set_index("stock_code")

    model.eval()
    all_pred_scaled = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_pred_scaled.append(pred)
    pred_scaled = np.concatenate(all_pred_scaled)

    test_stock_code = stock_code[test_mask]
    test_industry = industry[test_mask]
    test_y_raw = y_raw[test_mask]

    close_min = scalers.loc[test_stock_code, "close_min"].values
    close_max = scalers.loc[test_stock_code, "close_max"].values
    pred_raw = pred_scaled * (close_max - close_min) + close_min

    overall_mape = compute_mape(test_y_raw, pred_raw)
    print(f"\n=== Test set 整體 MAPE: {overall_mape:.4f} ({overall_mape*100:.2f}%) ===")

    industry_mape = {}
    for ind in np.unique(test_industry):
        mask = test_industry == ind
        industry_mape[ind] = compute_mape(test_y_raw[mask], pred_raw[mask])
        print(f"  {ind}: MAPE={industry_mape[ind]:.4f} ({industry_mape[ind]*100:.2f}%)")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "use_sentiment": not args.no_sentiment,
            "overall_mape": overall_mape,
            "industry_mape": industry_mape,
            "best_val_loss": best_val_loss,
            "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 模型與評估結果已存到 {args.out_dir}/")
