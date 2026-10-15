# -*- coding: utf-8 -*-
"""
LSTM股價預測（比照原論文：每檔股票各自訓練一個模型）
============================================================

目的：
    跟train_lstm.py（池化版）對照——這支腳本比照Jing et al. (2021)
    原論文的方法論，30檔股票各自獨立訓練一個LSTM，用原論文
    Table 6報告的最佳超參數（5層、20個hidden units、lr=0.01、
    epochs最多100、batch=10），驗證跟池化版本比起來結果如何。

    同樣支援 --no-sentiment 做消融實驗。

使用方式：
    python train_lstm_per_stock.py --out-dir lstm_perstock_with_sentiment
    python train_lstm_per_stock.py --no-sentiment --out-dir lstm_perstock_without_sentiment
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from prepare_lstm_dataset import TECH_FEATURES, SENTIMENT_FEATURE, build_stock_windows
from train_lstm import LSTMRegressor, compute_mape


def train_one_stock(X, y_scaled, split, hidden_size, num_layers, dropout, lr, batch_size, epochs, patience, device):
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    X_train = torch.tensor(X[train_mask], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_scaled[train_mask], dtype=torch.float32).to(device)
    X_val = torch.tensor(X[val_mask], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_scaled[val_mask], dtype=torch.float32).to(device)
    X_test = torch.tensor(X[test_mask], dtype=torch.float32).to(device)

    model = LSTMRegressor(X.shape[-1], hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(y_train)
    best_val_loss = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model(X_train[idx])
            loss = criterion(pred, y_train[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_test).cpu().numpy()
    return test_pred_scaled, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM股價預測（每檔股票各自訓練，比照原論文方法）")
    parser.add_argument("--input", type=str, default="final_dataset_breeze/all_stocks_final_dataset.csv")
    parser.add_argument("--out-dir", type=str, default="lstm_perstock_with_sentiment")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.10)
    # 原論文Table 6報告的最佳超參數
    parser.add_argument("--hidden-size", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-predictions", action="store_true",
                         help="額外存下每檔股票test集的逐日實際/預測價格，供多seed ensemble平均用")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 使用裝置：{device} ===")
    print(f"特徵版本：{'不含情緒因子（基準對照組）' if args.no_sentiment else '含情緒因子'}")
    print(f"超參數（比照原論文）：{args.num_layers}層、{args.hidden_size}個hidden units、lr={args.lr}、batch={args.batch_size}\n")

    df = pd.read_csv(args.input)
    df = df.dropna(subset=TECH_FEATURES).copy()

    results = []
    for i, ((code, name, industry), sub) in enumerate(df.groupby(["stock_code", "stock_name", "industry"]), 1):
        r = build_stock_windows(sub, args.window_size, args.train_frac, args.val_frac)
        if r is None:
            print(f"[{i}] {code} {name}：資料不足，跳過")
            continue

        X = r["X_without"] if args.no_sentiment else r["X_with"]
        pred_scaled, val_loss = train_one_stock(
            X, r["y_scaled"], r["split"], args.hidden_size, args.num_layers, args.dropout,
            args.lr, args.batch_size, args.epochs, args.patience, device,
        )

        test_mask = r["split"] == "test"
        y_test_raw = r["y_raw"][test_mask]
        pred_raw = pred_scaled * (r["close_max"] - r["close_min"]) + r["close_min"]
        mape = compute_mape(y_test_raw, pred_raw)

        results.append({"stock_code": code, "stock_name": name, "industry": industry,
                         "mape": mape, "n_test": int(test_mask.sum()), "best_val_loss": val_loss})
        print(f"[{i}/30] {code} {name}（{industry}）：MAPE={mape*100:.2f}%")

        if args.save_predictions:
            pred_dir = os.path.join(args.out_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            pred_df = pd.DataFrame({
                "test_idx": range(len(y_test_raw)),
                "actual": y_test_raw,
                "predicted": pred_raw,
            })
            pred_df.to_csv(os.path.join(pred_dir, f"{code}_{name}_predictions.csv"),
                            index=False, encoding="utf-8-sig")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out_dir, "per_stock_mape.csv"), index=False, encoding="utf-8-sig")

    overall_mape_simple = results_df["mape"].mean()
    overall_mape_weighted = np.average(results_df["mape"], weights=results_df["n_test"])
    industry_mape = results_df.groupby("industry")["mape"].mean().to_dict()

    print(f"\n=== 整體 MAPE（30檔平均）：{overall_mape_simple*100:.2f}% ===")
    print(f"=== 整體 MAPE（依test筆數加權）：{overall_mape_weighted*100:.2f}% ===")
    for ind, m in industry_mape.items():
        print(f"  {ind}: {m*100:.2f}%")

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "use_sentiment": not args.no_sentiment,
            "overall_mape": overall_mape_simple,
            "overall_mape_weighted": overall_mape_weighted,
            "industry_mape": industry_mape,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 結果已存到 {args.out_dir}/")
