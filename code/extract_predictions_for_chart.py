# -*- coding: utf-8 -*-
"""為PPT畫「實際vs預測收盤價」折線圖，重跑指定股票、seed=42，抓出test集的日期+實際價+預測價"""
import json
import numpy as np
import pandas as pd
import torch

from prepare_lstm_dataset import TECH_FEATURES, SENTIMENT_FEATURE, build_stock_windows
from train_lstm import LSTMRegressor, compute_mape
from train_lstm_per_stock import train_one_stock

SEED = 42
STOCKS = [("5522", "遠雄"), ("2330", "台積電")]

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("final_dataset_breeze/all_stocks_final_dataset.csv")
df = df.dropna(subset=TECH_FEATURES).copy()

out = {}
for code, name in STOCKS:
    sub = df[df["stock_code"].astype(str) == code].sort_values("Date").reset_index(drop=True)
    r = build_stock_windows(sub, window_size=10, train_frac=0.75, val_frac=0.10)
    pred_scaled, val_loss = train_one_stock(
        r["X_with"], r["y_scaled"], r["split"],
        hidden_size=20, num_layers=5, dropout=0.2, lr=0.01, batch_size=10,
        epochs=100, patience=15, device=device,
    )
    test_mask = r["split"] == "test"
    y_test_raw = r["y_raw"][test_mask]
    pred_raw = pred_scaled * (r["close_max"] - r["close_min"]) + r["close_min"]
    mape = compute_mape(y_test_raw, pred_raw)

    # 重建對應的日期：每個窗口t對應原始df第 window_size+t 列
    dates_all = sub["Date"].values
    window_size = 10
    n_seq = len(sub) - window_size
    test_idx = [t for t in range(n_seq) if r["split"][t] == "test"]
    test_dates = [str(dates_all[window_size + t]) for t in test_idx]

    out[code] = {
        "name": name,
        "mape": float(mape),
        "dates": test_dates,
        "actual": [float(x) for x in y_test_raw],
        "predicted": [float(x) for x in pred_raw],
    }
    print(f"{code} {name}: MAPE={mape*100:.2f}%  test筆數={len(test_dates)}")

with open("ppt_build/pred_chart_data.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("done")
