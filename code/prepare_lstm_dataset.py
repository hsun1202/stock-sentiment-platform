# -*- coding: utf-8 -*-
"""
LSTM訓練資料準備（技術指標 + 情緒因子 -> 下一日收盤價）
============================================================

目的：
    把final_dataset_breeze（或詞典版/BERT3版，供方法比較用）的
    每日技術指標+情緒因子，逐股票切成時間序列窗口，池化成單一
    訓練集，供train_lstm.py訓練。

設計重點：
    - 依時間順序切分train/val/test（不是隨機切），避免用未來
      資料預測過去造成的資料洩漏
    - 收盤價（預測目標）逐股票用「訓練集自己的」min/max做正規化，
      不用到val/test的統計量，正規化參數會存起來，供之後反正規化
      算真實股價MAPE用
    - 序列窗口不可跨股票邊界（每檔股票各自產生窗口後才池化）
    - 同時輸出「含情緒因子」（17特徵）與「不含情緒因子」（16特徵）
      兩組資料，供train_lstm.py做消融實驗比較

已知限制：
    16項技術指標的_scaled欄位是stock_indicators.py用「整檔股票
    全部599天」的min/max算的，不是只用訓練集算的，嚴格來說對
    test期間有輕微資訊洩漏。這是既有資料集本身的特性，不是這支
    程式新引入的問題；只有本程式新增的「收盤價目標」正規化有
    嚴格比照train-only原則。這點會在報告的限制章節說明。

使用方式：
    python prepare_lstm_dataset.py --input final_dataset_breeze/all_stocks_final_dataset.csv --out-dir lstm_data
"""

import os
import argparse
import numpy as np
import pandas as pd

TECH_FEATURES = [
    "MA5_scaled", "MA30_scaled", "MA60_scaled",
    "EMA5_scaled", "EMA30_scaled", "EMA60_scaled",
    "MACD_6_15_6_scaled", "MACD_12_26_9_scaled", "MACD_30_60_30_scaled",
    "RSI14_scaled", "WILLR14_scaled", "MOM14_scaled", "CMO14_scaled",
    "ULTOSC_7_14_28_scaled", "OBV_scaled", "ADOSC_3_10_scaled",
]
SENTIMENT_FEATURE = "daily_sentiment_factor"


def build_stock_windows(stock_df, window_size, train_frac, val_frac):
    """單一股票：正規化收盤價、切窗口、依時間分train/val/test。"""
    stock_df = stock_df.sort_values("Date").reset_index(drop=True)
    n = len(stock_df)
    n_seq = n - window_size  # 每個窗口對應一個「下一日」目標
    if n_seq <= 0:
        return None

    # 先算每個窗口目標（第window_size天之後）落在train/val/test的哪一段
    train_end = int(n_seq * train_frac)
    val_end = train_end + int(n_seq * val_frac)

    # 收盤價正規化：只用train部分的min/max
    train_close = stock_df["Close"].values[window_size:window_size + train_end]
    close_min, close_max = float(train_close.min()), float(train_close.max())
    close_range = (close_max - close_min) if close_max > close_min else 1.0

    close_all = stock_df["Close"].values
    close_scaled = (close_all - close_min) / close_range

    feat_with = stock_df[TECH_FEATURES + [SENTIMENT_FEATURE]].values.astype(np.float32)
    feat_without = stock_df[TECH_FEATURES].values.astype(np.float32)

    X_with, X_without, y_scaled, y_raw, split_idx = [], [], [], [], []
    for t in range(n_seq):
        X_with.append(feat_with[t:t + window_size])
        X_without.append(feat_without[t:t + window_size])
        y_scaled.append(close_scaled[t + window_size])
        y_raw.append(close_all[t + window_size])
        if t < train_end:
            split_idx.append("train")
        elif t < val_end:
            split_idx.append("val")
        else:
            split_idx.append("test")

    return {
        "X_with": np.array(X_with, dtype=np.float32),
        "X_without": np.array(X_without, dtype=np.float32),
        "y_scaled": np.array(y_scaled, dtype=np.float32),
        "y_raw": np.array(y_raw, dtype=np.float32),
        "split": np.array(split_idx),
        "close_min": close_min,
        "close_max": close_max,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="準備LSTM訓練資料（技術指標+情緒因子->下一日收盤價）")
    parser.add_argument("--input", type=str, default="final_dataset_breeze/all_stocks_final_dataset.csv")
    parser.add_argument("--out-dir", type=str, default="lstm_data")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    before = len(df)
    df = df.dropna(subset=TECH_FEATURES).copy()
    print(f"讀取 {before} 列，丟掉技術指標暖身期NaN後剩 {len(df)} 列")

    all_X_with, all_X_without, all_y_scaled, all_y_raw = [], [], [], []
    all_split, all_stock_code, all_industry = [], [], []
    scaler_records = []

    for (code, name, industry), sub in df.groupby(["stock_code", "stock_name", "industry"]):
        result = build_stock_windows(sub, args.window_size, args.train_frac, args.val_frac)
        if result is None:
            print(f"跳過 {code} {name}：資料量不足以切出窗口")
            continue

        n = len(result["y_scaled"])
        all_X_with.append(result["X_with"])
        all_X_without.append(result["X_without"])
        all_y_scaled.append(result["y_scaled"])
        all_y_raw.append(result["y_raw"])
        all_split.append(result["split"])
        all_stock_code.extend([code] * n)
        all_industry.extend([industry] * n)
        scaler_records.append({
            "stock_code": code, "stock_name": name, "industry": industry,
            "close_min": result["close_min"], "close_max": result["close_max"],
        })

    X_with = np.concatenate(all_X_with, axis=0)
    X_without = np.concatenate(all_X_without, axis=0)
    y_scaled = np.concatenate(all_y_scaled, axis=0)
    y_raw = np.concatenate(all_y_raw, axis=0)
    split = np.concatenate(all_split, axis=0)
    stock_code_arr = np.array(all_stock_code)
    industry_arr = np.array(all_industry)

    print(f"\n總序列數：{len(y_scaled)}（含情緒特徵維度={X_with.shape[-1]}，不含情緒={X_without.shape[-1]}）")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {(split == s).sum()} 筆")

    np.savez(
        os.path.join(args.out_dir, "lstm_dataset.npz"),
        X_with=X_with, X_without=X_without, y_scaled=y_scaled, y_raw=y_raw,
        split=split, stock_code=stock_code_arr, industry=industry_arr,
    )
    pd.DataFrame(scaler_records).to_csv(
        os.path.join(args.out_dir, "close_scalers.csv"), index=False, encoding="utf-8-sig"
    )

    print(f"\n[完成] 已輸出至 {args.out_dir}/lstm_dataset.npz 與 close_scalers.csv")
