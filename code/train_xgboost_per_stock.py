# -*- coding: utf-8 -*-
"""
用XGBoost（樹模型，沒有神經網路那種訓練隨機性雜訊）重跑跟LSTM同樣的
「每檔各自訓練」實驗，驗證情緒因子效果測不出來是不是LSTM架構本身的問題。

跟train_lstm_per_stock.py共用同一套資料切分邏輯（build_stock_windows），
只是把10天窗口攤平成一維特徵向量（而不是序列），符合樹模型的輸入格式。
樹模型給定同樣資料/參數，訓練結果幾乎完全確定，不需要多seed穩健性檢驗。

使用方式：
    python train_xgboost_per_stock.py --out-dir xgb_perstock_with
    python train_xgboost_per_stock.py --no-sentiment --out-dir xgb_perstock_without
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from prepare_lstm_dataset import TECH_FEATURES, build_stock_windows
from train_lstm import compute_mape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost股價預測（每檔股票各自訓練，跟LSTM同樣切分）")
    parser.add_argument("--input", type=str, default="final_dataset_267/all_stocks_final_dataset_267.csv")
    parser.add_argument("--out-dir", type=str, default="xgb_perstock_with")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.dropna(subset=TECH_FEATURES).copy()

    results = []
    feature_importance_sum = None
    for i, ((code, name, industry), sub) in enumerate(df.groupby(["stock_code", "stock_name", "industry"]), 1):
        r = build_stock_windows(sub, args.window_size, args.train_frac, args.val_frac)
        if r is None:
            print(f"[{i}] {code} {name}：資料不足，跳過")
            continue

        X = r["X_without"] if args.no_sentiment else r["X_with"]
        n_feat = X.shape[-1]
        X_flat = X.reshape(X.shape[0], -1)  # (n_seq, window_size*n_feat) 攤平成一維特徵向量

        split = r["split"]
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"

        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=20, eval_metric="rmse",
        )
        model.fit(X_flat[train_mask], r["y_scaled"][train_mask],
                  eval_set=[(X_flat[val_mask], r["y_scaled"][val_mask])], verbose=False)

        pred_scaled = model.predict(X_flat[test_mask])
        y_test_raw = r["y_raw"][test_mask]
        pred_raw = pred_scaled * (r["close_max"] - r["close_min"]) + r["close_min"]
        mape = compute_mape(y_test_raw, pred_raw)

        results.append({"stock_code": code, "stock_name": name, "industry": industry,
                         "mape": mape, "n_test": int(test_mask.sum())})
        print(f"[{i}] {code} {name}（{industry}）：MAPE={mape*100:.2f}%")

        imp = model.feature_importances_
        if feature_importance_sum is None:
            feature_importance_sum = np.zeros_like(imp)
        feature_importance_sum += imp

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out_dir, "per_stock_mape.csv"), index=False, encoding="utf-8-sig")

    overall_mape_simple = results_df["mape"].mean()
    overall_mape_weighted = np.average(results_df["mape"], weights=results_df["n_test"])
    industry_mape = results_df.groupby("industry")["mape"].mean().to_dict()

    print(f"\n=== 整體 MAPE（簡單平均）：{overall_mape_simple*100:.2f}% ===")
    print(f"=== 整體 MAPE（加權）：{overall_mape_weighted*100:.2f}% ===")

    # 情緒因子在特徵重要性排名（情緒因子是視窗內最後一個特徵、每個時間步的最後一維）
    feat_names = (TECH_FEATURES + (["sentiment"] if not args.no_sentiment else [])) * args.window_size
    avg_importance = feature_importance_sum / len(results)
    importance_df = pd.DataFrame({"feature": feat_names, "importance": avg_importance})
    importance_df = importance_df.groupby("feature")["importance"].sum().sort_values(ascending=False)
    print("\n=== 平均特徵重要性排名 ===")
    for feat, imp in importance_df.items():
        print(f"  {feat}: {imp:.4f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "use_sentiment": not args.no_sentiment,
            "overall_mape": overall_mape_simple,
            "overall_mape_weighted": overall_mape_weighted,
            "industry_mape": industry_mape,
            "feature_importance": importance_df.to_dict(),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 結果已存到 {args.out_dir}/")
