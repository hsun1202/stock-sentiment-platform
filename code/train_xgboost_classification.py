# -*- coding: utf-8 -*-
"""
把train_xgboost_per_stock.py的迴歸任務（預測精確收盤價）改成分類任務
（預測隔天收盤價比今天漲還是跌），其他資料切分、特徵、模型設定都一樣，
只換目標變數跟評估指標（Accuracy/F1，不是MAPE）。

目的：驗證「情緒因子測不出效果」是不是因為迴歸任務對雜訊比較敏感——
如果換成分類（漲跌方向）情緒因子的效果會不會比較明顯。
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from prepare_lstm_dataset import TECH_FEATURES, build_stock_windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost漲跌方向分類（每檔股票各自訓練）")
    parser.add_argument("--input", type=str, default="final_dataset_267/all_stocks_final_dataset_267.csv")
    parser.add_argument("--out-dir", type=str, default="xgb_classification_with")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.dropna(subset=TECH_FEATURES).copy()

    results = []
    feature_importance_sum = None
    for i, ((code, name, industry), sub) in enumerate(df.groupby(["stock_code", "stock_name", "industry"]), 1):
        sub_sorted = sub.sort_values("Date").reset_index(drop=True)
        r = build_stock_windows(sub, args.window_size, args.train_frac, args.val_frac)
        if r is None:
            print(f"[{i}] {code} {name}：資料不足，跳過")
            continue

        X = r["X_without"] if args.no_sentiment else r["X_with"]
        X_flat = X.reshape(X.shape[0], -1)
        n_seq = X.shape[0]

        # 重建漲跌方向目標：y_raw[t] 對應 close_all[t+window_size]，
        # 「昨天」是 close_all[t+window_size-1]，兩者比較決定漲跌
        close_all = sub_sorted["Close"].values
        ws = args.window_size
        prev_close = close_all[ws - 1: ws - 1 + n_seq]
        next_close = close_all[ws: ws + n_seq]
        y_direction = (next_close > prev_close).astype(int)

        split = r["split"]
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"

        if len(np.unique(y_direction[train_mask])) < 2:
            print(f"[{i}] {code} {name}：訓練集只有單一類別，跳過")
            continue

        model = xgb.XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr,
            subsample=0.8, colsample_bytree=0.8, random_state=args.seed,
            early_stopping_rounds=20, eval_metric="logloss",
        )
        model.fit(X_flat[train_mask], y_direction[train_mask],
                  eval_set=[(X_flat[val_mask], y_direction[val_mask])], verbose=False)

        pred_test = model.predict(X_flat[test_mask])
        y_test = y_direction[test_mask]
        acc = accuracy_score(y_test, pred_test)
        f1 = f1_score(y_test, pred_test, average="macro", zero_division=0)

        results.append({"stock_code": code, "stock_name": name, "industry": industry,
                         "accuracy": acc, "macro_f1": f1, "n_test": int(test_mask.sum())})
        print(f"[{i}] {code} {name}（{industry}）：Accuracy={acc*100:.2f}%  Macro-F1={f1:.4f}")

        imp = model.feature_importances_
        if feature_importance_sum is None:
            feature_importance_sum = np.zeros_like(imp)
        feature_importance_sum += imp

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out_dir, "per_stock_accuracy.csv"), index=False, encoding="utf-8-sig")

    overall_acc = results_df["accuracy"].mean()
    overall_f1 = results_df["macro_f1"].mean()
    industry_acc = results_df.groupby("industry")["accuracy"].mean().to_dict()

    print(f"\n=== 整體 Accuracy（簡單平均）：{overall_acc*100:.2f}% ===")
    print(f"=== 整體 Macro-F1（簡單平均）：{overall_f1:.4f} ===")

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
            "overall_accuracy": overall_acc,
            "overall_macro_f1": overall_f1,
            "industry_accuracy": industry_acc,
            "feature_importance": importance_df.to_dict(),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 結果已存到 {args.out_dir}/")
