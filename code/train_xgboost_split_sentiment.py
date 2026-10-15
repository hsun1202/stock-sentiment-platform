# -*- coding: utf-8 -*-
"""
測試「拆開正負面比例」這個特徵設計，比照Ko & Chang (2021)的做法——
他們用「正面留言比例」「負面留言比例」兩個獨立特徵，不是像我們現在
的daily_sentiment_factor把正負合併成一個淨值。合併後「正負都很多、
很兩極」跟「根本沒人討論」可能都接近0，資訊被抵銷掉；拆開來看看
是不是能讓情緒訊號更明顯。

三種特徵版本比較：
    baseline_no_sentiment : 只有16項技術指標
    combined              : 16項技術指標 + daily_sentiment_factor（現在的做法）
    split                 : 16項技術指標 + positive_ratio + negative_ratio（拆開）
"""
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from prepare_lstm_dataset import TECH_FEATURES


def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def build_windows(sub, feature_cols, window_size, train_frac, val_frac):
    sub = sub.sort_values("Date").reset_index(drop=True)
    n = len(sub)
    n_seq = n - window_size
    if n_seq <= 0:
        return None
    train_end = int(n_seq * train_frac)
    val_end = train_end + int(n_seq * val_frac)

    train_close = sub["Close"].values[window_size:window_size + train_end]
    close_min, close_max = float(train_close.min()), float(train_close.max())
    close_range = (close_max - close_min) if close_max > close_min else 1.0
    close_all = sub["Close"].values
    close_scaled = (close_all - close_min) / close_range

    feat = sub[feature_cols].values.astype(np.float32)
    X, y_scaled, y_raw, split = [], [], [], []
    for t in range(n_seq):
        X.append(feat[t:t + window_size].flatten())
        y_scaled.append(close_scaled[t + window_size])
        y_raw.append(close_all[t + window_size])
        split.append("train" if t < train_end else ("val" if t < val_end else "test"))
    return {
        "X": np.array(X), "y_scaled": np.array(y_scaled), "y_raw": np.array(y_raw),
        "split": np.array(split), "close_min": close_min, "close_max": close_max,
    }


def run_variant(df, feature_cols, label, window_size=10, train_frac=0.75, val_frac=0.10):
    results = []
    importance_sum = None
    for (code, name, industry), sub in df.groupby(["stock_code", "stock_name", "industry"]):
        r = build_windows(sub, feature_cols, window_size, train_frac, val_frac)
        if r is None:
            continue
        split = r["split"]
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=20, eval_metric="rmse",
        )
        model.fit(r["X"][train_mask], r["y_scaled"][train_mask],
                  eval_set=[(r["X"][val_mask], r["y_scaled"][val_mask])], verbose=False)
        pred_scaled = model.predict(r["X"][test_mask])
        pred_raw = pred_scaled * (r["close_max"] - r["close_min"]) + r["close_min"]
        mape = compute_mape(r["y_raw"][test_mask], pred_raw)
        results.append({"stock_code": code, "mape": mape, "n_test": int(test_mask.sum())})

        imp = model.feature_importances_
        if importance_sum is None:
            importance_sum = np.zeros_like(imp)
        importance_sum += imp

    results_df = pd.DataFrame(results)
    overall = results_df["mape"].mean()
    avg_imp = importance_sum / len(results)
    feat_names = feature_cols * window_size
    imp_df = pd.DataFrame({"feature": feat_names, "importance": avg_imp})
    imp_df = imp_df.groupby("feature")["importance"].sum().sort_values(ascending=False)

    print(f"\n=== {label} ===")
    print(f"股票數: {len(results)}  整體MAPE: {overall*100:.2f}%")
    print("特徵重要性排名:")
    for feat, imp in imp_df.items():
        marker = " <==" if feat in ("positive_ratio", "negative_ratio", "daily_sentiment_factor") else ""
        print(f"  {feat}: {imp:.4f}{marker}")
    return overall, imp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="final_dataset_267/all_stocks_final_dataset_267.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=TECH_FEATURES).copy()
    df["positive_ratio"] = np.where(df["total_comments"] > 0, df["positive_count"] / df["total_comments"], 0.0)
    df["negative_ratio"] = np.where(df["total_comments"] > 0, df["negative_count"] / df["total_comments"], 0.0)
    df["log_volume"] = np.log1p(df["total_comments"])
    df["extremity"] = df["positive_ratio"] + df["negative_ratio"]

    multi_features = ["daily_sentiment_factor", "positive_ratio", "negative_ratio", "log_volume", "extremity"]

    mape_none, _ = run_variant(df, TECH_FEATURES, "不含情緒（基準）")
    mape_combined, imp_combined = run_variant(df, TECH_FEATURES + ["daily_sentiment_factor"], "合併淨值分數（現行做法）")
    mape_split, imp_split = run_variant(df, TECH_FEATURES + ["positive_ratio", "negative_ratio"], "拆開正負面比例（比照Ko & Chang）")
    mape_multi, imp_multi = run_variant(df, TECH_FEATURES + multi_features, "多維情緒特徵（淨值+正負比例+討論量+極化度）")

    print("\n\n=== 總結對照 ===")
    print(f"不含情緒:         {mape_none*100:.2f}%")
    print(f"合併淨值分數:     {mape_combined*100:.2f}%  (差異 {(mape_combined-mape_none)*100:+.3f}pp)")
    print(f"拆開正負面比例:   {mape_split*100:.2f}%  (差異 {(mape_split-mape_none)*100:+.3f}pp)")
    print(f"多維情緒特徵:     {mape_multi*100:.2f}%  (差異 {(mape_multi-mape_none)*100:+.3f}pp)")
