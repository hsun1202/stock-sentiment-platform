# -*- coding: utf-8 -*-
"""
照論文（Jing, Wu & Wang 2021）：每檔各自訓練 LSTM 預測隔天收盤價、看 MAPE —— 換成 CMoney 貼文情緒，並與 PTT 在同一段期間比較。
設定、模型、切法全部沿用 lstm_price_v4.py（直接 import），只改：資料裁成 2020-2024（CMoney 只有這段）、多一組 +CMoney。
事先寫定的假說見 cmoney_full/PREREG_cmoney_full.md。
用法（在 D:/專題claude 執行）：python cmoney_full/lstm_price_cm.py --arm +CMoney --seed 0 --out-dir cmoney_full/lstm/+CMoney_seed0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
import lstm_price_v4 as L  # noqa: E402

PTT, CM = ["ptt_fac", "ptt_att"], ["cm_fac", "cm_att"]
CM_CLOSE = ["cmc_fac", "cmc_att"]   # 事後追加：只用 t 收盤前的 CMoney 貼文（避開「收盤前最後 20 篇」抽樣造成的 t+1 開盤前洩漏，見 xgb_leak_check.py）
ARMS = {"技術面": L.TECH, "+PTT": L.TECH + PTT, "+CMoney": L.TECH + CM, "+PTT+CMoney": L.TECH + PTT + CM,
        "+CMoney收盤前": L.TECH + CM_CLOSE}
START = "2020-01-01"


def load_both():
    df = L.load()
    cm = L.load("final_dataset_v3_cm2/all_stocks_v3.csv")
    assert (df.stock_code.values == cm.stock_code.values).all() and (df.Date.values == cm.Date.values).all()
    df["cm_fac"], df["cm_att"] = cm.ptt_fac.values, cm.ptt_att.values
    n = cm.ptt_pre_n + cm.ptt_intra_n
    p, q = cm.ptt_pre_pos + cm.ptt_intra_pos, cm.ptt_pre_neg + cm.ptt_intra_neg
    df["cmc_fac"] = np.where(p + q > 0, (p - q) / (p + q).replace(0, 1), 0.0)
    df["cmc_att"] = np.log1p(n).values
    return df[df.Date >= START].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-codes", default=None)
    a = ap.parse_args()
    hp = argparse.Namespace(hidden=20, layers=5, dropout=0.2, lr=0.01, batch=10, epochs=100, patience=15)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = os.environ.get("LSTM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("LSTM_THREADS", "2")))
    df = load_both()
    if a.only_codes:
        df = df[df.stock_code.isin(a.only_codes.split(","))]
    rows = []
    for code, sub in df.groupby("stock_code"):
        w = L.windows(sub, ARMS[a.arm], 10, 0.75, 0.10)
        if w is None:
            continue
        X, y, split, actual, prev, cmin, crng, dates = w
        te = split == "test"
        pred = L.train_one(X, y, split, hp, dev) * crng + cmin
        act, pv = actual[te], prev[te]
        rows.append({"stock_code": code, "n_test": int(te.sum()), "test_start": str(dates[te][0]),
                     "mape": float(np.mean(np.abs((act - pred) / act))),
                     "naive_mape": float(np.mean(np.abs((act - pv) / act))),
                     "dir_acc": float(np.mean(np.sign(pred - pv) == np.sign(act - pv)))})
        print(f"{code}: MAPE {rows[-1]['mape']:.2%}（天真 {rows[-1]['naive_mape']:.2%}）方向 {rows[-1]['dir_acc']:.1%}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(a.out_dir, "per_stock.csv"), index=False, encoding="utf-8-sig")
    json.dump({"arm": a.arm, "seed": a.seed, "n_stocks": len(r), "mape": float(r.mape.mean()),
               "naive_mape": float(r.naive_mape.mean()), "dir_acc": float(r.dir_acc.mean())},
              open(os.path.join(a.out_dir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[完成] {a.arm} seed{a.seed}：{len(r)} 檔，MAPE {r.mape.mean():.2%}、天真 {r.naive_mape.mean():.2%}、方向 {r.dir_acc.mean():.1%}")
