# -*- coding: utf-8 -*-
"""
十年資料（final_dataset_v3/all_stocks_v3.csv）的特徵與預測目標，給 lstm_direction_v4.py 共用。
特徵定義與 direction_test_v4.py（XGBoost 版）完全相同，兩個模型的比較才公平。
"""
import numpy as np
import pandas as pd


def build(path="final_dataset_v3/all_stocks_v3.csv", retail=False):
    df = pd.read_csv(path, dtype={"stock_code": str})
    df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
    g = df.groupby("stock_code")
    C = df["Close"]

    def lag(s, k):
        return s.groupby(df["stock_code"]).shift(k) if k else s

    def roll(s, w):
        return s.groupby(df["stock_code"]).transform(lambda x: x.rolling(w, min_periods=1).mean())

    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = np.log(C / g["Close"].shift(1))
    for k in (1, 2, 3, 4):
        feat[f"ret1_l{k}"] = lag(feat["ret1"], k)
    feat["ret5"] = np.log(C / g["Close"].shift(5))
    feat["ret10"] = np.log(C / g["Close"].shift(10))
    feat["gap"] = df["Open"] / g["Close"].shift(1) - 1
    feat["range"] = (df["High"] - df["Low"]) / C
    vol20 = g["Volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    feat["volr"] = np.log((df["Volume"] + 1) / (vol20 + 1))
    for c in ("MA5", "MA30", "MA60", "EMA5", "EMA30", "EMA60"):
        feat[f"{c}_r"] = df[c] / C - 1
    for c in ("MACD_6_15_6", "MACD_12_26_9", "MACD_30_60_30"):
        feat[f"{c}_r"] = df[c] / C
    feat["MOM14_r"] = df["MOM14"] / C
    feat["ADOSC_r"] = df["ADOSC_3_10"] / (vol20 + 1)
    for c in ("RSI14", "WILLR14", "CMO14", "ULTOSC_7_14_28"):
        feat[c] = df[c]
    TECH = list(feat.columns)

    def ptt_feats(n, pos, neg, tag):
        feat[f"{tag}_fac"] = np.where(pos + neg > 0, (pos - neg) / (pos + neg).replace(0, 1), np.nan)
        feat[f"{tag}_att"] = np.log1p(n)
        feat[f"{tag}_att5"] = roll(np.log1p(n), 5)
        return [f"{tag}_fac", f"{tag}_att", f"{tag}_att5"]

    n_c = df.ptt_pre_n + df.ptt_intra_n
    p_c = df.ptt_pre_pos + df.ptt_intra_pos
    q_c = df.ptt_pre_neg + df.ptt_intra_neg
    PTT_OPEN = ptt_feats(n_c + lag(df.ptt_pre_n, -1).fillna(0), p_c + lag(df.ptt_pre_pos, -1).fillna(0),
                         q_c + lag(df.ptt_pre_neg, -1).fillna(0), "ptt_open")   # t+1 開盤前可得

    CHIP = []
    for c in ("foreign", "trust", "dealer", "inst"):
        r = df[f"{c}_ratio"].clip(-1, 1)
        feat[f"{c}_r1"] = r
        feat[f"{c}_r5"] = roll(r, 5)
        CHIP += [f"{c}_r1", f"{c}_r5"]
    feat["margin_chg"] = df["margin_chg"].clip(-0.5, 0.5)
    feat["margin_chg5"] = roll(feat["margin_chg"], 5)
    feat["short_chg"] = df["short_chg"].clip(-1, 3)
    feat["short_margin_ratio"] = df["short_margin_ratio"].clip(0, 1)
    feat["margin_eligible"] = df["margin_eligible"]
    CHIP += ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]

    # 目標 B：隔天開盤→收盤（可交易），可用 t+1 開盤前的資訊（技術面 t、PTT 到 t+1 開盤前、t 盤後公布的籌碼）
    nextO, nextC = g["Open"].shift(-1), g["Close"].shift(-1)
    abn = df["abnormal_price"]
    valid = nextC.notna() & nextO.notna() & (nextC != nextO) & (lag(abn, -1).fillna(0) == 0) & (abn == 0) \
        & feat[TECH].notna().all(axis=1)
    y = (nextC > nextO).astype(int)
    meta = pd.DataFrame({"stock_code": df["stock_code"], "date": pd.to_datetime(df["Date"]),
                         "end_date": pd.to_datetime(g["Date"].shift(-1)), "valid": valid, "y": y})
    MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
    if retail:  # 散戶情緒版：散戶說的(PTT) vs 散戶做的(融資融券)，不放法人
        return feat, meta, {"技術面": TECH, "+PTT(散戶說)": TECH + PTT_OPEN,
                            "+融資融券(散戶做)": TECH + MARGIN, "+PTT+融資融券": TECH + PTT_OPEN + MARGIN}
    return feat, meta, {"技術面": TECH, "+PTT(開盤前可得)": TECH + PTT_OPEN,
                        "+籌碼(當天盤後公布)": TECH + CHIP, "+PTT+籌碼": TECH + PTT_OPEN + CHIP}
