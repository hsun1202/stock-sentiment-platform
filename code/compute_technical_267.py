# -*- coding: utf-8 -*-
"""
用 stocks_267/stocks/{code}/price/yahoo_{code}_{year}.csv 的本機股價資料
（組員提供，不用打yfinance API），套用 stock_indicators.py 既有的16項技術指標
公式，輸出跟 technical_output/ 同樣格式的 {code}_{name}_technical.csv。
"""
import os
import glob
import argparse
import pandas as pd

from stock_indicators import calculate_technical_indicators, normalize_indicators
from stock_list_267 import STOCK_LIST_267

INDICATOR_COLS = [
    "MA5", "MA30", "MA60", "EMA5", "EMA30", "EMA60",
    "MACD_6_15_6", "MACD_12_26_9", "MACD_30_60_30",
    "RSI14", "WILLR14", "MOM14", "CMO14", "ULTOSC_7_14_28",
    "OBV", "ADOSC_3_10",
]


def load_local_price(code, price_dir):
    files = sorted(glob.glob(os.path.join(price_dir, f"yahoo_{code}_*.csv")))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").set_index("date")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks-dir", default="stocks_267/stocks")
    parser.add_argument("--out-dir", default="technical_output_267")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ok, fail, no_price = 0, 0, 0
    for code, name, industry in STOCK_LIST_267:
        price_dir = os.path.join(args.stocks_dir, code, "price")
        df = load_local_price(code, price_dir)
        if df is None or len(df) < 60:  # 至少要有60天資料才能算MA60這種指標
            no_price += 1
            continue
        try:
            tech = calculate_technical_indicators(df)
            tech = normalize_indicators(tech, INDICATOR_COLS)
            tech = tech.reset_index().rename(columns={"date": "Date"})
            out_path = os.path.join(args.out_dir, f"{code}_{name}_technical.csv")
            tech.to_csv(out_path, index=False, encoding="utf-8-sig")
            ok += 1
        except Exception as e:
            print(f"[失敗] {code}{name}: {e}")
            fail += 1

    print(f"\n[完成] 成功 {ok}，資料不足跳過 {no_price}，計算失敗 {fail}，共 {len(STOCK_LIST_267)} 檔")


if __name__ == "__main__":
    main()
