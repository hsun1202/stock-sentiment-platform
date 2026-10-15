# -*- coding: utf-8 -*-
"""
台股技術指標抓取與計算
============================================================

目的：
    對應 Jing et al. (2021) 原論文 Table 5 選用的 16 個技術指標
    （趨勢型/動能型/成交量型），改用台股資料計算，
    並與 PTT 情緒因子按日期合併，組成可餵給 LSTM 的最終資料集。

使用方式：
    # 只抓股價+算指標
    python stock_indicators.py --stock 2330 --start 2026-06-01 --end 2026-07-30 --out 2330_technical.csv

    # 抓股價+算指標，並直接跟情緒分析結果合併
    python stock_indicators.py --stock 2330 --start 2026-06-01 --end 2026-07-30 \
        --sentiment 2330_sentiment_result.csv --out 2330_merged_dataset.csv

注意事項：
    yfinance 需要網路連線至 Yahoo Finance，請在有網路的環境執行
    （本機、非本沙盒環境）。台股代碼格式為 "{代碼}.TW"（上市）或
    "{代碼}.TWO"（上櫃），程式會先試 .TW，失敗再試 .TWO。
"""

import pandas as pd
import numpy as np
import argparse
import sys

try:
    import yfinance as yf
except ImportError:
    print("[錯誤] 缺少 yfinance 套件，請先執行：pip install yfinance")
    sys.exit(1)

# 註：原本使用 pandas_ta 套件計算指標，但該套件新版本要求 Python 3.12+，
# 與許多人常用的 Python 3.10 環境不相容。這裡改用 pandas/numpy 手刻公式，
# 不依賴任何額外套件，也讓每個指標的計算邏輯更透明、方便寫進報告說明。


def fetch_price_data(stock_code, start, end):
    """
    抓取台股歷史股價。
    先嘗試上市代碼格式 {code}.TW，失敗或抓不到資料再嘗試上櫃格式 {code}.TWO。
    """
    for suffix in [".TW", ".TWO"]:
        ticker = f"{stock_code}{suffix}"
        print(f"[嘗試] 抓取 {ticker} 的歷史股價...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is not None and not df.empty:
            print(f"[成功] 使用 {ticker}，共 {len(df)} 筆交易日資料")
            # yfinance 新版本回傳 MultiIndex 欄位，這裡攤平方便後續處理
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    print(f"[失敗] 無法取得股票代碼 {stock_code} 的資料，請確認代碼是否正確")
    return None


# ============================================================
# 以下為手刻技術指標公式（取代 pandas_ta），公式來源對應
# investopedia.com 標準定義，跟原論文 Jing et al. (2021) 一致
# ============================================================

def sma(series, length):
    """簡單移動平均 (Simple Moving Average)"""
    return series.rolling(window=length).mean()


def ema(series, length):
    """指數移動平均 (Exponential Moving Average)"""
    return series.ewm(span=length, adjust=False).mean()


def macd_line(series, fast, slow):
    """MACD 線 = 快線EMA - 慢線EMA（只取MACD主線，對應原論文只用MACD值）"""
    return ema(series, fast) - ema(series, slow)


def rsi(series, length=14):
    """相對強弱指標 (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def williams_r(high, low, close, length=14):
    """威廉指標 (Williams %R)"""
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def momentum(series, length=14):
    """動量指標 (Momentum)：今天收盤價 - N天前收盤價"""
    return series.diff(length)


def cmo(series, length=14):
    """錢德動量擺盪指標 (Chande Momentum Oscillator)"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=length).sum()
    loss = -delta.clip(upper=0).rolling(window=length).sum()
    return 100 * (gain - loss) / (gain + loss).replace(0, np.nan)


def ultimate_oscillator(high, low, close, fast=7, medium=14, slow=28):
    """終極波動指標 (Ultimate Oscillator)"""
    prior_close = close.shift(1)
    bp = close - pd.concat([low, prior_close], axis=1).min(axis=1)  # Buying Pressure
    tr = pd.concat([high, prior_close], axis=1).max(axis=1) - \
         pd.concat([low, prior_close], axis=1).min(axis=1)  # True Range

    avg_fast = bp.rolling(fast).sum() / tr.rolling(fast).sum()
    avg_medium = bp.rolling(medium).sum() / tr.rolling(medium).sum()
    avg_slow = bp.rolling(slow).sum() / tr.rolling(slow).sum()

    return 100 * (4 * avg_fast + 2 * avg_medium + avg_slow) / 7


def obv(close, volume):
    """能量潮 (On Balance Volume)"""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def chaikin_adosc(high, low, close, volume, fast=3, slow=10):
    """Chaikin A/D 震盪指標 (Chaikin Accumulation/Distribution Oscillator)"""
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_volume = money_flow_multiplier * volume
    ad_line = money_flow_volume.cumsum()
    return ema(ad_line, fast) - ema(ad_line, slow)


def calculate_technical_indicators(df):
    """
    計算 16 個技術指標，對應原論文 Table 5：
        趨勢型：MA(5/30/60), EMA(5/30/60), MACD(6,15,6)(12,26,9)(30,60,30)
        動能型：RSI(14), WILLR(14), MOM(14), CMO(14), ULTOSC(7,14,28)
        成交量型：OBV, ADOSC(3,10)
    """
    result = df.copy()

    # --- 趨勢指標 ---
    result["MA5"] = sma(df["Close"], 5)
    result["MA30"] = sma(df["Close"], 30)
    result["MA60"] = sma(df["Close"], 60)
    result["EMA5"] = ema(df["Close"], 5)
    result["EMA30"] = ema(df["Close"], 30)
    result["EMA60"] = ema(df["Close"], 60)

    result["MACD_6_15_6"] = macd_line(df["Close"], 6, 15)
    result["MACD_12_26_9"] = macd_line(df["Close"], 12, 26)
    result["MACD_30_60_30"] = macd_line(df["Close"], 30, 60)

    # --- 動能指標 ---
    result["RSI14"] = rsi(df["Close"], 14)
    result["WILLR14"] = williams_r(df["High"], df["Low"], df["Close"], 14)
    result["MOM14"] = momentum(df["Close"], 14)
    result["CMO14"] = cmo(df["Close"], 14)
    result["ULTOSC_7_14_28"] = ultimate_oscillator(df["High"], df["Low"], df["Close"], 7, 14, 28)

    # --- 成交量指標 ---
    result["OBV"] = obv(df["Close"], df["Volume"])
    result["ADOSC_3_10"] = chaikin_adosc(df["High"], df["Low"], df["Close"], df["Volume"], 3, 10)

    return result


def normalize_indicators(df, indicator_cols):
    """
    技術指標做 Min-Max 正規化到 [0,1]，對應原論文 Eq.(1)：
        x_scaled = (x - x_min) / (x_max - x_min)
    這是為了消除不同指標之間量綱（單位/數值範圍）差異過大的問題，
    例如 OBV 動輒幾百萬，但 RSI 只在 0~100 之間。
    """
    result = df.copy()
    for col in indicator_cols:
        col_min = result[col].min()
        col_max = result[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            result[f"{col}_scaled"] = 0.0
        else:
            result[f"{col}_scaled"] = (result[col] - col_min) / (col_max - col_min)
    return result


def merge_with_sentiment(technical_df, sentiment_path):
    """
    把技術指標資料跟情緒分析結果按日期合併。
    情緒分析結果格式來自 sentiment_lexicon.py 的輸出
    （欄位包含 board_date, text_sentiment_factor 等）。

    注意：PTT 的 board_date 格式是 "M/D"（無年份），股價資料的
    日期是完整 datetime index，這裡統一轉成 "M/D" 格式對齊。
    同一天如果有多篇文章，情緒因子取當天平均值。
    """
    sentiment_df = pd.read_csv(sentiment_path)

    # 把股價資料的日期轉成跟 PTT 一樣的 "M/DD" 格式以便合併
    # 注意：PTT 列表頁的日期格式「日」有補零（例如 "6/01"），
    # 但「月」沒有補零（例如 "6/01" 而非 "06/01"），
    # 這裡用 %-m（月不補零，Linux/Mac可用）+ %d（日補零）組合，
    # 若在 Windows 上執行 %-m 語法無效，改用下面的手動組法較保險。
    tech = technical_df.reset_index()
    date_col = tech.columns[0]  # yfinance 預設索引欄位通常叫 Date
    parsed_dates = pd.to_datetime(tech[date_col])
    tech["match_date"] = parsed_dates.dt.month.astype(str) + "/" + parsed_dates.dt.day.astype(str).str.zfill(2)

    # 同一天可能有多篇文章，先依日期彙總情緒因子（取平均）
    daily_sentiment = sentiment_df.groupby("board_date").agg(
        daily_sentiment_factor=("text_sentiment_factor", "mean"),
        daily_article_count=("text_sentiment_factor", "count"),
        daily_comment_count=("total_comments", "sum"),
    ).reset_index()

    merged = tech.merge(
        daily_sentiment, left_on="match_date", right_on="board_date", how="left"
    )

    # 沒有討論的交易日，情緒因子填 0（代表中性/無訊號），並標記文章數為 0
    merged["daily_sentiment_factor"] = merged["daily_sentiment_factor"].fillna(0.0)
    merged["daily_article_count"] = merged["daily_article_count"].fillna(0).astype(int)
    merged["daily_comment_count"] = merged["daily_comment_count"].fillna(0).astype(int)

    merged = merged.drop(columns=["match_date", "board_date"])
    return merged


import importlib.util
import os


def load_stock_list():
    """從 stock_volume_scanner.py 讀取已定案的30檔股票名單"""
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


def process_one_stock(code, name, start, end, out_dir, indicator_cols):
    """處理單一股票：抓股價、算指標、正規化、輸出"""
    price_df = fetch_price_data(code, start, end)
    if price_df is None:
        return False, 0

    tech_df = calculate_technical_indicators(price_df)
    tech_df = normalize_indicators(tech_df, indicator_cols)
    final_df = tech_df.reset_index()

    out_path = os.path.join(out_dir, f"{code}_{name}_technical.csv")
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return True, len(final_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="台股技術指標計算（支援單檔或批次30檔）")
    parser.add_argument("--stock", type=str, default=None,
                         help="單一股票代碼，例如 2330（不指定則自動跑批次30檔）")
    parser.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="結束日期 YYYY-MM-DD")
    parser.add_argument("--sentiment", type=str, default=None,
                         help="（僅單檔模式適用）情緒分析結果 CSV，提供則自動合併")
    parser.add_argument("--out", type=str, default=None, help="（僅單檔模式適用）輸出檔名")
    parser.add_argument("--out-dir", type=str, default="technical_output",
                         help="（批次模式適用）輸出資料夾")
    args = parser.parse_args()

    indicator_cols = [
        "MA5", "MA30", "MA60", "EMA5", "EMA30", "EMA60",
        "MACD_6_15_6", "MACD_12_26_9", "MACD_30_60_30",
        "RSI14", "WILLR14", "MOM14", "CMO14", "ULTOSC_7_14_28",
        "OBV", "ADOSC_3_10",
    ]

    if args.stock:
        # ---- 單檔模式（沿用原本邏輯）----
        out_path = args.out or f"{args.stock}_technical.csv"
        price_df = fetch_price_data(args.stock, args.start, args.end)
        if price_df is None:
            sys.exit(1)
        print("[計算] 技術指標中...")
        tech_df = calculate_technical_indicators(price_df)
        tech_df = normalize_indicators(tech_df, indicator_cols)
        if args.sentiment:
            print(f"[合併] 與情緒資料 {args.sentiment} 合併中...")
            final_df = merge_with_sentiment(tech_df, args.sentiment)
        else:
            final_df = tech_df.reset_index()
        final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[完成] 已輸出至 {out_path}，共 {len(final_df)} 筆交易日資料")

    else:
        # ---- 批次模式：跑完 stock_volume_scanner.py 裡定案的30檔股票 ----
        STOCK_LIST = load_stock_list()
        os.makedirs(args.out_dir, exist_ok=True)

        print(f"=== 批次計算技術指標：{len(STOCK_LIST)} 檔股票 ===")
        print(f"期間：{args.start} ~ {args.end}")
        print(f"輸出資料夾：{args.out_dir}/\n")

        success_count, skip_count, fail_count = 0, 0, 0

        for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
            out_path = os.path.join(args.out_dir, f"{code}_{name}_technical.csv")

            # 續傳機制：已經算過的股票直接跳過
            if os.path.exists(out_path):
                print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）已存在，跳過")
                skip_count += 1
                continue

            print(f"[{i}/{len(STOCK_LIST)}] 處理 {code} {name}（{industry}）...")
            try:
                ok, n_rows = process_one_stock(code, name, args.start, args.end,
                                                args.out_dir, indicator_cols)
                if ok:
                    print(f"    [完成] {n_rows} 筆交易日資料")
                    success_count += 1
                else:
                    print(f"    [失敗] 無法取得股價資料")
                    fail_count += 1
            except Exception as e:
                print(f"    [錯誤] {e}")
                fail_count += 1

        print(f"\n=== 批次完成 ===")
        print(f"成功：{success_count}，跳過（已完成）：{skip_count}，失敗：{fail_count}")
        if fail_count > 0:
            print(f"[提示] 有失敗的股票，重新執行同一指令會自動重試失敗的部分，不影響已完成的")
        print(f"\n[提醒] 前 60 個交易日的部分技術指標會是 NaN（例如 MA60 需要 60 天資料才能計算），")
        print(f"       訓練模型前記得處理缺失值（例如刪除前 60 筆，或用 dropna()）")
