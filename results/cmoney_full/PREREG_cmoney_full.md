# CMoney 完整資料：照論文的 LSTM（MAPE）＋ XGBoost 漲跌方向（事先寫定，2026-10-19，跑之前）

起因：交接文件待辦第 1 項。PTT 已用兩種作法測過（LSTM-MAPE 十年：技術面 13.24% → +PTT 13.08%，p=0.35；XGBoost ΔAUC ≈ 0），
CMoney 只做過舊版 58 檔 XGBoost（留言日期錯置）與新版迴歸（隔天報酬 t≈4–5，但幅度小、扣成本賺不到）。
簡報 v19 第 5 頁註明「CMoney 貼文尚未用 LSTM-MAPE 作法重跑」——這次補上。論文（Jing, Wu & Wang 2021）用的是股吧「貼文」，CMoney 同樣是貼文型論壇，比 PTT 留言更接近論文設定。

## 資料
- 265 檔（與 PTT 研究同一批），**2020-01 ~ 2024-12**（CMoney 只有這段；PTT 也裁成同一段，才能公平比較）。
- CMoney＝`final_dataset_v3_cm2`（清理後、新分類模型；約 50–59% 的股票日有貼文）；PTT＝`final_dataset_v3`。
- 情緒只用「隔天開盤前可得」的貼文／留言（同 lstm_price_v4.py、features_v4.py）。

## A. 照論文：每檔各自訓練 LSTM 預測隔天收盤價（`lstm_price_cm.py`）
- 設定完全沿用 `lstm_price_v4.py`：窗口 10 天、依時間切 75/10/15、5 層 20 hidden、lr 0.01、batch 10、early stopping 15；正規化只用訓練期。
- 測試期約為每檔最後 15%（約 2024-04 ~ 2024-12，約 180 個交易日；論文測 2 個月）。
- 四組：技術面 / +PTT / **+CMoney** / +PTT+CMoney；5 seeds。
- **主要檢定 L1：+CMoney 的平均 MAPE 低於技術面**（265 檔 × 5 seeds 配對，依股票群聚的配對 t；門檻 p<0.05 單一檢定）。
- 次要：天真預測（今天收盤當明天）MAPE；方向準確率；+PTT 與 +CMoney 誰比較好；贏天真預測的股票比例。

## B. XGBoost 漲跌方向（`xgb_direction_cm.py`）
- 目標、技術面特徵、模型同 `retail_test_v4.py`（隔天開盤→收盤漲跌；features_v4.build）。
- 只用 2020 以後的資料；測試年 2022、2023、2024（訓練 ≤ Y−1 上半年、驗證 Y−1 下半年）；5 seeds；依週群聚 bootstrap 300 次。
- **主要檢定 X1：+CMoney 相對技術面的 ΔAUC > 0**（三年合併，bootstrap 95% 區間不含 0 才算）。
- 次要：+PTT、+PTT+CMoney 的 ΔAUC；逐年。

## 預期（寫在跑之前）
- L1：**不會通過**。LSTM 預測價格水準在十年 PTT 版連天真預測都贏不了（MAPE 13% vs 1.2%），換情緒來源改不了這個結構問題；+CMoney 與技術面差距預期 < 0.5pp 且不顯著。
- X1：CMoney 迴歸有 t≈4–5 的隔天效果，所以 ΔAUC **可能是小的正值（+0.001 ~ +0.003）**；但只有三個測試年，區間可能含 0。預期 CMoney ≥ PTT。
