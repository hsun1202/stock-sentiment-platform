# 整合技術面分析與市場情緒之股票交易資訊平台

大學專題。以台股十年、全市場的資料，檢驗「市場情緒」能不能、以及該怎麼放進一個技術面＋情緒的交易資訊平台。

## 一句話結論

情緒指標（PTT、CMoney、媒體新聞、公司公告、隔夜漲幅）方向都對、統計上顯著，但市場反應得比讀者快，看到再進場扣掉成本就沒有了；扣成本後還有用的，是散戶實際的交易行為（融資融券）加技術面、持有一個月的選股。情緒的價值在**描述與提醒**，這決定了平台的設計。

## 起點

- 選讀論文：Jing, N., Wu, Z., & Wang, H. (2021). *A hybrid model integrating deep learning with investor sentiment analysis for stock price prediction.* Expert Systems with Applications, 178, 115019.（論壇情緒＋技術指標＋LSTM，加入情緒後 MAPE 0.0675 → 0.0449）
- 7 月概念驗證：台積電 2330、PTT 133 篇／1.8 萬則留言、35 個交易日。

## 主要結果

| 問題 | 結果 |
|---|---|
| 照論文作法在台股重做（LSTM 預測收盤價） | 265 檔、十年：MAPE 13.24% → 13.08%（不顯著）；「今天收盤當明天」只有 1.22%，模型 0 檔贏 |
| PTT 情緒預測隔天漲跌 | 6 種因子、3 種模型，7 個測試年都無效；59.5% 留言在收盤後才發 |
| 換分類器／平台（CMoney）／股價來源（TEJ）／目標（波動、成交量） | 結論都不變 |
| 大波動日是什麼在影響 | 公司公告與新聞方向可預期（265 檔歸納、889 檔驗證全過），但多在隔天開盤就反應完；超過一半的大波動日找不到消息 |
| 散戶交易行為 | 融資融券、散戶成交比重在全市場 1,144 檔每一年都有小訊號 |
| 交易頻率與成本 | 日頻扣成本全 ≤ 0；週頻 +0.09%/週（不顯著）；月頻 +0.63%/月贏一般股票（t=2.9），年化 22%，但相對加權指數不顯著 |
| 提醒類發現 | 沒消息又沒鎖漲停的大漲之後五天平均回吐（2015–2025 每年、2001–2013 驗證）；恐慌時技術面訊號較不可靠 |

完整數字、每項檢驗的設計與事先寫定的假說，見 [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)（依時間倒序的研究紀錄，最上面是總整理）。

## 資料夾

| 路徑 | 內容 |
|---|---|
| `code/` | 全部分析與爬蟲程式（Python／bash），檔名即用途；每支程式開頭的說明寫了目的、資料、事先寫定的假說與用法 |
| `results/` | 各檢驗的輸出紀錄（`.log`、`.out`、`*_results.json`），檔名與程式對應 |
| `slides/` | 報告簡報 `final_report_v19.pptx`、白話講稿 `講稿_v19.md`、產生簡報的 `build_deck_v19.js`（pptxgenjs） |
| `docs/` | `PROJECT_STATUS.md` 研究紀錄與交接文件 |

## 資料（未包含在本專案內）

原始資料因授權、版權或檔案大小，**不放在 GitHub**：

| 資料 | 來源 | 取得方式 |
|---|---|---|
| PTT 股板文章與留言 | PTT Stock 板 | `code/` 內的 PTT 爬蟲（`crawl_ptt_*`、`batch_crawler*`） |
| CMoney 股市同學會貼文 | CMoney | `code/crawl_cmoney_daily.py`（需自己登入取得 token，存成 `cmoney_token.json`，已列入 `.gitignore`） |
| 鉅亨網台股新聞 | 鉅亨網 | `code/crawl_cnyes_news.py` |
| 公司重大訊息 | 公開資訊觀測站 | `code/crawl_mops_material.py` |
| 股價、籌碼、股權分散、特殊事件 | TEJ Pro（學校授權） | 需有授權自行匯出，不可公開；轉檔程式見 `code/tej_*` |

程式預設在「資料所在的資料夾」執行（讀寫相對路徑），例如 `cd <資料夾> && python code/某程式.py` 前需先把程式放回資料夾或調整路徑。

## 環境

Python 3.10；主要套件見 `requirements.txt`。簡報：Node.js＋pptxgenjs（`slides/package.json`）。部分模型用到 NVIDIA GPU（XGBoost `device="cuda"`、BERT 微調）。
