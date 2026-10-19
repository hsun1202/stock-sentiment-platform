# 程式索引

自動產生（`make_code_index.py`）。說明取自每支程式開頭的註解；「結果」是 `results/` 裡同名的輸出檔。

## 資料蒐集：爬蟲與下載

| 程式 | 用途 | 結果 |
|---|---|---|
| `add_stock_scan.py` | 單一股票補掃工具 | — |
| `after_download_v4.sh` | 等 2015-2019 籌碼下載完成 → 重建十年資料集 → 跑逐年滾動分析。事件寫在 watchdog_download.log | — |
| `batch_crawler.py` | 30檔股票批次爬蟲（兩階段架構，2.5年份資料） | — |
| `batch_crawler_267.py` | 267檔股票批次爬蟲（兩階段架構，2020-2024年份資料，含推文使用者ID） | — |
| `check_crawl_full_progress.py` | 查詢PTT Stock板全板爬蟲（crawl_ptt_stock_full.py）的即時進度 使用方式： | — |
| `check_crawler_267_progress.py` | 查詢267檔PTT重爬任務（batch_crawler_267.py）的即時進度 使用方式： | — |
| `cmoney_probe.py` | CMoney 同學會可行性測試（小規模，約 200 次請求）：直接跳到指定交易日的收盤時間，只抓收盤前最後一頁。 要確認：1) 2015 年起抓不抓得到、每頁幾篇  2) 回傳文章的時間是否都在指定時間之前、跨了多久 3) 冷門股有沒有資料  4) 作者自己標的多空（bullOrBear）有多少比例 需要 cmoney_token.json 內有未過期的 access_token（由使用者登入後更新）。 輸出：cmoney_probe.… | `cmoney_probe.log` |
| `crawl_cmoney.py` | CMoney股市爆料同學會爬蟲（需要登入token，抓267檔股票2024年的討論） | — |
| `crawl_cmoney_2025.py` | CMoney 同學會「每個交易日收盤前」抽樣爬蟲——2025 年版（2025-01-01 ~ 2025-12-31；由 crawl_cmoney_daily.py 複製，只改期間、日曆、輸出資料夾） 作法：每檔股票從 2024 年底往回走。目前位置 T（時間）： 抓「T 之前最新的一頁（20 篇）」→ 若這頁最舊的文章還沒碰到上一個交易日收盤，就直接跳到上一個交易日收盤（熱門股：每天一頁）； 否則從這頁最舊的文章接著往回翻（冷門股：連續… | — |
| `crawl_cmoney_comments.py` | 補抓CMoney文章底下的留言（crawl_cmoney.py只抓了文章本身，沒抓留言內容） | — |
| `crawl_cmoney_daily.py` | CMoney 同學會「每個交易日收盤前」抽樣爬蟲（2020-01-01 ~ 2024-12-31，資料集內全部股票） 作法：每檔股票從 2024 年底往回走。目前位置 T（時間）： 抓「T 之前最新的一頁（20 篇）」→ 若這頁最舊的文章還沒碰到上一個交易日收盤，就直接跳到上一個交易日收盤（熱門股：每天一頁）； 否則從這頁最舊的文章接著往回翻（冷門股：連續翻頁，不漏文章）。 cursor ＝ 毫秒時間 × 1048.576（2026-0… | — |
| `crawl_cmoney_preopen.py` | CMoney 補抓：熱門股「開盤前被擠掉」的股票日，各補抓一頁 ——「當天 09:00 之前最新的 20 篇」。 原爬蟲（crawl_cmoney_daily.py）熱門股每天只抓收盤前最後 20 篇，盤中熱的日子開盤前貼文全被擠掉， 造成「t+1 開盤前」特徵偷看到 t+1 盤中熱度（見 cmoney_full/xgb_leak_check.log）。補抓後每個股票日開盤前都至少有最新的一頁。 目標：抓到 ≥20 篇、且最舊一篇已在當… | — |
| `crawl_cnyes_news.py` | 抓鉅亨網「台股」分類新聞（媒體新聞；補公開資訊觀測站重大訊息沒有的記者報導、法人觀點） 介面：GET https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?startAt=<unix>&endAt=<unix>&limit=30&page=N 實測可回溯到 2015 年；每頁最多 30 則；時間區間太長會被截斷 → 以 3 天為一個視窗逐頁抓（超過 30 頁再切成單日… | — |
| `crawl_mops_material.py` | 抓公開資訊觀測站「重大訊息」（歷史，逐公司逐年；只抓主旨，不抓全文） 介面：POST https://mops.twse.com.tw/mops/api/t05st01  {"companyId","year"(民國),"month":"all","firstDay":"","lastDay":"","dataType":"1"} 回傳每則：公司代號、名稱、發言日期(民國)、發言時間(時:分:秒)、主旨 輸出：mops_material… | — |
| `crawl_ptt_stock_full.py` | PTT Stock板全板爬蟲（不篩股票，含使用者ID，2020-2024） | — |
| `crawl_ptt_stock_full_v2.py` | PTT Stock板全板爬蟲 v2（不篩股票，含使用者ID、留言時間戳push_time、文章作者與內文body） | — |
| `crawl_pttweb_api.py` | PTT Stock板 鏡像站(pttweb.cc)爬蟲：補 PTT 本站已被清掉的舊文章（2015-2019初） | — |
| `download_2025.py` | 下載 2025 年資料（給「用 2015-2024 訓練、預測 2025」的真正樣本外測試）： - 證交所官方交易日曆 FMTQIK（每月）      -> chip_raw/calendar/YYYYMM.json、chip_raw/trading_days_2025.csv - 期交所 Put/Call 比（每月；build 的交易日曆 2020 年起用它） -> chip_raw/pcr/YYYYMM.csv - 證交所三大法人 … | `download_2025.log` |
| `download_chip_data.py` | 下載2020-2024籌碼面／市場情緒資料（原始檔，之後再解析）： - 證交所三大法人買賣超日報 T86  -> chip_raw/t86/YYYYMMDD.json - 證交所融資融券彙總 MI_MARGN    -> chip_raw/margin/YYYYMMDD.json - 期交所台指選擇權Put/Call比     -> chip_raw/pcr/YYYYMM.csv 交易日取自 taiex_2020_2024.csv。已存… | `download_chip_data.log` |
| `download_chip_data_2015.py` | 補下載 2015-2019 的籌碼面資料＋官方日行情（2020-2024 已由 download_chip_data.py 下載）： - 證交所三大法人買賣超日報 T86  -> chip_raw/t86/YYYYMMDD.json - 證交所融資融券彙總 MI_MARGN    -> chip_raw/margin/YYYYMMDD.json - 證交所每日收盤行情 MI_INDEX    -> chip_raw/mi_index/Y… | — |
| `download_daytrade.py` | 下載證交所「當日沖銷交易標的及成交量值」(TWTB4U)，2015-01-01 ~ 2024-12-31 每個交易日 -> chip_raw/daytrade/YYYYMMDD.json（含兩張表：大盤當沖統計、每檔當沖成交股數/買賣金額） 交易日曆用 market_sentiment/taiex_official.csv（證交所官方）。已存在的檔案會跳過，中斷後重跑同一指令即可接續。 間隔 DELAY 秒；非 OK 回應重試 3 次，… | `download_daytrade.log` |
| `download_intraday_index.py` | 下載證交所「每5秒指數統計」MI_5MINS_INDEX（2015-2024 每個交易日一次請求），整理成每分鐘加權指數 → market_sentiment/taiex_intraday_1min.csv（Date, time, taiex）；可續跑（已下載的日期跳過）。給盤中情緒（Renault 2017 式）檢驗用。 | `download_intraday_index.log` |
| `download_taifex_night.py` | 下載期交所「臺指期貨 TX 每日交易行情」（含一般 / 盤後 兩個交易時段），2017-05（夜盤開始）~2024-12，按月請求 -> chip_raw/taifex_tx/YYYYMM.csv（UTF-8） 期交所的「盤後」列歸在「次一個交易日」的日期下：該列是前一晚 15:00 ~ 當天 05:00 的夜盤（已確認：開盤價 = 前一個一般盤收盤）。 已存在的檔案會跳過，中斷後重跑同一指令即可接續。 | — |
| `download_us_daily.py` | 下載美股日資料（給「隔夜消息」對照組用）：S&P500、那斯達克、費城半導體、VIX 輸出 market_sentiment/us_daily.csv（Date=美國交易日，欄位為各指數收盤） | — |
| `stock_volume_scanner.py` | 30檔股票 PTT 討論量快篩工具 | — |
| `validate_pttweb.py` | 驗證 pttweb 鏡像站資料的品質，兩種對照： A. PTT本站殘留的2015-2018文章（ptt_stock_full_2015_2018/，自己的爬蟲v2抓的）vs 鏡像站同一篇 B. 2019-03（鏡像站 vs 同學的PTT本站資料）：文章覆蓋率（鏡像站多出多少後來被刪的文章）、同一篇留言是否一致 比對留言時看 (推噓, 使用者ID, 內容, MM/DD HH:MM) 的序列。 輸出：validate_pttweb_repo… | — |
| `watchdog_breeze.sh` | Breeze 情緒分類看門狗（2015-2024 版）：程式掛掉或卡住（log 20 分鐘沒更新）就自動重啟。 breeze_infer_267_fast.py 每 4000 則寫一次 .partial.csv，已完成的股票會跳過，所以重啟不會重算。 事件寫在 watchdog_breeze.log（START / RESTART / STALL / DONE / GIVEUP）。 | `watchdog_breeze.log` |
| `watchdog_download.sh` | 2015-2019 籌碼/官方日行情下載看門狗：掛掉就重啟（已下載的檔案會跳過）；log 40分鐘沒更新視為卡住 （下載程式被證交所擋時會退避最久15分鐘，所以門檻設寬一點）。事件寫在 watchdog_download.log | `watchdog_download.log` |
| `watchdog_pttweb.sh` | pttweb 爬蟲看門狗：爬蟲掛掉或卡住（log 10 分鐘沒更新）就自動重啟； 爬完後若還有沒抓到的文章（失敗的），再補跑，最多補 3 次。 事件寫在 watchdog_*.log（START / RESTART / STALL / RETRY / DONE / GIVEUP）。 2015-2019初舊文章：cd /d/專題claude && nohup bash watchdog_pttweb.sh > /dev/null 2>&1… | `watchdog_pttweb.log`、`watchdog_pttweb_deleted.log` |

## 資料清理與資料集建構

| 程式 | 用途 | 結果 |
|---|---|---|
| `audit_breeze_replies.py` | — | `audit_breeze_replies.log` |
| `audit_yahoo_2014_2019.py` | 稽核 Yahoo 股價 2014-06~2019-12：對照證交所官方交易日曆，找出 (1) 官方有開市但 Yahoo 大多數股票缺資料的日子（例如2020-2024版的2021-04-06） (2) Yahoo 有、官方沒開市的日子 (3) 大量股票「成交量=0且收盤價=前一天」的壞資料日（例如2021-08-17） Yahoo 原始資料快取在 yahoo_raw_2014_2025.pkl（之後 fix_prices_2015.py… | — |
| `build_2025.py` | 2025 年 CMoney：收盤前抽樣（cmoney_daily_2025/）＋熱門股開盤前補頁（cmoney_daily_pre_2025/）→ 清理 → 分類 → 股票日則數。 流程與 2020–2024 相同（cmoney_daily_build.py、cmoney_daily_build2.py、rebuild_cm3.py）：廣告/空白 → 清理 flag（與舊資料合併計算全體統計） → 分類（build2 最終模型的訓練設定… | — |
| `build_annotation_sample.py` | 人工標註樣本抽樣工具 | — |
| `build_chip_features.py` | 整理籌碼資料並併入v2資料集： 1. 補抓交易日曆(期交所PCR日期)中還沒下載的三大法人/融資融券(主要是2021-04-06) 2. 解析 chip_raw/t86(股)、chip_raw/margin(張) -> chip_daily.csv（每檔每天一列） 3. 併入 final_dataset_v2/all_stocks_v2.csv -> final_dataset_v2/all_stocks_v2_chip.csv 三大法… | — |
| `build_claude_label_sample.py` | 從全板2020-2024原始留言裡抽一批新樣本，供Claude本人標註 目的： 392筆人工標籤的訓練集太小，微調效果有限。這支程式從剛爬回來 的全板留言（ptt_stock_full_2020_2024/all_comments.csv）裡 抽一批全新、沒被標過的留言，讓Claude本人閱讀後直接標註情緒 （positive/negative/neutral），用來擴充微調訓練集。 | — |
| `build_dataset_v2.py` | 建立v2資料集（修正版）： 1. 用還原股價 prices_adj/ 重算16項技術指標（不做全期間min-max縮放，避免測試期資訊洩漏） 2. 標記異常價格日：還原後單日／報酬／>10.5%且非上市前5個交易日（減資、停牌復牌、資料錯誤） 3. PTT情緒：非交易日(週末/假日)的留言歸到「下一個交易日」，不再丟棄 輸出： technical_output_267_adj/{code}_{name}_technical.csv fi… | — |
| `build_dataset_v3.py` | 建立 v3 資料集：2015-01 ~ 2024-12 十年（v2 只有 2020-2024；v2 的程式與輸出保留不動） 1. 技術面：還原股價 prices_adj_2014/（fix_prices_2015.py，含Yahoo缺漏日修補、刪颱風假資料）重算16項指標 （2014-06 起算當暖機，不做全期間縮放） 2. 異常價格日：單日／報酬／ 超過「漲跌幅限制+0.5%」且非上市前5個交易日 漲跌幅限制 2015-06-01 以… | `build_dataset_v3.log`、`build_dataset_v3_2025.log` |
| `build_dataset_v3_2025.py` | 建立 v3 資料集：2015-01 ~ 2024-12 十年（v2 只有 2020-2024；v2 的程式與輸出保留不動） 1. 技術面：還原股價 prices_adj_2014/（fix_prices_2015.py，含Yahoo缺漏日修補、刪颱風假資料）重算16項指標 （2014-06 起算當暖機，不做全期間縮放） 2. 異常價格日：單日／報酬／ 超過「漲跌幅限制+0.5%」且非上市前5個交易日 漲跌幅限制 2015-06-01 以… | `build_dataset_v3_2025.log` |
| `build_fresh_sample.py` | 為「LLM zero-shot對照測試」抽一批全新樣本（不重複用過的250則） | — |
| `build_ptt_2015_2024.py` | PTT Stock板 2015-01-01 ~ 2024-12-31 合併＋清理＋留言拼接 | `build_ptt_2015_2024.log` |
| `chat_2025_prep.py` | 2025 年閒聊串（盤中/盤後閒聊）留言 → 交易日 → 抽樣（給 Breeze 分類），做「市場層級 PTT 情緒」的 2025 樣本外測試 來源：pttweb_stock_2025/raw_articles.jsonl（crawl_pttweb_api.py，只抓 500 篇閒聊串） 留言時間「MM/DD HH:MM」補年份：用文章發文年份；若月份早於發文月份（跨年）則 +1 年 交易日歸屬：同 market_sentiment_p… | — |
| `clean_cmoney_ads.py` | 清掉CMoney文章/留言裡看起來像廣告、推銷、機器人洗版的內容 | — |
| `clean_ptt_comments.py` | PTT 留言內容清理：去掉「沒有參考價值的廢話」，保留有情緒/看法的留言 | `clean_ptt_comments.log` |
| `cmoney_clean.py` | CMoney 同學會資料清理規則（2026-10-04；依 Claude 逐則閱讀 1,764 篇後歸納） bot    ：自動貼文（券商報告、盤後複習/盤中訊號、分點買賣超、三大法人/融資券數字、每日精選新聞、各式排行） promo  ：廣告/導流（加賴、拉群組、求追蹤按讚、推 App、直播/社團、短網址導流） link   ：只有網址（去掉網址後不到 8 個字） news   ：整段貼新聞（有記者/報導/通訊社字樣且超過 100 字… | — |
| `cmoney_daily_build.py` | CMoney 同學會（cmoney_daily/*.jsonl，2020-2024 收盤前抽樣）→ 情緒分類 → 股票日資料集 1) 去重、濾廣告/機器人貼文（沿用 clean_cmoney_ads.is_ad） 2) 用 Claude 金標微調的 BERT（只看內文版，3 seeds 平均）分類；另保留作者自標多空 bb 3) 依發文時間分到交易日：前一交易日收盤(13:30)後~當日開盤(09:00)前＝pre；09:00~13:3… | `cmoney_daily_build.log`、`cmoney_daily_build.out` |
| `cmoney_daily_build2.py` | CMoney 第二版：先清理（cmoney_clean.flag）→ 用「PTT 金標 + CMoney 金標」訓練 → 重新分類 → 重建資料集 CMoney 金標：cmoney_gold_all.csv（Claude 親自標 1,764 篇：160 個股票日 1,564 篇 + 隨機 200 篇） 1) 交叉驗證（依股票日分組 5 折，只在「清理後保留」的 CMoney 金標上評估）： 只用 PTT 金標訓練的舊模型（cmoney_… | `cmoney_daily_build2.log`、`cmoney_daily_build2.out` |
| `cmoney_date_fix_comments_only.py` | 只針對「留言」(不含文章全文)驗證CMoney日期錯置對同日訊號的影響——留言短、 長度分佈接近PTT，用已驗證過的classify()方法(精確長度分桶)速度應該接近PTT水準， 不會像混進長文章全文那樣退化。 | `cmoney_date_fix_comments_only.log` |
| `cmoney_date_fix_quicktest.py` | 快速驗證：CMoney的breeze_infer_cmoney.py把留言全部套用「文章日期」，忽略留言自己的 createTime，17.1%的留言其實是文章發布後1天以上才發的。這支腳本只挑留言量最大的幾檔 股票，用留言自己的真實日期重新分類，比較「修正日期後」跟「舊版(文章日期)」的 同日/落後情緒對隔日漲跌的AUC，看同日訊號是不是真的來自日期錯置。 | `cmoney_date_fix_quicktest.log` |
| `cnyes_build.py` | 鉅亨網台股新聞 → 清理 → 對到個股與交易日（給 cnyes_bigmove.py 用） 輸入：cnyes_news/*.jsonl（crawl_cnyes_news.py，188,441 則，2015~2025）、final_dataset_v3_tejall25（股名、交易日曆） 清理： 1. 去掉轉貼的公司公告（標題是「公司名：公告／代子公司／澄清…」且沒有來源）——那是重大訊息的重複，已在 mops_bigmove.py 分析… | `cnyes_build.log` |
| `compute_technical_267.py` | 用 stocks_267/stocks/{code}/price/yahoo_{code}_{year}.csv 的本機股價資料 （組員提供，不用打yfinance API），套用 stock_indicators.py 既有的16項技術指標 公式，輸出跟 technical_output/ 同樣格式的 {code}_{name}_technical.csv。 | — |
| `data_audit.py` | 專案資料總檢查：筆數、日期範圍、缺漏、重複、異常值、資料集之間的一致性。 | — |
| `daytrade_build.py` | 把 chip_raw/daytrade/*.json（證交所當日沖銷交易統計）整理成： chip_raw/daytrade_stock.csv   每檔每日：stock_code, Date, dt_shares, dt_buy, dt_sell chip_raw/daytrade_market.csv  大盤每日：Date, dt_shares, dt_shares_pct, dt_buy, dt_buy_pct, dt_sell, … | `daytrade_build.log` |
| `fix_prefix_collision.py` | 修正股票名稱前綴誤判問題 | — |
| `fix_prices.py` | 重建267檔「還原股價」(含現金股利還原)，並修補Yahoo兩天壞資料： - 2021-04-06：Yahoo整天缺資料 -> 用證交所官方日行情插入 - 2021-08-17：Yahoo成交量=0且收盤價複製前一天 -> 用官方日行情取代 還原方式：Yahoo的 Adj Close / Close 當作還原因子，開高低收同乘該因子。 補的那兩天用官方「漲跌價差」(除息日會自動相對除息參考價)算當日報酬，接在前一日還原價後面。 輸出 p… | — |
| `fix_prices_2015.py` | （2014-06 ~ 2025-01 版，給 2015-2024 十年分析用；原本 fix_prices.py 的 2017~ 版 prices_adj/ 保留不動） 重建267檔「還原股價」(含現金股利還原)，並修補Yahoo壞資料（audit_yahoo_2014_2019.py 對照證交所官方交易日曆找出）： - Yahoo 缺的官方開市日（多為週六補班開市日，如2016-01-30、2018-12-22；另有2019-09-09… | — |
| `fix_prices_2025.py` | 延伸還原股價到 2025-12（給「2015-2024 訓練、預測 2025」用）；作法同 fix_prices_2015.py，輸出到新資料夾 prices_adj_2025/（舊的不動） 1. yfinance 重新下載 267 檔 2014-06 ~ 2026-01（Adj Close / Close 當還原因子）→ yahoo_raw_2014_2026.pkl 2. 自動稽核：對照證交所官方交易日曆（chip_raw/cale… | — |
| `make_dataset_2025_with_ptt.py` | 把 2025 年 PTT 個股情緒（ptt_2025_stock_labels.csv）併進 final_dataset_v3_2025 → final_dataset_v3_2025_ptt/all_stocks_v3.csv （2015-2024 的 PTT 欄位不動；2025 原本幾乎為 0，改成新分類結果；欄位定義同 build_dataset_v3.py：pre＝前一收盤後~今天開盤前、intra＝盤中） | — |
| `market_sentiment_prep.py` | 大盤散戶情緒溫度計（1/3）：準備盤中/盤後閒聊串的留言 - 閒聊串：標題含「盤中閒聊／盤後閒聊」的每日討論串（2015-2024 共 5,095 篇、623 萬則留言），散戶即時心情最集中的地方 - 留言依「自己的時間」分到交易日 t 的時間窗 (t-1 收盤 13:30, t 收盤 13:30]（盤後閒聊的留言會算到隔一個交易日） → 交易日 t 的指數只用 t 收盤前已發的留言，預測 t 收盤之後的大盤不會偷看答案 - 每個交易日… | — |
| `match_ptt_2015_2024_267.py` | 把 PTT 2015-2024（已拼接＋內容清理，ptt_2015_2024/comments_clean.csv）比對到267檔股票。 標題比對：股票名稱規則沿用 match_ptt_articles_267.is_valid_match；股票代號改成較嚴格的 find_matches_v2： 舊規則是「標題含代號這串數字就算」，2015年資料加入後出問題——「[閒聊] 2015/07/27 盤中閒聊」被配到豐興(2015)、 「20… | `match_ptt_2015_2024_267.log` |
| `match_ptt_articles_267.py` | 把組員提供的10個batch zip（每個裡面裝~2.5萬篇PTT文章json，來源pttweb.cc mirror） 比對到267檔股票清單（stock_list_267.py），只用標題比對（比照batch_crawler.py的 is_valid_match邏輯），輸出跟batch_output/同樣格式的 {code}_{name}_comments.csv / _posts.csv，但因為原始資料沒有使用者ID、也沒有推噓類型… | — |
| `match_ptt_full_267.py` | 把PTT全板重爬（斷句修復版）資料比對到267檔股票，輸出格式與 match_ptt_articles_267.py 一致（{code}_{name}_comments.csv，欄位：stock_code,stock_name,article_id,resolved_date,content）， 可直接餵給 breeze_infer_267.py。標題比對規則沿用 match_ptt_articles_267.is_valid_matc… | — |
| `merge_datasets.py` | 技術指標 + 情緒因子 資料整合工具 | — |
| `merge_datasets_267.py` | merge_datasets.py 的267檔股票版本，邏輯一樣（技術指標為主軸、情緒因子 left join、無討論日期補0），但有兩個調整： 1. aggregate_daily_sentiment 分組計數欄位從 "link" 改成 "article_id" （breeze_infer_267.py 輸出的欄位名稱不同） 2. 技術指標篩選到只保留2022-2024（跟情緒分析的涵蓋範圍對齊）—— 技術指標本身有15年資料，但情緒… | — |
| `merge_finetune_labels.py` | 合併「392筆人工標籤」+「300筆Claude標籤」，供微調用 兩種標籤來源準確度不完全相同（人工是黃金標準；Claude標籤先前的 Kappa分析顯示跟人工只有中等一致性），所以保留source欄位，之後 如果想只用人工標籤驗證、或分析Claude標籤有沒有拖累微調效果， 都能回頭用source欄位切分。 | — |
| `prepare_lstm_dataset.py` | LSTM訓練資料準備（技術指標 + 情緒因子 -> 下一日收盤價） | — |
| `quality_check.py` | 批次資料品質自動檢查工具 | — |
| `rebuild_cm3.py` | 把補抓的開盤前貼文（cmoney_daily_pre/）併進 CMoney 資料 → final_dataset_v3_cm3/（cm2 不動，舊結果可重現）。 流程同 cmoney_daily_build.py / cmoney_daily_build2.py：去重（與原本已抓的比對）→ 廣告/空白 → 清理 flag（與舊資料合併計算， 舊文章維持原本的保留與否）→ 分類（同 build2 的最終模型：PTT 金標＋CMoney 金… | — |
| `rebuild_comments.py` | 留言資料重建工具 | — |
| `stitch_truncated_comments.py` | 嘗試把PTT因為單行字數上限被截斷的留言，跟後面緊接著的留言接回去。 | — |
| `stitch_truncated_comments_v2.py` | 用「使用者ID」重做斷句留言合併（v2，取代舊版stitch_truncated_comments.py） | — |
| `stock_indicators.py` | 台股技術指標抓取與計算 | — |
| `stock_list_267.py` | 267檔股票清單（來源：爬取股票清單.pdf，data/news_yahoo + checkpoint.json，2011-2026） | — |
| `stock_ptt_2025_prep.py` | 2025 年 PTT 個股留言（給「只用 PTT 情緒」的 2025 樣本外檢驗） 來源：pttweb_stock_2025_stocks/raw_articles.jsonl（只抓標題比對到 267 檔的 3,667 篇；比對規則＝match_ptt_2015_2024_267.find_matches_v2） 留言時間「MM/DD HH:MM」補年份（用發文年份，跨年 +1）；留言早於發文超過 1 分鐘視為時間異常、排除 交易日：(… | — |
| `tej_build_dataset.py` | 把資料集的「股價」換成 TEJ（調整股價(日)-除權息調整），其餘欄位（PTT、籌碼）完全不動 → 穩健性檢查用 價格欄位：Open/High/Low/Close（TEJ 還原價；無成交日的開高低以收盤價補）、Volume（千股×1000） 技術指標：用 TEJ 價格、同一支 calculate_technical_indicators 重算（TEJ 自 2014-01-02 起，暖機期比原本長） close_unadj：TEJ 真正的… | `tej_build_dataset.log` |
| `tej_build_fullmarket.py` | TEJ 新檢驗 (3)：全市場資料集（四碼普通股、含已下市），欄位格式與 final_dataset_v3_tej 相同，既有檢驗程式可直接讀。 股價/技術指標：tej_raw/tej_price_daily.pkl（還原價），作法同 tej_build_dataset.py 籌碼：tej_raw/tej_chip_daily.pkl → 法人買賣超（張×1000＝股）、融資融券餘額（張），衍生欄位定義同 build_dataset_v… | `tej_build_fullmarket.log` |

## 情緒分類：Breeze、BERT、微調、重標

| 程式 | 用途 | 結果 |
|---|---|---|
| `after_breeze_eval.sh` | 等情緒分類看門狗回報 DONE 後，自動跑相關性過濾的驗證（relevance_infer.py --eval，約數分鐘） | — |
| `after_market_classify.sh` | 等閒聊留言分類完成 → 自動跑大盤散戶情緒指數分析 | — |
| `benchmark_batch_size.py` | 測試不同batch size的吞吐量跟VRAM使用量，找出這張8.5GB顯卡撐得住的最快設定 | — |
| `breeze_infer.py` | 用Breeze-7B-Instruct對留言做情緒分類（強化版prompt：黑話/成語說明 + few-shot鎖輸出格式） | — |
| `breeze_infer_267.py` | breeze_infer_full.py 的267檔股票版本，邏輯完全一樣（含斷點續傳：每2000則 存一次 .partial.csv、整檔股票做完才寫最終 _sentiment.csv 並刪掉暫存檔、 中斷後重跑會自動接續進度、已完成的股票會直接跳過），只改了兩個地方： 1. 股票清單改讀 stock_list_267.py 的 STOCK_LIST_267（267檔） 2. aggregate_by_article 分組欄位從 "l… | — |
| `breeze_infer_267_fast.py` | breeze_infer_267.py 的加速版，提示詞/模型/解析規則完全沿用，輸出格式與斷點續傳也相同。 加速方式（不改變輸入token，結果與舊版應幾乎一致）： 1. 系統提示+few-shot（每則留言都相同的開頭）只跑一次forward，存成KV cache重複使用 2. 依token長度分桶，同長度的留言一批，不需要padding 3. 相同留言只分類一次（跨chunk、跨股票共用快取） | — |
| `breeze_infer_cmoney.py` | 用Breeze-7B對267檔股票的CMoney文章+留言做情緒推論（比照breeze_infer_267.py） | — |
| `breeze_infer_full.py` | 用Breeze-7B對30檔股票全部留言做情緒分類（批次生成版，含斷點續傳） | — |
| `chat_2025_classify.py` | 2025 年閒聊串抽樣留言 → Breeze-7B 強制三選一（與 2015-2024 的 market_sentiment_classify.py 完全相同的分類器與設定） 每 5,000 則存一次（chat_labels_2025.partial.csv），中斷後重跑自動接續。 | `chat_2025_classify.log` |
| `check_breeze_progress.py` | 查詢Breeze-7B批次任務的即時進度 使用方式： | — |
| `classifier_noise_test.py` | 分類器不夠準，最多會把 PTT 情緒的效果壓低多少？（測量誤差／衰減分析） 1) 用目前的 Breeze（constrained 三選一）重新分類 1,100 則 Claude 標註留言與 392 則人工標註留言 → 混淆矩陣、方向錯誤率 2) 模擬：以實際的「每個股票日留言數」分布，產生真實標籤 → 依混淆矩陣產生分類器標籤， 比較「真實淨情緒」與「分類器淨情緒」；衰減係數 b＝真實淨情緒對分類器淨情緒的迴歸斜率。 觀察到的迴歸係數 … | `classifier_noise_test.log` |
| `debug_first_token.py` | — | — |
| `emotion_classify.py` | PTT 留言細分情緒分類（8類）：多(看多) 空(看空) 貪(貪婪/興奮) 慌(恐慌) 追(FOMO/怕錯過) 絕(絕望/認賠) 諷(反串/嘲諷) 中(中立/無關) 強制八選一：比較模型回答這 8 個字的機率（都是單一 token），另存各類機率（之後可算「恐慌程度」等連續指數） 驗證：python emotion_classify.py --eval [--model breeze／qwen]   → 對 emotion/gold_3… | — |
| `eval_breeze_finetune.py` | 比較Breeze-7B微調前後，在held-out val集上的準確率/Macro-F1 val.csv是完全沒被拿去訓練的60筆人工標籤，用同一批資料、同一套 prompt格式，分別跑「原版zero-shot」跟「LoRA微調後」，才能公平 比較微調到底有沒有用。 | — |
| `eval_constrained.py` | — | `eval_constrained.log`、`eval_constrained_v2.log` |
| `eval_constrained_v2.py` | — | `eval_constrained_v2.log` |
| `finetune_breeze_lora.py` | 用392筆人工標註資料，對Breeze-7B做QLoRA微調（情緒分類任務） | — |
| `finetune_gold_cv.py` | 用 Claude 金標微調中文 BERT 做三分類（正/負/中），並用「依股票日分組的 5 折交叉驗證」公平評估。 金標：relabel/sample.csv + relabel/labels_XX.txt（5,194 則、400 個股票日，有股票名與標題） claude_all_labels.csv（1,100 則，只有留言文字；只放訓練、不參與評估） 輸入兩種版本：content＝只看留言；context＝「股票名｜標題｜推噓」+ … | `finetune_gold_cv.log`、`finetune_gold_cv.out` |
| `finetune_gold_infer.py` | 用全部 Claude 金標（重標 5,194 則 + 舊金標 1,100 則）訓練最終的微調 BERT（多個 seed 平均）， 重新分類 2015-2024 全部個股留言，並重建資料集的 PTT 欄位。 輸入：sentiment_output_267_2015_2024/*_comment_labels.csv（原 Breeze 標籤）、ptt_2015_2024/articles.csv（標題） 輸出：sentiment_outpu… | `finetune_gold_infer.out` |
| `ft_rebuild_dataset.py` | 由 sentiment_output_267_ft/ 的逐則標籤重建 final_dataset_v3_ft/all_stocks_v3.csv（只換 ptt_* 欄位）。 先用其中的 breeze 欄重算，確認與原資料集一致（流程驗證），再換成微調標籤。 python ft_rebuild_dataset.py | — |
| `infer_bert_sentiment.py` | 用微調後的BERT模型，對30檔股票的全部留言做情緒推論 | — |
| `intraday_classify_capped.py` | 開盤半小時留言分類（加速版）：每個交易日最多分類 150 則（intraday_classify_full.py 全量 80 萬則要約3小時）。 - 沿用全量版已完成的部分（first30_labels.partial.csv 對應依 comment_time 排序後的前 N 則，那些日子全部保留） - 其餘日子每天隨機抽樣，讓「已完成 + 新抽樣」每天合計 ≤ 150 則（原本每日抽樣只有約35則、表態約23則，雜訊大） 輸出：mar… | `intraday_classify_capped.log` |
| `intraday_classify_full.py` | 盤中即時情緒：把「盤中/盤後閒聊」在 09:00~09:30 的全部留言分類（先前每日只抽樣約35則），降低開盤情緒的抽樣雜訊。 同一個 Breeze 強制三選一分類器、batch 16；每 20,000 則存檔可續跑 → market_sentiment/first30_labels.csv | `intraday_classify_full.log` |
| `market_sentiment_classify.py` | 大盤散戶情緒溫度計（2/3）：用與個股分析相同的 Breeze-7B 強制三選一分類器，分類閒聊串抽樣留言。 每 20,000 則存一次（market_sentiment/chat_labels.partial.csv），中斷後重跑自動接續。 | `market_sentiment_classify.log` |
| `news_sentiment_classify.py` | 新聞情緒：用 LLM 判斷 PTT [新聞] 標題對相關公司/台股是 利多、利空 還是 無關（參考 Lopez-Lira & Tang 2023） - 對象：PTT 股板 [新聞] 原發文（不含 Re:/Fw:）2015-2024 共約 8.3 萬篇；其中 1.38 萬篇標題比對得到 267 檔股票 - 模型：Breeze-7B（與留言情緒同一個模型），強制三選一：比較模型回答「好」「壞」「無」的機率 news_score = log … | `news_sentiment_classify.log` |
| `prepare_bert_dataset.py` | BERT弱監督訓練資料準備 | — |
| `prepare_bert_dataset_3class.py` | BERT訓練資料準備（V3：三分類，納入neutral，修正V2的分布偏移問題） | — |
| `prepare_bert_dataset_lexicon.py` | BERT訓練資料準備（V2：改用詞典法判斷結果當標籤，而非推/噓） | — |
| `prepare_breeze_finetune_data.py` | 準備Breeze-7B LoRA微調用的train/val切分 用 all_human_labels.csv（392筆，已修正5筆客觀標錯的版本）切出 train/val，val完全不參與訓練，用來公平比較「微調前 vs 微調後」 的zero-shot準確率有沒有真的進步。 | — |
| `prepare_human_finetune_data.py` | 用人工標註結果準備BERT微調資料（在既有詞典蒸餾模型基礎上，用真實標籤再微調） | — |
| `prepare_human_from_scratch.py` | 合併兩批人工標註（250+150），準備「從零訓練」BERT用的資料 | — |
| `relabel_analysis.py` | Claude 重新標註 vs 原本的 Breeze 分類器：更準的標籤會不會改變 PTT 情緒的結論？ 讀 relabel/sample.csv 與 relabel/labels_XX.txt（已標的批次都會用到；p=看多、n=看空、u=中立） 1) 留言層級：一致率、混淆矩陣、方向相反比例 2) 股票日層級（同一批留言）：Claude 淨情緒 vs Breeze 淨情緒 的相關與斜率 b（測量誤差的衰減係數） → 真實係數 ≈ 觀察係… | `relabel_analysis.log` |
| `relabel_sample_build.py` | 抽樣給 Claude 重新標註（更準的分類器檢驗） 母體：2015-2024 個股留言（sentiment_output_267_2015_2024/*_comment_labels.csv，含 Breeze 標籤）， 依留言時間歸到交易日（同 build_dataset_v3.py：前一交易日 13:30 後 ~ 今天 13:30），只取「今天收盤前」的留言（pre+intra） 抽樣：Breeze 有表態（正+負≥1）的股票日中隨機… | — |
| `relevance_infer.py` | 留言「是否與股市/投資相關」的第二道過濾（Breeze-7B，強制二選一：比較模型回答「是」與「否」的機率） | — |
| `scratch_claude_labels.py` | 將Claude逐則閱讀後產生的標籤填回claude_label_batch.csv | — |
| `scratch_claude_labels2.py` | 將Claude逐則閱讀後產生的第二批800筆標籤填回claude_label_batch2.csv | — |
| `sentiment_lexicon.py` | 台股 PTT 留言文字情緒分析 (V6：權重詞典 + 成語覆寫機制) | — |
| `stock_ptt_2025_classify.py` | 2025 年 PTT 個股留言 → Breeze-7B 強制三選一（與 2015-2024 個股分析相同的分類器與設定） 同一則留言若因文章提到多檔股票而重複，只分類一次再對應回去。每 5,000 則存一次，中斷後重跑自動接續。 可選環境變數 CAP：每個（股票, 交易日）最多抽幾則（預設不抽樣）。 | `stock_ptt_2025_classify.log` |
| `test_bert_cases.py` | — | — |
| `test_bert_cases_3class.py` | — | — |
| `test_breeze_cases.py` | — | — |
| `test_constrained_prod.py` | — | `test_constrained_prod.log` |
| `test_fast_speed.py` | — | `test_fast_speed.log` |
| `test_fast_vs_old.py` | — | `test_fast_vs_old.log` |
| `test_logit_mode.py` | — | `test_logit_mode.log` |
| `test_padded_batching.py` | — | `test_padded_batching.log` |
| `test_taiwan_llm_accuracy.py` | 用跟Breeze-7B一樣的prompt設計（同樣的system prompt、few-shot範例）， 在Taiwan-LLM上跑一次392筆人工標籤，直接比較Accuracy/Macro-F1， 不用fine-tune、單純看兩個模型zero-shot的表現差異。 | — |
| `train_bert_sentiment.py` | BERT情緒分類模型微調 | — |

## 預測檢驗：PTT／CMoney 情緒 → 漲跌與報酬

| 程式 | 用途 | 結果 |
|---|---|---|
| `after_lstm_intraday.sh` | 等原論文版 LSTM 全部跑完（lstm_price_v4/_run.log 出現 SUMMARY_DONE）→ 分類開盤半小時全部留言（GPU） | — |
| `afterhours_nightfut.py` | 控制美股隔夜報酬後：PTT「盤後/隔夜閒聊」情緒 → 隔天大盤漲跌（承 afterhours_replication.py） 論文（Taiwan FinBERT + CNN-BiLSTM-SA）：盤後閒聊情緒＋開高低收 → 預測加權指數隔天漲跌，準確率 87.5%→90.62%（疑似測試集僅約 32 天）。 本腳本用我們已有的閒聊情緒標籤（market_sentiment/chat_labels.csv）＋官方加權指數，做嚴格版本： -… | `afterhours_nightfut.log`、`afterhours_nightfut_results.json` |
| `afterhours_replication.py` | 輕量重測：PTT「盤後閒聊」情緒 → 隔天大盤漲跌（對照 Discover Computing 2025 的設定） 論文（Taiwan FinBERT + CNN-BiLSTM-SA）：盤後閒聊情緒＋開高低收 → 預測加權指數隔天漲跌，準確率 87.5%→90.62%（疑似測試集僅約 32 天）。 本腳本用我們已有的閒聊情緒標籤（market_sentiment/chat_labels.csv）＋官方加權指數，做嚴格版本： - 情緒只用… | `afterhours_replication.log`、`afterhours_replication_results.json` |
| `afterhours_us_control.py` | 控制美股隔夜報酬後：PTT「盤後/隔夜閒聊」情緒 → 隔天大盤漲跌（承 afterhours_replication.py） 論文（Taiwan FinBERT + CNN-BiLSTM-SA）：盤後閒聊情緒＋開高低收 → 預測加權指數隔天漲跌，準確率 87.5%→90.62%（疑似測試集僅約 32 天）。 本腳本用我們已有的閒聊情緒標籤（market_sentiment/chat_labels.csv）＋官方加權指數，做嚴格版本： -… | `afterhours_us_control.log`、`afterhours_us_control_kw.log`、`afterhours_us_control_kw_results.json`、`afterhours_us_control_results.json` |
| `afterhours_us_control_kw.py` | 控制美股隔夜報酬後：PTT「盤後/隔夜閒聊」情緒 → 隔天大盤漲跌（承 afterhours_replication.py） 論文（Taiwan FinBERT + CNN-BiLSTM-SA）：盤後閒聊情緒＋開高低收 → 預測加權指數隔天漲跌，準確率 87.5%→90.62%（疑似測試集僅約 32 天）。 本腳本用我們已有的閒聊情緒標籤（market_sentiment/chat_labels.csv）＋官方加權指數，做嚴格版本： -… | `afterhours_us_control_kw.log`、`afterhours_us_control_kw_results.json` |
| `backtest_v4.py` | 扣交易成本回測：用 LSTM 正式版（lstm_direction_v4.py，10 seeds 平均）的測試期預測，2018~2024 七年 策略（每個交易日 t 收盤後決定，t+1 開盤買進、收盤賣出＝當沖，每檔等權）： 做多 top k%：預測上漲機率最高的前 k%（k = 1, 5, 10） 多空 k%：做多前 k%、放空後 k%（現股當沖先賣後買），兩邊各半資金 對照：全部股票等權買進（大盤當沖基準）；只用技術面的 LSTM … | `backtest_v4.log`、`backtest_v4.run.log`、`backtest_v4_results.json` |
| `cmoney_daily_test.py` | CMoney 同學會 vs PTT：今天收盤前的淨情緒 → 隔天報酬（2020-2024，同一套方法，與 ptt_cc_deepdive.py 相同的變數定義） 來源：p20＝PTT(Breeze)；cm＝CMoney×微調 BERT；cmbb＝CMoney×作者自標多空 報酬用超額（個股 − 當天全體平均）；迴歸依交易日群聚標準誤；控制：今天超額、／今天超額／、漲停、跌停、量能、文章量、規模 另：收盤買「淨情緒>0」隔天開盤/收盤賣的… | `cmoney_daily_test.log`、`cmoney_daily_test_tej.log`、`cmoney_daily_test_tej.out`、`cmoney_daily_test_v2.log` |
| `cmoney_direction_test.py` | 用CMoney已爬到的74檔資料做同日vs落後情緒的漲跌方向檢驗， 驗證CMoney有精確時間戳（createTime）是否能避開PTT那種「同日訊號、落後就消失」的 時間洩漏疑慮。 | `cmoney_direction_test.log`、`cmoney_direction_test_results.json` |
| `compare_classmate_clean.py` | 比對同學的清理版（ptt_classmate_clean/）與同學原始版（ptt_classmate_2019_2024/）：刪了哪些文章/留言、有沒有改內容。 | — |
| `compare_classmate_ptt.py` | 比對同學的 PTT 資料（ptt_classmate_2019_2024/stock_YYYY.jsonl）與自己的 2020-2024 全板資料 （ptt_stock_full_2020_2024/all_posts.csv + all_comments.csv）。 | — |
| `compare_lstm_results.py` | 比較「含情緒因子」vs「不含情緒因子」LSTM模型的MAPE 使用方式： | — |
| `daytrade_test.py` | 檢驗推測：「台股當沖多 → 資訊消化快 → 散戶情緒/籌碼訊號衰減快」 資料：證交所每日當沖統計（chip_raw/daytrade_stock.csv，由 daytrade_build.py 產生）＋ final_dataset_v3 十年個股資料 設定：沿用 retail_test_v4.py（XGBoost、逐年滾動、測試年 2018~2024、週 block bootstrap），預測對象有兩個： B  = 隔天（t+1）開盤→… | `daytrade_test.log`、`daytrade_test_results.json`、`daytrade_test_run.log` |
| `direction_stratified.py` | 漲跌方向檢驗的分層版：只看「真的有PTT情緒資訊」的股票/股票日。 top20 / top50 : 只用訓練期(<=2022-06)留言量最高的K檔股票（訓練與測試都在這些股票內） days_eval     : 全部股票訓練，但只在「近3日留言>=20」的股票日上評估 active_only   : 訓練/驗證/測試都只用「近3日留言>=20」的股票日 | `direction_stratified.log`、`direction_stratified_results.json` |
| `direction_test.py` | 隔日漲跌方向預測：情緒因子有沒有增量資訊？（比舊版更嚴謹） - 特徵皆為無單位比值/報酬（不使用整段樣本min-max縮放，避免洩漏） - 所有股票合併訓練、依日期切分（train<=2022-06, val 2022-07~12, test 2023-01~2024-12） - 情緒分不同落後天數：lag0(同日,可能含事後留言=洩漏診斷), lag1, lag2；另有「只有討論量」對照 - 指標：AUC、log-loss、Accur… | `direction_test.log`、`direction_test_results.json`、`direction_test_v2.log`、`direction_test_v2_results.json` |
| `direction_test_v2.py` | 用v2資料(還原股價、修補壞資料、週末留言歸到下一交易日)重跑「技術面＋PTT」漲跌預測， 對照修正前 direction_test.py 的結果。 - 預測目標：隔天(h=1)、一週後(h=5個交易日) 收盤是否高於今天 - 排除：標籤期間內有 abnormal_price(減資/停牌復牌/資料錯誤) 的列 - 切分：train 標籤結束日<=2022-06-30、val 2022-07~12(標籤結束日<=2022-12-31)、t… | `direction_test_v2.log`、`direction_test_v2_results.json` |
| `direction_test_v3.py` | v2+籌碼資料：比較各類資訊對個股漲跌(隔天h=1、一週後h=5)的預測貢獻。 組合：技術面 / +PTT / +籌碼 / +籌碼+PCR / 全部，另外拆開 PTT情緒方向 vs PTT討論熱度。 所有非技術面特徵都用「當天收盤後已公布」的資訊（PTT用落後1天，避免同日洩漏疑慮）。 XGBoost、10 seeds；ΔAUC 用按週與按月 block bootstrap；並分2023、2024兩年各算一次。 | `direction_test_v3.log`、`direction_test_v3_results.json` |
| `direction_test_v3b.py` | (v3b) 籌碼訊號的穩健性檢查。原始說明： v2+籌碼資料：比較各類資訊對個股漲跌(隔天h=1、一週後h=5)的預測貢獻。 組合：技術面 / +PTT / +籌碼 / +籌碼+PCR / 全部，另外拆開 PTT情緒方向 vs PTT討論熱度。 所有非技術面特徵都用「當天收盤後已公布」的資訊（PTT用落後1天，避免同日洩漏疑慮）。 XGBoost、10 seeds；ΔAUC 用按週與按月 block bootstrap；並分2023、2… | `direction_test_v3b.log`、`direction_test_v3b_results.json` |
| `direction_test_v4.py` | v4：十年資料（2015-2024，final_dataset_v3/all_stocks_v3.csv）逐年滾動檢驗「技術面＋市場情緒」對個股隔天漲跌的預測力 設計（沿用 v2/v3b 的特徵與模型，改成 walk-forward）： - 測試年 Y = 2018 ~ 2024：訓練用 2015-01 ~ (Y-1)-06 的標籤、驗證(early stopping)用 (Y-1) 下半年、測試 Y 全年 → 7 個不重疊的測試年，涵… | `direction_test_v4.log`、`direction_test_v4_results.json` |
| `emotion_market_analysis.py` | 市場散戶「細分情緒」每日指數（盤中/盤後閒聊，emotion/chat_emotion.csv，每日150則，Breeze 八類機率） 指數（交易日 t：前一交易日收盤後 ~ t 收盤前的留言；各類機率的當日平均）： 恐慌 p_慌、貪婪 p_貪、FOMO p_追、絕望 p_絕、反串 p_諷、看多 p_多、看空 p_空 貪婪恐慌差 = (貪+追+多) − (慌+絕+空) (1) 合理性：與當日/過去5日大盤報酬的相關；大事件期間各指數的 … | `emotion_market_analysis.log` |
| `extract_predictions_for_chart.py` | 為PPT畫「實際vs預測收盤價」折線圖，重跑指定股票、seed=42，抓出test集的日期+實際價+預測價 | — |
| `granger_ptt.py` | 格蘭傑因果檢定：PTT 情緒/討論熱度 與 個股報酬，誰領先誰？（十年資料 2015-2024） 時間對齊（依留言自己的時間，build_dataset_v3.py 已分好時段）： 交易日 t 的 PTT = 前一交易日收盤(13:30)後 ~ t 收盤前發的留言（ptt_pre + ptt_intra） 交易日 t 的報酬 r_t = log(Close_t / Close_{t-1})   → 兩者涵蓋同一段時間 變數：sent_t… | `granger_ptt.log`、`granger_ptt.run.log`、`granger_ptt_results.json` |
| `holdout_2025_ptt_test.py` | 真正的樣本外：用 2015-2024 訓練（2024 下半年當驗證/early stopping），預測 2025 全年（模型從沒看過 2025） 資料：final_dataset_v3_2025/all_stocks_v3.csv（build_dataset_v3_2025.py；2015-2024 與原資料列數相同） PTT 2025 年沒有資料 → 只比較 技術面 / +融資融券 / +法人(對照) 1) 方向：隔天開盤→收盤漲跌… | `holdout_2025_ptt_test.log` |
| `holdout_2025_test.py` | 真正的樣本外：用 2015-2024 訓練（2024 下半年當驗證/early stopping），預測 2025 全年（模型從沒看過 2025） 資料：final_dataset_v3_2025/all_stocks_v3.csv（build_dataset_v3_2025.py；2015-2024 與原資料列數相同） PTT 2025 年沒有資料 → 只比較 技術面 / +融資融券 / +法人(對照) 1) 方向：隔天開盤→收盤漲跌… | `holdout_2025_test.log` |
| `intraday_sentiment_test.py` | 盤中即時散戶情緒（仿 Renault 2017, JBF）：開盤前半小時的情緒變化，能不能預測收盤前半小時的大盤報酬？ 資料：盤中/盤後閒聊留言（market_sentiment/chat_labels.csv，每日抽樣≤300則已分類）＋證交所每分鐘加權指數（taiex_intraday_1min.csv） 每個交易日 d： S_night  ＝ 前一交易日 13:30 ~ d 09:00 的留言淨情緒 (正−負)/(正+負) S_f… | `intraday_sentiment_test.log` |
| `lag_timestamp_test.py` | 用「留言自己的時間戳」檢驗：先前 PTT 同日(lag0)情緒訊號，是真的預測力，還是時間錯置（事後留言被算到發文當天）？ | `lag_timestamp_test.log`、`lag_timestamp_test_results.json` |
| `learning_curve_test.py` | 訓練集大小會不會影響結果？（學習曲線） 目標：隔天開盤→收盤漲跌（同 retail_test_v4）；測試年 2022、2023、2024；驗證期＝前一年下半年。 訓練期一律到「前一年 6 月底」為止，往前取 1 年、2 年、4 年、全部（2015 起）四種長度。 比較：技術面 AUC、加 PTT 的 ΔAUC、加融資融券的 ΔAUC；模型 XGBoost（2 seeds）與邏輯迴歸。 若「資料越多 → PTT 增量越大」，代表訓練資料… | `learning_curve_test.log` |
| `logreg_direction_v4.py` | 邏輯迴歸版（最簡單的基準）：與 retail_test_v4.py（XGBoost）完全相同的特徵、目標（隔天開盤→收盤漲跌）、 逐年滾動（測試年 2018~2024）、週 block bootstrap，只把模型換成邏輯迴歸，方便與 XGBoost、LSTM 並排比較。 前處理：inf→缺值，訓練集中位數補值，標準化後截尾到 ±5； 超參數：L2 懲罰強度 C 用驗證期（前一年下半年）AUC 在 {0.001, 0.01, 0.1, … | `logreg_direction_v4.log`、`logreg_direction_v4_results.json` |
| `longshort_sentiment_test.py` | 橫截面多空組合檢驗（Baker & Wurgler式investor sentiment標準檢定法）： 不問「這檔股票會不會漲」，而是問「情緒偏多的股票，隔天表現有沒有比情緒偏空的股票好」。 每天把當天有討論的股票依情緒因子分成「偏多組」(>0)和「偏空組」(<0)，算兩組隔日 報酬的價差(long-short spread)，累積成時間序列後做t檢定。這個設計會自動抵銷當天 大盤整體漲跌(跟直接比較兩組的相對表現，不是各自的絕對方向)… | — |
| `lstm_direction_v4.py` | LSTM 正式版：十年資料（2015-2024）逐年滾動，預測個股「隔天開盤→收盤」漲跌（可交易）， 比較 技術面 / +PTT / +籌碼 / +PTT+籌碼，並與 XGBoost 版（direction_test_v4.py）對照。 為什麼不照原論文預測股價/看MAPE：舊版 LSTM（per-stock 價格迴歸）的 MAPE 被「價格水準外推誤差」主導 （直接用昨天收盤價 MAPE 1.34%，LSTM 11.5%），偵測不到情… | `lstm_direction_v4.log`、`lstm_direction_v4_3seeds.log`、`lstm_direction_v4_results.json`、`lstm_direction_v4_results_3seeds.json` |
| `lstm_price_cm.py` | 照論文（Jing, Wu & Wang 2021）：每檔各自訓練 LSTM 預測隔天收盤價、看 MAPE —— 換成 CMoney 貼文情緒，並與 PTT 在同一段期間比較。 設定、模型、切法全部沿用 lstm_price_v4.py（直接 import），只改：資料裁成 2020-2024（CMoney 只有這段）、多一組 +CMoney。 事先寫定的假說見 cmoney_full/PREREG_cmoney_full.md。 | — |
| `lstm_price_v4.py` | 原論文作法（Jing, Wu & Wang 2021）用十年資料重跑：每檔股票各自訓練一個 LSTM，預測「隔天收盤價」，看 MAPE 與舊版 train_lstm_per_stock.py 相同：per-stock、窗口10天、依時間切 train 75% / val 10% / test 15%、 論文 Table 6 超參數（5層、20 hidden、lr 0.01、batch 10、最多100 epochs、early stop… | — |
| `margin_return_test.py` | 融資融券能不能「預測報酬」（不只漲跌方向）—— 日頻，隔天開盤→收盤報酬（＝當沖型交易），2018~2024 (a) 單一變數排序：每天依 融資增減 / 融券增減 / 券資比 把股票分五組，看隔天開盤→收盤平均報酬（等權） (b) 模型排序：用 daytrade_test.py 存好的 XGBoost 預測（技術面 vs +融資融券），每天取預測前20%/後20%，看報酬差 報酬為毛報酬（未扣成本）；台股當沖來回成本約 0.32%（六折… | `margin_return_test.log` |
| `market_ptt_2025_test.py` | 市場層級 PTT 情緒的 2025 年「真正樣本外」檢驗（事先寫定的假說，來自 2020-2024 的發現） H1：情緒上升日（dS＝今天淨情緒 − 前20日平均 > 0）的隔天，加權指數收盤→收盤報酬高於其他日子 （2015-2024 全期差 +8.5 bp，t=2.0；2015-2019 −6.0；2020-2024 +22.7 bp，t=3.4） H2：規則「dS>0 做多、<0 做空」／「dS>0 才做多」在 2025 年優於買… | `market_ptt_2025_test.log` |
| `market_ptt_test.py` | 大盤層級：只用 PTT 閒聊串情緒（收盤前可得）預測隔天加權指數（漲跌方向與報酬） 情緒：market_sentiment/chat_labels.csv 依「前一交易日收盤後 ~ 今天收盤」的留言（每天抽樣≤300則，Breeze 三分類） S＝(正−負)/(正+負)；dS＝S − 前20日平均（相對變化） 目標：隔天 收盤→收盤、隔夜（收盤→09:01）、盤中（09:01→收盤） 1) 迴歸（Newey-West 5）：隔天報酬 ~… | `market_ptt_test.log` |
| `market_sentiment_analysis.py` | 大盤散戶情緒溫度計（3/3）：建立每日指數，檢驗能否預測加權指數（2015-2024） 成分（交易日 t；只用 t 收盤前可得的資訊）： 散戶說  ptt_mood  ：閒聊串抽樣留言（每日≤300則）淨情緒 (正−負)/(正+負)，取5日平均 散戶關注 ptt_att  ：閒聊串全部留言數 log，減去過去60日平均（異常熱度） 散戶做  margin_buy：全市場融資買進金額 / 大盤成交金額（散戶借錢買股的熱度），取5日平均 m… | `market_sentiment_analysis.log`、`market_sentiment_analysis.run.log` |
| `news_drift_robust.py` | 新聞情緒 20 日漂移的穩健性檢驗：日曆時間投資組合（calendar-time portfolio） 每個交易日 d：持有「進場日在過去 1~20 個交易日內、且 LLM 判為 好（或 壞）」的股票（等權）， 當日報酬（收盤→收盤）減 267 檔等權平均 → 每日超額報酬序列；Newey-West(20) t 值；逐年拆開。 進場日同 news_sentiment_analysis.py：發文後第一個能成交的開盤；持有期從進場日收盤後… | — |
| `news_sentiment_analysis.py` | 新聞情緒分析（news_sentiment_classify.py 的結果）：三種檢驗 (1) 個股新聞事件研究：比對到 267 檔的 [新聞]，依 LLM 判斷「好/壞/無」，看之後的「超額報酬」 進場 = 發文後第一個能成交的開盤（09:00 前發文 → 當天開盤；之後 → 下一交易日開盤），不偷看 報酬：進場日 開盤→收盤（1日）、開盤→第5/第20個交易日收盤；超額 = 個股 − 267檔等權平均（同一段期間） t 值：依進場日… | `news_sentiment_analysis.log` |
| `ptt_attention_spike.py` | PTT 討論量暴增（爆量）事件：之後報酬怎麼走？依情緒正負分開（Da-Engelberg-Gao 2011：關注度 → 短期推升 → 之後反轉） 爆量定義：今天收盤前的留言數 ≥5 則，且 log(1+留言數) 比該股過去 60 天平均高 2 個標準差以上（只用過去資料） 報酬：每日超額（個股 − 當天全體平均），事件日 0＝爆量當天收盤；看第 −20 ~ +20 天累積超額 分組：爆量且淨情緒 ≥0.3（偏多）／≤ −0.3（偏空）／… | `ptt_attention_spike.log` |
| `ptt_cc_deepdive.py` | 深入檢驗：PTT 淨情緒（今天收盤前的留言）→ 隔天報酬的正向關係，是真的資訊，還是「今天漲跌／漲停／關注度」的代理？ 樣本：final_dataset_v3（2015-2024），有表態留言（正+負≥1）的股票日 報酬一律用「超額」＝個股 − 當天全體股票平均（去掉大盤漂移） 1) 迴歸（依交易日群聚標準誤）：隔天超額 ~ 淨情緒 + 今天超額 + ／今天超額／ + 漲停/跌停 + 量能 + 留言量 + 規模 2) 拆成 隔夜（收盤→… | `ptt_cc_deepdive.log`、`ptt_cc_deepdive_ft.log`、`ptt_cc_deepdive_tej.log`、`ptt_cc_deepdive_tejft.log` |
| `ptt_event_study.py` | PTT 情緒事件研究＋加強控制（接續 ptt_cc_deepdive.py） A) 加強控制：除了今天漲跌，再控制過去 5/20/60 天超額報酬、20 天波動、量能、留言量、規模、漲跌停 → 淨情緒還剩多少？ B) 事件研究：以「留言日收盤」為第 0 天，強正（≥3則表態、淨≥0.5）vs 強負（淨≤−0.5），看第 −20 ~ +20 天的累積超額報酬 → 若事件前就大漲/大跌（情緒跟著價格走）、事件後持續，代表情緒是落後指標＋動能… | `ptt_event_study.log` |
| `ptt_rule_test.py` | 只用 PTT 情緒預測漲跌方向 / 投資報酬（不搭配技術面） (1) 交易規則（不需訓練，測 2015-2024 十年）：PTT 淨情緒 (正−負)/(正+負) 大於 0 就買、小於 0 就賣（放空） A 開盤當沖：用到「隔天開盤前」的留言 → 隔天開盤進場、收盤出場（成本：當沖 32 bp 六折 / 44 bp 無折扣） B 收盤持有一天：只用到「今天收盤前」的留言 → 今天收盤進場、隔天收盤出場（成本：一般交易 47 bp 六折 /… | `ptt_rule_test.log`、`ptt_rule_test_ft.log`、`ptt_rule_test_ft.out`、`ptt_rule_test_tej.log` |
| `retail_chase_reversal.py` | 把「事件」和「散戶」接起來：沒消息的大漲之後會回吐（bigmove_decompose / newsmove_backtest 的發現）——散戶追得越兇，回吐越多嗎？ 樣本：大漲日（超額>0 的大波動日）、當天沒收漲停。主要看「沒消息」的（歸因 4/5/6），並列「有消息」的當對照。 散戶追價的三個量法（都是大漲當天收盤後就知道的）： M 融資增減：當天融資餘額變化率（只取可信用交易的股票）——散戶借錢追 S 散戶成交比重異常：當天（1… | `retail_chase_reversal.log`、`retail_chase_reversal_out889.log` |
| `retail_contrarian.py` | 散戶「反指標」檢驗：散戶越樂觀，之後幾個月越差？（十年資料 2015-2024，月頻率） 理論：Baker & Wurgler 等——散戶情緒造成的錯誤定價會在數週到數月後修正，所以情緒高 → 之後報酬低（反向）。 先前只測過隔天/一週（沒效果），這裡測 1~3 個月。 每月底 m、每檔股票： 散戶說的：當月 PTT 淨情緒 (正−負)/(正+負)（依留言時間歸屬；當月表態留言 ≥20 則才算） 散戶做的：當月融資餘額變化率（月底 / … | `retail_contrarian.log`、`retail_contrarian_results.json` |
| `retail_test_v4.py` | 十年資料逐年滾動：「散戶情緒」兩種量法 vs 法人，對個股隔天開盤→收盤漲跌的預測力（同 direction_test_v4 的設定） 散戶說的：PTT（開盤前可得）；散戶做的：融資融券（融資增減、融券增減、券資比；t 盤後公布） 對照：法人（外資、投信、自營商買賣超/成交量；t 盤後公布） | `retail_test_v4.log`、`retail_test_v4_ft.log`、`retail_test_v4_ft.out`、`retail_test_v4_results.json` |
| `return_regression_test.py` | 改成「預測報酬大小（漲跌幅 %）」而不是只猜漲跌方向，也不預測股價水準（原論文作法） 資料：final_dataset_v3（265檔，2015-2024）；特徵同 features_v4.py。逐年滾動（測試年 2018~2024；訓練≤前一年6月、驗證前一年下半年）。 兩個目標（只用當時已知的資訊）： cc ＝ 隔天漲跌幅（收盤→隔天收盤）：技術面(t)、融資融券/法人用前一天(t-1)公布的、PTT 只到 t 收盤前的留言 oc … | `return_regression_test.log` |
| `run_lstm_price_v4.sh` | 原論文作法（per-stock LSTM 預測收盤價）十年版：4組（散戶情緒版）× 10 seeds，4個平行。已完成(有metrics.json)的自動跳過，可重複執行。 | — |
| `run_lstm_seeds_stitched.sh` | 20組種子 x (含/不含情緒) 的per-stock LSTM，資料為PTT全板斷句修復+強制三選一分類器(2020-2024)。 已完成的組合(有metrics.json)會自動跳過，可安全重複執行。 | `run_lstm_seeds_stitched.out` |
| `schedule_lstm_price_v4.sh` | 等到凌晨 00:00 再開始跑（使用者晚上要用電腦） | — |
| `sentiment_factor_zoo.py` | 情緒因子比較：同一套預測設定下，比較「不同方式量出來的 PTT 散戶情緒因子」誰有預測力（十年 2015-2024，逐年滾動 2018~2024） 動機：目前只用了一種因子（分類器三分類 → (正−負)/(正+負)），分類器準確率 59%；可能不是 PTT 沒資訊，而是因子沒量對。 預測設定（同 direction_test_v4 / retail_test_v4 的目標 B）：t+1 開盤→收盤漲跌；技術面基準；XGBoost 5 s… | `sentiment_factor_zoo.log`、`sentiment_factor_zoo.run.log`、`sentiment_factor_zoo_results.json` |
| `sign_strategy_test.py` | 最簡單的交易規則回測：「預測會漲就買、預測會跌就賣」（日頻，隔天開盤進場、收盤出場＝當沖，2018~2024） 預測：daytrade_test.py 存好的 XGBoost 上漲機率（逐年滾動、樣本外），p > 0.5 視為預測會漲。 A 多空版：p>0.5 開盤買、收盤賣；p<0.5 開盤先賣、收盤買回（當沖先賣後買）。 B 只做多：p>0.5 才買，否則不交易（空手）。 每天把所有部位等權平均成組合報酬；每筆部位都是一次當沖來回，… | `sign_strategy_test.log` |
| `size_group_test.py` | 依規模分組檢驗「PTT 在小型股有沒有增量」（日頻版；用 daytrade_test.py 已存好的 XGBoost 預測，不重訓） 規模代理：60日成交值中位數（close_unadj × 成交量；Yahoo 成交量壞值當缺值），每天橫斷面分三組：0=最小、1=中、2=最大。 目標：B＝隔天開盤→收盤、B2＝隔兩天開盤→收盤（同 daytrade_test.py 的設定，測試年 2018~2024）。 各組 ΔAUC＝(加入某訊號) … | `size_group_test.log` |
| `spike_direction_test.py` | 異常情緒/討論量偵測：不是每天都看情緒因子，只在「討論量相對該股近20日均值 出現異常尖峰」的日子才納入情緒特徵，其餘日子視為沒有訊號。邏輯：多數日子的 留言是雜訊，只有聲量異常放大時才可能反映真正的重大消息或群眾情緒轉向。 | `spike_direction_test.log`、`spike_direction_test_results.json` |
| `split5_test.py` | 「訓練 5 年、預測 5 年」：2015-2019 訓練一次（2019 下半年當驗證/early stopping），模型固定，直接預測 2020-2024 與逐年滾動（每年重訓，retail_test_v4 / return_regression_test）對照。 方向：隔天開盤→收盤漲跌，XGBoost（3 seeds）與邏輯迴歸，AUC 與「加某訊號 − 只有技術面」ΔAUC（週 bootstrap） 報酬：隔天開盤→收盤報酬，X… | `split5_test.log` |
| `stock_ptt_2025_test.py` | 只用 PTT 個股情緒：2025 年「真正樣本外」檢驗（事先寫定的假說，來自 2015-2024） H（主要）：淨情緒（今天收盤前留言）→ 隔天「隔夜」超額報酬的係數 > 0 2015-2024：控制今天/漲跌停/留言量 +4.6 bp（t=3.8）；再加過去5/20天報酬 +2.7~2.8 bp（t≈2.3） 另外報告：隔天收盤→收盤、隔天盤中；規則「情緒>0買、<0賣」毛/淨報酬；強正/強負事件後 1/5/20 天累積超額 兩期用完… | `stock_ptt_2025_test.log` |
| `summarize_cm.py` | 彙總 cmoney_full/lstm/ 的 per-stock LSTM 結果（每檔先對 seeds 平均，再跨股票做配對 t），寫 cmoney_full/lstm_summary.log | — |
| `summarize_lstm_price_v4.py` | 彙總 run_lstm_price_v4.sh：各組 MAPE / 方向準確率（10 seeds），與「技術面」配對比較（同種子相減），並列天真預測。 | `summarize_lstm_price_v4.log` |
| `summarize_seeds.py` | 彙整多組種子的「含情緒 vs 不含情緒」LSTM整體MAPE，算平均差異、95%信賴區間、配對t檢定、Wilcoxon。 | — |
| `taiex_direction_test.py` | 把256檔PTT情緒合併成「全市場」每日情緒(每天中位數708則留言，遠比單一個股密集)， 測試能不能預測大盤(TAIEX/^TWII)隔日漲跌，而不是個股——模仿Chen & Hsu(2025)用 大盤而非個股當預測目標的做法，用跟direction_test.py同一套嚴謹方法(XGBoost、多seed、 按週block bootstrap)驗證換題目後情緒因子是否還是測不出效果。 | `taiex_direction_test.log`、`taiex_direction_test_results.json` |
| `train_lstm.py` | LSTM股價預測模型訓練（技術指標 + 情緒因子 -> 下一日收盤價） | — |
| `train_lstm_per_stock.py` | LSTM股價預測（比照原論文：每檔股票各自訓練一個模型） | — |
| `train_xgboost_classification.py` | 把train_xgboost_per_stock.py的迴歸任務（預測精確收盤價）改成分類任務 （預測隔天收盤價比今天漲還是跌），其他資料切分、特徵、模型設定都一樣， 只換目標變數跟評估指標（Accuracy/F1，不是MAPE）。 | — |
| `train_xgboost_per_stock.py` | 用XGBoost（樹模型，沒有神經網路那種訓練隨機性雜訊）重跑跟LSTM同樣的 「每檔各自訓練」實驗，驗證情緒因子效果測不出來是不是LSTM架構本身的問題。 | — |
| `train_xgboost_split_sentiment.py` | 測試「拆開正負面比例」這個特徵設計，比照Ko & Chang (2021)的做法—— 他們用「正面留言比例」「負面留言比例」兩個獨立特徵，不是像我們現在 的daily_sentiment_factor把正負合併成一個淨值。合併後「正負都很多、 很兩極」跟「根本沒人討論」可能都接近0，資訊被抵銷掉；拆開來看看 是不是能讓情緒訊號更明顯。 | — |
| `vol_volume_test.py` | 預測目標換成「波動」與「成交量」：PTT 情緒／留言量對隔天個股波動、成交量有沒有增量？（文獻：Antweiler & Frank 2004 留言板對波動的預測力強於對報酬） 資料：final_dataset_v3_tej（265 檔、PTT=Breeze、TEJ 股價）＋ tej_raw/tej_chip_daily.pkl（散戶成交比重）；CMoney 用 final_dataset_v3_tejcm2（2020-24，只做迴歸）。… | `vol_volume_test.log`、`vol_volume_test.out`、`vol_volume_test_results.json` |
| `volatility_test_v4.py` | PTT 討論熱度/情緒能不能預測「波動」（大動作），而不是漲跌方向？（十年資料，逐年滾動 2018~2024） 動機：文獻線索——情緒濃烈/討論爆量不代表方向對，但可能代表「接下來會有大動作」。 如果成立，PTT 在平台上可當「風險提醒」（這檔討論爆量，注意波動），研究也不只有負面結論。 公平的基準：波動有很強的「群聚」（今天大動作 → 明天也容易大動作），所以技術面基準額外加入 近期已實現波動（5/20日報酬標準差、平均振幅、今天／報… | `volatility_test_v4.log`、`volatility_test_v4.run.log`、`volatility_test_v4_results.json` |

## 共用工具

| 程式 | 用途 | 結果 |
|---|---|---|
| `analyze_v2.py` | — | — |
| `features_v4.py` | 十年資料（final_dataset_v3/all_stocks_v3.csv）的特徵與預測目標，給 lstm_direction_v4.py 共用。 特徵定義與 direction_test_v4.py（XGBoost 版）完全相同，兩個模型的比較才公平。 | — |
| `run_with_ft.py` | 不改原程式，改用「微調分類器」版資料集重跑既有檢驗： 讀檔時把 final_dataset_v3/ 換成 final_dataset_v3_ft/；寫出的 .log/.csv/.png/.pkl 檔名加上 _ft（不覆蓋原結果）。 | — |

## TEJ 全市場與散戶交易行為

| 程式 | 用途 | 結果 |
|---|---|---|
| `tej_fullmarket_test.py` | TEJ 新檢驗 (3)：把「散戶做的」檢驗擴到全市場（四碼普通股 1,144 檔、含已下市；final_dataset_v3_tejall），回應「265 檔事後選定、倖存者偏差」的限制。 全市場沒有 PTT 配對，所以只檢驗技術面、融資融券、法人買賣超、散戶成交比重。 | `tej_fullmarket_test.log`、`tej_fullmarket_test.out`、`tej_fullmarket_test_results.json` |
| `tej_oos2025_test.py` | 2025 年真正樣本外：散戶「做」的訊號（融資融券＋散戶成交比重）在完全沒看過的一年、全市場還成不成立？ 資料：final_dataset_v3_tejall25（TEJ 全市場四碼普通股，2015~2025）。2025 年在所有先前的分析、調參、特徵選擇中都沒有用過。 模型與特徵完全沿用 tej_fullmarket_test.py / tej_retail_daily.py（不做任何修改或調參）；訓練到 2024-06、驗證 202… | `tej_oos2025_test.log`、`tej_oos2025_test_results.json` |
| `tej_retail_daily.py` | TEJ 新檢驗 (1)-日頻：散戶成交比重（＝100−法人成交比重，t 日盤後可得）對隔天報酬有沒有預測力，方法比照融資融券。 資料：final_dataset_v3_tej/all_stocks_v3.csv（265 檔、TEJ 還原股價）＋ tej_raw/tej_chip_daily.pkl；測試期 2018~2024。 | `tej_retail_daily.log`、`tej_retail_daily.out`、`tej_retail_daily_results.json` |
| `tej_retail_group.py` | TEJ 新檢驗 (2)：依「散戶持股比例」分組，看情緒（PTT、CMoney）與融資融券的效果在散戶多的股票是不是比較強。 資料：tej_raw/tej_股權分散_年.csv（年資料）；股票日資料 final_dataset_v3_tej（PTT、融資融券）、final_dataset_v3_tejcm2（CMoney 清理後×新模型）。 散戶持股比例＝持股未滿 400 張的股東合計持股比率（＝100 − 400 張以上各級距比率）；Y… | `tej_retail_group.log` |
| `tej_retail_monthly.py` | TEJ 新檢驗 (1)-月頻：散戶成交比重（＝100−法人成交比重）對「下個月」個股報酬橫斷面有沒有預測力，方法比照 monthly_ml_test.py。 資料：final_dataset_v3_tej/all_stocks_v3.csv（265 檔、TEJ 還原股價）＋ tej_raw/tej_chip_daily.pkl；股票×月面板 2015~2024。 | `tej_retail_monthly.log` |

## 事件：公告、新聞、大波動日

| 程式 | 用途 | 結果 |
|---|---|---|
| `attention_disposition_study.py` | 注意股／處置股事件研究（散戶熱炒的股票被交易所點名之後，股價怎麼走？） 資料：TEJ 調整股價(日) 的「注意股票(A)」「處置股票(D)」旗標。目前只有 2001~2013 那份匯出檔有這兩欄（tej_raw/tej_price_daily_pre2014.pkl）。 定義：超額＝個股 − 當天全市場四碼普通股等權平均（用 TEJ 報酬率欄）。 注意股事件：旗標 A 出現且前一個交易日沒有 A（一波的第一天）；依事件前 5 天累積超額… | `attention_disposition_study_pre2014.log` |
| `bigmove_decompose.py` | 剩下四分之三的大波動日是什麼？——把每個大波動日依序歸因，並檢驗「沒有消息的大波動之後會不會反轉」。 資料：final_dataset_v3_tejall25（全市場、TEJ 還原價）、tej_raw/tej_基本資料.csv（TEJ 子產業 249 類；同類不足 4 檔者改用 TEJ 產業 97 類）、 mops_bigmove_matched*.csv（公司重大訊息對到的影響日）、cnyes_bigmove_matched*.csv… | `bigmove_decompose.log`、`bigmove_decompose_out889.log` |
| `bigmove_news.py` | 新方向初探（2026-10-08 使用者討論後提出）：先找「股價波動最大的日子」，回頭看那幾天的新聞，找關鍵字規律，看是什麼在影響股價。 這是探索性、描述性的分析（沒有事先寫定的假說；找到的規律要另外用沒看過的資料驗證）。 資料：final_dataset_v3_tej（265 檔、TEJ 股價、2015-2024）＋ news_sentiment/news_scores.csv（PTT 股板 [新聞] 文章標題 83,002 篇，其中… | `bigmove_news.log` |
| `bigup_reversal_pre2014.py` | 樣本外檢查：沒收漲停的大漲日之後會不會回吐？（2001~2013，完全沒看過；這段期間沒有新聞資料，所以只能驗「不分有沒有消息」的版本） 背景：newsmove_backtest.py（2015~2025）發現——剔除收漲停的之後，沒消息的大漲日 5 天後超額 −0.6%（兩組股票、11 年每年為負），有消息的約 0。 2015~2025 全體（有消息約占兩成）加權後約 −0.45%。 事先寫定的假說（2026-10-10，看 2001… | `bigup_reversal_pre2014.log` |
| `buyback_backtest.py` | 庫藏股公告後續漲：實際進出場回測（扣成本、排除買不到的） 事件：tej_raw/tej_events.csv 的「庫藏股」（公告日）；股價：2001~2013 用 tej_raw/tej_price_daily_pre2014.pkl，2015~2025 用 final_dataset_v3_tejall25。 第 0 天＝公告日（非交易日順延，最多 7 天）。公告日沒有時間，所以最早只允許在「第 +1 天收盤」進場（保守：假設都是第 … | `buyback_backtest.log` |
| `cnyes_bigmove.py` | 大波動日 → 回頭看媒體新聞（鉅亨網）→ 類別／語氣／關鍵字規律 → 是什麼在影響；並與公司重大訊息合併看能解釋多少大波動日。 資料：cnyes_matched.pkl（cnyes_build.py）、final_dataset_v3_tejall25（TEJ 還原價）、mops_bigmove_matched*.csv（重大訊息對到的股票日）。 定義同 mops_bigmove.py：超額＝個股−全市場等權；大波動日＝／超額／>自己 … | `cnyes_bigmove.log`、`cnyes_bigmove.out`、`cnyes_bigmove_out889.log`、`cnyes_bigmove_out889.out` |
| `mops_bigmove.py` | 大波動日 → 回頭看公司重大訊息 → 找類別與關鍵字規律 → 看是什麼在影響（使用者 2026-10-08 提出的新方向；用公開資訊觀測站重大訊息，每則有時分秒） 資料：mops_material/*.jsonl（crawl_mops_material.py；265 檔、2015~2025 的重大訊息主旨）＋ final_dataset_v3_tejall25（TEJ 還原價；只取原 265 檔）。 作法： 影響日：盤前（前一交易日 1… | `mops_bigmove.log`、`mops_bigmove.out`、`mops_bigmove_out889.log`、`mops_bigmove_out889.out` |
| `negnews_backtest.py` | 「負面新聞之後會續跌」——能不能執行？（cnyes_bigmove 的發現：開盤前已發布的負面新聞，影響日之後 4 天再 −0.43%〔889 檔〕／−0.11%〔265 檔〕） 這批資料兩組股票都已看過，所以這裡不是驗證，只是把「理論上的續跌」換成「實際能做的」算一次。 事先寫定的規則（2026-10-10，跑之前寫下，不搜尋參數）： 事件：影響日開盤前已發布的負面語氣新聞（同 cnyes_bigmove.py 的 NEG 詞表、且不… | `negnews_backtest.log` |
| `newsmove_backtest.py` | 「有消息的大漲之後會續漲」——實際進出場回測（排除漲停買不到的、扣成本） 事件：bigmove_decompose.csv（265 檔）與 bigmove_decompose_out889.csv（889 檔）裡的大漲日（x>0），依歸因分「有消息」（1 公司公告、2 媒體新聞）與「沒消息」（4、5、6）。 有消息的定義＝影響當天的公告/新聞在當天收盤前就已發布，所以收盤時是知道的。 股價：final_dataset_v3_tejall… | `newsmove_backtest.log` |
| `tej_event_oos_pre2014.py` | 用完全沒看過的年份（2013 年以前）驗證事件研究的結果。 背景：tej_event_study.py 在 2015~2025 的資料上（探索性）看到「庫藏股公告後第 2~5 天還有 +0.65% 的超額報酬」。 那十一年的資料全部看過了，所以請使用者另外匯出 2013 年以前的 TEJ 還原股價（tej_raw/TEJ20261008113623.zip），在跑之前先寫定假說。 | `tej_event_oos_pre2014.log` |
| `tej_event_study.py` | TEJ「特殊事件日期資料庫」事件研究：減資、股利、庫藏股、現增、危機出事、MSCI 納入/剔除 的「公告日」前後，股價怎麼動？ （新方向「大波動日 → 是什麼在影響」的第一步：先用有明確日期與類別的事件，量每一類事件的平均影響。探索性。） 資料：tej_raw/20261008232334DataExport.csv（使用者 2026-10-08 匯出，18 個類別、全部歷史）→ 整理成 tej_raw/tej_events.csv 股… | `tej_event_study.log` |

## 交易頻率：月頻、週頻選股

| 程式 | 用途 | 結果 |
|---|---|---|
| `monthly_fullmarket_addinfo.py` | 把「媒體新聞」與「公司公告」也拉長成月頻，加進月頻全市場選股模型（monthly_fullmarket_strategy.py），有沒有再多一點？ （使用者 2026-10-12 問：其他訊號如果把時間拉長呢？日頻時新聞與公告方向都對、但開盤就反應完；月頻時成本低很多，值得重測。） 面板、模型、切分、成本、下市處理、可交易範圍 完全同 monthly_fullmarket_strategy.py；只多加特徵。 新增特徵（都只用當月最後一… | `monthly_fullmarket_addinfo.log` |
| `monthly_fullmarket_checks.py` | 月頻全市場選股（monthly_fullmarket_strategy.py）的第二輪檢查——2026-10-12，跑之前寫定，全部照實報： (1) 月底收漲停買不到：把月底當天收漲停的股票從買進名單剔除（不遞補），看占多少、績效差多少。 (2) 拿掉 2018 年（那年模型只用 2015~2016 兩年訓練，而且相對基準贏最多）：2019~2025 還贏不贏。 (3) 分前後段：2018~2021、2022~2025。 (4) 不用機… | `monthly_fullmarket_checks.log` |
| `monthly_fullmarket_robust.py` | 月頻全市場選股（monthly_fullmarket_strategy.py，S1 通過）的穩健性檢查——2026-10-11，跑之前寫定要檢查的四件事，全部照實報： (a) 換基準：相對「市值加權」（接近加權指數，這幾年被台積電拉高）還贏不贏？ (b) 只看不在原 265 檔內的股票（月頻分析以前只在 265 檔做過，這些股票對月頻模型是沒看過的）。 (c) 晚一天進場：月底收盤算出名單，隔一個交易日收盤才買（持有到下個月第一個交易日… | `monthly_fullmarket_robust.log` |
| `monthly_fullmarket_strategy.py` | 月頻、全市場、只做多、扣成本：技術面＋散戶交易行為 每月選股，能不能贏過大盤？ （前面所有日頻做法都輸給成本；月頻是唯一毛報酬幅度夠大的地方，且從未在全市場測過、未扣成本，2025 年完全沒用過。） 資料：final_dataset_v3_tejall25（TEJ 全市場四碼普通股 1,154 檔、2015~2025、還原價、含已下市）。 面板：股票×月；特徵只用該月最後一個交易日（含）以前的資訊。 特徵：技術面 11 個（同 mont… | `monthly_fullmarket_strategy.log` |
| `monthly_ml_test.py` | 月頻機器學習檢驗（Gu-Kelly-Xiu 式）：預測「下個月」個股報酬的橫斷面排序，看 PTT / 融資融券 / 法人 有沒有增量 目的：回答「結果不好是不是因為預測任務設計（日頻漲跌、股票混在一起）」——改成文獻中機器學習有效的月頻橫斷面設定重測。 資料：final_dataset_v3/all_stocks_v3.csv（265檔、2015-2024）彙總成「股票×月」面板；特徵只用該月最後一個交易日（含）以前的資訊。 目標：下個… | `monthly_ml_test.log` |
| `monthly_size_test.py` | 月頻機器學習檢驗（Gu-Kelly-Xiu 式）：預測「下個月」個股報酬的橫斷面排序，看 PTT / 融資融券 / 法人 有沒有增量 目的：回答「結果不好是不是因為預測任務設計（日頻漲跌、股票混在一起）」——改成文獻中機器學習有效的月頻橫斷面設定重測。 資料：final_dataset_v3/all_stocks_v3.csv（265檔、2015-2024）彙總成「股票×月」面板；特徵只用該月最後一個交易日（含）以前的資訊。 目標：下個… | `monthly_size_test.log` |
| `overnight_sentiment_monthly.py` | 用「隔夜報酬」當散戶情緒指標（不靠文字、每檔每天都有），在月頻、全市場檢驗。 文獻：Aboody, Even-Tov, Lehavy & Trueman (2018, JFQA) 隔夜報酬是個股層級的情緒指標，短期延續、長期反轉； Zhang, Tsai, Weng & Tsai (2023, PBFJ) 台股也成立，且散戶成交比重越高越明顯；Lou, Polk & Skouras (2019, JFE) 隔夜/盤中的拉鋸。 資料：fi… | `overnight_sentiment_monthly.log` |
| `weekly_fullmarket_strategy.py` | 週頻、全市場、只做多、扣成本的選股（使用者 2026-10-15 問「那如果看週的呢」） 作法與 monthly_fullmarket_strategy.py 相同，只是把「每月」換成「每週」（以每週最後一個交易日為準）。 資料：final_dataset_v3_tejall25（TEJ 全市場 1,154 檔、2015~2025）＋ 新聞 cnyes_bigmove_matched*.csv ＋ 公告 mops_bigmove_mat… | `weekly_fullmarket_strategy.log`、`weekly_fullmarket_strategy.out` |

## 情緒當開關

| 程式 | 用途 | 結果 |
|---|---|---|
| `sentiment_regime_robust.py` | 「情緒當開關」的穩健性檢查（sentiment_regime_test.py 三個主要檢定都通過之後做；2026-10-09，跑之前寫定要檢查什麼） 問題：(1) 情緒狀態會連續好幾天相同，逐日 t 可能高估顯著性 → 改用 Newey-West（HAC，落後 20 天）標準誤。 (2) 情緒跟著行情走（恐慌高＝剛大跌、波動大）。技術面訊號好壞的差異，會不會只是「市場狀態」造成、跟情緒本身無關？ → 迴歸：隔天技術面 Rank IC ~… | `sentiment_regime_robust.log` |
| `sentiment_regime_test.py` | 情緒當「開關」：市場情緒高或低的時候，技術面訊號的準確度一不一樣？（真正的「整合技術面與情緒」：情緒不是加進去的變數，而是決定技術面可不可信的狀態） 文獻：Stambaugh, Yu & Yuan (2012) 異象在高情緒之後較強；Antoniou, Doukas & Subrahmanyam (2013) 動能只在樂觀期有效。 資料：daytrade_preds/B_{Y}.pkl（技術面 XGBoost 對隔天開盤→收盤上漲機率的… | `sentiment_regime_test.log` |

## 專案維護

| 程式 | 用途 | 結果 |
|---|---|---|
| `make_code_index.py` | 產生 GitHub 專案用的程式索引 code/INDEX.md：每支程式的用途（取程式開頭說明的第一段）、對應的結果檔，依主題分組。 由 sync_github.sh 在每次同步前自動執行，所以新程式會自動出現在索引裡。 | — |
| `sync_github.sh` | 把專題資料夾的最新程式、結果紀錄、文件、最新版簡報同步到 GitHub 私人專案（只同步白名單，不含原始資料與 token） 來源：D:\專題claude    匯出：D:\專題claude_github（git 專案，remote＝GitHub 私人 repo） 由工作排程器定時執行（見 PROJECT_STATUS.md），也可以手動跑。 | — |

## 其他

| 程式 | 用途 |
|---|---|
| `lowfreq_cm.py` | 把交易拉長：週頻、月頻選股，看 CMoney（與 PTT）情緒在「技術面＋散戶行為」之上有沒有增量。事先寫定見 PREREG_cmoney_full.md 第 D 節。 做法同 weekly_fullmarket_strategy.py / monthly_fullmarket_strategy.py；股票＝265 檔（CMoney 有資料）、期間 2020–2024、測試 2022–2024。 |
| `lowfreq_fm_check.py` | 事後探索（看到 lowfreq_cm 單一變數 IC 後才追加）：CMoney 討論量／淨情緒對下期報酬的效果，控制規模、成交值、波動、動能後還在不在？ 每期橫斷面迴歸（變數先轉成橫斷面排名再標準化），Fama-MacBeth 平均係數與 t。 |
| `lowfreq_net_sort.py` | 事後探索：依 CMoney 當期淨情緒分 5 組，看下期報酬（不用模型、不需訓練，所以 2020–2024 全期都能看；但規則是看到 IC 後才挑的，要標明）。 |
| `oos_2025.py` | 2025 年真正樣本外：CMoney 淨情緒對下期報酬（事先寫定見 PREREG_cmoney_full.md 第 E 節）。 變數、分組、Fama-MacBeth 完全照 lowfreq_net_sort.py／lowfreq_fm_check.py；資料 final_dataset_v3_cm3x（2025＝build_2025.py）。 樣本外＝下期報酬落在 2025 年的期間：月頻形成月 2024-12 ~ 2025-11（12… |
| `quarterly_vol.py` | 季頻高低波動選股 vs 大盤（不扣手續費）。事先寫定的假說見 PREREG_quarterly_vol.md。 每季底依過去 3 個月日報酬標準差分 5 組、等權、持有一季；和 0050 含息、^TWII、市值加權比。 cd D:/專題claude && python quarterly_vol/quarterly_vol.py |
| `run_2025_chain.sh` | 2025 年 CMoney：收盤前抽樣爬蟲 → 熱門股開盤前補一頁。之後的分類與樣本外檢驗另外跑（build_2025.py、oos_2025.py）。 |
| `run_cmoney_full.sh` | CMoney 完整資料：LSTM 5 組 × 5 seeds（GPU 1 條線＋CPU 8 個平行）→ 彙總。已完成的自動跳過，可重複執行。 實測：這個小 LSTM（batch 10）GPU 單一任務最快（每檔約 6 秒），但 GPU 上多開幾個不會變快；CPU 16 核可同時跑 8 個。 CM_TAG=cm3 → 讀補抓後的資料集、結果放 cmoney_full/lstm_cm3/ |
| `run_preopen_chain.sh` | 補抓開盤前 → 分類＋重建 final_dataset_v3_cm3 → XGBoost 與洩漏檢查（cm3）。LSTM 不在這裡（使用者晚上說了再跑：CM_TAG=cm3 bash run_cmoney_full.sh）。 |
| `sent_strategies.py` | 用 CMoney 淨情緒選股的 7 種策略（事先寫定見 PREREG_cmoney_full.md 第 F 節；清單、參數不增不改）。 CM_TAG=cm3 → 只有 2020–2024（樣本內）；CM_TAG=cm3x → 2020–2025，另外單獨報 2025（樣本外，主要檢定 S2、S7）。 |
| `xgb_direction_cm.py` | XGBoost 隔天開盤→收盤漲跌：CMoney 貼文情緒 vs PTT，同一批 265 檔、同一段 2020-2024。 設定同 retail_test_v4.py（features_v4.build 的技術面特徵與目標 B、XGB 超參數、依週 bootstrap），只改： 只用 2020 以後的資料（CMoney 只有這段）、測試年 2022-2024、多一組 +CMoney。事先寫定的假說見 cmoney_full/PREREG… |
| `xgb_leak_check.py` | 檢查 xgb_direction_cm.py 的 +CMoney 是否來自抽樣洩漏（事後追加的檢查，2026-10-19，看到 X1 通過後才寫——要如實標註）。 疑慮：CMoney 爬蟲每個交易日只抓「收盤前最後 20 篇」。t+1 開盤前的貼文有沒有被抓到，取決於 t+1 盤中貼文多不多（盤中熱 → 開盤前的被擠掉）， 而 t+1 盤中熱不熱鬧跟 t+1 漲跌有關 → 「t+1 開盤前」特徵偷看到了 t+1 的資訊。PTT 是全量抓… |
| `xgb_strategy_cm.py` | 拿 XGBoost 漲跌模型（技術面 vs +CMoney，cm3）去預測隔天開盤→收盤的漲跌與報酬，並做當沖回測。事先寫定見 PREREG_cmoney_full.md 第 C 節。 模型、特徵、切法同 xgb_direction_cm.py（測試 2022–2024、5 seeds）。 |
