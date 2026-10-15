# 資料蒐集與處理細節

原始資料不在 GitHub 上。這份文件記錄每種資料怎麼取得、踩過的坑，讓之後能重建。

## PTT 股板

- 本站：`batch_crawler_267.py`（267 檔、文章＋留言＋使用者 ID）、`crawl_ptt_stock_full_v2.py`（全板、含留言時間戳）。
- 本站 2019-03 以前的一般文章已被看板上限清掉，舊文由鏡像站 pttweb 補：`crawl_pttweb_api.py`。
- 個股配對：只用**文章標題**比對代號或股名（代號前後不能接數字、年份型代號不能是日期的一部分；修正前曾誤配 49 萬則）。
- 斷句修復：PTT 長留言會被拆成多行，`stitch_truncated_comments_v2.py` 接回。
- 留言時間戳：早期爬蟲沒存，造成「同日訊號」其實是收盤後留言的時間錯置；後來重爬補上，並一律只用隔天開盤前的留言。

## CMoney 股市同學會

- `crawl_cmoney_daily.py`：2020-01-01 到 2024-12-31，每檔股票從 2024 年底往回走；熱門股每個交易日只抓收盤（13:30）前最後一頁 20 篇，冷門股連續翻頁。
- 需要使用者本人登入後從瀏覽器取出 token 存成 `cmoney_token.json`（**不可上傳**，已列入 `.gitignore`）；token 約一天到期，401 時程式會暫停等檔案更新。
- 坑：cursor ≈ 毫秒 × 1048.576 只適合「跳到某時間」，接續翻頁要用 API 回傳的 nextCursor。
- 資料從 2020-01 開始，2019 年以前抓不到。
- 清理（`cmoney_clean.py`）：120.8 萬 → 107.1 萬篇，移除重複貼文、廣告導流、新聞轉貼、機器人與行銷帳號。

## 鉅亨網台股新聞

- `crawl_cnyes_news.py`：`GET https://api.cnyes.com/media/api/v1/newslist/category/tw_stock_news?startAt&endAt&limit=30&page`，可回溯到 2015。
- 坑：①時間篩選只到「天」；②每個查詢最多 30 頁（第 31 頁回 422）；③每分鐘約 30 次以上請求就回 HTTP 500（statusCode 4008）→ 單支、每 2.4 秒一次、出錯等 90 秒；④`tw_stock` 分類同一則會重複 4 次，要用 `tw_stock_news`；⑤舊新聞的股票標記是空的。
- 清理與配對：`cnyes_build.py`（去掉轉貼公告與券商制式評論；股名比對先量精確率）。

## 公開資訊觀測站重大訊息

- `crawl_mops_material.py`：`POST https://mops.twse.com.tw/mops/api/t05st01`，逐公司逐民國年，回主旨＋日期＋時分秒；舊的 ajax 網址會被擋。
- 全市場 1,154 家 × 11 年＝12,694 次請求，兩支並行（`--reverse` 從尾端）約 5 小時，零失敗。

## TEJ Pro（學校授權，不可公開散布）

由有授權的使用者在 TEJ Pro 匯出，轉檔程式：

| 資料表 | 轉檔／使用 |
|---|---|
| 籌碼分佈（法人／信用／一般現股成交比重、融資融券、持股率） | `tej_raw/tej_chip_daily.pkl` |
| 調整股價（日）－除權息調整（2014–2025；另 2001–2013） | `tej_build_dataset.py`、`tej_build_fullmarket.py` |
| 股權分散（年） | `tej_retail_group.py` |
| 基本資料（上市別、下市日、產業） | `bigmove_decompose.py` |
| 特殊事件日期資料庫（9 類事件） | `tej_event_study.py` |

- TEJ 匯出檔為 UTF-16、Tab 分隔、缺值為 "."；部分為 cp950、逗號分隔、數字含千分位。
- 以 TEJ 檢查原 Yahoo 還原價：約 1% 的股票日報酬有出入，對結果影響在小數點後一位。

## 證交所／期交所公開資料

三大法人買賣超（T86）、融資融券彙總（MI_MARGN）、選擇權 Put/Call 比、當沖統計（TWTB4U）、每 5 秒指數（MI_5MINS_INDEX）、台指期夜盤：`download_*.py`。

## 情緒分類

| 方法 | 用在 | 程式 |
|---|---|---|
| Breeze（台灣大型語言模型）三分類 | PTT 個股留言 231 萬則、閒聊串、新聞標題 | `breeze_infer_267_fast.py`、`market_sentiment_classify.py`、`news_sentiment_classify.py` |
| 8 類細分情緒 | PTT 閒聊串 | `emotion_classify.py` |
| Claude 逐則重標（標準答案） | 5,194 則 PTT 留言 | `relabel_sample_build.py`、`relabel_analysis.py` |
| 微調中文 BERT | 重標全部 PTT 留言、CMoney | `finetune_gold_cv.py`、`finetune_gold_infer.py` |
| 新聞語氣詞表 | 鉅亨網標題 | `cnyes_bigmove.py` |
