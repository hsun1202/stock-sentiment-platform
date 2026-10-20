# 整合技術面分析與市場情緒之股票交易資訊平台

[English version below](#technical-analysis-and-market-sentiment-for-a-taiwan-stock-information-platform)

大學專題：以台股十年、全市場的資料，檢驗「市場情緒」能不能預測股價，以及情緒應該怎麼放進一個「技術面＋情緒」的交易資訊平台。
後期每個檢驗都先把假設與通過門檻寫定，再在沒看過的股票或年份上驗證；通過與未通過都照實記錄。

組員專案（PTT 開盤前討論量、大漲跌日的 LLM 歸因）：[jameschen108/tw-stock-sentiment](https://github.com/jameschen108/tw-stock-sentiment)

## 背景

- **選讀論文**：Jing, Wu & Wang（2021）*A hybrid model integrating deep learning with investor sentiment analysis for stock price prediction*，Expert Systems with Applications 178, 115019。用東方財富股吧 88 萬則貼文的情緒加上 16 項技術指標，以 LSTM 預測隔天收盤價，加入情緒後 MAPE 0.0675 → 0.0449。是本專題「技術面＋情緒雙軌整合」平台的依據。
- **7 月概念驗證**：台積電 2330、PTT 133 篇／1.8 萬則留言、35 個交易日；流程跑通，動能指標與情緒同向（相關約 0.2），但樣本太少不能談預測。後續工作：資料拉長、驗證情緒的預測力。

## 結論

| 研究 | 問題 | 結論 | 文件 |
|---|---|---|---|
| 一、重做選讀論文 | 加情緒能讓 LSTM 預測股價更準嗎？ | **不能**。台股 265 檔、十年：MAPE 13.24% → 13.08%（不顯著）；「今天收盤當明天」只有 1.22%，模型 0 檔贏過。換成最接近論文股吧貼文的 CMoney（2020–2024）：10.64% → 10.56%（p=0.44），天真 1.33%，結論相同 | [01_replication.md](docs/01_replication.md) |
| 二、散戶「說」的 | PTT 情緒能預測隔天漲跌、報酬、波動、成交量嗎？ | **不能**。59.5% 留言在收盤後才發；換分類器、換股價來源、換目標都一樣；2025 樣本外殘留效果反轉（CMoney 見研究八） | [02_sentiment.md](docs/02_sentiment.md) |
| 三、散戶「做」的 | 融資融券、散戶成交比重有預測力嗎？ | **有，但很小**。全市場 1,144 檔每一年都讓漲跌預測變準（ΔAUC +0.003），沒被挑進樣本的 879 檔一樣；散戶持股多的股票更明顯 | [03_retail_trading.md](docs/03_retail_trading.md) |
| 四、大波動日 | 大漲大跌是什麼在影響？之後怎麼走？ | **公告與新聞方向可預期**（265 檔歸納、889 檔驗證全過），但多在隔天開盤就反應完；超過一半的大波動日找不到消息；**沒消息又沒鎖漲停的大漲之後平均回吐**（24 年幾乎每年） | [04_large_moves.md](docs/04_large_moves.md) |
| 五、公司事件 | 庫藏股、減資等事件公告後怎麼走？ | 庫藏股公告後續漲（2001–2013 驗證 +1.23%，t=5.9），但扣成本後贏不了大盤 | [05_events.md](docs/05_events.md) |
| 六、交易頻率 | 拉長持有時間，扣成本後還剩多少？ | 日頻全 ≤ 0、週頻剩一點；**月頻技術面＋散戶行為選股扣成本後每月贏一般股票 0.63%（t=2.9），年化 22%**，但相對加權指數不顯著；文字類資訊在任何時間尺度都加不上去 | [06_holding_period.md](docs/06_holding_period.md) |
| 七、大盤層級 | 盤中、盤後情緒與細分情緒指數 | 恐慌指數在股災時準確升高（描述力好）；盤中反指標顯著但只有 1 bp；重測 2025 論文，控制美股後情緒增量幾乎消失；情緒當「開關」大半是市場狀態的影子 | [07_market_level.md](docs/07_market_level.md) |
| 八、CMoney 完整檢驗 | 換一個散戶平台（貼文型、2020–2025）結論一樣嗎？ | **比 PTT 有用一點、但賺不到錢**。漲跌模型 AUC 每年 +0.003（PTT 0；發現並修正抽樣洩漏後仍成立）；散戶越悲觀的股票下個月越差，控制規模動能後仍顯著，2025 樣本外方向重現（FM t=1.93）但 12 個月證據不足；日頻扣當沖成本全虧，週／月／季選股增量不顯著，7 種情緒選股策略 2025 都沒有可靠超額；反過來買最悲觀的會一直輸 | [09_cmoney.md](docs/09_cmoney.md) |
| 九、季頻高低波動 | 組員說高低波動季頻能贏大盤，我們的資料呢？ | **不扣成本也沒贏含息 0050**（2016–2025：低波動 +11.5%、高波動 +14.6%、0050 +19.3%）；低波動 Sharpe 最高、回落最小；高波動只贏不含息的加權指數（股利差） | [10_quarterly_volatility.md](docs/10_quarterly_volatility.md) |

**一句話**：情緒指標方向都對、統計上顯著，但市場比讀者快，看到再進場扣掉成本就沒有了；扣成本後還有用的是散戶實際的交易行為加技術面、持有一個月。情緒的價值在**描述與提醒**，這決定了平台的設計。

所有事先登記的假說、門檻與結果（含未通過）：[docs/prereg/PREREGISTRATION.md](docs/prereg/PREREGISTRATION.md)
依時間倒序的完整研究紀錄：[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)

## 資料

原始資料**不放在 GitHub**（TEJ 為學校授權、爬來的原文有版權、檔案數 GB）。取得方式與踩過的坑見 [docs/08_data_collection.md](docs/08_data_collection.md)。

| 資料 | 規模 | 期間 | 來源 |
|---|---|---|---|
| PTT 股板留言 | 2,021 萬則（對到個股 231 萬則） | 2015–2025 | PTT Stock 板、pttweb 鏡像站 |
| CMoney 股市同學會 | 約 179 萬篇貼文（含開盤前補頁；平台 2020 年以前沒有貼文） | 2020–2025 | CMoney（需本人登入取 token） |
| 鉅亨網台股新聞 | 18.8 萬則（對到個股 9.4 萬則） | 2015–2025 | 鉅亨網 |
| 公司重大訊息 | 40.9 萬則、1,154 家 | 2015–2025 | 公開資訊觀測站 |
| 股價、籌碼、融資融券、成交比重 | 全市場 1,154 檔（含下市）、270 萬股票日 | 2014–2025 | TEJ Pro |
| 舊年份股價（含注意股、處置股） | 1,055 檔 | 2001–2013 | TEJ Pro |
| 特殊事件 | 4.6 萬筆、9 類 | 1971–2026 | TEJ Pro |
| 三大法人、融資融券、當沖、每分鐘指數、台指夜盤、美股 | — | 2015–2024 | 證交所、期交所 |

## 共通設計

- **不偷看未來**：逐年往前滾動（用過去訓練、預測下一年）；情緒只用隔天開盤前已發布的內容；籌碼只用當天盤後已公布的數字；特徵正規化只用訓練期。
- **時間對齊**：開盤前與盤中發布的消息算當天，收盤後的算下一個交易日。
- **超額報酬**：個股減當天全市場等權平均。
- **開發與驗證分開**：規則在原 265 檔或 2015–2019 找，一字不改套到另外 889 檔、2020–2025、2001–2013 或 2025。
- **多重檢定**：同批 k 個假設用 Bonferroni（p < 0.05/k）；t 值依交易日或事件月份群聚。
- **可交易性**：扣來回成本 0.47%（另報 0.585%）；剔除當天收在漲停／跌停、實際買不到或賣不掉的情況；下市當期記 −50%。
- **對照**：一律與「只有技術面」或「今天收盤當明天」比較。

## 執行

程式在 [`code/`](code/)，每支程式的用途與對應結果見 [`code/INDEX.md`](code/INDEX.md)（自動產生）。多數程式在資料夾根目錄執行、讀寫相對路徑；結果寫到同名的 `.log`／`_results.json`，已收在 [`results/`](results/)。

```bash
pip install -r requirements.txt

# 1. 資料蒐集
python code/batch_crawler_267.py          # PTT 267 檔文章與留言
python code/crawl_pttweb_api.py           # PTT 舊文（鏡像站）
python code/download_chip_data.py         # 證交所三大法人、融資融券（另有 _2015 補舊年）
python code/crawl_cmoney_daily.py         # CMoney 同學會（需 cmoney_token.json）
python code/crawl_cnyes_news.py           # 鉅亨網新聞（約 7 小時，務必單支慢速）
python code/crawl_mops_material.py --all  # 公開資訊觀測站重大訊息（全市場約 5 小時）
python code/tej_build_dataset.py          # TEJ 匯出檔 → 265 檔資料集（需授權資料）
python code/tej_build_fullmarket.py       # TEJ → 全市場資料集

# 2. 情緒分類
python code/breeze_infer_267_fast.py      # Breeze 三分類（GPU）
python code/finetune_gold_cv.py           # 微調 BERT 的交叉驗證；finetune_gold_infer.py 重標全部
python code/cmoney_clean.py               # CMoney 清理
python code/cnyes_build.py                # 新聞清理與配股票

# 3. 資料集
python code/build_dataset_v3.py           # → final_dataset_v3/all_stocks_v3.csv

# 4. 各研究（見 docs/ 與 code/INDEX.md）
python code/lstm_price_v4.py --arm 技術面 --seed 0   # 研究一（批次：bash code/run_lstm_price_v4.sh）
python code/retail_test_v4.py                         # 研究二、三：XGBoost 逐年滾動
python code/tej_fullmarket_test.py                    # 研究三：全市場
python code/mops_bigmove.py && python code/mops_bigmove.py out    # 研究四：公告（探索 → 驗證）
python code/cnyes_bigmove.py && python code/cnyes_bigmove.py out  # 研究四：新聞
python code/bigmove_decompose.py out                  # 研究四：歸因與之後走勢
python code/tej_event_oos_pre2014.py                  # 研究五：庫藏股樣本外
python code/monthly_fullmarket_strategy.py            # 研究六：月頻選股
python code/weekly_fullmarket_strategy.py             # 研究六：週頻
python code/sentiment_regime_test.py                  # 研究七：情緒當開關
CM_TAG=cm3 bash code/run_cmoney_full.sh              # 研究八：CMoney LSTM（其餘見 docs/09_cmoney.md 第 11 節）
python code/xgb_direction_cm.py && python code/oos_2025.py   # 研究八：漲跌模型、2025 樣本外
python quarterly_vol/quarterly_vol.py                 # 研究九：季頻高低波動

# 5. 簡報
cd slides && npm install && node build_deck_v19.js
```

環境：Python 3.10；部分模型用 NVIDIA GPU（XGBoost `device="cuda"`、BERT、Breeze）；簡報用 Node.js＋pptxgenjs。
本專案沒有自動測試（程式未寫成套件）；正確性靠事先登記、樣本外驗證、與 TEJ 交叉核對資料。

## 平台設計

| 模組 | 內容 | 依據 |
|---|---|---|
| 情緒指數模組 | 散戶溫度計、恐慌／貪婪等 8 類情緒、個股討論熱度與淨情緒、隔夜情緒 | 研究二、七：描述力強 |
| 技術面＋籌碼看板 | 技術指標、融資融券、散戶成交比重、月頻選股分數 | 研究三、六：唯一扣成本後還有用的；標示不保證贏大盤 |
| 事件提示 | 公告類別的歷史方向、「沒消息的大漲之後通常回吐」、恐慌時訊號可靠度下降 | 研究四、五、七：樣本外驗證過 |

所有功能標示「這是描述與提醒，不是漲跌預測」。

## 限制

- PTT 個股留言集中在少數熱門股（台積電占三成）；全市場沒有 PTT 配對。
- 新聞只有鉅亨網、只看標題；「找不到消息」可能只是沒抓到。
- 情緒分類器準確率約六成（重標與微調實驗顯示對結論影響小）。
- 研究一主分析用 PTT 留言，論文用股吧貼文；最接近的 CMoney 貼文只有 2020 年以後（已重跑，結論相同）。
- CMoney 熱門股是抽樣（每天收盤前 20 篇＋開盤前 20 篇）；265 檔是事後挑的 PTT 熱門股，策略比大盤有股票池偏誤。
- 「沒消息的大漲會回吐」在 2015 年以後的切法不是事先寫定的；舊年份上的版本才是。
- 月頻選股靠模型組合多個特徵，可解釋性較差；最近兩年輸加權指數。

## 資料夾

| 路徑 | 內容 |
|---|---|
| `code/` | 全部程式（213 支）與自動產生的 `INDEX.md` |
| `results/` | 各檢驗的輸出紀錄 |
| `docs/` | 各研究說明（01–10）、事先登記假說總表、完整研究紀錄 |
| `slides/` | 報告簡報、白話講稿、產生簡報的程式 |

本專案由作者電腦上的 `code/sync_github.sh` 每小時自動同步（程式、結果、文件、最新版簡報；上傳前檢查不含資料檔與金鑰）。

---

# Technical Analysis and Market Sentiment for a Taiwan Stock Information Platform

Undergraduate capstone project. Using ten years of Taiwan stock data across the whole market, we test whether retail investor sentiment predicts stock prices, and how sentiment should be presented in a "technical + sentiment" trading information platform. In the later stages every test had its hypotheses and thresholds written down before running, and was validated on unseen stocks or years; passes and failures are both reported.

**Starting point.** Jing, Wu & Wang (2021, *Expert Systems with Applications* 178, 115019) report that adding forum sentiment to 16 technical indicators in an LSTM lowers next-day closing-price MAPE from 0.0675 to 0.0449.

**Findings.**

| Study | Question | Finding |
|---|---|---|
| 1. Replication | Does sentiment improve LSTM price prediction? | No. 265 Taiwan stocks, 10 years: MAPE 13.24% → 13.08% (n.s.); the naive "today's close = tomorrow's" forecast has 1.22% and the LSTM never beats it. With CMoney posts (closest to the paper's forum posts, 2020–2024): 10.64% → 10.56% (p = 0.44), naive 1.33%. |
| 2. What retail investors say | Does PTT sentiment (CMoney: see 8) predict next-day direction, returns, volatility or volume? | No. 59.5% of comments are posted after the close; results hold after changing the classifier, platform, price source and target; the small residual effect reverses in 2025. |
| 3. What retail investors do | Do margin trading and the retail share of volume predict returns? | Yes, but small: ΔAUC ≈ +0.003 every year across 1,144 stocks, including 879 stocks never used in development; stronger where retail ownership is high. |
| 4. Large moves | What drives large price moves, and what happens next? | Announcement and news categories have predictable signs (developed on 265 stocks, all six pre-registered tests pass on 889 unseen stocks) but are priced at the next open; over half of large moves have no identifiable news; large up-moves with no news and no limit-up close partially reverse over the next five days (almost every year over 24 years). |
| 5. Corporate events | What happens after buyback announcements etc.? | Buyback announcements are followed by further gains (+1.23% on days 2–5, 2001–2013 out of sample) that do not beat the market after costs. |
| 6. Holding period | How much survives costs at longer horizons? | Daily: nothing. Weekly: a little (n.s.). Monthly technical + retail-trading stock selection beats the average stock by 0.63% per month after costs (t = 2.9; 22% annualised) but not the cap-weighted index; text-based information adds nothing at any horizon. |
| 7. Market level | Intraday / after-hours sentiment and emotion indices | Fear index spikes accurately in crashes (good description); an intraday contrarian signal is significant but ~1 bp; a 2025 paper's after-hours result largely disappears after controlling for US markets; sentiment as a "regime switch" is mostly a proxy for recent market moves. |
| 8. CMoney (second retail platform) | Does a post-based forum (2020–2025) change the conclusion? | Slightly more informative than PTT, still not profitable: ΔAUC ≈ +0.003 every year (PTT 0; holds after fixing a sampling leak); stocks with the most pessimistic posts underperform next month even after size/momentum controls, and the sign repeats out of sample in 2025 (FM t = 1.93, but only 12 months); daily trading loses after costs, and weekly/monthly/quarterly selection gains are not significant; none of seven sentiment strategies earns a reliable excess return in 2025. Buying the most pessimistic stocks consistently loses. |
| 9. Quarterly volatility sort | A teammate reported high/low-volatility portfolios beat the market quarterly | Before costs, neither beats 0050 with dividends (2016–2025: low-vol 11.5%, high-vol 14.6%, 0050 19.3%); low-vol has the best Sharpe and smallest drawdown; high-vol only beats the price index without dividends. |

**Bottom line.** Sentiment indicators point the right way and are statistically significant, but the market moves faster than readers; after costs, only retail trading behaviour plus technical indicators at a monthly horizon remains useful. Sentiment is best used to describe and alert, which shapes the platform design.

Raw data (PTT, CMoney, cnyes, MOPS, TEJ) are not included for licensing and copyright reasons; see `docs/08_data_collection.md` for how to rebuild them.
