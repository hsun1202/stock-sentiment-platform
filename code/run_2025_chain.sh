#!/bin/bash
# 2025 年 CMoney：收盤前抽樣爬蟲 → 熱門股開盤前補一頁。之後的分類與樣本外檢驗另外跑（build_2025.py、oos_2025.py）。
cd /d/專題claude
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> cmoney_full/_run.log; }
log "2025 CRAWL START"
PYTHONUTF8=1 python -u cmoney_full/crawl_cmoney_2025.py --rate 15 --workers 12 > cmoney_full/crawl_2025.out 2>&1
n=$(ls cmoney_daily_2025/*.done 2>/dev/null | wc -l)
if [ "$n" -lt 267 ]; then log "2025 CRAWL_INCOMPLETE ($n/267，重跑會續抓)"; exit 1; fi
log "2025 CRAWL_DONE"
PRE_SRC=cmoney_daily_2025 PRE_OUT=cmoney_daily_pre_2025 PRE_YEAR=2025 PYTHONUTF8=1 python -u cmoney_full/crawl_cmoney_preopen.py --rate 15 --workers 12 > cmoney_full/crawl_pre_2025.out 2>&1
grep -q "全部完成" cmoney_daily_pre_2025/crawl.log && log "2025 PREOPEN_DONE" || { log "2025 PREOPEN_INCOMPLETE"; exit 1; }
log "2025 CHAIN_DONE"
