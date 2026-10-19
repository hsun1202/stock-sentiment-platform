#!/bin/bash
# 補抓開盤前 → 分類＋重建 final_dataset_v3_cm3 → XGBoost 與洩漏檢查（cm3）。LSTM 不在這裡（使用者晚上說了再跑：CM_TAG=cm3 bash run_cmoney_full.sh）。
cd /d/專題claude
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> cmoney_full/_run.log; }
log "CHAIN START（補抓 → 重建 cm3 → XGBoost）"
PYTHONUTF8=1 python -u cmoney_full/crawl_cmoney_preopen.py --rate 15 --workers 12 > cmoney_full/crawl_preopen.out 2>&1
if ! grep -q "全部完成" cmoney_daily_pre/crawl.log; then log "CRAWL_INCOMPLETE（重跑本腳本會續抓）"; exit 1; fi
log CRAWL_DONE
PYTHONUTF8=1 python -u cmoney_full/rebuild_cm3.py > cmoney_full/rebuild_cm3.out 2>&1 && log REBUILD_DONE || { log REBUILD_FAILED; exit 1; }
CM_TAG=cm3 PYTHONUTF8=1 python -u cmoney_full/xgb_direction_cm.py > cmoney_full/xgb_direction_cm_cm3.out 2>&1 && log "XGB_DONE (cm3)" || log "XGB_FAILED (cm3)"
CM_TAG=cm3 PYTHONUTF8=1 python -u cmoney_full/xgb_leak_check.py > cmoney_full/xgb_leak_check_cm3.out 2>&1 && log "LEAKCHECK_DONE (cm3)" || log "LEAKCHECK_FAILED (cm3)"
log CHAIN_DONE
