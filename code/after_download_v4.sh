#!/usr/bin/env bash
# 等 2015-2019 籌碼下載完成 → 重建十年資料集 → 跑逐年滾動分析。事件寫在 watchdog_download.log
cd "/d/專題claude" || exit 1
until grep -q "DONE\|GIVEUP" watchdog_download.log 2>/dev/null; do sleep 60; done
grep -q "GIVEUP" watchdog_download.log && exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') START build_dataset_v3" >> watchdog_download.log
PYTHONUTF8=1 python -u build_dataset_v3.py > build_dataset_v3.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') BUILD_DONE exit=$?" >> watchdog_download.log
PYTHONUTF8=1 python -u direction_test_v4.py > direction_test_v4.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') V4_DONE exit=$?" >> watchdog_download.log
