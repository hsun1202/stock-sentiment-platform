#!/usr/bin/env bash
# 等情緒分類看門狗回報 DONE 後，自動跑相關性過濾的驗證（relevance_infer.py --eval，約數分鐘）
cd "/d/專題claude" || exit 1
until grep -q "DONE\|GIVEUP" watchdog_breeze.log 2>/dev/null; do sleep 60; done
grep -q "GIVEUP" watchdog_breeze.log && exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') START relevance eval" >> watchdog_breeze.log
PYTHONUTF8=1 python -u relevance_infer.py --eval > relevance_eval.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') EVAL_DONE exit=$?" >> watchdog_breeze.log
