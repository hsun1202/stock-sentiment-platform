#!/usr/bin/env bash
# 等原論文版 LSTM 全部跑完（lstm_price_v4/_run.log 出現 SUMMARY_DONE）→ 分類開盤半小時全部留言（GPU）
cd "/d/專題claude" || exit 1
until grep -q "SUMMARY_DONE" lstm_price_v4/_run.log 2>/dev/null; do sleep 120; done
PYTHONUTF8=1 python -u intraday_classify_full.py > intraday_classify_full.log 2>&1
echo "FIRST30_DONE exit=$?" >> intraday_classify_full.log
