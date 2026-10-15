#!/usr/bin/env bash
# 等閒聊留言分類完成 → 自動跑大盤散戶情緒指數分析
cd "/d/專題claude" || exit 1
until grep -q "完成 →" market_sentiment_classify.log 2>/dev/null; do sleep 60; done
PYTHONUTF8=1 python -u market_sentiment_analysis.py > market_sentiment_analysis.run.log 2>&1
echo "ANALYSIS_DONE exit=$?" >> market_sentiment_classify.log
