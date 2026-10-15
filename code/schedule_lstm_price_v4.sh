#!/bin/bash
# 等到凌晨 00:00 再開始跑（使用者晚上要用電腦）
cd /d/專題claude
target=$(date -d "tomorrow 00:00" +%s); now=$(date +%s)
[ $target -gt $now ] && sleep $((target - now))
bash run_lstm_price_v4.sh
