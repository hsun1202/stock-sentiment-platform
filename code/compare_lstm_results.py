# -*- coding: utf-8 -*-
"""
比較「含情緒因子」vs「不含情緒因子」LSTM模型的MAPE
============================================================
使用方式：
    python compare_lstm_results.py
"""

import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-dir", type=str, default="lstm_model_with_sentiment")
    parser.add_argument("--without-dir", type=str, default="lstm_model_without_sentiment")
    args = parser.parse_args()

    with open(f"{args.with_dir}/metrics.json", encoding="utf-8") as f:
        m_with = json.load(f)
    with open(f"{args.without_dir}/metrics.json", encoding="utf-8") as f:
        m_without = json.load(f)

    print("=" * 60)
    print("LSTM 股價預測：含情緒因子 vs 不含情緒因子 MAPE比較")
    print("=" * 60)
    print(f"\n整體 MAPE")
    print(f"  含情緒因子　　：{m_with['overall_mape']*100:.2f}%")
    print(f"  不含情緒因子　：{m_without['overall_mape']*100:.2f}%")
    diff = (m_with['overall_mape'] - m_without['overall_mape']) * 100
    verdict = "情緒因子有幫助" if diff < 0 else "情緒因子沒幫助（甚至更差）"
    print(f"  差異　　　　　：{diff:+.2f}個百分點（{verdict}）")

    print(f"\n各產業 MAPE 對照")
    print(f"{'產業':<12}{'含情緒':>10}{'不含情緒':>10}{'差異(pp)':>12}{'情緒有沒有幫助':>16}")
    industries = sorted(m_with["industry_mape"].keys())
    for ind in industries:
        w = m_with["industry_mape"][ind] * 100
        wo = m_without["industry_mape"][ind] * 100
        d = w - wo
        v = "有" if d < 0 else "沒有"
        print(f"{ind:<12}{w:>9.2f}%{wo:>9.2f}%{d:>+11.2f}pp{v:>16}")

    print(f"\ntrain/val/test 樣本數：{m_with['n_train']}/{m_with['n_val']}/{m_with['n_test']}")
