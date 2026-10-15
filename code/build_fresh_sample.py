# -*- coding: utf-8 -*-
"""
為「LLM zero-shot對照測試」抽一批全新樣本（不重複用過的250則）
============================================================

目的：
    之前的annotation_sample.csv（250則）已經被人工標註、也在對話
    裡被完整看過，不能再拿來測試新方法（會有答案外洩問題）。
    這支程式重新從batch_output抽一批候選池，排除掉跟舊樣本重複
    的留言內容，再隨機抽出全新的一批供LLM zero-shot測試+人工
    標註比對用。

使用方式：
    python build_fresh_sample.py --total 150
"""

import os
import argparse
import importlib.util
import pandas as pd


def load_stock_list():
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抽全新樣本供LLM zero-shot對照測試")
    parser.add_argument("--batch-dir", type=str, default="batch_output")
    parser.add_argument("--old-sample", type=str, default="annotation_sample.csv")
    parser.add_argument("--out", type=str, default="fresh_sample.csv")
    parser.add_argument("--pool-per-stock", type=int, default=500)
    parser.add_argument("--total", type=int, default=150)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    STOCK_LIST = load_stock_list()
    old = pd.read_csv(args.old_sample)
    used_content = set(old["content"].astype(str).str.strip().tolist())
    print(f"排除舊樣本內容：{len(used_content)} 則")

    pool_frames = []
    for code, name, industry in STOCK_LIST:
        path = os.path.join(args.batch_dir, f"{code}_{name}_comments.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["stock_code", "resolved_date", "content"])
        df["content"] = df["content"].astype(str).str.strip()
        df = df[df["content"].str.len() > 0]
        df = df[~df["content"].isin(used_content)]
        if len(df) == 0:
            continue
        n = min(args.pool_per_stock, len(df))
        sampled = df.sample(n=n, random_state=args.seed)
        sampled["industry"] = industry
        sampled["stock_name"] = name
        pool_frames.append(sampled)

    pool = pd.concat(pool_frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["content"])
    print(f"候選池大小（已排除重複）：{len(pool)}")

    picked = pool.sample(n=min(args.total, len(pool)), random_state=args.seed).reset_index(drop=True)
    picked.insert(0, "id", range(1, len(picked) + 1))
    picked["human_label"] = ""

    out_cols = ["id", "stock_code", "stock_name", "industry", "resolved_date", "content", "human_label"]
    picked[out_cols].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"已輸出 {len(picked)} 則至 {args.out}")
