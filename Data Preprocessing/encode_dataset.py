#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将原始 mianzhen.csv / shezhen.csv → dataset_encoded.csv
用法:
    python encode_dataset.py --csv mianzhen.csv --map label_map.csv --out dataset_encoded.csv
"""
import pandas as pd, csv, argparse, re
from pathlib import Path

def load_map(map_csv: str) -> dict[tuple[str, str], int]:
    """读取 label_map.csv → {(task, label): idx}"""
    df = pd.read_csv(map_csv)
    return {(row['task'], row['label']): int(row['index'])
            for _, row in df.iterrows()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='原始标注 CSV')
    ap.add_argument('--map', default='label_map.csv')
    ap.add_argument('--out', default='dataset_encoded.csv')
    return ap.parse_args()

def _parse_csv(path: Path) -> pd.DataFrame:
    """读取单个 csv，把第 1 行作为子标签合并进列名后返回 DataFrame"""
    df = pd.read_csv(path)
    sub_headers = df.iloc[0]                   # 第 1 行：子标签
    new_cols, current_parent = [], None

    for col, sub in zip(df.columns, sub_headers):
        col_clean = str(col).strip()           # 父标签
        if not col_clean.startswith('Unnamed'):
            current_parent = col_clean         # 更新当前父标签

        if col_clean == '图片名':
            new_cols.append('图片名')
            continue

        sub_clean = str(sub).strip()
        new_cols.append(f'{current_parent}_{sub_clean}')

    df.columns = new_cols
    df = df.iloc[1:].reset_index(drop=True)    # 丢掉子标签行
    return df

def main():
    args = parse_args()
    label_map = load_map(args.map)

    df = _parse_csv(args.csv)                       # 用你已有的两行表头解析函数

    rows = []
    for _, row in df.iterrows():
        sample = {'image': row['图片名']}
        for col in df.columns:
            if col == '图片名': continue
            cell = str(row[col]) if not pd.isna(row[col]) else ''
            # 支持英文, 中文，顿号
            labels = re.split(r'[,\uFF0C\u3001]+', cell.strip())
            idxs   = sorted({label_map.get((col, lab.strip())) 
                             for lab in labels if lab.strip()})
            sample[col] = ' '.join(map(str, idxs))   # 多个索引空格分隔
        rows.append(sample)

    pd.DataFrame(rows).to_csv(args.out, index=False, encoding='utf-8')
    print(f'✔️  已输出 {args.out}，共 {len(rows)} 行')

if __name__ == '__main__':
    main()
