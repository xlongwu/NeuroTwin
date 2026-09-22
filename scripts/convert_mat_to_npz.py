# coding=utf-8
"""将散落的小 mat 文件按被试聚合为 npz，减少训练时的小文件 I/O。

源数据（只读，不修改）：
    data/Mask/{HC,MDD}/Mask_{subj_id}.mat
    data/ROISignals_window/{group}/ROISignals_{subj_id}-{i}.mat  (i=1..total_windows)

输出：
    data/npz_cache/{group}/{subj_id}.npz
        sc      —— 原始 Mask 矩阵（未归一化，归一化仍由 dataloader 统一执行）
        windows —— [total_windows, F, S] float32（每窗统一为 [F=ROI 数, S=时间点]）

与 utils/dataloader.py 的数据发现/校验逻辑保持一致：
被试 id 取 Mask 文件名去 'Mask_' 前缀；窗口不足的被试跳过（训练时同样不可用）。
已存在的目标 npz 自动跳过，脚本可中断续跑。
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataloader import NeuroTwinDataset  # noqa: E402  复用 mat 提取逻辑，保证转换与训练读取一致


def convert_subject(task):
    subj_id, sc_path, window_paths, out_path, seq_len = task
    ds = NeuroTwinDataset.__new__(NeuroTwinDataset)  # 仅用其 mat 提取方法，绕过 __init__
    ds.cache_in_memory = False
    ds._mat_cache = {}
    sc = ds._read_mat_array(sc_path, preferred_keys=['SC', 'sc_matrix', 'mask', 'Mask'])
    sc = np.asarray(sc)

    windows = []
    for w_path in window_paths:
        arr = ds._read_mat_array(w_path, preferred_keys=['ROISignals', 'data', 'bold'])
        arr = np.asarray(arr)
        if arr.shape[0] == seq_len:
            arr = arr.T                      # [S, F] -> [F, S]
        elif arr.shape[1] != seq_len:
            raise ValueError(f"{subj_id} 窗口 {w_path.name} 形状无法推断: {arr.shape}")
        windows.append(arr.astype(np.float32))

    windows = np.stack(windows, axis=0)      # [W, F, S]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, sc=sc, windows=windows)
    return subj_id, windows.shape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', default=str(PROJECT_ROOT / 'data'))
    p.add_argument('--groups', nargs='+', default=['HC', 'MDD'])
    p.add_argument('--total_windows', type=int, default=9)
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    data_root = Path(args.data_root)
    for group in args.groups:
        sc_dir = data_root / 'Mask' / group
        window_dir = data_root / 'ROISignals_window' / group
        out_dir = data_root / 'npz_cache' / group

        sc_files = sorted(sc_dir.glob('Mask_*.mat'), key=lambda p: p.stem)
        if not sc_files:
            print(f"[{group}] 未找到 Mask 文件，跳过")
            continue

        tasks, skipped_window, done_before = [], 0, 0
        for sc_file in sc_files:
            subj_id = sc_file.stem.replace('Mask_', '').strip()
            out_path = out_dir / f"{subj_id}.npz"
            if out_path.exists():
                done_before += 1
                continue
            window_paths = [window_dir / f"ROISignals_{subj_id}-{i}.mat"
                            for i in range(1, args.total_windows + 1)]
            if not all(f.exists() for f in window_paths):
                skipped_window += 1
                continue
            tasks.append((subj_id, sc_file, window_paths, out_path, args.seq_len))

        print(f"[{group}] Mask 被试 {len(sc_files)} | 已转换 {done_before} | 缺窗跳过 {skipped_window} | 待转换 {len(tasks)}")
        if not tasks:
            continue

        t0 = time.time()
        with Pool(args.workers) as pool:
            for n, (subj_id, shape) in enumerate(pool.imap_unordered(convert_subject, tasks), 1):
                if n % 200 == 0 or n == len(tasks):
                    print(f"[{group}] {n}/{len(tasks)} | 最近: {subj_id} {shape} | {time.time()-t0:.0f}s")

        total_mb = sum(f.stat().st_size for f in out_dir.glob('*.npz')) / 1024 / 1024
        print(f"[{group}] 完成 | 输出 {len(list(out_dir.glob('*.npz')))} 个 npz | 共 {total_mb:.0f} MB | 耗时 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
