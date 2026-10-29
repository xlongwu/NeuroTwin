"""Data-scale conversion matching the evaluation loader's float32 raw path."""
import numpy as np


def calibrate_raw(raw: np.ndarray, length: int = 32) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(raw, dtype=np.float32).T)
    cal = x[:, :length]
    mu = cal.mean(axis=1, keepdims=True)
    scale = np.sqrt(cal.var(axis=1, keepdims=True) + 1e-6)
    return ((x-mu)/scale).T
