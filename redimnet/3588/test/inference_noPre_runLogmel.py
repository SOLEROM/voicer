#!/usr/bin/env python3
"""
infer_logmel_rknn_fp16.py
-------------------------
Run a ReDimNet-NoMel RKNN model on a *pre-computed* log-Mel tensor (.npy)
using **FP-16** throughput.

Workflow
--------
# After compute_logmel(wave) returns [1,1,60,134] (NCHW, float32):
np.save("clip_logmel.npy", logmel_nchw.astype(np.float16))

Usage
-----
python infer_logmel_rknn_fp16.py  model.rknn  logmel_X  [rk3588]

The .npy file must be NCHW [B,1,60,134], float32 or float16.
"""

# ───────────────────────── Imports ─────────────────────────
import os
import sys
import numpy as np
from rknn.api import RKNN

# os.environ["RKNN_LOG_LEVEL"] = "3"

# ─────────────────── helper: load log-Mel ───────────────────
def load_logmel(path: str) -> np.ndarray:
    """
    Load .npy tensor and return float16 **NCHW** [B,1,60,134].
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError(f"{path}: expected NCHW [B,1,60,134], got {arr.shape}")
    return arr.astype(np.float16, copy=False)

# ─────────────────── main routine ───────────────────
def main(rknn_path: str, npy_path: str, target: str = "rk3588"):
    print(f"[1/3] Loading RKNN model: {rknn_path}")
    rk = RKNN()
    if rk.load_rknn(rknn_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN model")

    print(f"[2/3] Initialising runtime on target: {target}")
    if rk.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    logmel16 = load_logmel(npy_path)
    print(f"[INFO] log-Mel tensor shape {logmel16.shape}, dtype {logmel16.dtype}")

    print("[3/3] Running inference …")
    emb16 = rk.inference(inputs=[logmel16], data_format='nchw')[0] \
              .astype(np.float16, copy=False)
    rk.release()

    print(f"\n✅ Embedding shape: {emb16.shape}, dtype {emb16.dtype}")
    print(f"   min={emb16.min():.4f}  max={emb16.max():.4f}")
    print("   preview:", emb16[0, :10])

# ───────────────────── CLI wrapper ─────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_file  = sys.argv[1]
    logmel_npy  = sys.argv[2]
    target_chip = sys.argv[3] if len(sys.argv) > 3 else "rk3588"

    main(model_file, logmel_npy, target_chip)
