#!/usr/bin/env python3
"""
infer_logmel_rknn.py
--------------------
Run a ReDimNet-NoMel RKNN model on a *pre-computed* log-Mel tensor saved
as an .npy file (created earlier with np.save).

Typical tensor to save beforehand
---------------------------------
# After compute_logmel(wave) returning [1,1,60,200] (NCHW):
np.save("test_logmel.npy", logmel_nchw)

Usage
-----
python infer_logmel_rknn.py  model.rknn  logmel.npy  [rk3588]
"""

# ───────────────────────────── Imports ──────────────────────────────
import os
import sys
import numpy as np
from rknn.api import RKNN

os.environ["RKNN_LOG_LEVEL"] = "3"

# ────────────────────────── helper: load & check ────────────────────
def load_logmel(path: str) -> np.ndarray:
    """
    Loads an .npy file and guarantees a 4-D float32 array ready for RKNN.
    Accepts either NCHW [1,1,60,200] **or** NHWC [1,60,200,1].
    Returns NHWC (batch, H, W, C).
    """
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.float32:
        print("[INFO] Converting tensor to float32 …")
        arr = arr.astype(np.float32)

    if arr.ndim != 4:
        raise ValueError("Expected 4-D tensor in the .npy file "
                         "(NCHW or NHWC). Got shape {}.".format(arr.shape))

    return arr

# ────────────────────────── main routine ────────────────────────────
def main(rknn_path: str, npy_path: str, target: str = "rk3588"):
    print(f"[1/3] Loading RKNN model: {rknn_path}")
    rk = RKNN()
    if rk.load_rknn(rknn_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN model")

    print(f"[2/3] Initialising runtime on target: {target}")
    if rk.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    # Load pre-computed log-Mel tensor
    logmel_nhwc = load_logmel(npy_path)
    print(f"[INFO] Loaded log-Mel tensor with shape {logmel_nhwc.shape}")

    # ───────── Run inference ─────────
    print("[3/3] Running inference …")
    emb = rk.inference(inputs=[logmel_nhwc])[0]
    rk.release()

    print(f"\n✅ Embedding shape: {emb.shape}")
    print(f"   min={emb.min():.4f}  max={emb.max():.4f}")
    print("   preview:", emb[0, :10])

# ─────────────────────────── CLI wrapper ────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_file = sys.argv[1]
    logmel_npy = sys.argv[2]
    target_chip = sys.argv[3] if len(sys.argv) > 3 else "rk3588"

    main(model_file, logmel_npy, target_chip)
