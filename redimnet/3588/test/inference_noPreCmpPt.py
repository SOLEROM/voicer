#!/usr/bin/env python3
"""
compare_logmel_to_ref_fp16.py
-----------------------------
Compare a pre-computed log-Mel tensor (.npy) against a reference embedding
(.pt) using a ReDimNet-NoMel RKNN – **all data passed as float16**.

Usage
-----
python compare_logmel_to_ref_fp16.py  model.rknn  logmel.npy  ref_embed.pt  [rk3588]

Notes
-----
• `logmel.npy` may be float32 or float16; it will be converted to float16 NHWC
  ([1, 60, 200, 1]) before inference.
• `ref_embed.pt` is expected to have been saved with `torch.save(tensor, path)`
  (float32 or float16).  It is converted to float16 on load.
• Cosine similarity is finally computed in float32 for accuracy.
"""

# ───────────────────────── Imports ─────────────────────────
import os
import sys
import numpy as np
from rknn.api import RKNN
import torch                        # only for torch.load()

os.environ["RKNN_LOG_LEVEL"] = "3"
_EPS = 1e-6                         # for cosine similarity

# ─────────────────── helper: log-Mel loader ───────────────────
def load_logmel(path: str) -> np.ndarray:
    """
    Returns NHWC float16 tensor, shape [B, 60, 200, 1].
    Accepts stored NCHW/NHWC, float32/float16.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 4:
        raise ValueError(f"{path}: expected 4-D tensor, got {arr.shape}")

    # Ensure NHWC layout
    if arr.shape[-1] == 1:          # already NHWC
        nhwc = arr
    elif arr.shape[1] == 1:         # NCHW → NHWC
        nhwc = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"{path}: cannot infer channel position")

    return nhwc.astype(np.float16, copy=False)


# ─────────────────── helper: reference loader ───────────────────
def load_ref_embedding(pt_path: str) -> np.ndarray:
    """
    Loads a .pt tensor (float32 or float16) and returns float16 NumPy array.
    """
    try:
        ref = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load reference: {e}")

    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().to(torch.float16).numpy()
    else:
        ref = np.asarray(ref, dtype=np.float16)

    return ref


# ─────────────────── cosine similarity (fp16→fp32) ───────────────────
def cosine_similarity(a16: np.ndarray, b16: np.ndarray) -> float:
    a = a16.astype(np.float32, copy=False).flatten()
    b = b16.astype(np.float32, copy=False).flatten()
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))


# ──────────────────── main routine ────────────────────
def main(rknn_model: str, npy_path: str, ref_pt: str, target: str):
    print(f"[1/4] Loading RKNN model: {rknn_model}")
    rk = RKNN()
    if rk.load_rknn(rknn_model) != 0:
        raise RuntimeError("❌ Failed to load RKNN model")

    print(f"[2/4] Initialising runtime for target: {target}")
    if rk.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    # ── Load FP-16 log-Mel tensor & run inference ──
    logmel16 = load_logmel(npy_path)                 # float16 NHWC
    print(f"[INFO] log-Mel tensor shape {logmel16.shape}, dtype {logmel16.dtype}")

    print("[3/4] Running inference …")
    probe_emb16 = rk.inference(inputs=[logmel16])[0].astype(np.float16, copy=False)
    rk.release()

    # ── Load reference embedding ──
    ref_emb16 = load_ref_embedding(ref_pt)
    print(f"[INFO] Reference embedding shape {ref_emb16.shape}, dtype {ref_emb16.dtype}")

    # ── Cosine similarity ──
    sim = cosine_similarity(probe_emb16, ref_emb16)
    print(f"\n✅ Cosine Similarity : {sim:.4f}")
    print(f"🔎 Distance (1 - sim): {1 - sim:.4f}")


# ──────────────────── CLI wrapper ────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    model_file  = sys.argv[1]
    logmel_npy  = sys.argv[2]
    ref_pt      = sys.argv[3]
    target_chip = sys.argv[4] if len(sys.argv) > 4 else "rk3588"

    main(model_file, logmel_npy, ref_pt, target_chip)
