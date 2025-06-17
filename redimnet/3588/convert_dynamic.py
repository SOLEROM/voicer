#!/usr/bin/env python3
"""
convert_onnx_to_rknn.py – ONNX ➜ RKNN for a *fixed* 200-frame NHWC log-mel
spectrogram  [B, 60, 200, 1].

Usage
-----
python convert_onnx_to_rknn.py   model.onnx   rk3588   fp   model.rknn

positional args
---------------
  model.onnx   path to the exported NHWC ONNX
  platform     rk3562 | rk3566 | rk3568 | rk3576 | rk3588 | rv11xx | rk1808
  dtype        fp | i8 | u8      (fp = FP16, no quantisation)
  out.rknn     output file (default: model.rknn)

Optional environment variables
------------------------------
  DATASET_TXT  points to a dataset.txt for calibration (default: dataset.txt)
"""
from __future__ import annotations
import os, sys, tempfile, atexit
from pathlib import Path

import numpy as np
from rknn.api import RKNN

# ---------- constants ----------
FIXED_SHAPE_NHWC = [1, 60, 200, 1]      # B H W C
VALID_PLATFORMS  = {
    'rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588',
    'rv1103', 'rv1106', 'rv1109', 'rv1126', 'rk1808',
}
DEFAULT_OUT   = 'model.rknn'
DEFAULT_DTYPE = 'fp'
DATASET_TXT   = os.getenv('DATASET_TXT', 'dataset.txt')
# ---------------------------------


def parse_cli():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    onnx_path = sys.argv[1]
    platform  = sys.argv[2].lower()
    if platform not in VALID_PLATFORMS:
        sys.exit(f'ERROR: unknown platform {platform!r}')

    dtype = sys.argv[3].lower() if len(sys.argv) > 3 else DEFAULT_DTYPE
    if dtype not in ('fp', 'i8', 'u8'):
        sys.exit('ERROR: dtype must be fp / i8 / u8')

    out_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT
    quant    = dtype in ('i8', 'u8')
    return onnx_path, platform, quant, out_path


def first_sample_path():
    """
    Returns an existing sample from dataset.txt or creates a random
    [1,60,200,1] tensor saved as *.npy for the overflow check.
    """
    if os.path.isfile(DATASET_TXT):
        with open(DATASET_TXT) as f:
            p = f.readline().strip()
        if p and os.path.isfile(p):
            return p

    arr = np.random.randn(*FIXED_SHAPE_NHWC).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(
        suffix='.npy', delete=False, prefix='_tmp_sample_', dir='.')
    np.save(tmp.name, arr)
    tmp.close()
    atexit.register(lambda: os.remove(tmp.name))
    return tmp.name


def main():
    onnx_path, platform, do_quant, out_path = parse_cli()
    rknn = RKNN(verbose=True)

    print('[1/5] rknn.config()')
    rknn.config(
        mean_values=[[0.0]],         # single-channel spectrogram
        std_values=[[1.0]],
        target_platform=platform,
        optimization_level=0
    )

    print('[2/5] rknn.load_onnx()', onnx_path)
    if rknn.load_onnx(onnx_path, input_size_list=[FIXED_SHAPE_NHWC]) != 0:
        sys.exit('load_onnx failed')

    print('[3/5] rknn.build()', 'with quant' if do_quant else 'FP16')
    if rknn.build(do_quantization=do_quant,
                  dataset=DATASET_TXT if do_quant else None) != 0:
        sys.exit('build failed')

    sample = first_sample_path()
    print('[4/5] rknn.accuracy_analysis() – simulator run on', sample)
    rknn.accuracy_analysis(inputs=[sample], target=None)

    print('[5/5] rknn.export_rknn()', out_path)
    if rknn.export_rknn(out_path) != 0:
        sys.exit('export failed')

    print('✔ RKNN saved at', out_path)
    rknn.release()


if __name__ == '__main__':
    main()
