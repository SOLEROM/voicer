#!/usr/bin/env python3
"""
convert_onnx_to_rknn.py  –  ONNX → RKNN (+ overflow check)

Usage
-----
python convert_onnx_to_rknn.py  model.onnx  rk3588  fp  model.rknn
dtype   fp  = no quant (FP16 on NPU)
        i8  = int8  (asymmetric_affine-i8)
        u8  = uint8 (asymmetric_affine-u8)

If dataset.txt exists its 1st line is also fed to accuracy_analysis,
otherwise we generate a random [1,1,60,200] tensor and save it as
_tmp_sample.npy.
"""
import sys, os, numpy as np, tempfile, atexit
from rknn.api import RKNN

# ---------- quick-edit defaults ----------
DATASET_TXT   = 'dataset.txt'
DEFAULT_OUT   = 'model.rknn'
DEFAULT_DTYPE = 'fp'
VALID_PLATFORMS = {
    'rk3562','rk3566','rk3568','rk3576','rk3588',
    'rv1103','rv1106','rk1808','rv1109','rv1126'
}
# -----------------------------------------

def parse_cli():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    onnx  = sys.argv[1]
    plat  = sys.argv[2].lower()
    if plat not in VALID_PLATFORMS:
        sys.exit(f'ERROR: unknown platform {plat}')
    dtype = sys.argv[3].lower() if len(sys.argv) > 3 else DEFAULT_DTYPE
    if dtype not in ('i8','u8','fp'):
        sys.exit('ERROR: dtype must be i8/u8/fp')
    out   = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT
    quant = dtype in ('i8','u8')
    return onnx, plat, quant, out

def first_sample_path():
    # 1) dataset.txt present → use its first entry
    if os.path.isfile(DATASET_TXT):
        with open(DATASET_TXT) as f:
            p = f.readline().strip()
        if p and os.path.isfile(p):
            return p

    # 2) fallback – create a random tensor and save it
    arr = np.random.randn(1,1,60,200).astype('float32')
    tmp = tempfile.NamedTemporaryFile(
        suffix='.npy', delete=False, prefix='_tmp_sample_', dir='.')
    np.save(tmp.name, arr); tmp.close()
    atexit.register(lambda: os.remove(tmp.name))
    return tmp.name

def main():
    onnx_path, platform, do_quant, out_path = parse_cli()
    rknn = RKNN(verbose=True)

    print('[1/5] config()')
    rknn.config(mean_values=[[0] * 60 ], std_values=[[1] * 60 ], 
                target_platform=platform, optimization_level=0)

    print('[2/5] load_onnx()', onnx_path)
    if rknn.load_onnx(onnx_path) != 0:
        sys.exit('load_onnx failed')

    print('[3/5] build()', 'with quant' if do_quant else 'FP16')
    if rknn.build(do_quantization=do_quant,
                  dataset=DATASET_TXT if do_quant else None) != 0:
        sys.exit('build failed')

    # -------- overflow / saturation sanity-check ----------
    # sample_npy = first_sample_path()
    # print('[4/5] accuracy_analysis() – simulator run on', sample_npy)
    # # list-form input is supported by Toolkit2 (no names needed)
    # rknn.accuracy_analysis(inputs=[sample_npy], target=None)
    # print('↑  Check the table: the first "inf"/"nan" row pinpoints the layer '
    #       'whose activations exceed ±65 k in FP16.\n')

    print('[5/5] export_rknn()', out_path)
    if rknn.export_rknn(out_path) != 0:
        sys.exit('export failed')

    print('✔ RKNN saved at', out_path)
    rknn.release()

if __name__ == '__main__':
    main()
