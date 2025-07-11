


## ubuntu target

flash from update prebuild
 sudo ./upgrade_tool uf  ../../firmware/ubuntu/unpack_Luckfox_Pico_Ultra_W_Ubuntu_eMMC/Luckfox_Pico_Ultra_W_Ubuntu_eMMC/ultra_w_ubuntu_emmc/update.img

```
Welcome to Ubuntu 22.04.3 LTS (GNU/Linux 5.10.160 armv7l)

jammy-updates/main armhf 

```

## can use python runtime?

```
Toolkit / Toolkit-2 – whenever you need to convert or quantise a model. Do this on a desktop-class machine.

Toolkit-Lite / Lite-2 – when you only need to run an already-converted .rknn on the embedded target. Minimal dependencies, tiny footprint.

C RKNN Runtime – if you want to skip Python on the device altogether (e.g. production firmware).

```



```
rknn-toolkit2 (v1.6+)
    Full SDK for RKNPU-2 – conversion, quant, analysis, + PC inference
    RK3562/66/68, RK3588, RV1103/1106
    x86-64, aarch64 only (no armv7l wheels) 

rknn-toolkit-lite2
    Python inference-only runtime for RKNPU-2
    same chips as Toolkit-2
    aarch64 wheels
        https://pypi.org/project/rknn-toolkit-lite2/
        Currently only support on:
                AArch64 Linux
                Python 3.7/3.8/3.9/3.10/3.11/3.12

```

* Luckfox Pico Ultra is RV1106 – single-core Cortex-A7 (32-bit)
* It needs the RKNPU-2 runtime (Toolkit-Lite 2), but the published Lite 2 wheels are 64-bit only.
* The old Lite (v1.x) armv7l wheels do exist, but they drive RKNPU-1 only—they cannot talk to the RV1106 NPU firmware.



## versions

on target:
5118caecec7f4da43e1b74029c5cebce  /oem/usr/lib/librknnmrt.so
strings /oem/usr/lib/librknnmrt.so | grep "librknn"
librknnmrt version: 1.6.0 (2de554906@2024-01-17T14:53:41)

on build-rknn docker:
strings /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so | grep version
librknnmrt version: 1.6.0 (9a7b5d24c@2023-12-13T17:33:10)

on build-root docker:
5118caecec7f4da43e1b74029c5cebce  /home/user/shared/luckfox-pico/output/out/oem/usr/lib/librknnmrt.so

* sol: align runtime and compile versions

```
strings lib/librknnmrt.so  | grep version
strings ReDimNet_no_mel.rknn | grep -m1 "version"
```


## plan to use C RKNN Runtime

### driver support

> lsmod | grep rkn
rknpu   

>? cat /proc/rknpu/version 
RKNPU driver: v0.9.2


### runtime libs

find / -name *rknn*.so
/oem/usr/lib/librknnmrt.so

> strings /oem/usr/lib/librknnmrt.so | grep "librknn"
librknnmrt version: 1.6.0 (2de554906@2024-01-17T14:53:41)



### cross-compile

* is on glibc / uClibc system?

```
[root@luckfox root]# ldd /lib/libc.so.0 
	ld-uClibc.so.1 => /lib/ld-uClibc.so.1 (0xa6fc6000)
```



### dsp functions


### senity compare

Validation workflow
unit-test (ctest) so CI keeps you safe.

### performace

FFT	Replace KissFFT with CMSIS-DSP NEON routines
Memory	Use rknn_create_mem + rknn_set_input_mem for zero-copy input buffers 


## build RKNN module

on rknn docker:
    
    conda activate RKNN-Toolkit2
    cd ~/shared/voicer/redimnet/1106
    cp -ar ../3588/cal .
    python convert.py ../wrkB0/ReDimNet_no_mel_fp16.onnx rv1106 i8 ReDimNet_no_mel.rknn

## run

./run ReDimNet_no_mel.rknn  test000.wav


# toolchian 2.3 / uclibc / buildroot

on docker of buildroot/ubu + toolchain 2.3

```
strings ReDimNet_no_mel.rknn | grep version
,"target_platform":["rv1106"],"version":"2.3.2",

home/user/shared/rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so 
librknnmrt version: 2.3.2 (429f97ae6b@2025-04-09T09:11:49)
```

## debug level

* use env var: export RKNN_LOG_LEVEL=1