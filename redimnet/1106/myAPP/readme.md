# probe

## build

* pay attention as may build for glibc or uclibc

```
RKNN_SDK=/home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api

|-- aarch64
|   `-- librknnrt.so
|-- armhf
|   `-- librknnrt.so
|-- armhf-uclibc
|   |-- librknnmrt.a
|   `-- librknnmrt.so
`-- include
    |-- rknn_api.h
    |-- rknn_custom_op.h
    `-- rknn_matmul_api.h
```

###  uclibc

```
CROSS_COMPILE=> /home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-

rknn_api.h      =>  $(RKNN_SDK)/include

rknnmrt         => $(RKNN_SDK)/armhf-uclibc
```

### glibc 

```
CROSS_COMPILE   =>  ? 

rknn_api.h      =>  $(RKNN_SDK)/include

rknnmrt         => $(RKNN_SDK)/armhf
```
