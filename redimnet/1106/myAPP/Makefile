###############################################################################
#  RV1106 build – runtime app (run)  +  DSP unit-tests (test_dsp)
###############################################################################

# ── tool-chain prefix (hard-float uClibc) ────────────────────────────────────
CROSS_COMPILE := /home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
SYSROOT := 	/home/user/shared/luckfox-pico/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot


CXX      := $(CROSS_COMPILE)g++
STRIP    := $(CROSS_COMPILE)strip

# ── RKNN runtime (uClibc) ───────────────────────────────────────────────────
RKNN_SDK := /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
RKNN_INC := $(RKNN_SDK)/include
RKNN_LIB := $(RKNN_SDK)/armhf-uclibc          # contains librknnmrt.so

# ── compiler / linker flags ─────────────────────────────────────────────────
## kissfft must be before RKNN includes to override the default FFT implementation;
CXXFLAGS := -O2 -Wall -std=c++17 --sysroot=$(SYSROOT) -Idsp/kissfft -I$(RKNN_INC)


## libsndfile conf Buildroot image BR2_PACKAGE_LIBSNDFILE=y (Audio/Sound -*- libsndfile)
LDFLAGS_RUN  := -L$(RKNN_LIB) -L$(SYSROOT)/usr/lib -lrknnmrt -lsndfile -lpthread -lm -ldl
LDFLAGS_TEST := -L$(SYSROOT)/usr/lib  -lsndfile -lpthread -lm -ldl	 # no RKNN needed

# ── libs ────────────────────────────────────────────────────────────────
# KISS FFT library from mborgerding/kissfft/master
KISS_SRC  := dsp/kissfft/kiss_fft.c dsp/kissfft/kiss_fftr.c


# ── sources ────────────────────────────────────────────────────────────────
SHARED_SRC := dsp/preemph.cc dsp/stft.cc dsp/melbank.cc dsp/quant.cc $(KISS_SRC)

RUN_SRC    := main.cc $(SHARED_SRC)
TEST_SRC   := test_dsp.cc $(SHARED_SRC)

RUN_OBJ    := $(RUN_SRC:.cc=.o)
TEST_OBJ   := $(TEST_SRC:.cc=.o)

# ── targets ────────────────────────────────────────────────────────────────
RUN_TARGET   := run
TEST_TARGET  := test_dsp

.PHONY: all clean install
all: $(RUN_TARGET) $(TEST_TARGET)

$(RUN_TARGET): $(RUN_OBJ)
	$(CXX) $^ $(LDFLAGS_RUN)  -o $@

$(TEST_TARGET): $(TEST_OBJ)
	$(CXX) $^ $(LDFLAGS_TEST) -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c                 
	$(CXX) $(CXXFLAGS) -x c -c $< -o $@

install: $(RUN_TARGET) $(TEST_TARGET)
	mkdir -p out
	$(STRIP) -s $(RUN_TARGET)  -o out/$(RUN_TARGET)
	$(STRIP) -s $(TEST_TARGET) -o out/$(TEST_TARGET)
	@echo "→ out/$(RUN_TARGET) and out/$(TEST_TARGET) ready."

clean:
	rm -f $(RUN_OBJ) $(TEST_OBJ) $(RUN_TARGET) $(TEST_TARGET)
	rm -rf out
