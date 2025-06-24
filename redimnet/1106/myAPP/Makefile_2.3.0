###############################################################################
#  RV1106 build — runtime app (run) + unit-test (test_dsp)
###############################################################################

# ── tool-chain ---------------------------------------------------------------
CROSS_COMPILE := /home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-

SYSROOT := /home/user/shared/luckfox-pico/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot

CXX   := $(CROSS_COMPILE)g++
STRIP := $(CROSS_COMPILE)strip

# ── RKNN runtime (the one that matches your model) ---------------------------
# contains librknnmrt.so
RKNN_LIBDIR := /home/user/shared/rknn_model_zoo/install/rv1106_linux_armv7l/rknn_yolov5_demo/lib
# contains librknn_api.h
RKNN_INC    := /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include

# ── compiler / linker flags --------------------------------------------------
#       KissFFT path first (overrides stub in RKNN headers)
CXXFLAGS := -O2 -Wall -std=c++17 \
            --sysroot=$(SYSROOT) \
            -Idsp/kissfft -I$(RKNN_INC)

# ELF searches $ORIGIN/lib at runtime
LDFLAGS_RUN  := -L$(RKNN_LIBDIR) -Wl,-rpath,'$$ORIGIN/lib' \
                -lrknnmrt -L$(SYSROOT)/usr/lib -lsndfile -lpthread -lm -ldl
LDFLAGS_TEST :=                 -L$(SYSROOT)/usr/lib -lsndfile -lpthread -lm -ldl

# ── KissFFT (vendored) -------------------------------------------------------
KISS_SRC := dsp/kissfft/kiss_fft.c dsp/kissfft/kiss_fftr.c
KISS_OBJ := $(KISS_SRC:.c=.o)

# ── project sources ----------------------------------------------------------
SHARED_SRC := dsp/preemph.cc dsp/stft.cc dsp/melbank.cc dsp/quant.cc
RUN_SRC    := main.cc      $(SHARED_SRC)
TEST_SRC   := test_dsp.cc  $(SHARED_SRC)

RUN_OBJ  := $(RUN_SRC:.cc=.o)  $(KISS_OBJ)
TEST_OBJ := $(TEST_SRC:.cc=.o) $(KISS_OBJ)

# ── targets ------------------------------------------------------------------
RUN_TARGET  := run
TEST_TARGET := test_dsp

.PHONY: all clean install
all: $(RUN_TARGET) $(TEST_TARGET)

$(RUN_TARGET): $(RUN_OBJ)
	$(CXX) $^ $(LDFLAGS_RUN)  -o $@

$(TEST_TARGET): $(TEST_OBJ)
	$(CXX) $^ $(LDFLAGS_TEST) -o $@

# pattern rules
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -x c -c $< -o $@

# ── install bundle -----------------------------------------------------------
install: $(RUN_TARGET) $(TEST_TARGET)
	mkdir -p out/lib
	# copy & strip binaries
	$(STRIP) -s $(RUN_TARGET)  -o out/$(RUN_TARGET)
	$(STRIP) -s $(TEST_TARGET) -o out/$(TEST_TARGET)
	# include matching runtime library
	cp  $(RKNN_LIBDIR)/librknnmrt.so  out/lib/
	@echo "→ out/ directory ready:"
	@echo "   out/$(RUN_TARGET)  out/$(TEST_TARGET)  out/lib/librknnmrt.so"

clean:
	rm -f $(RUN_OBJ) $(TEST_OBJ) $(RUN_TARGET) $(TEST_TARGET)
	rm -rf out
