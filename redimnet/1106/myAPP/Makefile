###############################################################################
#  RV1106 build — runtime app (run) + unit-tests (test_dsp, test_dummy)
###############################################################################

# ── tool-chain --------------------------------------------------------------
CROSS_COMPILE := /home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-

SYSROOT := /home/user/shared/luckfox-pico/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot

CXX   := $(CROSS_COMPILE)g++
STRIP := $(CROSS_COMPILE)strip

# ── RKNN runtime (the one that matches your model) --------------------------
RKNN_LIBDIR := /home/user/shared/rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api/armhf-uclibc
RKNN_INC    := /home/user/shared/rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api/include

# ── compiler / linker flags -------------------------------------------------
CXXFLAGS := -O2 -Wall -std=c++17 \
            --sysroot=$(SYSROOT) \
            -Idsp/kissfft -I$(RKNN_INC) -I$(SYSROOT)/usr/include

#  run  → needs RKNN + libsndfile
LDFLAGS_RUN   := -L$(RKNN_LIBDIR) -Wl,-rpath,'$$ORIGIN/lib' \
                 -lrknnmrt -L$(SYSROOT)/usr/lib -lsndfile -lpthread -lm -ldl
#  test_dsp → only DSP code + libsndfile
LDFLAGS_TEST  :=                 -L$(SYSROOT)/usr/lib -lsndfile -lpthread -lm -ldl
#  test_dummy → RKNN only, no DSP / sndfile
LDFLAGS_DUMMY := -L$(RKNN_LIBDIR) -Wl,-rpath,'$$ORIGIN/lib' \
                 -lrknnmrt -L$(SYSROOT)/usr/lib -lpthread -lm -ldl

# ── KissFFT (vendored) ------------------------------------------------------
KISS_SRC := dsp/kissfft/kiss_fft.c dsp/kissfft/kiss_fftr.c
KISS_OBJ := $(KISS_SRC:.c=.o)

# ── project sources ---------------------------------------------------------
SHARED_SRC  := dsp/preemph.cc dsp/stft.cc dsp/melbank.cc dsp/quant.cc
RUN_SRC     := main.cc      $(SHARED_SRC)
TEST_SRC    := test_dsp.cc $(SHARED_SRC)
DUMMY_SRC   := test_dummy.cc

RUN_OBJ     := $(RUN_SRC:.cc=.o)   $(KISS_OBJ)
TEST_OBJ    := $(TEST_SRC:.cc=.o)  $(KISS_OBJ)
DUMMY_OBJ   := $(DUMMY_SRC:.cc=.o)

# ── targets -----------------------------------------------------------------
RUN_TARGET    := run
TEST_TARGET   := test_dsp
DUMMY_TARGET  := test_dummy            

.PHONY: all clean install
all: $(RUN_TARGET) $(TEST_TARGET) $(DUMMY_TARGET)

$(RUN_TARGET): $(RUN_OBJ)
	$(CXX) $^ $(LDFLAGS_RUN)   -o $@

$(TEST_TARGET): $(TEST_OBJ)
	$(CXX) $^ $(LDFLAGS_TEST)  -o $@

$(DUMMY_TARGET): $(DUMMY_OBJ)          
	$(CXX) $^ $(LDFLAGS_DUMMY) -o $@

# pattern rules
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -x c -c $< -o $@

# ── install bundle ----------------------------------------------------------
install: $(RUN_TARGET) $(TEST_TARGET) $(DUMMY_TARGET)
	mkdir -p out/lib
	# copy & strip binaries
	$(STRIP) -s $(RUN_TARGET)    -o out/$(RUN_TARGET)
	$(STRIP) -s $(TEST_TARGET)   -o out/$(TEST_TARGET)
	$(STRIP) -s $(DUMMY_TARGET)  -o out/$(DUMMY_TARGET)
	# include matching runtime library
	cp  $(RKNN_LIBDIR)/librknnmrt.so  out/lib/
	@echo \"→ out/ directory ready:\"
	@echo \"   out/$(RUN_TARGET)  out/$(TEST_TARGET)  out/$(DUMMY_TARGET)  out/lib/librknnmrt.so\"

clean:
	rm -f $(RUN_OBJ) $(TEST_OBJ) $(DUMMY_OBJ) \
	      $(RUN_TARGET) $(TEST_TARGET) $(DUMMY_TARGET)
	rm -rf out
