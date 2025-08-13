#!/usr/bin/env python3
"""
peak_limiter2.py – peak limiter with FIFO keep-alive + live debug
"""

import os, sys, time, errno, numpy as np

IN_FIFO  = "/tmp/in.pipe"
OUT_FIFO = "/tmp/out.pipe"

RATE   = 48_000
CHUNK  = 480                 # 10 ms
LIMIT  = 30_000
DTYPE  = np.int16
BYTES  = CHUNK * DTYPE().nbytes

# --- Open both FIFOs read-write so open() never fails ---
try:
    fin  = os.open(IN_FIFO,  os.O_RDWR | os.O_NONBLOCK)
    fout = os.open(OUT_FIFO, os.O_RDWR | os.O_NONBLOCK)
except FileNotFoundError as e:
    sys.exit(f"FIFO missing: {e.filename}.  mkfifo first.")

print("[Limiter] running  chunk=%d  limit=%d" % (CHUNK, LIMIT))
have_reader = False
next_print  = time.time() + 0.5
peak_hist   = []

while True:
    # read() returns b'' when no reader yet; poll until PipeWire connects
    try:
        buf = os.read(fin, BYTES)
    except BlockingIOError:
        time.sleep(0.001)
        continue

    if not buf:
        time.sleep(0.01)
        continue

    if not have_reader:
        print("♦ PipeWire connected to both FIFOs")
        have_reader = True

    x = np.frombuffer(buf, dtype=DTYPE)
    peak = int(np.abs(x).max())
    if peak > LIMIT:
        gain = LIMIT / peak
        x = (x * gain).astype(DTYPE)
    else:
        gain = 1.0

    os.write(fout, x.tobytes())
    peak_hist.append(peak)

    now = time.time()
    if now >= next_print:
        rms = int(np.sqrt((np.square(x, dtype=np.int32).mean())))
        print(f"[{time.strftime('%H:%M:%S')}] peak={max(peak_hist):5d} "
              f"gain={gain:4.2f}  rms={rms:5d}")
        peak_hist.clear()
        next_print = now + 0.5
