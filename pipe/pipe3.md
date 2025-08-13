

```
        +------------------+              +------------------+
Mic --->|  py-input (Sink) |---/tmp/in.pipe--> Python limiter \
        +------------------+              |                  | ->/tmp/out.pipe--> py-output (Source)--> Apps / Speakers
                                          +------------------+


```


mkfifo /tmp/in.pipe  /tmp/out.pipe
chmod 777 /tmp/in.pipe  /tmp/out.pipe

python code program (peak_limiter.py)

```
#!/usr/bin/env python3
import os, numpy as np

CHUNK  = 480          # 10 ms  @ 48 kHz
LIMIT  = 30000        # peak ceiling
dtype  = np.int16
bytes_ = CHUNK * dtype().nbytes

with open("/tmp/in.pipe",  "rb", buffering=0) as fin, \
     open("/tmp/out.pipe", "wb", buffering=0) as fout:

    while True:
        buf = fin.read(bytes_)
        if not buf:
            break          # pipe closed
        samples = np.frombuffer(buf, dtype=dtype)

        peak = np.abs(samples).max()
        if peak > LIMIT:
            samples = (samples * (LIMIT / peak)).astype(dtype)

        fout.write(samples.tobytes())

print("end of run..")


```




## hand test


python3 peak_limiter.py &

** Upstream replaced the old pipe-source/pipe-sink helpers with the more general libpipewire-module-pipe-tunnel starting around PipeWire 0.3.60

### (1)

# ❶ Source: PW will READ from /tmp/out.pipe and publish “py-output”
pw-cli load-module libpipewire-module-pipe-tunnel \
  'tunnel.mode=source pipe.filename=/tmp/out.pipe \
   audio.rate=48000 audio.format=S16 audio.channels=1 \
   node.name=py-output'


# ❷ Sink: PW will WRITE into /tmp/in.pipe and publish “py-input”
pw-cli load-module libpipewire-module-pipe-tunnel \
  'tunnel.mode=sink  pipe.filename=/tmp/in.pipe  \
   audio.rate=48000 audio.format=S16 audio.channels=1 \
   node.name=py-input'


pw-cli ls Node | grep py-
 		node.name = "py-input"
 		node.name = "py-output"

### (2)

pw-link alsa_input.platform-es8316-sound.HiFi__hw_rockchipes8316__source:capture_FL \
        py-input:playback_0



pw-link py-output:capture_0 \
        alsa_output.platform-es8316-sound.HiFi__hw_rockchipes8316__sink:playback_FL
pw-link py-output:capture_1 \
        alsa_output.platform-es8316-sound.HiFi__hw_rockchipes8316__sink:playback_FR

pw-link -l

### test loopback

cat /tmp/in.pipe > /tmp/out.pip
//need to hear myself


### (3)      
chmod +x peak_limiter.py
./peak_limiter.py


## (1) auto

# ~/.config/pipewire/pipewire.conf.d/30-py-tunnel.conf
# Two nodes:
#   py-input  : PW writes → /tmp/in.pipe   (Sink)
#   py-output : PW reads  ← /tmp/out.pipe  (Source)

context.modules = [
  { name = libpipewire-module-pipe-tunnel
    args = {
      tunnel.mode   = sink
      file.path     = "/tmp/in.pipe"
      audio.format  = S16LE
      audio.rate    = 48000
      audio.channels = 1
      node.name     = "py-input"
    }
  }
  { name = libpipewire-module-pipe-tunnel
    args = {
      tunnel.mode   = source
      file.path     = "/tmp/out.pipe"
      audio.format  = S16LE
      audio.rate    = 48000
      audio.channels = 1
      node.name     = "py-output"
    }
  }
]
