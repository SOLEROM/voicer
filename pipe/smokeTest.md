# voice 

* smoke test

```
# list devices
arecord -l
aplay -l

## test with  noice file
sox -n -r 48000 -c 2 -b 16 test_noise.wav synth 3 whitenoise

aplay test_noise.wav
aplay -D hw:4,0 test_noise.wav

## record
arecord -D hw:4,0 -f S16_LE -c 2 -r 48000 -d 5 test.wav
aplay test.wav

```
