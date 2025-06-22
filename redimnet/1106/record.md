# 

## 

```
> arecord -l
**** List of CAPTURE Hardware Devices ****
card 0: rv1106acodec [rv1106-acodec], device 0: ffae0000.i2s-rv1106-hifi ff480000.acodec-0 [ffae0000.i2s-rv1106-hifi ff480000.acodec-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

> cat /proc/asound/devices
  0: [ 0]   : control
 16: [ 0- 0]: digital audio playback
 24: [ 0- 0]: digital audio capture


```

## sox utility

* Show statistics 
    > sox file.wav -n stat

    
    Capture a short noise sample:

sox input.wav -n trim 0 0.5 noiseprof noise.prof

    Apply noise reduction:

sox input.wav cleaned.wav noisered noise.prof 0.3
