# test what running



``` 
// See who has the sound devices open
sudo fuser -v /dev/snd/* 

// running deamons
ps -e | grep -E 'pipewire|wireplumber|pulseaudio|jackd'

//
arecord -L


// Identify the kernel driver
cat /proc/asound/cards

```

## PipeWire

```
> pw-cli info
> pw-top
```


## PulseAudio


```
pactl info
```

## jack

```
jack_lsp -c

```

## 