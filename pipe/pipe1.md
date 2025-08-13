
```
┌─────────────┐          raw PCM            ┌───────────────┐       processed PCM       ┌──────────────┐
│  pw-cat -r  │──► /tmp/in.pcm  ──►  python │  process.py   │ ──► /tmp/out.pcm ──► pw-cat -w final.wav
│ (capture)   │                             │ (your logic)  │                          (one WAV file)
└─────────────┘                              └───────────────┘                          └──────────────┘
```

