# run on target

## data format

```
PyTorch default is NCHW:

    Shape: [Batch, Channels, Height, Width]
onnx model : [B=1, 1, n_mels, time_frames]


RKNN default is NHWC:

    Shape: [Batch, Height, Width, Channels]
```

set our rknn to NCHW:

```
rknn.inference(inputs=[inp], data_format='nchw') ##!!!
```

## target

* log additions on the target
```
TBD
```

* run env on target:
```
source  /local/voicer/venvRKNN/bin/activate
cd /local/voicer/3566
```



## convert

* use embedding_testRob1.torch result embedding to test rkkn convert

* run:

fp for B2

```
python convert.py  ../wrkB2/ReDimNet_no_mel_fp16.onnx rk3566 fp B2 ReDimNet_no_mel.rknn 


python convert.py  ../wrkB2/ReDimNet_no_mel_fp16.onnx rk3566 fp B2 ReDimNet_no_mel.rknn ../wrkB2/audio/logmel_testRob1.npy  ../wrkB2/audio/embedding_testRob1.torch 

🔎  Δmax=0.006991  Δmean=0.002093  cosine=0.999999
✅ RKNN output matches reference within tolerance.
✔ RKNN saved at ReDimNet_no_mel.rknn
EOF!


```

i8 

```

TBD

```

## unit test

* [see test here](./test/readme.md)


## inference test

```
python inference_rknn.py ReDimNet_no_mel.rknn testRob1.wav --target rk3566


python inference_rknn.py ReDimNet_no_mel.rknn ../wrkB0/audio/testRob1.wav
python inference_rknn.py ReDimNet_no_mel.rknn ../wrkB0/audio/testRob1.wav  ../wrkB0/audio/testRob2.wav
```


## live run

* on hw record for reference:
    arecord -l //list devices
    arecord -D hw:4,0 --dump-hw-params  // dump params
    arecord -D hw:4,0 -f S16_LE -c2 -r16000 -d 10 localREC.wav

* for long rec average each 2sec parts:
    python prepareEmbFromRec.py  ReDimNet_no_mel.rknn localREC.wav refEmbed.tor
    python live_compare2Emb.py  ReDimNet_no_mel.rknn refEmbed.tor rk3566 2 modelB2

* run live:
    python live_compare2Wav.py  ReDimNet_no_mel.rknn localREC.wav rk3566 2 modelB2

    

