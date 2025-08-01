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

## 3588

* log additions on the target
```
pip install rknn-toolkit2 numpy soundfile 
pip install rknn-toolkit2 soundfile numpy scipy sounddevice
sudo apt install libportaudio2 portaudio19-dev
pip install --force-reinstall sounddevice
```

* run env on target:
```
source  /home/rock/proj/voiceRKNN/venvRKNN/bin/activate
cd /home/rock/proj/voiceRKNN/voicer/redimnet/3588
```



## convert

* use embedding_testRob1.torch result embedding to test rkkn convert

* run:
```
python convert.py ../wrkB0/ReDimNet_no_mel_fp16.onnx rk3588 fp ReDimNet_no_mel.rknn

python convert.py \
       ../wrkB0/ReDimNet_no_mel_fp16.onnx rk3588 fp ReDimNet_no_mel.rknn \
       ../wrkB0/audio/logmel_testRob1.npy  ../wrkB0/audio/embedding_testRob1.torch
```

```
python convert.py ../wrkB0/ReDimNet_no_mel_fp16.onnx rk3588 i8 ReDimNet_no_mel.rknn

python convert.py ../wrkB0/ReDimNet_no_mel_fp16.onnx rk3588 i8 ReDimNet_no_mel.rknn \
       ../wrkB0/audio/logmel_testRob1.npy  ../wrkB0/audio/embedding_testRob1.torch
```

## unit test

* [see test here](./test/readme.md)


## inference test

```
python inference_rknn.py ReDimNet_no_mel.rknn ../wrkB0/audio/testRob1.wav
python inference_rknn_2.py ReDimNet_no_mel.rknn ../wrkB0/audio/testRob1.wav  ../wrkB0/audio/testRob2.wav
```


## live run

* on hw record for reference:
    arecord -l //list devices
    arecord -D hw:4,0 --dump-hw-params  // dump params
    arecord -D hw:4,0 -f S16_LE -c2 -r16000 -d 10 localREC.wav

* for long rec average each 2sec parts:
    python prepareEmbFromRec.py  ReDimNet_no_mel.rknn localREC.wav refEmbed.tor

* run live:
    python live_compare2Wav_B2.py  ReDimNet_no_mel.rknn localREC.wav rk3566
    python live_compare2Emb_B2.py  ReDimNet_no_mel.rknn refEmbed.tor rk3566

    

