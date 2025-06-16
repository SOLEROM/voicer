# run on target

## data format

PyTorch default is NCHW:

    Shape: [Batch, Channels, Height, Width]
onnx model : [B=1, 1, n_mels, time_frames]


RKNN default is NHWC:

    Shape: [Batch, Height, Width, Channels]


## 3588

source  /home/rock/proj/voiceRKNN/venvRKNN/bin/activate
pip install rknn-toolkit2 numpy soundfile 
pip install rknn-toolkit2 soundfile numpy scipy sounddevice
sudo apt install libportaudio2 portaudio19-dev
pip install --force-reinstall sounddevice


python inference_rknn.py ReDimNet_no_mel.rknn_3588 test00.wav


python inference_rknn_2.py  ReDimNet_no_mel.rknn_3588 test00.wav me2.wav rk3588


#### live

arecord -D hw:4,0 -f S16_LE -c2 -r16000 -d 5 rec_me.wav
python live_compare_rknn.py   ReDimNet_no_mel.rknn_3588 rec_me.wav rk3588
