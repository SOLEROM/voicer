

├── wavs/                
│   ├── alice01.wav
│   ├── bob02.wav
│   └── …
└── gen_calib_npy.py     

50–200 calibration samples usually give the same accuracy;

pip install numpy librosa soundfile
python gen_calib_npy.py --wav-dir wavs/