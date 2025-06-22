mkdir -p dsp/kissfft
wget -qO dsp/kissfft/kiss_fft.h  https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fft.h
wget -qO dsp/kissfft/kiss_fft.c  https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fft.c
wget -qO dsp/kissfft/kiss_fftr.h https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fftr.h
wget -qO dsp/kissfft/kiss_fftr.c https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fftr.c

wget -qO dsp/kissfft/_kiss_fft_guts.h https://raw.githubusercontent.com/mborgerding/kissfft/master/_kiss_fft_guts.h
wget -qO dsp/kissfft/kiss_fft_log.h   https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fft_log.h