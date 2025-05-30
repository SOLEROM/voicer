# About

* https://github.com/IDRnD/ReDimNet

##

* spectrogram processing, particularly in extracting utterance-level speaker representations.
	* Speaker representations refer to numerical features that uniquely describe a person's voice.
	* is this person who they claim to be?)
	* speaker identification (who is speaking?
	* Utterance-level representations mean that the model processes an entire spoken phrase, rather than just short frames of speech.
	* aggregates features across the whole utterance to create a single compact vector that represents the speaker
	* can reduce noise and variability, focusing on the speaker’s identity rather than background sounds or temporary speech variations.

* model sizes ranging from 1 to 15 million parameters 
* computational complexities between 0.5 to 20 GMACs : Giga Multiply-Accumulates
	* represent the number of floating-point or integer operations needed for inference.
	* billion operations per forward pass
	* More GMACs increase latency unless an efficient accelerator is used
	* Lower GMAC models consume less energy 

```
example:
	device with 10 TOPS => 
		0.5 GMAC model in 0.05 ms.
		20 GMAC model in 2 ms.	

```

* state-of-the-art performance in speaker recognition 




## get model with cache setting

```
>python dwModel_M.py                
Downloading: "https://github.com/IDRnD/ReDimNet/releases/download/latest/M-vb2+vox2+cnc-ft_mix.pt" to 
/data/proj/voice/redimnet/models/checkpoints/M-vb2+vox2+cnc-ft_mix.pt
```



