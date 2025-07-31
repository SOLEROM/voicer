# tests

## inference_Pre_wavVsEmb.py      
* input: model.rknn  inputWAV.wav embed_X
* run pre calc then compare cosine with ref embedding

## inference_noPre_runLogmel.py  
* input:  model.rknn  logmel_X
* NO pre calc ; load logmel data file and run inference on that

## inference_noPre_logVsEmb.py  
* input: model.rknn  logmel_X  embed_X
* NO pre calc ; load logmel data file and compare cosine with ref pt
