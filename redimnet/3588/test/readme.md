# tests

## inference_compareWithPt.py      
* input: model.rknn  probe.wav  ref_embed.pt
* run pre calc then compare cosine wiht ref pt

## inference_noPreUseNPY.py  
* input:  model.rknn  clip_logmel.npy
* NO pre calc ; load logmel data file and run inference on that

## inference_noPreCmpPt.py  
* input: model.rknn  logmel.npy  ref_embed.pt
* NO pre calc ; load logmel data file and compare cosine with ref pt
