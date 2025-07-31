echo "========================================================"
echo "senity tests..."
echo "========================================================"
echo "comapre same voice to itself"
python inference_rknn_2.py  ReDimNet_no_mel.rknn ../wrkB0/audio/test000.wav ../wrkB0/audio/test000.wav 2>>/dev/null | grep Sim
echo "compare voice with host embedding (with pre)"
python test/inference_Pre_wavVsEmb.py ReDimNet_no_mel.rknn  ../wrkB0/audio/testRob1.wav  ../wrkB0/audio/embedding_testRob1.torch  2>>/dev/null | grep Sim
echo "compare logmel with embedding (no pre)"
python test/inference_noPre_logVsEmb.py  ReDimNet_no_mel.rknn ../wrkB0/audio/logmel_test_human1_1.npy ../wrkB0/audio/embedding_test_human1_1.torch 2>>/dev/null | grep Sim

echo "========================================================"
echo "same voices..."
echo "========================================================"
echo "rob-rob"
python inference_rknn_2.py  ReDimNet_no_mel.rknn ../wrkB0/audio/testRob1.wav  ../wrkB0/audio/testRob2.wav 2>>/dev/null | grep Cosine 
echo "hum1-hum1"
python inference_rknn_2.py  ReDimNet_no_mel.rknn ../wrkB0/audio/test_human1_1.wav  ../wrkB0/audio/test_human1_2.wav 2>>/dev/null | grep Cosine 
echo "hum2-hum2"
python inference_rknn_2.py  ReDimNet_no_mel.rknn ../wrkB0/audio/test_human2_1.wav  ../wrkB0/audio/test_human2_2.wav 2>>/dev/null | grep Cosine


echo "========================================================"
echo "diffs..."
echo "========================================================"
echo rob-hum1
python inference_rknn_2.py  ReDimNet_no_mel.rknn ../wrkB0/audio/testRob1.wav ../wrkB0/audio/test_human1_1.wav 2>>/dev/null | grep Cosine 
echo rob-hum2
python inference_rknn_2.py  ReDimNet_no_mel.rknn ../wrkB0/audio/testRob1.wav ../wrkB0/audio/test_human2_1.wav 2>>/dev/null | grep Cosine 
echo hum1-hum2
python inference_rknn_2.py  ReDimNet_no_mel.rknn ../wrkB0/audio/test_human1_1.wav ../wrkB0/audio/test_human2_1.wav 2>>/dev/null | grep Cosine 


