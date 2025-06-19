import os
from typing import Callable
import torch
import numpy as np

def test_all_voices(extract_speaker_embedding_function:Callable, cosine_similarity_function:Callable ,save_embeddings:bool=False):
    """
    
    """
    pwd_path = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(pwd_path, '../audio')
    
    embed0 = extract_speaker_embedding_function(wav_path=os.path.join(audio_path,'test000.wav'))

    embed1 = extract_speaker_embedding_function(wav_path=os.path.join(audio_path,'testRob1.wav'))
    embed2 = extract_speaker_embedding_function(wav_path=os.path.join(audio_path,'testRob2.wav'))
    
    embed3 = extract_speaker_embedding_function(wav_path=os.path.join(audio_path,'test_human1_1.wav'))
    embed4 = extract_speaker_embedding_function(wav_path=os.path.join(audio_path,'test_human1_2.wav'))

    embed5 = extract_speaker_embedding_function(wav_path=os.path.join(audio_path,'test_human2_1.wav'))
    embed6 = extract_speaker_embedding_function(wav_path=os.path.join(audio_path,'test_human2_2.wav'))
    
    print("**************************************************************************")
    print("*************************   compare summary ******************************")
    print("**************************************************************************")
    print("====>>>> should be similar:")
    print(f"Similarity (robot1 to robot2 ): {cosine_similarity_function(embed1, embed2)}")
    print(f"Similarity (human1 to human1 ): {cosine_similarity_function(embed3, embed4)}")
    print(f"Similarity (human2 to human2 ): {cosine_similarity_function(embed5, embed6)}")
    print("====>>>> should be differnet:")
    print(f"Similarity (robot to human1  ): {cosine_similarity_function(embed1, embed3)}")
    print(f"Similarity (robot to human2  ): {cosine_similarity_function(embed1, embed5)}")
    print(f"Similarity (human1 to human2 ): {cosine_similarity_function(embed3, embed5)}")
    
    if save_embeddings:
        names = [
            'embedding_test000',
            'embedding_testRob1',
            'embedding_testRob2',
            'embedding_test_human1_1',
            'embedding_test_human1_2',
            'embedding_test_human2_1',
            'embedding_test_human2_2',
        ]
        embeds = [embed0, embed1, embed2, embed3, embed4, embed5, embed6]

        for name, emb in zip(names, embeds):
            torch_path = os.path.join(audio_path, f"{name}.torch")
            npy_path   = os.path.join(audio_path, f"{name}.npy")
            torch.save(emb, torch_path)
            # np.save(npy_path, emb.cpu().numpy())
        
        

    response_embed_dic = {
        'embed0': embed0,
        'embed1': embed1,
        'embed2': embed2,
        'embed3': embed3,
        'embed4': embed4,
        'embed5': embed5,
        'embed6': embed6
    } 
    
    return response_embed_dic

