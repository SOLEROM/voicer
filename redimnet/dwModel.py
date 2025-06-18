import torch

# model_name='M' # ~b3-b4 size
# train_type='ft_mix'
# dataset='vb2+vox2+cnc'

# model_name='b2' # ~b2
# train_type='ptn'
# dataset='vox2'

model_name='B0'
train_type='ptn'
dataset='vox2'

torch.hub.set_dir('/data/proj/voice/redimnet/models')

model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name=model_name, 
                       train_type=train_type, 
                       dataset=dataset)

model.eval()



