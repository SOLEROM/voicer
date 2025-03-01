import torch

model_name='M' # ~b3-b4 size
train_type='ft_mix'
dataset='vb2+vox2+cnc'

torch.hub.set_dir('/data/deep/redimnet/models')

model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name=model_name, 
                       train_type=train_type, 
                       dataset=dataset)

model.eval()


