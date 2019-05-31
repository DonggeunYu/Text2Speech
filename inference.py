import torch

from hparams import hparams
from tacotron.tacotron import Tacotron
from den

checkpoint_file = './checkpoint_path/checkpoint_0'
hparams.update({"num_speakers": len(config.data_paths)})

model = Tacotron(hparams).cuda()
model.load_state_dict(torch.load(checkpoint_file)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow_256channels.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)