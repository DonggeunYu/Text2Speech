#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import sys
import soundfile as sf

from hparams import hparams
from tacotron.tacotron import Tacotron
from waveglow.denoiser import Denoiser
from text import text_to_sequence
from text.symbols import symbols

import matplotlib.pylab as plt

import IPython.display as ipd

import sys

sys.path.append('waveglow/')
import numpy as np
import torch

from utils.audio_processing import griffin_lim
from text import text_to_sequence
from waveglow.denoiser import Denoiser


# In[2]:
hparams['smaple_rate'] = 22050

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
    plt.show()

# In[3]:


checkpoint_path = './checkpoint_path/checkpoint_30000'
waveglow_path = 'waveglow/waveglow_256channels.pt'
num_speakers = 2
speaker_id = 0
text = "설문을 만들었습니다."

# In[4]:

cuda1 = torch.device('cuda:1')

model = Tacotron(hparams, len(symbols), num_speakers=num_speakers).cuda(1)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

# In[7]:


waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()

for m in waveglow.modules():
    if 'Conv' in str(type(m)):
        setattr(m, 'padding_mode', 'zeros')

for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

# In[ ]:


sequence = np.array(text_to_sequence(text))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
speaker_id = np.array([speaker_id])
speaker_id = speaker_id.reshape(speaker_id.shape[0], -1)

speaker_id = torch.autograd.Variable(torch.from_numpy(speaker_id)).cuda().long()
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, speaker_id)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))

import soundfile as sf
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
sf.write('tone_440.wav', audio[0].float().data.cpu().numpy(), hparams['sample_rate'], format='WAV', endian='LITTLE', subtype='PCM_16') # 깨지지 않음
print(np.shape(audio))