import numpy as np
from utils import audio
from scipy import signal
from hparams import hparams, hparams_debug_string
import matplotlib.pyplot as plt

def preemphasis(wav, k, preemphasize=True):
	if preemphasize:
		return signal.lfilter([1, -k], [1], wav)
	return wav

x = audio.load_wav("./kss/1/1_0000.wav", sr=44100)
mel_spectrogram = audio.melspectrogram(x, hparams).astype(np.float32)
linear_spectrogram = audio.linearspectrogram(x, hparams).astype(np.float32)
print(mel_spectrogram)
print(linear_spectrogram)
plt.plot(mel_spectrogram)
#plt.plot(linear_spectrogram)
plt.show()