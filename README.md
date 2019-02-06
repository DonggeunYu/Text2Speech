# Text2Speech

제작중임...

Text to Speech로 text를 음성으로 변환해주는 기술이다. Text to Speech 기술은 크게 Google의 Tacotron, Baidu의 Deep Voice가 양대산맥 형태를 이루고 있습니다. 이들 중 Tacotron-2를 Base Model로 개발하기로 했습니다.



#### Model Architecture

![](https://camo.githubusercontent.com/7bdc61ffb468c3daf1af3b5cef2ccc16c3473cd9/68747470733a2f2f707265766965772e6962622e636f2f625538734c532f5461636f74726f6e5f325f4172636869746563747572652e706e67)



## Preprocess

1. Trim lead/trail silences
2. Pre-emphasize
3. Rescale wav
4. Mu-law quantize or mulaw or raw
5. Compute the mel scale spectrogram from the wav
6. Compute the linear scale spectrogram from the wav
7. Time resolution adjustment
8. Save



####Trim lead/trail silences

wav의 앞과 뒤의 침묵을 다듬는다.

M-AILABS dataset을 사용할 경우 시작과 종료시 0.5 침묵을 다듬을 때 유용합니다.

#### Pre-emphasize

인간의 음성 생성 메커니즘은 에너지를 주파수에 걸쳐 떨어 뜨려 음향 모델의 정보량을 줄입니다. 특히 높은 주파수는 낮은 주파수에 비해 에너지가 적어 선형 예측 모델에서 좋지 않은 결과를 얻습니다.

이를 극복하기 위해 신호에 high pass filter를 적용하여 이러한 성분을 향상시키고 훨씬 균일하게 분산 된 스펙트럼을 얻습니다. 이를 pre-emphasizing 단계라고 합니다.



![](http://latex2png.com/output//latex_a5af090f15e35ca2a460fecf6766b2dd.png)

Where α = hparams.preemphasize = 0.97

![](https://i.stack.imgur.com/UFYmc.png)

Source: https://dsp.stackexchange.com/questions/43616/pre-emphasizing-in-speech-recognition



#### Rescale wav

wav의 파형을 -1~1범위 값으로 압축합니다.

[Trim lead/trail silences](####Trim lead/trail silences) 와 [Pre-emphasize](####Pre-emphasize) 의 값을 따로 계산한다.

![](http://latex2png.com/output//latex_be736ce1b94bb3865bd363dba52e1ca9.png)

Where α = hparams.rescaling_max = 0.999



#### Mu-law quantize

raw는 더 나은 품질이지만 학습 시간이 많이 소요됩니다. mulaw-quantize는 학습하기가 쉽지만 품질은 낮습니다.

μ-law algorithm은 오디오 신호의 동적 범위를 축소시킵니다.

source: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm



mulaw-quantize, mulaw, raw중 한가지를 선택합니다.

|      종류      | mulaw | out shape |                  특징                  |
| :------------: | :---: | :-------: | :------------------------------------: |
| mulaw quantize |   O   |  0 ~ mu   | wav에서 음성의 시작과 끝이 다듬어진다. |
|     mulaw      |   O   |  -1 ~ 1   |                                        |
|      raw       |   X   |  -1 ~ 1   |                                        |



where μ = hparams.quantize_channels = 2**16

Explanation: 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255



##### mulaw quantize

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2df208f7dd18fc678447dbffac60b8ca21eaffba)

scale [-1, 1] to [0, mu]

![](http://latex2png.com/output//latex_1a8dccf27b28ad1482ac164da330ec87.png)



앞 부분에서 뒤로 뒷 부분에서 앞으로 ![](http://latex2png.com/output//latex_e88260e85bf5cd4429f2912fb2863410.png) 를 처음으로 만족할때를 음성의 시작과 끝으로 정의한다.

where α = params.silence_threshold = 2



```python
wav = wav[start: end]
preem_wav = preem_wav[start: end]
out = out[start: end] #mulaw quantize의 두 번째 수식
```



##### mulaw

out:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2df208f7dd18fc678447dbffac60b8ca21eaffba)



##### raw

out: wav

rescale된 wav가 out가 됩니다.



#### Compute the mel scale spectrogram from the wav

STFT: 긴 시간의 신호를 짧은 시간 간격의 여러 신호로 나누고 각각의 신호에 대해 행하는 푸리에 변환. 신호의 진동수가 시간에 따라 어떻게 변하는지 알 수 있다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/d7573db711f34a739f0ecb5ecbfe42ab03227b70)

source: https://en.wikipedia.org/wiki/Short-time_Fourier_transform



amplitude to decibel:

![](http://latex2png.com/output//latex_0f6462aaf10b074baa6b878cb5413f7c.png)

where α = params.silence_threshold = -100



#### Compute the linear scale spectrogram from the wav

mel scale spectrogram과 같은 기능이다.



```python
assert linear_frames == mel_frames
```

으로 mel scale spectrogram과 linear scale spectrogram을 비교한다.

[Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2) 에서 사용한다.



#### Time resolution adjustment

audio mel spectrogram로 그램 사이의 시간 해상도를 조절한다.

오디오의 길이가 hop size의 배수가되도록하여 우리가 사용할 수 있도록한다.



#### Save

.npy 확장자를 사용하여 audio, mel spectrogram, linear spectrogram을 저장한다.



## References and Resources

#### GitHub

- [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [keithito/tacotron](https://github.com/keithito/tacotron)

#### Paper

- [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
- [Fast Wavenet](https://arxiv.org/pdf/1611.09482.pdf)