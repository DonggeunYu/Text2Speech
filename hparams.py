
hparams = {
  "name": "Tacotron-Wavenet-Vocoder",

  "cleaners": "korean_cleaners",

  "skip_path_filter": False,
  "use_lws": False,

  'learning_rate': 1e-4,
  'weight_decay': 1e-6,

  "max_wav_value": 32768.0,
  "sample_rate": 44800,
  "filter_length": 1024,
  "hop_length": 256,
  "win_length": 1024,
  "n_mel_channels": 80,
  "mel_fmin": 0.0,
  "mel_fmax": 8000.0,

  "batch_size": 32,

  "preemphasize": False,
  "preemphasis": 0.97,
  "min_level_db": -100,
  "ref_level_db": 20,
  "signal_normalization": False,
  "allow_clipping_in_normalization": False,
  "symmetric_mels": True,
  "max_abs_value": 4,

  "rescaling": True,
  "rescaling_max": True,

  "trim_silence": True,
  "trim_fft_size": 512,
  "trim_hop_size": 128,
  "trim_top_db": 23,

  "clip_mels_length": True,
  "max_mel_frames": 1000,

  "l2_regularization_strength": 0,
  "sample_size": 15000,
  "silence_threshold": 0,

  "filter_width": 2,
  "gc_channels": 32,

  "input_type": "raw",
  "scalar_input": True,

  "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
  "residual_channels": 32,
  "dilation_channels": 32,
  "quantization_channels": 256,
  "out_channels": 30,
  "skip_channels": 512,
  "use_biases": True,

  "initial_filter_width": 32,
  "upsample_factor": [5, 5, 12],

  "wavenet_batch_size": 8,
  "store_metadata": False,
  "tacotron_decay_steps": 100000,

  "wavenet_learning_rate": 1e-2,
  "wavenet_decay_rate": 0.5,
  "wavenet_decay_steps": 300000,

  "wavenet_clip_gradients": False,

  "optimizer": "adam",
  "momentum": 0.9,
  "max_checkpoints": 3,

  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "use_fixed_test_inputs": False,

  "initial_learning_rate": 1e-2,
  "decay_learning_rate_mode": 0,
  "initial_data_greedy": True,
  "initial_phase_step": 8000,
  "main_data_greedy_factor": 0,
  "main_data": [""],
  "prioritize_loss": False,

  "model_type": "deepvoice",
  "speaker_embedding_size": 16,

  "embedding_size": 512,
  "dropout_prob": 0.5,

  "enc_prenet_sizes": [256, 128],
  "enc_bank_size": 16,
  "enc_bank_channel_size": 128,
  "enc_maxpool_width": 2,
  "enc_highway_depth": 4,
  "enc_rnn_size": 128,
  "enc_proj_sizes": [128, 128],
  "enc_proj_width": 3,
  "encoder_lstm_units" : 256,

    "enc_conv_num_layers" : 3,
    "enc_conv_kernel_size" : 5,
    "enc_conv_channels" : 512,
    "tacotron_zoneout_rate" : 0.1,

    "n_frames_per_step": 1,  # currently only 1 is supported
    "decoder_rnn_dim": 1024,
    "prenet_dim": 256,
    "max_decoder_steps": 1000,
    "gate_threshold": 0.5,
    "p_attention_dropout": 0.1,
    "p_decoder_dropout": 0.1,

  "attention_type": "bah_mon_norm",
  "attention_size": 256,
  "attention_state_size": 256,
  "attention_rnn_dim": 1024,
  "attention_dim": 128,

  "attention_location_n_filters": 32,
  "attention_location_kernel_size": 31,

  "dec_layer_num": 2,
  "dec_rnn_size": 256,
  "decoder_lstm_units": 1024,

  "dec_prenet_sizes": [256, 128],
  "post_bank_size": 8,
  "post_bank_channel_size": 128,
  "post_maxpool_width": 2,
  "post_highway_depth": 4,
  "post_rnn_size": 128,
  "post_proj_sizes": [256, 80],
  "post_proj_width": 3,

  "postnet_embedding_dim": 512,
  "postnet_kernel_size": 5,
  "postnet_n_convolutions": 5,

  "reduction_factor": 5,

  "min_tokens": 30,
  "min_iters": 30,
  "max_iters": 200,
  "skip_inadequate": False,

  "griffin_lim_iters": 60,
  "power": 1.5,

  "recognition_loss_coeff": 0.2,
  "ignore_recognition_level": 0,

  "num_freq": 1.,
  "frame_shift_ms": 1.,
  "frame_length_ms": 1.,

  "mask_padding": True
}

hparams["num_freq"] = 1024
hparams['frame_shift_ms'] = hparams['hop_length'] * 1000.0/ hparams['sample_rate']      # hop_size=  sample_rate *  frame_shift_ms / 1000
hparams['frame_length_ms'] = hparams['win_length'] * 1000.0/ hparams['sample_rate']
