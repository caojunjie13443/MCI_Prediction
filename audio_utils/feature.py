# Last updated on 2022/07/21
# Author: Bingli
# 本文件输入为录音文件，用于声学特征的提取。

import random
import statistics

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np

class AudioUtil():
  """Return the processed audio features. """

  # Load an audio file. Return the signal as a tensor and the sample rate
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)
  
    # Since Resample applies to a single channel, we resample one channel at a time
  @staticmethod
  def resample(aud, newsr = 16000):
    sig, sr = aud
    if (sr == newsr):
      # Nothing to do
      return aud
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])

    return ((resig, newsr))

  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)

    
  # Shifts the signal to the left or right by some percent. 
  # Values at the end are 'wrapped around' to the start of the transformed signal.
  @staticmethod
  def time_shift(aud, shift_limit, shift_random):
    sig,sr = aud
    _, sig_len = sig.shape
    random.seed(42)
    SHIFT_RANDOM = random.random()
    shift_amt = int(shift_random * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

    
  # Generate a Spectrogram
  @staticmethod
  def feat_melspec(aud, n_mels=128, n_fft=2048, hop_len=512):
    sig, sr = aud
    top_db = 80

    # spec shape: [channel=1, n_mels, time]
    spec = T.MelSpectrogram(sr, 
                            n_fft=n_fft, 
                            normalized = True, 
                            hop_length=hop_len, 
                            n_mels=n_mels)(sig)

    # Convert to decibels
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
    sgram = spec[0].reshape(spec[0].shape[1], 128, 1)
    
    # print(sgram.shape)
    return (sgram)

  # Generate a MFCC
  @staticmethod
  def feat_mfcc(aud, n_mels=128, n_fft=2048, n_mfcc=128, hop_len=512):
    sig, sr = aud
    mfcc = T.MFCC(sr, n_mfcc,
                            melkwargs={
                            "n_fft": n_fft,
                            "n_mels": n_mels,
                            "hop_length": hop_len,
                            "mel_scale": "htk",
                        },)(sig)

    return (mfcc)
    

  # Generate a Pitch
  @staticmethod
  def feat_pitch(aud):
    sig, sr = aud
    pitch = F.detect_pitch_frequency(sig, sr)

    return (pitch)
  

  # Augment the Spectrogram by masking out some sections of it in both the frequency
  # dimension(horizontal) and the time dimension(vertical) to prevent overfitting.
  # The masked sections are replaced with the mean value.
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    aug_spec = aug_spec[0].reshape(aug_spec[0].shape[1], 128)

    return aug_spec
  
  @staticmethod
  def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled
