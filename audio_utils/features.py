# Last updated on 2022/07/21
# Author: Bingli
# 本文件输入为录音文件，用于声学特征的提取。

import random
import statistics

import torch
import torchaudio


class AudioUtil():
  """Return the processed spectrograms. """

  # Load an audio file. Return the signal as a tensor and the sample rate
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)
  # # Convert the given audio to the desired number of channels
  # @staticmethod
  # def rechannel(aud, new_channel):
  #   sig, sr = aud

  #   if (sig.shape[0] == new_channel):
  #     # Nothing to do
  #     return aud
    
  #   if (new_channel == 1):
  #     # Convert from stereo to mono by selecting only the first channel
  #     resig = sig[:1, :]
  #   else:
  #     # Convert from mono to stereo by duplicating the first channel
  #     # Convert to 3 channels to fit required shape of model
  #     resig = torch.cat([sig, sig, sig])
  #     # resig = torch.cat([sig, sig])
  #   return ((resig, sr))


  # Since Resample applies to a single channel, we resample one channel at a time
  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud
    if (sr == newsr):
      # Nothing to do
      return aud
    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])
    return ((resig, newsr))
  
  # corvert the given audio to three channels with different frames
  @staticmethod
  def reshape(aud):
    sig, sr = aud
    unit = int(sig.shape[1]/3)
    sig1, sig2, sig3 = sig[:, :unit], sig[:, unit:unit*2], sig[:, unit*2:]
    # sig1, _ = torchaudio.load(audio_file, frame_offset=0, num_frames=duration_unit)
    # sig2, _ = torchaudio.load(audio_file, frame_offset=sr * duration_unit, num_frames=duration_unit)
    # sig3, _ = torchaudio.load(audio_file, frame_offset=2 * sr * duration_unit, num_frames=duration_unit)
    # return (sig1, sig2, sig3)
    # concat three sigs
    resig = torch.cat([sig1, sig2, sig3])
    return ((resig, sr))
    
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
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

    
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

    return aug_spec
