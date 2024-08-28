# Last updated on 2022/07/21
# Author: Bingli
# 本文件用于准备训练数据集。

import random

from torch.utils.data import Dataset

from .features import AudioUtil

# Audio Dataset

class SoundDS(Dataset):
  """Return the dataset for training. """

  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration_unit = 20
    self.sr = 48000
    self.channel = 3
    self.shift_pct = 0.4
    SEED = 42
    random.seed(SEED)
    self.SHIFT_RANDOM = random.random()        
  # Number of items in dataset
  def __len__(self):
    return len(self.df)    
    
  # Get i'th item in dataset
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with the relative path
    # audio_file = self.data_path + self.df.loc[idx, 'file'] + '.wav'
    audio_file = self.data_path + self.df.loc[idx, 'file']
    
    # Get the Class ID
    class_id = self.df.loc[idx, 'label']
    
    # Make all sounds have the same number of channels and same sample rate. 
    # Unless the sample rate is the same, the pad_trunc will still result in 
    # arrays of different lengths, even though the sound duration is the same.
    aud = AudioUtil.open(audio_file)
    reaud = AudioUtil.resample(aud, self.sr)
    reshape = AudioUtil.reshape(reaud)
    dur_aud = AudioUtil.pad_trunc(reshape, self.duration_unit)
    shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct, self.SHIFT_RANDOM)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    return aug_sgram, class_id
