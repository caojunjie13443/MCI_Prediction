from torch.utils.data import Dataset

from audiofeature import AudioUtil
import random
import torch
import numpy as np

# Audio Dataset
n_fft = 2048
win_length = None
hop_length = 512
n_lfcc = 256

class AudioDS(Dataset):
  """Return the dataset for training. """

  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 60
    self.sr = 48000
    self.channel = 1
    self.shift_pct = 0.4
    self.n_fft = 2048
    self.win_length = None
    self.hop_length = 512

    self.n_mels = 128
    self.n_mfcc = 128
    self.n_lfcc = 128


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
    
    # # Get the Class ID
    class_id = self.df.loc[idx, 'label']
    
    # Make all sounds have the same number of channels and same sample rate. 
    # Unless the sample rate is the same, the pad_trunc will still result in 
    # arrays of different lengths, even though the sound duration is the same.
    aud = AudioUtil.open(audio_file)

    re_aud = AudioUtil.resample(aud, newsr = 8000)
    dur_aud = AudioUtil.pad_trunc(re_aud, self.duration)
    
    sgram = AudioUtil.feat_melspec(dur_aud, n_mels=128, n_fft=2048, hop_len=512)
    # sgram_scaled = AudioUtil.spec_to_image(sgram, n_mels=128, n_fft=2048, hop_len=512)
    # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    # mfcc = AudioUtil.feat_mfcc(aud, n_mels=128, n_fft=2048, n_mfcc=128, hop_len=512)
    # pitch = AudioUtil.feat_pitch(aud)
    # feat_lstm = sgram.reshape(1, sgram.shape[1], 128)
    # feat_lstm = np.array([sgram[0][:, 1].numpy(), mfcc[0][:, 1].numpy()]).reshape(1, 128, 2)
    data = sgram.numpy()

    # return sgram

    return data
