# coding=utf-8
import time
import random
import collections
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchmetrics
from torch.utils.data import random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from audio_utils.dataset import SoundDS
from audio_utils.predict import OutputPred
from audio_utils.train import train_model
from model_utils.config_utils import load_model_configs
from model_utils.random_seed import setup_seed


def load_dataset(configs):
    all_df = pd.read_csv(configs.audio_test_file)
    all_df = all_df.dropna()
    all_df = all_df.reset_index(drop=True)
    audio_df = all_df[['file_name', 'label']].copy()
    audio_df.columns = ['file', 'label']

    train, validate, test = np.split(all_df.sample(frac=1, random_state=configs.seed), [int(.6*len(all_df)), int(.8*len(all_df))])
    train_idx, val_idx, test_idx = train.index, validate.index, test.index
    myds = SoundDS(audio_df, configs.wav_audio_dir)
    train_ds, val_ds, test_ds = Subset(myds, train_idx), Subset(myds, val_idx), Subset(myds, test_idx)

    batch_size = configs.batch_size
    dataloaders = {
        'train':
        torch.utils.data.DataLoader(train_ds,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0),
        'validation':
        torch.utils.data.DataLoader(val_ds,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0), 
        'test':
        torch.utils.data.DataLoader(test_ds,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0) 
    }

    return dataloaders


def finetune_pretrain_model(dataloaders, configs):
    assert configs.model in ["resnet18", "resnet50", "resnet101", "alexnet", "vgg"]

    if 'resnet' in configs.model:
        if configs.model == "resnet18":
            model = torchvision.models.resnet18(weights=None)
            model.load_state_dict(torch.load(configs.pretrained_model_dir + 'resnet18-f37072fd.pth'))
        elif configs.model == "resnet50":
            model = torchvision.models.resnet50(weights=None)
            model.load_state_dict(torch.load(configs.pretrained_model_dir + 'resnet50-0676ba61.pth'))
        elif configs.model == "resnet101":
            model = torchvision.models.resnet101(weights=None)
            model.load_state_dict(torch.load(configs.pretrained_model_dir + 'resnet18-f37072fd.pth'))
        # 修改输入shape和输出shape
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2),
                    nn.LogSoftmax(dim=1))

    elif configs.model == "alexnet":
        model = torchvision.models.alexnet(weights=None)
        model.load_state_dict(torch.load(configs.pretrained_model_dir + 'alexnet-owt-7be5be79.pth'))
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
                                nn.Linear(num_ftrs, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 2),
                                nn.LogSoftmax(dim=1))

    elif configs.model == "vgg16":
        model = torchvision.models.vgg16(weights=None)
        model.load_state_dict(torch.load(configs.pretrained_model_dir + 'vgg16-397923af.pth'))
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
                                nn.Linear(num_ftrs, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 2),
                                nn.LogSoftmax(dim=1))

    model = train_model(model, dataloaders, configs)
        
    return model


def inference(model, dataloaders, writer, configs):
    if configs.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    total_t, correct_t = 0, 0
    true_positive, pred_positive, positive = 0, 0, 0
    with torch.no_grad():
        model.eval()  # 验证模式
        for data_t, target_t in (dataloaders['test']):
            data_t, target_t = data_t.to(device), target_t.to(device)
            positive += float(torch.sum(target_t==1).item())
            outputs_t = model(data_t)
            _, pred_t = torch.max(outputs_t, dim=1)
            correct_t  += float(torch.sum(pred_t==target_t).item())
            total_t  += float(target_t.size(0))
            true_positive += float(torch.sum((pred_t + target_t)==2).item())
            pred_positive += float(torch.sum(pred_t==1).item())

        val_acc = correct_t / total_t
        val_recall = true_positive / positive
        val_recall = true_positive / positive if positive else 0
        val_precision = true_positive / pred_positive if pred_positive else 0
        val_f1 = 2 * (val_recall * val_precision) / (val_recall + val_precision) if (val_recall + val_precision) else 0

        writer.write(f'seed: {configs.seed}, acc:{val_acc:.4f}, recall:{val_recall:.4f}, precision:{val_precision:.4f}, f1-score:{val_f1:.4f}\n')
    
    print("End inference!")


if __name__=='__main__':
    torch.cuda.empty_cache()
    gc.collect()

    config_file = "./configs/configs.yml"
    configs = load_model_configs(config_file, 'yml')
    assert configs.replicate_time > 0

    for i in range(configs.replicate_time):
        configs.seed += 1
        setup_seed(configs.seed)  # 设置随机种子
        dataloaders = load_dataset(configs)  # 读取数据集

        best_fituned_model = finetune_pretrain_model(dataloaders, configs)  # 训练最优微调模型

        fintuned_model_id_dir = configs.fintuned_model_dir + str(configs.language_test_id) + '/' + configs.model + '/'
        test_file_path = os.path.join(fintuned_model_id_dir, f'test.txt')
        pred_writer = open(test_file_path, 'a', encoding='utf-8')

        inference(best_fituned_model, dataloaders, pred_writer, configs)
        pred_writer.close()
        
        torch.cuda.empty_cache()
        gc.collect()
