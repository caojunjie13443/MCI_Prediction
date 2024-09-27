import os
import copy
import gc
import random
import time

import numpy as np
import torch
import torch.nn as nn


def train_model(model, dataloaders, configs):
    # 定义数据集&设置超参
    lr, n_epochs = float(configs.lr), int(configs.epochs)  # 学习率  训练轮数
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

    if configs.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
    for param in model.parameters():  # 参数可学习
        param.requires_grad = True

    # 创建语言测试对应的文件夹  fintuned_model/language_test_id/model_type/
    fintuned_model_id_dir = configs.fintuned_model_dir + str(configs.language_test_id) + '_' + configs.metric + '/' + configs.model + '/'
    os.makedirs(fintuned_model_id_dir, exist_ok=True)
    # 存储ckpts文件, 路径名: dir/test_id/model_seed.pt
    model_save_path = os.path.join(fintuned_model_id_dir, (configs.model + f'_{configs.seed}.pt'))
    # 训练日志和验证日志
    train_file_path = os.path.join(fintuned_model_id_dir, f'train_{configs.seed}.txt')
    eval_file_path = os.path.join(fintuned_model_id_dir, f'eval.txt')
    train_writer = open(train_file_path, 'w', encoding='utf-8')
    eval_writer = open(eval_file_path, 'w', encoding='utf-8')

    total_step = len(dataloaders['train'])  # 每轮的step总数
    best_eval_metric, global_step = 0, 0  # z最优准确率、总step
    best_ckpt = None  # 保存最优ckpt, 用于测试集测试

    for epoch in range(1, n_epochs+1):
        running_loss, correct, total = 0.0, 0, 0  # 记录loss, 记录正确数量, 记录总数量
        print(f'Epoch {epoch}\n')

        for batch_idx, (data_, target_) in enumerate(dataloaders['train']):
            global_step += 1  # 更新总迭代次数
            data_, target_ = data_.to(device), target_.to(device)  # 载入数据和标签
            optimizer.zero_grad()  # 梯度清空
            outputs = model(data_)  # 输出模型预测结果

            loss = criterion(outputs, target_)  # 通过损失函数计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss  += float(loss.detach().item())  # 记录loss
            _, pred = torch.max(outputs, dim=1)  # 得到预测结果
            correct  += float(torch.sum(pred==target_).item())  # 记录预测正确数量
            total  += float(target_.size(0))  # 累加样本数

            train_writer.write(f'Epoch [{epoch}/{n_epochs}], Step [{batch_idx}/{total_step}], global_step: {global_step}, loss: {loss.item():.4f}\n')
            
            if global_step % configs.eval_every_steps == 0 and global_step != 0:  # 间隔指定step进行验证集验证
                with torch.no_grad():
                    model.eval()  # 验证模式
                    total_t, correct_t = 0, 0
                    true_positive, pred_positive, positive = 0, 0, 0
                    for data_t, target_t in (dataloaders['validation']):
                        data_t, target_t = data_t.to(device), target_t.to(device)
                        positive += float(torch.sum(target_t == 1).item())
                        outputs_t = model(data_t)
                        _, pred_t = torch.max(outputs_t, dim=1)
                        correct_t  += float(torch.sum(pred_t == target_t).item())
                        total_t  += float(target_t.size(0))
                        true_positive += float(torch.sum((pred_t + target_t) == 2).item())
                        pred_positive += float(torch.sum(pred_t == 1).item())

                    val_acc = correct_t / total_t
                    val_recall = true_positive / positive if positive else 0
                    val_precision = true_positive / pred_positive if pred_positive else 0
                    val_f1 = 2 * (val_recall * val_precision) / (val_recall + val_precision) if (val_recall + val_precision) else 0

                    if configs.metric == 'recall':
                        val_metric = val_recall
                    elif configs.metric == 'precision':
                        val_metric = val_precision
                    elif configs.metric == 'accuracy':
                        val_metric = val_acc
                    elif configs.metric == 'f1':
                        val_metric = val_f1

                    if val_metric > best_eval_metric:
                        best_eval_metric = val_metric
                        eval_writer.write(f'Epoch: {epoch}/{n_epochs}, Step: {batch_idx}/{total_step}, global_step: {global_step}, selection_metric: {configs.metric}\n')
                        eval_writer.write(f'seed: {configs.seed}, acc:{val_acc:.4f}, recall:{val_recall:.4f}, precision:{val_precision:.4f}, f1-score:{val_f1:.4f}\n')
                        eval_writer.write('\n')
                        best_ckpt = copy.deepcopy(model)
                        torch.save(model, model_save_path)
                        print(f'Improvement-Detected, save-model at global-step:{global_step}')

        train_writer.write('\n')
        train_writer.write(f'Epoch [{epoch}/{n_epochs}], Epoch_Acc: {(correct / total):.4f}, Epoch_Loss: {(running_loss / total_step):.4f}')
        train_writer.write('\n')
        
        model.train()

    train_writer.close()
    eval_writer.close()

    return  best_ckpt
