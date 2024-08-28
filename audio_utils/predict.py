# Last updated on 2022/07/21
# Author: Bingli
# 本文件用于输出预测结果和打印识别报告。

import torch
from sklearn.metrics import classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
def OutputPred(model, val_dl):
  """Print classification report and Return prediction values. """

  correct_prediction = 0
  total_prediction = 0

  scores = []
  y_probs = []
  y_preds = []
  y_true = []
  # Disable gradient updates
  with torch.no_grad():
    for i, data in enumerate(val_dl):
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data
      # inputs = inputs.view(-1,5532,64)#lstm需要改
      inputs, labels = inputs.to(device), labels.to(device)
      # inputs, labels = data[0], data[1]

      # # Normalize the inputs
      # inputs_m, inputs_s = inputs.mean(), inputs.std()
      # inputs = (inputs - inputs_m) / inputs_s

      # # Get predictions
      outputs = model(inputs)
      _, preds = torch.max(outputs.data, 1)
      probs = [i[1] for i in torch.softmax(outputs, 1)]
      # y_pred_probs = torch.sigmoid(outputs)
      # # print(y_pred_probs)
      # probs = torch.softmax(outputs, 1)
      # probs, preds = probs.topk(1, 1)
      # print(probs, preds)
      # # Get the predicted class with the highest score
      # score, prediction = torch.max(outputs, 1)

      # # Count of predictions that matched the target label
      # correct_prediction += (prediction == labels).sum().item()

      # total_prediction += prediction.shape[0]

      # scores += score.cpu().tolist()
      # probs += y_pred_probs.cpu().tolist()
      # y_preds += torch.max(outputs,1)[1].cpu().tolist()
      # y_true += labels.tolist()
      # output = model.forward(inputs)
      # probs = torch.nn.functional.softmax(output, dim=1)
      # print(probs)
      # conf, classes = torch.max(probs, 1)
      # print(classes, conf)
      y_probs.extend([i.item() for i in probs])
      y_preds.extend([i.item() for i in preds])
      y_true.extend([i.item() for i in labels])
      # scores.extend(conf.item())

  # acc = correct_prediction/total_prediction
  # print(len(y_preds), len(probs), len(y_true))
  print(classification_report(y_true, y_preds))

  #print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
  # return probs, y_preds, scores

  return y_probs, y_preds, y_true
