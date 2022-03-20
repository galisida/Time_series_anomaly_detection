import pandas as pd
import torch
import os
from matplotlib import pyplot as plt
from utils.data_prepare import get_batch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import wandb



# def plot_and_loss(eval_model, data_source, epoch, criterion, input_window, timestamp, scaler, dim, threshold=None):
def plot_and_loss(eval_model, data_source, epoch, criterion, input_window, scaler, dim, labels):
    model_type = eval_model.model_type
    eval_model.eval()
    # print('---------------------------------')
    # print('data_source shape:', data_source.shape)
    # print('data_source[[0]]:', data_source[[0]].shape)
    data_source = torch.cat((data_source[[0]], data_source, data_source[[-1]]), 0)


    # data_source = np.concentrate(data_source[])
    total_loss = 0.
    # test_result = torch.Tensor(0)
    # truth = torch.Tensor(0)
    print('data_source shape:', data_source.shape)
    input_dim = data_source.shape[1]
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1, input_window)
            output = eval_model(data)
            if i == 0:
                print('output shape:', output.shape)
                if output.shape[2] == 1:
                    flag = True
                    test_result = torch.cat((output[0].view(-1), output[:-1].view(-1).cpu()), 0)
                    truth = target.view(-1)
                else:
                    print('output[[0]].shape:', output[[0]].shape)
                    print('output[:-1].shape:', output[:-1].shape)
                    test_result = torch.cat((output[[0]].squeeze(1), output[:-1].squeeze(1).cpu()), 0)
                    truth = target.squeeze(1)
                    # test_result = torch.cat((output[0].view(-1), test_result.view(-1).cpu()), 0)
            total_loss += criterion(output, target).item()
            if output.shape[2] == 1:
                test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
                truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            else:
                test_result = torch.cat((test_result, output[[-1]].squeeze(1).cpu()), 0)
                truth = torch.cat((truth, target[[-1]].squeeze(1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    # len(test_result)

    # plt.plot(truth[:500], color="blue")
    # plt.plot(truth[:1000], color="blue")
    # test_result = torch.cat((test_result.view(-1).cpu(), test_result[-1].view(-1)), 0)
    print('truth shape:', truth.shape)
    print('test_result shape:', test_result.shape)
    if len(truth.shape) == 1:
        truth = scaler.inverse_transform(truth.reshape(-1, 1))
        test_result = scaler.inverse_transform(test_result.reshape(-1, 1))
    else:
        truth = scaler.inverse_transform(truth)
        test_result = scaler.inverse_transform(test_result)

    # truth = truth.reshape(-1)
    # test_result = test_result.reshape(-1)
    print('output truth shape:', truth.shape)
    print('output test_result shape:', test_result.shape)

    plt.plot(truth, color="blue")
    plt.plot(test_result, color="red")


    # plt.plot(test_result - truth, color="green")
    # wandb.log({"test_result - truth": (test_result - truth)})

    # save loss
    print('test_result[0].shape, type', test_result[0].shape, test_result[0].dtype)
    print('test_result.shape, type', test_result.shape, test_result.dtype)
    # test_result = torch.cat((test_result[0], test_result), 0)
    print("loss shape: ", (test_result - truth).shape)
    # res = pd.DataFrame({"date": timestamp.values[:len(truth)], "truth": truth[:, 0], "test_result": test_result[:, 0], "loss": (test_result - truth)[:, 0]})
    # res = pd.DataFrame({"truth": truth[:, 0], "test_result": test_result[:, 0], "loss": (test_result - truth)[:, 0]})
    # if os.path.exists("res") == False:
    #     os.mkdir("res")
    # res_csv_path = "res/test_loss_" + str(dim) + ".csv"
    # res_csv_path = "res/test_loss_" + model_type + ".csv"
    # with open(res_csv_path, "w") as f:
    #     res.to_csv(res_csv_path)

    loss_value = np.abs(test_result - truth)
    # desc_idx = loss_value.argsort()[::-1]
    # threshold = loss_value[desc_idx[1000]]  # todo:怎么找阈值？能自适应吗？统计方法？要好好想想
    # print('---------------------------------')
    # print("threshold: ", threshold)
    # print('---------------------------------')
    # pred = np.where(test_result - truth > threshold, 1, 0)
    # label = pd.read_csv('dataset/cpu4.csv')['label'].values

    clf = svm_c(loss_value, labels)
    pred = clf.predict(loss_value)



    exp_precision = cal_precision(pred, labels)
    exp_recall = cal_recall(pred, labels)
    exp_acc = cal_acc(pred, labels)
    exp_f1 = cal_f1(pred, labels)
    print('precision: ', exp_precision, ' recall: ', exp_recall, ' acc: ', exp_acc, ' f1: ', exp_f1)
    # wandb.log({"precision": cal_precision(pred, label), "recall": cal_recall(pred, label), "acc": cal_acc(pred, label), "f1": cal_f1(pred, label)})
    exp_out = pd.DataFrame({'precision': [cal_precision(pred, labels)], 'recall': [cal_recall(pred, labels)], 'acc': [cal_acc(pred, labels)], 'f1': [cal_f1(pred, labels)]})
    exp_out_path = "exp/exp_out_" + str(epoch) + " model_" + model_type + ".csv"
    exp_out.to_csv(exp_out_path, index=False)

    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    # plt.xticks(ticks=range(len(truth)), labels=timestamp.values[:len(truth)], rotation=90)

    if not os.path.exists("graph"):
        os.mkdir("graph")
    plt.savefig('graph/transformer-epoch%d_%s_%s.png' % (epoch, dim, model_type))
    plt.close()

    return total_loss / i


def cal_precision(pred, label):
    precision = precision_score(label, pred)
    return precision


def cal_recall(pred, label):
    recall = recall_score(label, pred)
    return recall


def cal_acc(pred, label):
    acc = accuracy_score(label, pred)
    return acc


def cal_f1(pred, label):
    f1 = f1_score(label, pred)
    return f1


def svm_c(input_data, labels):
    print('---------------------------------')
    print('start SVM')
    print('---------------------------------')
    x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, random_state=123)
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉，n_jobs=-1，多核计算
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = RandomizedSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('精度为%s' % score)

    return clf