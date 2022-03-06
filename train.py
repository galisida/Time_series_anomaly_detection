import copy
import math
import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from utils.data_prepare import get_data
from model.Transformer import TransAm
from utils.train import train
from utils.plot_and_loss import plot_and_loss
from utils.reconstruct import predict_future
from utils.eval import evaluate

import wandb


# wandb.init(project="my-test-project", entity="cocoshe")

# wandb.config = {
#   "learning_rate": 0.006,
#   "epochs": 100,
#   "batch_size": 64
# }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    # parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    # parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate.') # seems not good
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--input_window', type=int, default=2, help='Number of input steps.')
    parser.add_argument('--output_window', type=int, default=1, help='Number of prediction steps, '
                                                                     'in this model its fixed to one.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch_size.')
    parser.add_argument('--model', type=str, default="weights/best_model.pth", help='Load model path.(default: '
                                                                                    'weights/best_model.pth)')

    parser.add_argument('--port_id', type=str, default=None, help='port_id.')
    parser.add_argument('--polution_id', type=str, default=None, help='polution_id.')
    parser.add_argument('--date_s', type=str, default=None, help='start of the date.')
    parser.add_argument('--date_e', type=str, default=None, help='end of the date.')
    parser.add_argument('--dim', type=str, default=None, help='choose one dim(input dim name).')
    parser.add_argument('--company_id', type=str, default=None, help='company_id.')
    # parser.add_argument("--config", help="train config file path")
    # parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


# torch.manual_seed(0)
# np.random.seed(0)

if not os.path.exists("weights"):
    os.mkdir("weights")

# 格式化成2016-03-20_11:45:39形式
present = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())
logDir = os.getcwd() + os.sep + 'weights' + os.sep + present
print(logDir)

if not os.path.exists(os.getcwd() + os.sep + "weights" + os.sep + present):
    os.mkdir(logDir)


# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)


def train_(args, req_json, choice, preloaded_csv, meta):
    # def main(args):
    # input_window = 5  # number of input steps
    # output_window = 1  # number of prediction steps, in this model its fixed to one
    # batch_size = 10
    input_window = args.input_window
    output_window = args.output_window
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # web 版本的参数
    if req_json['port_id'] is not None:
        args.port_id = req_json['port_id']
    if req_json['date_s'] is not None:
        args.date_s = req_json['date_s']
    if req_json['date_e'] is not None:
        args.date_e = req_json['date_e']
    if req_json['company_id'] is not None:
        args.company_id = req_json['company_id']
    if req_json['polution_id'] is not None:
        args.polution_id = req_json['polution_id']
    args.dim = req_json['dim']

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    train_data, val_data, timestamp, scaler = get_data(args, input_window, output_window, device=device,
                                                       preloaded_csv=preloaded_csv, meta=meta)
    # 全为0的情况
    if train_data is None:
        return None

    model = TransAm().to(device)

    criterion = nn.MSELoss()
    # lr = 0.005

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=lr, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.95)

    best_val_loss = float("inf")
    # epochs = 100  # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, input_window, model, optimizer, criterion, scheduler, epoch, batch_size)

        if epoch % 5 == 0:
            val_loss, res = plot_and_loss(model, val_data, epoch, criterion, input_window, timestamp, scaler, args.dim,
                                          choice, lr=lr)
            predict_future(model, val_data, 200, input_window)
            save_path = "weights" + os.sep + present + os.sep + "trained-for-" + str(epoch) + "-epoch.pth"
            torch.save(model.state_dict(), save_path)
        else:
            val_loss = evaluate(model, val_data, criterion, input_window)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} '.format(epoch, (
                time.time() - epoch_start_time), val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            save_path = "weights" + os.sep + "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print("save successfully")

        scheduler.step()

    res['date'] = res['date'].astype(str)
    return res
    # src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
    # out = model(src)
    #
    # print(out)
    # print(out.shape)

    # save_path = "weights/last_model.pth"
    # torch.save(model.state_dict(), save_path)
    # print("save successfully")


# if __name__ == "__main__":
def run_model(req_json, preloaded_csv):
    args = parse_args()
    # print(args)
    dim_names = ['concentration', 'amount']
    date_names = ['date_s_1', 'date_e_1', 'date_s_2', 'date_e_2']
    # 非总览页面
    if req_json['polution_id'] != '':
        meta = 'single'
        res = single_polution_method(req_json, preloaded_csv, args, dim_names, date_names, meta)
        return res, meta, None
    else:  # 总览页面
        res = []
        meta = 'all'
        polution_ids = preloaded_csv['polution_id'].dropna().unique()
        print(polution_ids)
        for i in range(len(polution_ids)):
            args.polution_id = polution_ids[i]
            print('current polution_id: ', args.polution_id)
            res.append(single_polution_method(req_json, preloaded_csv, args, dim_names, date_names, meta))
        return res, meta, polution_ids

    # main(args)


def single_polution_method(req_json, preloaded_csv, args, dim_names, date_names, meta):
    res = []
    for dim_name in dim_names:
        for i in range(0, len(date_names), 2):
            req_json['date_s'] = req_json[date_names[i]]
            req_json['date_e'] = req_json[date_names[i + 1]]
            req_json['dim'] = dim_name
            choice = i // 2 + 1
            res.append(train_(args, req_json, choice, preloaded_csv=preloaded_csv, meta=meta))
            if res[-1] is None:
                res = res[:-1]

    return res
