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

wandb.init(project="my-test-project", entity="cocoshe")

wandb.config = {
  "learning_rate": 0.006,
  "epochs": 100,
  "batch_size": 64
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.006, help='Initial learning rate.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--input_window', type=int, default=20, help='Number of input steps.')
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

if not os.path.exists(os.getcwd() + os.sep +  "weights" + os.sep + present):
    os.mkdir(logDir)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)


def main(args):
    # input_window = 20  # number of input steps
    # output_window = 1  # number of prediction steps, in this model its fixed to one
    # batch_size = 10
    input_window = args.input_window
    output_window = args.output_window
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data, timestamp, scaler = get_data(args, input_window, output_window, device=device)
    model = TransAm().to(device)

    criterion = nn.MSELoss()
    # lr = 0.005
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.95)

    best_val_loss = float("inf")
    # epochs = 100  # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, input_window, model, optimizer, criterion, scheduler, epoch, batch_size)

        if epoch % 10 == 0:
            val_loss = plot_and_loss(model, val_data, epoch, criterion, input_window, timestamp, scaler, args.dim)
            predict_future(model, val_data, 200, input_window)
            save_path = "weights" + os.sep + present + os.sep +"trained-for-" + str(epoch) + "-epoch.pth"
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

    # src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
    # out = model(src)
    #
    # print(out)
    # print(out.shape)


    # save_path = "weights/last_model.pth"
    # torch.save(model.state_dict(), save_path)
    # print("save successfully")

if __name__ == "__main__":
    args = parse_args()
    # print(args)
    main(args)