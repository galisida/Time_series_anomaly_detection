import argparse
import math
import time
import torch
import torch.nn as nn
import numpy as np
from utils.data_prepare import get_data
from model.Transformer import TransAm
from utils.train import train
from utils.plot_and_loss import plot_and_loss
from utils.reconstruct import predict_future
from utils.eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--input_window', type=int, default=100, help='Number of input steps.')
    parser.add_argument('--output_window', type=int, default=1, help='Number of prediction steps, '
                                                                     'in this model its fixed to one.')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of batch_size.')
    parser.add_argument('--model', type=str, default="weights/best_model.pth", help='Load model path.(default: '
                                                                                    'weights/best_model.pth)')
    # parser.add_argument("--config", help="train config file path")
    # parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


# torch.manual_seed(0)
# np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)

# weight_path = "weights/best_model"


def main(args):
    input_window = 100  # number of input steps
    output_window = 1  # number of prediction steps, in this model its fixed to one
    batch_size = 10
    weight_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransAm().to(device).load_state_dict(torch.load(weight_path))
    train_data, val_data, timestamp = get_data(args, input_window, output_window, device=device, preloaded_csv=None)

    criterion = nn.MSELoss()

    epochs = 1  # The number of epochs

    epoch_start_time = time.time()

    val_loss, res = plot_and_loss(model, val_data, 1, criterion, input_window, timestamp)
    predict_future(model, val_data, 200, input_window)
    val_loss = evaluate(model, val_data, criterion, input_window)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'
          .format(1, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    # src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
    # out = model(src)
    #
    # print(out)
    # print(out.shape)


if __name__ == "__main__":
    args = parse_args()
