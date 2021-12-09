import math
import time
import os
import torch
import torch.nn as nn
import numpy as np
from utils.data_prepare import get_data
from model.Transformer import TransAm
from utils.train import train
from utils.plot_and_loss import plot_and_loss
from utils.reconstruct import predict_future
from utils.eval import evaluate

torch.manual_seed(0)
np.random.seed(0)

if not os.path.exists("weights"):
    os.mkdir("weights")

# 格式化成2016-03-20 11:45:39形式
present = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
if not os.path.exists("weights/" + present):
    os.mkdir("weights/" + present)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)

input_window = 100  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, val_data = get_data(input_window, output_window, device=device)
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float("inf")
epochs = 100  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data, input_window, model, optimizer, criterion, scheduler)  # , epoch=100, batch_size=10)

    if epoch % 1 == 0:
        val_loss = plot_and_loss(model, val_data, epoch, criterion, input_window)
        predict_future(model, val_data, 200, input_window)
        save_path = "weights/" + present + "/trained-for-" + str(epoch) + "-ecpoch.pth"
        torch.save(model.state_dict(), save_path)
    else:
        val_loss = evaluate(model, val_data, criterion, input_window)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
            time.time() - epoch_start_time), math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        save_path = "weights/best_model.pth"
        torch.save(model.state_dict(), save_path)
        # print("save successfully")

    scheduler.step()

# src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
# out = model(src)
#
# print(out)
# print(out.shape)


# save_path = "weights/last_model.pth"
# torch.save(model.state_dict(), save_path)
# print("save successfully")
