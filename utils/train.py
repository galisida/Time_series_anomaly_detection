import math
import time

import torch
import wandb

from utils.data_prepare import get_batch


# TODO(done): 实例化model并传入, optimizer, criterion, scheduler, epoch, batch_size
def train(train_data, input_window, model, optimizer, criterion, scheduler, epoch=100, batch_size=10):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size, input_window)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()

        # wandb.log({"loss": loss})
        #
        # Optional
        # wandb.watch(model)


        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        # log_interval = int(len(train_data) / batch_size / 5)
        log_interval = 10
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} '.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()
