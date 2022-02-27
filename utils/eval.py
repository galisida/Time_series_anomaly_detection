import torch
from utils.data_prepare import get_batch


# TODO(done): add criterion
def evaluate(eval_model, data_source, criterion, input_window):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size, input_window)
            output = eval_model(data)  # [seq_len(input_window), batch_size, dim]
            # print(data.shape)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)
