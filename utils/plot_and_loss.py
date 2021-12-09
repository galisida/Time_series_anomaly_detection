import torch
import os
from matplotlib import pyplot as plt
from utils.data_prepare import get_batch


# TODO(done): add criterion
def plot_and_loss(eval_model, data_source, epoch, criterion, input_window):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1, input_window)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    len(test_result)

    plt.plot(test_result, color="red")
    # plt.plot(truth[:500], color="blue")
    # plt.plot(truth[:1000], color="blue")
    plt.plot(truth, color="blue")


    plt.plot(test_result - truth, color="green")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')

    if not os.path.exists("../graph"):
        os.mkdir("../graph")

    plt.savefig('graph/transformer-epoch%d.png' % epoch)
    plt.close()

    return total_loss / i
