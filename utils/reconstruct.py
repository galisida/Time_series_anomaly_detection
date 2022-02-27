import torch
import os
from matplotlib import pyplot as plt
from utils.data_prepare import get_batch


# TODO(done): add input_window
# predict the next n steps based on the input data
def predict_future(eval_model, data_source, steps, input_window):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source, 0, 1, input_window)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    # I used this plot to visualize if the model pics up any long therm struccture within the data.
    plt.plot(data, color="red")
    plt.plot(data[:input_window], color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')

    if not os.path.exists("../graph"):
        os.mkdir("../graph")

    plt.savefig('graph/transformer-future%d.png' % steps)
    plt.close()
