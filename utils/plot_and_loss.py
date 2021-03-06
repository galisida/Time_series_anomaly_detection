import pandas as pd
import torch
import os
from matplotlib import pyplot as plt
from utils.data_prepare import get_batch
# import wandb


def plot_and_loss(eval_model, data_source, epoch, criterion, input_window, timestamp, scaler, dim, choice, lr):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1, input_window)
            output = eval_model(data)
            if i == 0:
                test_result = torch.cat((output[0].view(-1), output[:-1].view(-1).cpu()), 0)
                truth = data.view(-1)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    # len(test_result)

    # plt.plot(truth[:500], color="blue")
    # plt.plot(truth[:1000], color="blue")
    truth = scaler.inverse_transform(truth.reshape(-1, 1))
    test_result = scaler.inverse_transform(test_result.reshape(-1, 1))
    truth = truth.reshape(-1)
    test_result = test_result.reshape(-1)

    date_list = timestamp.values[:len(truth)]
    plt.figure(figsize=(10, 8))
    l1, = plt.plot(date_list, truth, color="blue")
    l2, = plt.plot(date_list, test_result, color="red")
    plt.xticks(rotation=45)

    # plt.legend('origin_sequence', 'reconstructed_sequence')
    plt.legend(handles=[l1, l2], labels=['origin_sequence', 'reconstructed_sequence'], loc='upper right')

    # plt.plot(test_result - truth, color="green")
    # wandb.init(project="my-project")
    # # wandb.log({'Origin_sequence': truth})
    # # wandb.log({'Reconstructed_sequence': test_result})
    # wandb.log({'Origin_sequence': truth, 'Reconstructed_sequence': test_result, "Abnormal_value": (test_result - truth)})

    # save loss
    print("loss shape: ", (test_result - truth).shape)
    res = pd.DataFrame({"date": timestamp.values[:len(truth)], "truth": truth, "test_result": test_result, "loss": test_result - truth})
    if os.path.exists("res") == False:
        os.mkdir("res")
    # res_csv_path = "res/test_loss_" + str(dim) + "_" + "epoch" + str(epoch) + "_" + str(lr) + ".csv"
    res_csv_path = "res/epoch%s_lr%s_inputWindow%s_optimizer_3.csv" % (str(epoch), str(lr), str(input_window))
    with open(res_csv_path, "w") as f:
        res.to_csv(res_csv_path)

    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    # plt.xticks(ticks=range(len(truth)), labels=timestamp.values[:len(truth)], rotation=90)

    if not os.path.exists("graph"):
        os.mkdir("graph")
    plt.savefig('graph/transformer-epoch%d_%s_date_%d.png' % (epoch, dim, choice))
    plt.close()

    return total_loss / i, res
