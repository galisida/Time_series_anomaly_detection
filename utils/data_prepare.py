import torch
import torch.nn as nn
import numpy as np


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
# TODO: add output_window
def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


# TODO(done): add input && output_window, device
def get_data(input_window, output_window, device='cpu'):
    # construct a littel toy dataset
    time = np.arange(0, 400, 0.1)
    amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))

    from sklearn.preprocessing import MinMaxScaler

    # loading weather data from a file
    from pandas import read_csv
    # series = read_csv('dataset/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    # series = read_csv('dataset/test.CSV')
    series = read_csv('dataset/test2.CSV')
    # series = read_csv('dataset/all_data.csv')
    series = series.fillna(0)
    series = series['concentration']
    print("series: ", series)
    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # sampels = 2600
    # train_data = amplitude[:sampels]
    # test_data = amplitude[sampels:]
    train_data = test_data = amplitude

    # convert our train data into a pytorch train tensor
    train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[
                     :-output_window]  # todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack..

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window, output_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    return train_sequence.to(device), test_data.to(device)


# TODO(done): add input_window
def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]  # 这里的sql_len是指source的seq_len, 其实应该还是batch_size
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target
