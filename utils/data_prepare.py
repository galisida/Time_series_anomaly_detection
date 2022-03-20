import torch
import torch.nn as nn
import numpy as np
import pandas as pd


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
    return torch.FloatTensor(np.array(inout_seq))


# TODO(done): add input && output_window, device
def get_data(args, input_window, output_window, device='cpu'):
    # construct a littel toy dataset
    # time = np.arange(0, 400, 0.1)
    # amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))

    from sklearn.preprocessing import MinMaxScaler

    # loading weather data from a file
    from pandas import read_csv
    # series = read_csv('dataset/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    # series = read_csv('dataset/test.CSV')
    # series = read_csv('dataset/cpu4.csv')
    series = read_csv('dataset/mammography_label.csv', header=None)
    # series = read_csv('dataset/all_data.csv')
    print('df.head():\n', series.head())
    # timestamp = series['timestamp']
    dim_name = args.dim
    if dim_name is not None:
        series = series[dim_name]
        print("dim_name: ", dim_name)
    # labels = series[]
    # series = series['value']
    print('series.shape:', series.shape)
    # print(series.iloc[:, -1].head())
    labels = series.iloc[:, -1]
    print(labels.head())
    print()
    series = series.iloc[:, :-1]
    print('-----------------------------------------------')
    print("series_value: ", series[:10])
    print("series.shape: ", series.shape)
    print('-----------------------------------------------')


    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if len(series.shape) == 1:
        amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    else:
        amplitude = scaler.fit_transform(series.to_numpy())
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # sampels = 2600
    # train_data = amplitude[:sampels]
    # test_data = amplitude[sampels:]
    train_data = test_data = amplitude

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window, output_window)
    test_data = test_data[:-output_window]

    # return train_sequence.to(device), test_data.to(device), timestamp, scaler
    return train_sequence.to(device), test_data.to(device), scaler, labels


# TODO(done): add input_window
def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]  # 这里的sql_len是指source的seq_len, 其实应该还是batch_size
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    # print('-----------------------------------------------')
    # print('data.shape: ', data.shape)
    # print('input.shape: ', input.shape)
    # print('-----------------------------------------------')
    if len(input.shape) == 4:
        input = input.squeeze(2)
        target = target.squeeze(2)
    return input, target
