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
    return torch.FloatTensor(inout_seq)


# TODO(done): add input && output_window, device
def get_data(args, input_window, output_window, device='cpu', preloaded_csv=None, meta=None):
    # construct a littel toy dataset
    # time = np.arange(0, 400, 0.1)
    # amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))

    from sklearn.preprocessing import MinMaxScaler

    # loading weather data from a file
    from pandas import read_csv
    # series = read_csv('dataset/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    # series = read_csv('dataset/test.CSV')
    if (args.port_id or args.date_s or args.date_e) is None:
        series = read_csv('dataset/test2.CSV')
        print('running on test dataset!!!!!!!!!!!!!!!!!!!!!')
        print('running on test dataset!!!!!!!!!!!!!!!!!!!!!')
        print('running on test dataset!!!!!!!!!!!!!!!!!!!!!')
    else:
        # series = read_csv('dataset/all_data.csv')
        series = preloaded_csv
        series['company_id'] = series['company_id'].astype('string')
        series['port_id'] = series['port_id'].astype('string')
        series['polution_id'] = series['polution_id'].astype('string')
        series['date'] = pd.to_datetime(series['date'], format='%d/%m/%Y %H:%M:%S')
        series_port_id = series[series['port_id'].str.contains(args.port_id)]
        print('find port id.')
        series_polution_id = series_port_id[series_port_id['polution_id'].str.contains(args.polution_id)]
        print('find polution id.')
        series = series_polution_id[series_polution_id['company_id'].str.contains(args.company_id)]
        print('find company id.')
        # 1/2/2020 --> 日/月/年 -->2020年2月1日
        st = pd.to_datetime(args.date_s, format='%d/%m/%Y')
        data_s_ = str(st.day) + '/' + str(st.month) + '/' + str(st.year)
        st_ = pd.to_datetime(data_s_, format='%d/%m/%Y')

        ed = pd.to_datetime(args.date_e, format='%d/%m/%Y')
        data_e_ = str(ed.day + 2) + '/' + str(ed.month) + '/' + str(ed.year)
        ed_ = pd.to_datetime(data_e_, format='%d/%m/%Y')

        series = series.loc[(series['date'] >= st_) &
                            (series['date'] <= ed_)]
    series = series.sort_values(by='date')
    series.reset_index(drop=True, inplace=True)
    # series = read_csv('dataset/all_data.csv')
    print('dateframe:')
    print(series)

    if len(series[args.dim].unique()) == 1:
        print('all data are ', series[args.dim].unique())
        if meta == 'all':
            return None, None, None, None
    # series = series.fillna({'concentration': 0, 'amount': 0})  # after preload
    timestamp = series['date']
    dim_name = args.dim
    if dim_name is not None:
        series = series[dim_name]
        print("dim_name: ", dim_name)
    else:
        series = series['concentration']
    # series = series['amount']
    print('-----------------------------------------------')
    print("series: ", series)
    print("series.shape: ", series.shape)
    print('-----------------------------------------------')

    print(series.to_numpy().shape)
    if series.to_numpy().shape[0] == 0:
        print('no data!')
        return None, None, None, None
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
    train_sequence = train_sequence[:-output_window]

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window, output_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device), timestamp, scaler


# TODO(done): add input_window
def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]  # 这里的sql_len是指source的seq_len, 其实应该还是batch_size
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target
