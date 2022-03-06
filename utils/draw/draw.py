import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker
import pandas as pd
import numpy as np
import re
import os


csv_list = os.listdir('./res')
csv_list.sort()
print(csv_list)


date_list = pd.read_csv('./res/' + csv_list[0])['date'].values.tolist()
truth = pd.read_csv('./res/' + csv_list[0])['truth'].values.tolist()
print(date_list)
print(len(date_list))
print(truth)
print(len(truth))


print("----------------------------------------------------")
lstm_df = pd.read_csv('../../model/lstm.csv')
lstm_test = lstm_df['test_result'].values.tolist()
print(lstm_test)
print(len(lstm_test))



for csv_name in csv_list:
    temp = re.findall(r'\d+', csv_list[0])
    params = []
    param_lr = temp[1] + "." + temp[2]
    params.append(temp[0])
    params.append(param_lr)
    params.append(temp[3])
    # print(params)  # epoch  lr  input_window

    df = pd.read_csv('./res/' + csv_name)
    # print(df)
    test_result = df['test_result'].values.tolist()
    # print(test_result)
    print(len(test_result))





c = (
    Line()
    .add_xaxis(date_list)
    .add_yaxis("origin_sequence", truth,
               is_smooth=False,
               is_symbol_show=False,
               linestyle_opts=opts.LineStyleOpts(width=2),
               color='blue'
            )
    # lstm_result
    # .add_yaxis("LSTM_reconstructed_sequence", lstm_test
    #            , is_smooth=False,
    #            is_symbol_show=False,
    #            color='green'
    # )

    .set_global_opts()

)

optimizer_list = ['AdamW', 'SGD', 'ASGD', 'Adadelta']
index = 0
for i in range(len(csv_list[:-1])):
    temp = re.findall(r'\d+', csv_list[i])
    print(temp)
    param_lr = temp[1] + "." + temp[2]
    param_epoch = temp[0]
    param_input_window = temp[3]
    df = pd.read_csv('./res/' + csv_list[i])
    test_result = df['test_result'].values.tolist()


    # if param_epoch != "30" or param_lr != "0.0005":
    # if param_epoch != "50" or param_input_window != "2" or param_lr != "0.001" or temp[4] != '0':
    if len(temp) != 5:
    # if param_input_window != '5' or param_lr != "0.0005":
        continue
    c.add_yaxis(
        # "ep: " + param_epoch + " lr: " + param_lr + " iw: " + param_input_window,
        # " input_window: " + param_input_window,
        # "epoch: " + param_epoch,
        # "lr: " + param_lr,
        optimizer_list[index ],
        test_result,
        is_smooth=False,
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=2),

    )
    index += 1


c.set_global_opts(
    xaxis_opts=opts.AxisOpts(
        type_="category",
        boundary_gap=False,
        axistick_opts=opts.AxisTickOpts(is_show=False),
        axisline_opts=opts.AxisLineOpts(is_on_zero=False),
    ),
    yaxis_opts=opts.AxisOpts(
        type_="value",
        axistick_opts=opts.AxisTickOpts(is_show=True),
        axisline_opts=opts.AxisLineOpts(is_on_zero=True),
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
)

c.render("./test_result.html")

    # .add_yaxis(
    #     "origin_sequence",
    #     truth,
    #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
    # )
    # .add_yaxis(
    #     "商家B",
    #     Faker.values(),
    #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
    # )
    # .set_global_opts(title_opts=opts.TitleOpts(title="Line-MarkLine"))
    # .render("line_markline.html")

