import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable


data_csv = pd.read_csv('../res/test_data.csv')
# df_csv = data_csv.loc[data_csv['port_id'].isin(['64000000000600100000 ']),['concentration']]
date_list = data_csv['date'].values
truth = data_csv['truth'].values.astype('float32')
print('date_list:\n ', date_list)
print('truth:\n ', truth)
# df_csv = data_csv['']
#plt.plot(df_csv)
#plt.savefig(r"C:\Users\dayan\PycharmProjects\pythonProject\graph/initial.jpg")
# 数据预处理
# df_csv = df_csv.dropna()  # 滤除缺失数据
# dataset = df_csv.values   # 获得csv的值
# dataset = dataset.astype('float32')
dataset = truth
max_value = np.max(dataset)  # 获得最大值
min_value = np.min(dataset)  # 获得最小值
scalar = max_value - min_value  # 获得间隔数量
# dataset = list(map(lambda x: x / scalar, dataset))  # 归一化
dataset = (dataset - min_value) / scalar  # 归一化
#plt.plot(dataset)
#plt.savefig(r"C:\Users\dayan\PycharmProjects\pythonProject\graph/afterdeal.jpg")
#print(dataset)

def create_dataset(dataset, look_back=2): #通过前面几条的数据来预测下一条的数据，look_back设置具体的把前面几条的数据作为预测的输入data_X，而输出就是下一条data_Y
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 创建好输入输出
data_X, data_Y = create_dataset(dataset)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

#
train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

#搭建模型
class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


model = lstm(2, 4, 1, 2)

#设置交叉熵损失函数和自适应梯度下降算法
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 开始训练
import time
time_start = time.time()

for e in range(300):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 10 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item
        ()))
time_end = time.time()
print('训练时间：', time_end - time_start)

model = model.eval() # 转换成测试模式

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = model(var_data)  # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()

# 画出实际结果和预测的结果
dataset = dataset * scalar + min_value
pred_test = pred_test * scalar + min_value

plt.figure(figsize=(10, 8))
print("-----------------------------------")
print(len(truth))
print(len(pred_test))
l1, = plt.plot(date_list, truth, color="blue")
l2, = plt.plot(date_list, np.append(pred_test[:2], pred_test), color="red")
plt.legend(handles=[l1, l2], labels=['origin_sequence', 'reconstructed_sequence'], loc='upper right')

plt.xticks(rotation=45)
# plt.xticks(ticks=range(len(truth)), labels=timestamp.values[:len(truth)], rotation=90)

plt.grid(True, which='both')
plt.axhline(y=0, color='k')
# plt.plot(dataset, 'b', label='origin_sequence')
# plt.plot(pred_test, 'r', label='reconstructed_sequence')
# plt.legend(loc='best')
plt.savefig('../graph/lstm.jpg')
plt.close()



res = pd.DataFrame({"test_result": np.append(pred_test[:2], pred_test)})
csv_name = lstm

res_csv_path = './lstm.csv'
with open(res_csv_path, "w") as f:
    res.to_csv(res_csv_path)


