import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=2, num_layers=1):
        super(LSTM, self).__init__()
        self.model_type = 'lstm'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # print('x.shape:', x.shape)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# model = LSTM()
# data = torch.randn(100, 20, 1)  # [seq_len, batch_size, input_size]
# print(data.shape)
# output = model(data)
# print(output.shape)
