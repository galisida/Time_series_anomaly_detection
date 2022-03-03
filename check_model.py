import torch
from thop import profile, clever_format

from model.Transformer import TransAm

model = TransAm()
print(model)

inputs = torch.randn([20, 1, 1])
flops, params = profile(model, inputs=(inputs,))
macs, params = clever_format([flops, params], "%.3f") # 格式化输出
print('flops', macs)  # 计算量
print('params:',params)  # 模型参数量
# 136.5 MFLOPs
# 6.58 M params

