import torch
from thop import profile
from model import Transformer

model = Transformer.TransAm()
print(model)

# input_data = torch.randn([20, 1, 250])
# output_data = model(input_data)
# print(input_data)
# print(output_data)

flops, params = profile(model, inputs=(20, 1, 250))
print("flops: ", flops)
print("params: ", params)