import torch
import pandas as pd
adapters_weights = torch.load('/home/yongxi/work/O-LoRA/exp/sdlora/order_2/outputs/3-agnews/adapter/adapter_model.bin', map_location='cpu')
# adapters_weights = torch.load('/home/yongxi/work/O-LoRA/exp/debug/order_1/outputs/1-dbpedia/adapter/adapter_model.bin', map_location='cpu')
# print(adapters_weights.keys())
for k,v in adapters_weights.items():
        print('-'*40)
        print(k,v.shape,v)
# csv = pd.read_csv('analyze/sdlora.csv')
#task=2 direction=dir_0
# value= csv[(csv['task']==2) & (csv['direction']=='dir_1')]['value']
# for i in value:
#     print(i)
# print("length:")
# print(len(value))
# print("max:",max(value))
# print("min:",min(value))
# print("mean:",sum(value)/len(value))