import os
# Fix OpenCL ICD override bug in Conda environment
for var in ['OCL_ICD_VENDORS', 'OCL_ICD_VENDORS_RESET', 'OCL_ICD_FILENAMES_RESET']:
    if var in os.environ:
        del os.environ[var]

import numpy as np
import json, pickle
from time import time
import torch

import sys; sys.path.append('../')
from graphcast import ncutil, forcefield, graphcast

# 1. 加载权重与配置
model_root = '../weights/lite_6mesh_37level_0.25deg'
graph_body = dict(np.load(model_root + '/GraphBody.npz'))
norm_vector = dict(np.load(model_root + '/NormVector.npz'))

with open(model_root + '/DataConfig.json') as f:
    data_config = json.loads(f.read())

# 2. 提取需要的各种变量 Key 列表与层级
const_keys = list(data_config['const_vars'].keys())
dynamic_keys = list(data_config['dynamic_vars'].keys())
force_keys = list(data_config['force_vars'].keys())
all_keys = const_keys + dynamic_keys + force_keys

levels = data_config['levels']

# 3. 时间步解析
ds_data = ncutil.nc.Dataset('../weights/testdata_and_normvector/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc')
t0, t1 = ncutil.get_times(ds_data)[:2]
ts = t1 + (np.arange(4) + 1) * (t1 - t0) # next 4 step

# 4. 纯净的数据提取
print("Loading data...")
const = ncutil.get_data(ds_data, t0, keys=const_keys, levels=levels)
dynamic_t1 = ncutil.get_data(ds_data, t0, keys=dynamic_keys, levels=levels)
dynamic_t2 = ncutil.get_data(ds_data, t1, keys=dynamic_keys, levels=levels)

# 强迫场保持原逻辑
force_t1 = forcefield.force_5cn(t0, data_config['lats'], data_config['lons'])
force_t2 = forcefield.force_5cn(t1, data_config['lats'], data_config['lons'])

# 5. 初始化模型与加载权重
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.xpu.is_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

print("Initializing model...")
net = graphcast.GraphCastModel(graph_body, norm_vector, layers=16) # 16 layers default
net.load_state_dict(torch.load(model_root + '/GraphWeight.pth'))
net.to(device=device, dtype=torch.bfloat16)
net.eval()

# 6. 自回归推理循环
print("Starting inference...")
start = time()
for step, t in enumerate(ts):
    with torch.no_grad():
        print(f"start step {step + 1} ")
        force_next = forcefield.force_5cn(t, data_config['lats'], data_config['lons'])
        x_input = const, dynamic_t1, dynamic_t2, force_t1, force_t2, force_next
        tensor_input = [torch.tensor(i, dtype=torch.bfloat16, device=device) for i in x_input]
        dynamic_next = net.forward(*tensor_input)
        force_t1, force_t2 = force_t2, force_next
        dynamic_t1, dynamic_t2 = dynamic_t2, dynamic_next.to(torch.float32).cpu().numpy()
        
        # Free GPU memory immediately
        del tensor_input, x_input, dynamic_next
        if device.type == 'xpu':
            torch.xpu.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

print('cost:', time()-start)
print(f"Prediction Mean: {dynamic_t2.mean()} | Min: {dynamic_t2.min()} | Max: {dynamic_t2.max()}")
