#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import argparse
from typing import List, Union
import matplotlib.pyplot as plt

# import torchonn as onn
# from torchonn.models import ONNBaseModel
# from torchonn.op.mzi_op import project_matrix_to_unitary


import torch
from torch import Tensor, nn
from torch.types import Device, _size
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
# from torchonn.layers import MZILinear
# from torchonn.models import ONNBaseModel
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

from configs.config import configs


# ### Initilization

# In[ ]:


# Init logging
import logging

logger = logging.getLogger(__name__)  # Use the current module's name
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
logger.addHandler(handler)
acc_example = 0.95  # Replace with your actual accuracy calculation
logger.info(f"Current accuracy: %{acc_example}")  # Log as info
# logger.debug("Current accuracy: %.2f", accuracy)  # Log as info


# In[ ]:


# parser = argparse.ArgumentParser()
# parser.add_argument("config", metavar="FILE", help="config file")
# args = parser.parse_args(args=['configs/eeg_pt.yml'])
# args.config

parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")
# parser.add_argument('--run-dir', metavar='DIR', help='run directory')
# parser.add_argument('--pdb', action='store_true', help='pdb')
args = parser.parse_args(args=['configs/eeg_pt.yml'])
args, opts = parser.parse_known_args()

configs.load(args.config, recursive=True)
configs.update(opts)

# if torch.cuda.is_available() and int(configs.run.use_cuda):
#     torch.cuda.set_device(configs.run.gpu_id)
#     device = torch.device("cuda:" + str(configs.run.gpu_id))
#     torch.backends.cudnn.benchmark = True
# else:
#     device = torch.device("cpu")
#     torch.backends.cudnn.benchmark = False

# if int(configs.run.deterministic) == True:
#     set_torch_deterministic()


# In[26]:


import yaml

parser = argparse.ArgumentParser()
parser.add_argument("config_file", metavar="FILE", help="config file")
# parser.add_argument('--run-dir', metavar='DIR', help='run directory')
# parser.add_argument('--pdb', action='store_true', help='pdb')
args = parser.parse_args(args=['configs/eeg_torch.yml'])
# args, opts = parser.parse_known_args()
# f = 'configs/eeg_pt.yml'
with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)


# In[27]:


config


# ### Dataset

# In[ ]:


class customDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
#         self.annotations = pd.read_csv(label_dir)
        self.data_dir = data_dir   # './data/origin_csv/train'
        self.transform = transform
        self.files = os.listdir(self.data_dir)
        self.annotations = pd.read_csv(label_dir)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.files[index])
        data = pd.read_csv(data_path)
        data = torch.tensor(data.values, dtype=torch.float32)
        file_name = self.files[index]
        
        label = torch.tensor(int(label_dic[self.annotations.iloc[index,1]]))
        
        if self.transform:
            data = self.transform(data)
            
        return (data.t(), label, file_name)


# In[ ]:


train_label_dir = './data/train_label.csv'
train_data_dir = './data/origin_csv/train/'

eval_label_dir = './data/eval_label.csv'
eval_data_dir = './data/origin_csv/eval/'

label_dic = {'normal':0, 'abnormal':1}

    
# transform = transforms.Compose([
#     transforms.MinMaxScaler(feature_range=(0, 1)),
#     transforms.ToTensor(),
# ])

train_dataset = customDataset(data_dir=train_data_dir, label_dir=train_label_dir)
eval_dataset = customDataset(data_dir=eval_data_dir, label_dir=eval_label_dir)
# combined_dataset = ConcatDataset([train_dataset, eval_dataset])


# ### Define model

# In[ ]:


### Define operation in auto-encoder
class Mat_mul(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
#         self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weight + self.bias
#         return torch.mul(input, self.weight, self.bias)
    
### Define auto-encoder    
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Mat_mul(input_size, hidden_size),
            nn.ReLU()
        )
        self.encoder_2 = nn.Sequential(
            Mat_mul(int(input_size/2), hidden_size),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            Mat_mul(hidden_size, input_size),
            nn.ReLU()
        )
        self.decoder_2 = nn.Sequential(
            Mat_mul(int(input_size/2), input_size),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
#         z = self.encoder_2(z)
        x_hat = self.decoder(z)
#         x_hat = self.decoder_2(x_hat)
        return x_hat


### Define transformer_classifier
class transformer_classifier(nn.Module):
    def __init__(self, input_size, classes):
        super(transformer_classifier, self).__init__()
        self.au = AutoEncoder(input_size=1000, hidden_size=256)  
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, classes)

    def forward(self, x):
        z = self.transformer_encoder(x)
        z = self.flatten(z)
        y = self.linear(z)
        return y
    
classifier = transformer_classifier(256*19, 2).to('cuda')

