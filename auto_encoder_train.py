#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
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


# ### Initilization

# In[2]:


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


# ## Auto-Encoder

# #### Load raw data
data_dir = './data/origin_csv/train'
files = os.listdir(data_dir)
logger.debug("Num of files: %", len(files))
label_dir = './data/label.csv'
annotations = pd.read_csv(label_dir)
logger.debug("Num of files: %", annotations.shape)
# In[3]:


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


# In[4]:


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
combined_dataset = ConcatDataset([train_dataset, eval_dataset])


# #### Define auto-encoder model

# In[5]:


# define

class Mat_mul(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
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


# x = torch.randn(100, 10)

# define ae model
ae = AutoEncoder(input_size=1000, hidden_size=256).to('cuda')


# use ae encoder
# z = ae.encoder(x)

# print(z.shape)


# In[6]:


init_lr = 1e-4
epochs = 100
batch_size = 4096
step = 0

dataloader = DataLoader(dataset=combined_dataset, batch_size=batch_size, \
                                  shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(),betas=(0.9,0.9),lr=init_lr)

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=0, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if max_iter == 0:
        raise Exception("MAX ITERATION CANNOT BE ZERO!")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    logger.debug(f'lr=: {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ## Train auto-encoder

# In[7]:


model_params = {
    'model_state_dict': ae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'learning_rate': optimizer.param_groups[0]['lr']
}
min_loss = 0.5

for epoch in range(epochs):
    
    poly_lr_scheduler(optimizer, init_lr=init_lr, iter=epoch, max_iter=epochs)
    for batch_index, (data,_,_) in enumerate(dataloader, 0):
        data = data.to('cuda')
#         data = data.to('cuda')
        x_hat = ae(data)
#         logger.debug("x_hat, shape=%", x_hat.shape)
        loss = criterion(x_hat, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info(f'epoch={epoch}, loss={loss}')
    torch.save(model_params, './weights/model_params_latest.pth')
    if min_loss > loss:
        torch.save(model_params, './weights/model_params_best.pth')
        min_loss = loss


# ## Save auto-encoder
torch.save(ae.state_dict(), './weights/ae_model_weights.pth')
# ## Load auto-encoder

# In[ ]:





# ## auto-encoder inference for transformer

# In[ ]:


def ae_infer(data_path:str, result_path:str, label_dir:str):
    
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    enc_dataset = customDataset(data_dir=str(data_path), label_dir=label_dir)
    # Define column names (optional, but recommended)
    channels = ['Fp1', 'Fp2', 'F3','F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
                'T5', 'T6', 'Fz', 'Cz', 'Pz']
    ae_inference  = AutoEncoder(input_size=1000, hidden_size=256).to('cuda')
    ae_inference.load_state_dict(torch.load('./weights/ae_model_weights.pth'))
    ae_inference.eval()
    
    for batch_index, (data,label,file_name) in enumerate(enc_dataset, 0):
        data = data.to('cuda')
        z = ae_inference.encoder(data)   # 19*500
    #     z = ae_inference.encoder_2(z)   # 19*256
    #     logger.debug(file_name)  # Log as info
    #     logger.debug(z.shape)  # Log as info
        z = z.t().cpu()
        z = z.detach().numpy()
        
        df = pd.DataFrame(z, columns=channels)

        # Save as CSV file
        df.to_csv(result_path+file_name, index=False) 

#  training data encoding
train_label_dir = './data/train_label.csv'
train_data_dir = './data/origin_csv/train/'
train_result_dir = './data/encodered_csv/train/'
ae_infer(train_data_dir, train_result_dir, train_label_dir)

#  evaluation data encoding
eval_label_dir = './data/eval_label.csv'
eval_data_dir = './data/origin_csv/eval/'
eval_result_dir = './data/encodered_csv/eval/'
ae_infer(eval_data_dir, eval_result_dir, eval_label_dir)

# Load data first
label_dir = './data/label.csv'
data_dir = './data/origin_csv/train'
result_dir = './data/encodered_csv/train/'
label_dic = {'normal':0, 'abnormal':1}

if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)

enc_dataset = customDataset(data_dir=data_dir, label_dir=label_dir)

for batch_index, (data,label,file_name) in enumerate(enc_dataset, 0):
    data = data.to('cuda')
    z = ae_inference.encoder(data)   # 19*500
#     z = ae_inference.encoder_2(z)   # 19*256
#     logger.debug(file_name)  # Log as info
#     logger.debug(z.shape)  # Log as info
    z = z.t().cpu()
    z = z.detach().numpy()

# Define column names (optional, but recommended)
    channels = ['Fp1', 'Fp2', 'F3','F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
                'T5', 'T6', 'Fz', 'Cz', 'Pz']

    # Create DataFrame
    df = pd.DataFrame(z, columns=channels)

    # Save as CSV file
    df.to_csv(result_dir+file_name, index=False) 

    
    # convert tensor to encodered .csv
    
    
# #### Train transformer
# signal dataset in pytorch
label_dir = './data/label.csv'
data_dir = './data/csv/'


class customDataset(Dataset):
    def __init__(self, data_dir, transform=None):
#         self.annotations = pd.read_csv(label_dir)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.data_dir))
    
    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.annotations.iloc[index,1])
        data = pd.read_csv(data_path)
#         y_label = torch.tensor(int(self.annotations.iloc[index,2]))
        
        if self.transform:
            data = self.transform(data)
            
        return data

dataset = customDataset(data_dir=data_dir)

# train_set, test_set = torch.utils.data.random_split(dataset, [1067, 22000])

# # For dataset
# train_user = ['00007', '00008', '00009', '00010', '00014', '00015', '00016', \
#               '00017', '00018', '00019', '00020', '00021', \
#               '00022', '00027', '00028', '00030', '00031', '00032', '00033', '00036', \
#               '00037', '00038', '00039', '00040']
# val_user = ['00024', '00025', '00026', '00034', '00035']
# classes = {'dislike':0, 'like':1}

# data_path = Path('../Neuromarketing/raw_preference')  # need to modify
# # label_path = Path('./labels')
# result_dir_path = '../Neuromarketing/result/'

# if os.path.exists(result_dir_path):
#     shutil.rmtree(result_dir_path)
# os.mkdir(result_dir_path)

# # Load data and labels
# train_data_files = []
# val_data_files = []
# for file in os.listdir(data_path):
#     for user in train_user:
#         if file.startswith(user):
#             train_data_files.append(file)
#     for user in val_user:
#         if file.startswith(user):
#             val_data_files.append(file)


# # Preprocessing of training dataset

# # train_data_files = os.listdir(data_path)
# bands = {'raw': 701, 'zero':0, 'delta':1, 'theta':2, 'alpha':3, 'beta':4, 'gamma':5}
# # bands_len = [0, 38, 76, 146, 279, 538]
# band = 'raw'

# if band == 'raw':
#     train_data = np.zeros((len(train_data_files), bands['raw'], 4), dtype='float16')
# else:
#     train_data = np.zeros((len(train_data_files),14,bands_len[bands[band]]-(bands_len[bands[band]-1])), dtype='float16')


# # We have column features saved as csv files but sklearn needs row features.
# train_labels = np.zeros(len(train_data_files))
# # print(data.shape)
# for i,file in enumerate(train_data_files):
# #     print(i)
#     if band == 'raw':
#         df = pd.read_csv(data_path / file)
# #         print(df.loc[:, ["O1", "O2","F3", "F4"]])
#         df = pd.DataFrame(df.loc[:, ["O1","O2","F3","F4"]]).to_numpy()
#         train_data[i,:] = df
# #         print(train_data)
#     else:
#     # load data with specific band
#         train_data[i,:] = pd.read_csv(data_path / file, header=None)[bands_len[bands[band]-1]:bands_len[bands[band]]].T
    
# #     train_data[i,:] = preprocessing.normalize(train_data[i,:], norm='l2')
#     pre_label = file.split('_')[1]
#     train_labels[i] = classes[pre_label]
# #     data = pd.read_csv(data_path / file, header=None)

# logger.debug(f'train_data shape: {train_data.shape}, train_label shape: {train_labels.shape}')


# # Preprocessing for test dataset
# if band == 'raw':
#     val_data = np.zeros((len(val_data_files), bands['raw'], 4), dtype='float16')
# else:
#     val_data = np.zeros((len(train_data_files),14,bands_len[bands[band]]-(bands_len[bands[band]-1])), dtype='float16')
# val_labels = np.zeros(len(val_data_files))
# for i,file in enumerate(val_data_files):
# #     print(i)
#     if band == 'raw':
#     # load data with fixed length
#         df = pd.read_csv(data_path / file)
#         df = pd.DataFrame(df.loc[:, ["O1", "O2", "F3", "F4"]]).to_numpy()
#         val_data[i,:] = df
#     else:
#     # load data with specific band
#         val_data[i,:] = pd.read_csv(data_path / file, header=None)[bands_len[bands[band]-1]:bands_len[bands[band]]].T

    
# #     val_data[i,:] = preprocessing.normalize(val_data[i,:], norm='l2')
#     pre_label = file.split('_')[1]
#     val_labels[i] = classes[pre_label]
# #     data = pd.read_csv(data_path / file, header=None)
# logger.debug(f'val_data shape: {val_data.shape}, val_label shape: {val_labels.shape}')

# import pickle
# import joblib

# from sklearn import svm

# svm_classifier = svm.SVC(kernel='rbf', C=1.0, tol=1e-6)
# print(train_data.shape)
# train_data = train_data.reshape((1,-1))
# print(train_data.shape)
# svm_classifier.fit(train_data, train_labels)
# ### Normalize dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.reshape((len(train_data), -1)))
X_test = scaler.fit_transform(val_data.reshape((len(val_data), -1)))   
logger.debug(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
# ### Positional encoding

# ### raw data to tensor

# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train_1 = X_train[:, 0:X_train.shape[1]//2]
X_train_2 = X_train[:, X_train.shape[1]//2:]


X_test_1 = X_test[:, 0:X_test.shape[1]//2]
X_test_2 = X_test[:, X_test.shape[1]//2:]

y_train = train_labels
y_test = val_labels

X_train_1 = torch.FloatTensor(X_train_1).to('cuda')
X_train_2 = torch.FloatTensor(X_train_2).to('cuda')
X_test_1 = torch.FloatTensor(X_test_1).to('cuda')
X_test_2 = torch.FloatTensor(X_test_2).to('cuda')

y_train = torch.LongTensor(y_train).to('cuda')
y_test = torch.LongTensor(y_test).to('cuda')

input_dim = X_train_1.shape[1]
output_dim = 2
logger.debug(f'X_train_1.shape: {X_train_1.shape}, X_train_2.shape: {X_train_2.shape}, y_train.shape: {y_train.shape},X_test_1.shape: {X_test_1.shape}, X_test_2.shape: {X_test_2.shape}, y_test.shape: {y_test.shape}')


# ### Build Optimizer and lr
init_lr = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),betas=(0.9,0.9),lr=init_lr)

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      
                      max_iter=0, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if max_iter == 0:
        raise Exception("MAX ITERATION CANNOT BE ZERO!")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    logger.debug(f'lr=: {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# ### define  training
def train_network(model,optimizer,criterion,
                  X_train_1,X_train_2,y_train,
                  X_test_1,X_test_2,y_test,
                  num_epochs,train_losses,test_losses):
    for epoch in range(num_epochs):
        
        # update lr_rate
#         poly_lr_scheduler(optimizer, init_lr=init_lr, iter=epoch, max_iter=num_epochs)
        
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        
        #forward feed
        output_train = model(X_train_1, X_train_2)

        #calculate the loss
        loss_train = criterion(output_train, y_train)
        


        #backward propagation: calculate gradients
        loss_train.backward()

        #update the weights
        optimizer.step()

        
        output_test = model(X_test_1, X_test_2)
        loss_test = criterion(output_test,y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")
# ### train

# In[ ]:


num_epochs = 500
train_losses = np.zeros(num_epochs)
test_losses  = np.zeros(num_epochs)

train_network(model,optimizer,criterion,
              X_train_1,X_train_2,y_train,
              X_test_1,X_test_2,y_test,
              num_epochs,train_losses,test_losses)


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()


# In[ ]:


predictions_train = []
predictions_test =  []
with torch.no_grad():
    predictions_train = model(X_train_1, X_train_2)
    predictions_test = model(X_test_1, X_test_2)


# In[ ]:


def get_accuracy_multiclass(pred_arr,original_arr):
    if len(pred_arr)!=len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred= []
    # we will get something like this in the pred_arr [32.1680,12.9350,-58.4877]
    # so will be taking the index of that argument which has the highest value here 32.1680 which corresponds to 0th index
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
    #here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)


# In[ ]:


train_acc = get_accuracy_multiclass(predictions_train.cpu(),y_train.cpu())
test_acc  = get_accuracy_multiclass(predictions_test.cpu(),y_test.cpu())


# In[ ]:


logger.info(f"Training Accuracy: {round(train_acc*100,3)}")
logger.info(f"Test Accuracy: {round(test_acc*100,3)}")


# ### Evolutionary algorithm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the task (binary classification)
def generate_data(num_samples=100):
    # Generate random data for a binary classification task
    X = torch.randn(num_samples, 2)
    y = (X[:, 0] * X[:, 1] > 0).float()  # Simple XOR-like classification
    return X, y

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Define the evolutionary algorithm
def evolutionary_algorithm(population_size, generations, mutation_rate):
    # Initialize the population with random neural networks
    population = [SimpleNN() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate the fitness of each individual in the population
        fitness_scores = []
        for individual in population:
            X, y = generate_data()
            predictions = individual(X)
            loss = nn.BCELoss()(predictions, y.view(-1, 1))
            fitness_scores.append(-loss.item())  # Negative because we want to maximize fitness

        # Select the top-performing individuals as parents
        selected_parents = np.argsort(fitness_scores)[-int(0.2 * population_size):]

        # Create offspring through crossover and mutation
        offspring = []
        for _ in range(population_size - len(selected_parents)):
            parent1 = population[np.random.choice(selected_parents)]
            parent2 = population[np.random.choice(selected_parents)]
            child = SimpleNN()

            # Crossover (swap parameters)
            for child_param, parent1_param, parent2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                crossover_mask = (torch.rand_like(child_param.data) > 0.5).float()
                child_param.data = crossover_mask * parent1_param.data + (1 - crossover_mask) * parent2_param.data

            # Mutation (randomly perturb parameters)
            for child_param in child.parameters():
                mutation_mask = (torch.rand_like(child_param.data) < mutation_rate).float()
                mutation = torch.randn_like(child_param.data) * mutation_mask
                child_param.data += mutation

            offspring.append(child)

        # Replace the old population with the new population
        population = selected_parents + offspring

    # Return the best-performing individual
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual

# Example usage
best_model = evolutionary_algorithm(population_size=50, generations=50, mutation_rate=0.1)

# Now you can use the best_model for further processing or inference
