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
# from torchonn.layers import MZILinear
# from torchonn.models import ONNBaseModel


import torch
from torch import Tensor, nn
from torch.types import Device, _size
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
test_num = 0.95  # Replace with your actual accuracy calculation
logger.info(f"Current accuracy: {test_num:.2f}")  # Log as info
# logger.debug("Current accuracy: %.2f", accuracy)  # Log as info


# ## Load encodered data

# In[3]:


train_label_dir = './data/train_label.csv'
train_data_dir = './data/encodered_csv/train/'

eval_label_dir = './data/eval_label.csv'
eval_data_dir = './data/encodered_csv/eval/'

label_dic = {'normal':0, 'abnormal':1}


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

# dataset = customDataset(data_dir=data_dir, label_dir=label_dir)
train_dataset = customDataset(data_dir=train_data_dir, label_dir=train_label_dir)
eval_dataset = customDataset(data_dir=eval_data_dir, label_dir=eval_label_dir)


# In[4]:


epochs = 200
batch_size = 128
step = 0
init_lr = 1e-4

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, \
                                  shuffle=True)

eval_loader = DataLoader(dataset=eval_dataset, shuffle=True)

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


# ## Train transformer

# #### define model

# In[5]:


# encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to('cuda')
# out = encoder_layer(src)
class transformer_classifier(nn.Module):
    def __init__(self, input_size, classes):
        super(transformer_classifier, self).__init__()
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
optimizer = torch.optim.Adam(classifier.parameters(),betas=(0.9,0.9),lr=init_lr)
criterion = nn.CrossEntropyLoss()


# #### Train

# In[ ]:


# train_loader = DataLoader(dataset=dataset, batch_size=batch_size, \
#                                   shuffle=True)

# src = torch.rand(18, 128).to('cuda')
# label = torch.tensor([0]).to('cuda')
# x_hat = ae.encoder(data)

min_loss = 1
for epoch in range(epochs):
    # Training loop
    poly_lr_scheduler(optimizer, init_lr=init_lr, iter=epoch, max_iter=epochs)
    for batch_index, (data,target,_) in enumerate(train_loader, 0):
#     for batch_index, data in enumerate(train_loader, 0):
        data, target = data.to('cuda'), target.to('cuda')
        y = classifier(data)
#         logger.debug(f"y size:{y.shape}, tatget size{target.shape}")
        train_loss = criterion(y, target)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
#         logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
#     logger.info(f"Training Loss: {loss}")
    if epoch%5==0:
    # Validation loop
#         classifier.eval()  # Set the model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        accuracy = 0
        with torch.no_grad():
            for batch_index, (data,target,_) in enumerate(eval_loader, 0):
                data, target = data.to('cuda'), target.to('cuda')
                outputs = classifier(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)  # Total number of samples
                correct += (predicted == target).sum().item()  # Count correct predictions

        val_loss /= len(eval_loader)
        accuracy = 100 * correct / total
        logger.info(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
                    
    torch.save(classifier.state_dict(), './weights/transformer_params_latest.pth')
    if train_loss < min_loss:
        torch.save(classifier.state_dict(), './weights/transformer_params_best.pth')
        min_loss = train_loss


# ## Save transformer weights

# In[ ]:


torch.save(classifier.state_dict(), './weights/transformer_weights.pth')


# ### Normalize dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.reshape((len(train_data), -1)))
X_test = scaler.fit_transform(val_data.reshape((len(val_data), -1)))   
logger.debug(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
# ### Positional encoding
def positional_encoding(max_length, d_model, model_type='sinusoidal'):
    """
    Generates positional encodings for a given maximum sequence length and model dimensionality.

    Args:
        max_length (int): The maximum length of the sequence.
        d_model (int): The dimensionality of the model.
        model_type (str): The type of positional encoding to use. Defaults to 'sinusoidal'.

    Returns:
        numpy.ndarray: The positional encoding matrix of shape (max_length, d_model).
    """

    if model_type == 'sinusoidal':
        pe = np.zeros((max_length, d_model))
        position = np.arange(0, max_length, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
    else:
        raise ValueError("Unsupported model_type: {}".format(model_type))

    return pe

pe_train = positional_encoding(X_train.shape[0], X_train.shape[1])
pe_test = positional_encoding(X_test.shape[0], X_test.shape[1])
# Add positional encoding to the signal
X_train =  X_train + pe_train # Add corresponding row of pe matrix
X_test =  X_test + pe_test

# def positional_encoding(max_length, d_model, model_type='sinusoidal'):
#     for i, signal in enumerate(signal_dataset):
#         if len(signal) <= max_length:
#             # Pad shorter signals with zeros
#             signal = np.pad(signal, (0, max_length - len(signal)), mode='constant')
#         else:
#             # Truncate longer signals
#             signal = signal[:max_length]


# ### Build Optimizer and lr

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

