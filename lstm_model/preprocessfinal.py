import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.deterministic = True

def getbatch():
    oct_data = np.loadtxt("lstminput.txt", delimiter='\t')
    unfold_timestep = 5
    total_timestep = unfold_timestep + 1
    batch_size = 25
    
    batch_sample =random.sample(range(0, oct_data.shape[0]-total_timestep), batch_size)
    list_top =[]
    
    for i in batch_sample:
        lis =[]
        for j in range(i, i+total_timestep):
            lis.append(oct_data[j,:])    
        list_top.append(lis)
    data_inp = np.array(list_top)
    # Data format: batch, seq_len, input_size
    # LSTM input format: seq_len, batch, input_size
    data = torch.from_numpy(data_inp[:, :, :])
    lstm_inp = data.permute(1, 0, 2)
    return lstm_inp
