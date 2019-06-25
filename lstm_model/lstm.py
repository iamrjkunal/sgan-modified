import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd import Variable

dataset = np.loadtxt("lstminput.txt", delimiter='\t')
timestep = dataset.shape[0]
train = Variable(torch.from_numpy(dataset[:, :]))
# trainfinal = Variable(torch.from_numpy(dataset[timestep: timestep +1, :-1]))
# testfinal_gt = Variable(torch.from_numpy(dataset[timestep: timestep +1, [-1]]))




def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)



class LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, batch_size =1, output_dim=1, num_layers=1):
        
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        mlp_dims = [hidden_dim + 34,512, 1]
        self.mlp = make_mlp(mlp_dims)

    def init_hidden(self):
        
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))

    def forward(self, input1, input2):
        
        lstm_out, hidden = self.lstm(input1)
#         print(hidden[0].view(-1,self.hidden_dim,).size(), input2.view(-1,34).size())
#         exit()

        mlp_inp = torch.cat( [ hidden[0].view(-1,self.hidden_dim), input2.view(-1,34)], dim=1)
        mlp_inp = torch.cat([mlp_inp, mlp_inp], dim =0)
#         print(mlp_inp.size())
#         exit()
        y_pred = self.mlp(mlp_inp)
        return y_pred

model = LSTM(input_dim=35,hidden_dim=64, batch_size=1,output_dim=1, num_layers=1)

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr= 0.1)

# num_epochs = 100
# block_size = 6
# num_samples = int(timestep/6)
# samples = [train[x:x+block_size] for x in np.random.randint(timestep ,size=num_samples)]

# for t in range(num_epochs):
#     for sample in samples:
#         inp1 = sample[:-1, :].view(1,5,35)
#         test = sample[[-1],:].view(35,1)
#         inp2 = test[:-1,:]
#         model.hidden = model.init_hidden()
#         optimiser.zero_grad()
#         y_pred = model(inp1.float(), inp2.float())
#         loss = loss_fn(y_pred, test[34].data)
#         if t % 10 == 0:
#             print("Epoch ", t, "MSE: ", loss.item())
#         loss.backward()
#         optimiser.step()
        
num_epochs = 2


for i in train:
#         print(i)
#         print(i+1)
#         exit()
    inp1 = i.view(1,1,35)
    test = (i+1).view(35,1)
    inp2 = test[:-1,:]
    model.hidden = model.init_hidden()
    y_pred = model(inp1.float(), inp2.float())
    test2 = torch.cat([test[34], test[34]], dim =0).view(2,1)
    loss = loss_fn(y_pred, test2.float())
    print("MSE: ", loss.item())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()