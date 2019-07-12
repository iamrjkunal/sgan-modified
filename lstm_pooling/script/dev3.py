import os
import numpy as np
import pandas as pd
#import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.deterministic = True

oct_data = np.loadtxt("../dataset/lstminput.txt", delimiter='\t')
# =============================================================================
# parser = argparse.ArgumentParser()
# 
# parser.add_argument('--time_step', default=3, type=int)
# parser.add_argument('--batch_size', default=10, type=int)
# 
# args = parser.parse_args()
# unfold_timestep = args.time_step
# total_timestep = unfold_timestep + 1
# batch_size = args.batch_size
# 
# =============================================================================
#k = random.sample(range(1, 34), 1)[0]
k = 34
sensor_list = random.sample(range(0, 34), k)
sensor_list.sort()

sensor_list.append(34)
with open("sensor_list_dev3.txt", "w") as file:
    file.write(str(sensor_list))
<<<<<<< HEAD
=======
    
# Loading list
# with open("sensor_list_dev3.txt", "r") as file:
#     sensor_list = eval(file.readline())

>>>>>>> b7554263bd65057dbfd4d0339bb2b5027b906d15
unfold_timestep = 3
def getbatch():
    unfold_timestep = 3
    total_timestep = unfold_timestep + 1
    sample_t0 =random.sample(range(0, oct_data.shape[0]-total_timestep), 1)[0]
    lis =[]
    for j in range(sample_t0, sample_t0+total_timestep):
        lis.append(oct_data[j,sensor_list])    
    data_inp = np.array(lis)
    # Data format: batch, seq_len, input_size
    # LSTM input format: seq_len, batch, input_size
    data = torch.from_numpy(data_inp[:, :])
    lstm_inp = data.view(total_timestep,len(sensor_list),1)
    return lstm_inp

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



class Encoder(nn.Module):
    def __init__(self, input_dim=1, embedding_dim = 64, hid_dim = 64, pool_dim= 64, mlp_dim = 512, n_layers = 1, activation='relu', batch_norm=True, dropout = 0):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        
        self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    hid_dim=self.hid_dim,
                    mlp_dim=mlp_dim,
                    pool_dim=pool_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                ).cuda()
        self.spatial_embedding = nn.Linear(1, embedding_dim).cuda()
        
        mlp_dims = [hid_dim + pool_dim, mlp_dim, hid_dim]
        self.mlp = make_mlp( mlp_dims, activation=activation, batch_norm=batch_norm, dropout=dropout )
        
        self.hidden = torch.zeros( self.n_layers, k+1, self.hid_dim ).requires_grad_().to(self.device)
        self.cell = torch.zeros( self.n_layers, k+1, self.hid_dim ).requires_grad_().to(self.device)
        
    def forward(self, src):
        
        #src = [src sent len, batch size, input_dim]
        ts_len = src.size()[0]
    
        for t in range(0, ts_len):    
            outputs, (self.hidden, self.cell) = self.rnn(src[t,:,:].view(1,k+1,1), (self.hidden, self.cell))
            pool_h = self.pool_net(self.hidden, src[t,:,:])
            mlp_inp = torch.cat( [self.hidden.view(-1, self.hid_dim), pool_h], dim=1)
            mlp_out = self.mlp(mlp_inp)
            self.hidden = torch.unsqueeze(mlp_out, 0)
        self.hidden = self.hidden.detach() 
        self.cell = self.cell.detach()
        #outputs = [src sent len, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        #outputs are always from the top hidden layer
        return self.hidden, self.cell

class Decoder(nn.Module):
    def __init__(self, input_dim =1, embedding_dim=64,output_dim =1, hid_dim =64, n_layers =1, dropout =0):
        super(Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        self.spatial_embedding = nn.Linear(1, embedding_dim).cuda()
        self.out = nn.Linear((k+1)*hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [1, batch size, input_dim]
#         output, (hidden, cell) = self.rnn(input, (hidden, cell))
#         print(output.size(), output.squeeze(0).size())
 #       prediction = self.out(output.squeeze(0)[[-1],:])
        emb = self.spatial_embedding(input)
        prediction = self.out(torch.cat([emb.squeeze(0), hidden.squeeze(0)[[-1],:]], dim=0).view(-1))
#         print(emb.squeeze(0).size(), hidden.squeeze(0)[[-1],:].size())
#         print(torch.cat([emb.squeeze(0), hidden.squeeze(0)[[-1],:]], dim=0).view(-1).size())
#         exit()
        #prediction = [batch size, output dim]
#         return prediction, hidden, cell
        return prediction

class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, hid_dim=64, mlp_dim=512, pool_dim=64,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = mlp_dim
        self.hid_dim = hid_dim
        self.pool_dim = pool_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + hid_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, pool_dim]

        self.spatial_embedding = nn.Linear(1, embedding_dim).cuda()
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout).cuda()

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, src):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, hid_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, pool_dim)
        """
        pool_h = []
        
        num_sensors = h_states.size()[1]
        curr_hidden = h_states.view(-1, self.hid_dim).cuda()
        curr_end_pos = src.view(num_sensors, 1).cuda()
        # Repeat -> H1, H2, H1, H2
        curr_hidden_1 = curr_hidden.repeat(num_sensors, 1).cuda()
        # Repeat position -> P1, P2, P1, P2
        curr_end_pos_1 = curr_end_pos.repeat(num_sensors, 1).cuda()
        # Repeat position -> P1, P1, P2, P2
        curr_end_pos_2 = self.repeat(curr_end_pos, num_sensors).cuda()
        curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
        mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)
        curr_pool_h = curr_pool_h.view(num_sensors, num_sensors, -1).max(1)[0]
        pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0).cuda()
        return pool_h

        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, encode_inp, decode_inp):
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        
    
        hidden, cell = self.encoder(encode_inp)
        #hidden = [n layers, batch size, hid dim]
#         hidden = hidden[:,:-1,:]
#         cell = cell[:,:-1,:]
#         output, hidden, cell = self.decoder(decode_inp, hidden, cell)
        output = self.decoder(decode_inp, hidden, cell)
        return output


enc = Encoder()
dec = Decoder()
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)



optimizer = optim.Adam(model.parameters())

criterion = nn.MSELoss(size_average=False)

# class RMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss(size_average=False)
        
#     def forward(self,yhat,y):
#         return torch.sqrt(self.mse(yhat,y))
# criterion = RMSELoss()

torch.backends.cudnn.benchmark = True

def train(model,train_num_batches, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i in range(0,train_num_batches):
#         print(i)
        batch_iter = getbatch()
        if torch.cuda.is_available():
            batch_iter = batch_iter.cuda()
        encode_inp = batch_iter[:-1,:,:]
        decode_inp = batch_iter[[-1],:-1,:]
        real_output = batch_iter[[-1],[-1],:].view(1)
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            encode_inp = encode_inp.cuda()
            decode_inp = decode_inp.cuda()
            real_output = real_output.cuda()
        
        predicted_output = model(encode_inp.float(), decode_inp.float())
#         print(predicted_output.size(), real_output.size())
#         exit()
        loss = criterion(predicted_output.float(), real_output.float())
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / train_num_batches

def evaluate(model, valid_num_batches, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i in range(0,valid_num_batches):
#             print(i)
            batch_iter = getbatch()
            if torch.cuda.is_available():
                batch_iter = batch_iter.cuda()
            encode_inp = batch_iter[:-1,:,:]
            decode_inp = batch_iter[[-1],:-1,:]
            real_output = batch_iter[[-1],[-1],:].view(1)
            if torch.cuda.is_available():
                encode_inp = encode_inp.cuda()
                decode_inp = decode_inp.cuda()
                real_output = real_output.cuda()
        
            predicted_output = model(encode_inp.float(), decode_inp.float())
            loss = criterion(predicted_output.float(), real_output.float())
            epoch_loss += loss.item()
        
    return epoch_loss / valid_num_batches


N_EPOCHS = 10000
CLIP = 1

best_valid_loss = float('inf')

train_num_batches = 1000
valid_num_batches = 200
test_num_batches = 400

for epoch in range(N_EPOCHS):  
      
    train_loss = train(model, train_num_batches, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_num_batches, criterion)    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkout = 'sensor_lstm_' + str(unfold_timestep) + 'tspool3.pt'
        torch.save(model.state_dict(), checkout)
    print("####Epoch:", epoch+1)
    print("Train Loss :", train_loss)
    print("Val Loss:", valid_loss)

model.load_state_dict(torch.load(checkout))
best_test_loss = float('inf')
for i in range(5):
    test_loss = evaluate(model, test_num_batches, criterion)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
print("Test Loss:", best_test_loss)
