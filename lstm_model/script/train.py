import os
import numpy as np
import pandas as pd
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.deterministic = True

oct_data = np.loadtxt("../dataset/lstminput.txt", delimiter='\t')
parser = argparse.ArgumentParser()

parser.add_argument('--time_step', default=3, type=int)
parser.add_argument('--batch_size', default=10, type=int)

args = parser.parse_args()
unfold_timestep = args.time_step
total_timestep = unfold_timestep + 1
batch_size = args.batch_size

def getbatch():
    
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Encoder(nn.Module):
    def __init__(self, input_dim=35, hid_dim = 128 , n_layers =1, dropout = 0):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size, input_dim]
        
        outputs, (hidden, cell) = self.rnn(src)
        
        #outputs = [src sent len, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        #outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim =34, output_dim =1, hid_dim =128, n_layers =1, dropout =0):
        super(Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        
        self.out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [1, batch size, input_dim]
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        #prediction = [batch size, output dim]
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, encode_inp, decode_inp):
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(encode_inp)
        
        output, hidden, cell = self.decoder(decode_inp, hidden, cell)
        return output


enc = Encoder(n_layers =1)
dec = Decoder(n_layers =1)
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)



optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss(size_average=False)


def train(model,train_num_batches, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i in range(0,train_num_batches):
        batch_iter = getbatch()
        if torch.cuda.is_available():
            batch_iter = batch_iter.cuda()
        encode_inp = batch_iter[:-1,:,:]
        decode_inp = batch_iter[[-1],:,:-1]
        real_output = batch_iter[[-1],:,[-1]].view(-1,1)
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            encode_inp = encode_inp.cuda()
            decode_inp = decode_inp.cuda()
            real_output = real_output.cuda()
        
        predicted_output = model(encode_inp.float(), decode_inp.float())
#         print(predicted_output.size(), real_output.view(-1,1).size())
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
            batch_iter = getbatch()
            if torch.cuda.is_available():
                batch_iter = batch_iter.cuda()

            encode_inp = batch_iter[:-1,:,:]
            decode_inp = batch_iter[[-1],:,:-1]
            real_output = batch_iter[[-1],:,[-1]].view(-1,1)
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
        checkout = 'sensor_lstm' + str(unfold_timestep) + 'ts.pt'
        torch.save(model.state_dict(), checkout)
    print("####Epoch:", epoch+1)
    print("Train Loss :", train_loss)
    print("Val Loss:", valid_loss)

