# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 02:56:50 2018

@author: Cong
"""

import os
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, n_layers=1):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, n_layers)
        
    def forward(self, x):
        # input:  (time_steps, batch_size, feat_dim=input_size)
        # output: (time_steps, batch_size, hidden_size)
        outputs, hidden = self.rnn(x)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size=50, output_size=2, n_layers=1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(output_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, code_vec, target):
        target_len = target.shape[0] # (time_steps(seq_len), batch_size, feat_dim)
        batch_size = code_vec.shape[1] # (1, batch_size, hidden_size)
        
        decoder_input = torch.zeros(1, batch_size, self.output_size).float()
        decoder_hidden = code_vec
        decoder_outputs = torch.zeros(
            target_len,
            batch_size,
            self.output_size
        )  # (time_steps, batch_size, feat_dim)
        
        # unfold the decoder RNN on the time dimension
        for t in range(target_len):
            rnn_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            rnn_output = rnn_output.squeeze(0) # squeeze the time dimension
            decoder_outputs[t] = self.out(rnn_output) # (batch_size, output_size)
            decoder_input = target[t].unsqueeze(0) # teacher forcing
        
        return decoder_outputs, decoder_hidden


class Seq2SeqAE(nn.Module):
    def __init__(self, hidden_size=50, feat_dim=3):
        super(Seq2SeqAE, self).__init__()
        self.encoder = Encoder(hidden_size=hidden_size, input_size=feat_dim)
        self.decoder = Decoder(hidden_size=hidden_size, output_size=feat_dim)

    def forward(self, inputs, target):
        encoder_outputs, encoder_hidden = self.encoder.forward(inputs)
        decoder_outputs, decoder_hidden = self.decoder.forward(encoder_hidden, target)
        return decoder_outputs
    
    
# one-way distance between two trajectories
def aeowd(traj1, traj2, feat_dim=3, hidden_size=50, iteration=3000, loss_th=1e-5, learning_rate=0.001):
    
    # models
    model = Seq2SeqAE(hidden_size=hidden_size, feat_dim=feat_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # position embedding
    pos_embed1 = np.arange(0, len(traj1), dtype=np.float).reshape(-1,1) / len(traj1)
    traj1 = np.c_[traj1, pos_embed1]
    pos_embed2 = np.arange(0, len(traj2), dtype=np.float).reshape(-1,1) / len(traj2)
    traj2 = np.c_[traj2, pos_embed2]
    
    # data processing
    input_seq = torch.from_numpy(traj1).unsqueeze(1).float()
    input_eval = torch.from_numpy(traj2).unsqueeze(1).float()
    
    # train
    for e in range(iteration):
        # (seqlen, batch, feature)
        out = model(input_seq, input_seq)
        loss = criterion(out, input_seq)
        if loss.item() < loss_th:
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 50 == 0:
            print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))
    
    # evaluate
    model.train(False)
    out_eval = model(input_eval, input_eval)
    loss_eval = criterion(out_eval, input_eval)

    return np.abs(loss_eval.item() - loss.item())


# bi-directional distance between two trajectories
def aebid(traj1, traj2, feat_dim=3, hidden_size=50, iteration=3000, loss_th=1e-5, learning_rate=0.001):
    
    # models
    model = Seq2SeqAE(hidden_size=hidden_size, feat_dim=feat_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # position embedding
    pos_embed1 = np.arange(0, len(traj1), dtype=np.float).reshape(-1,1) / len(traj1)
    traj1 = np.c_[traj1, pos_embed1]
    pos_embed2 = np.arange(0, len(traj2), dtype=np.float).reshape(-1,1) / len(traj2)
    traj2 = np.c_[traj2, pos_embed2]
    
    # data1
    input_seq = torch.from_numpy(traj1).unsqueeze(1).float()
    input_eval = torch.from_numpy(traj2).unsqueeze(1).float()
    
    # train1
    for e in range(iteration):
        # (seqlen, batch, feature)
        out = model(input_seq, input_seq)
        loss = criterion(out, input_seq)
        if loss.item() < loss_th:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 50 == 0:
            print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))
    delta1 = loss.item()
    
    # evaluate1
    model.train(False)
    out_eval = model(input_eval, input_eval)
    loss12 = criterion(out_eval, input_eval)
    
    # data2
    input_seq = torch.from_numpy(traj2).unsqueeze(1).float()
    input_eval = torch.from_numpy(traj1).unsqueeze(1).float()
    
    # train2
    model.train(True)
    for e in range(iteration):
        # (seqlen, batch, feature)
        out = model(input_seq, input_seq)
        loss = criterion(out, input_seq)
        if loss.item() < loss_th:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 50 == 0:
            print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))
    delta2 = loss.item()
    
    # evaluate2
    model.train(False)
    out_eval = model(input_eval, input_eval)
    loss21 = criterion(out_eval, input_eval)

    return np.abs(loss12.item() + loss21.item() - delta1 - delta2)


# pair distances between two sets
def aecdist(train_set, test_set, feat_dim=3, 
            hidden_size=50, iteration=3000, loss_th=1e-5, learning_rate=0.001):
    
    train_num = len(train_set)
    test_num = len(test_set)
    
    # models
    model = Seq2SeqAE(hidden_size=hidden_size, feat_dim=feat_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # compute distances
    print('compute distances between %d and %d samples' % (train_num,test_num))
    dist_mat = np.zeros((train_num, test_num))
    
    for i in range(train_num):
        print('--', i)
        train_traj = train_set[i]
        pos_embed = np.arange(0, len(train_traj), dtype=np.float).reshape(-1,1) / len(train_traj)
        train_traj = np.c_[train_traj, pos_embed]
        
        # train
        model.train(mode=True)
        input_seq = torch.from_numpy(train_traj).unsqueeze(1).float()
        for e in range(iteration):
            # (seqlen, batch, feature)
            out = model(input_seq, input_seq)
            loss = criterion(out, input_seq)
            if loss.item() < loss_th:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (e + 1) % 20 == 0:
                print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))
        
        # evaluate
        model.eval()
        for j in range(test_num):
            if (j+1) % 50 == 0:
                print(j+1)
            test_traj = test_set[j]
            pos_embed_test = np.arange(0, len(test_traj), dtype=np.float).reshape(-1,1) / len(test_traj)
            test_traj = np.c_[test_traj, pos_embed_test]
            
            input_eval = torch.from_numpy(test_traj).unsqueeze(1).float()
            out = model(input_eval, input_eval)
            loss_eval = criterion(out, input_eval)
            
            dist_mat[i,j] = np.abs(loss_eval.item() - loss.item())
    return dist_mat


# pair distances between a set itself
def aepdist(trajs, modeldir='./statedicts/', feat_dim=3, 
            hidden_size=50, iteration=3000, loss_th=1e-5, learning_rate=0.001):
    
    traj_num = len(trajs)
    
    # models
    model = Seq2SeqAE(hidden_size=hidden_size, feat_dim=feat_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # compute distance matrix
    print('training...')
    
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    self_loss = np.zeros((traj_num,))
    for i in range(traj_num):
        print('--', i)
        source = trajs[i]
        pos_embed = np.arange(0, len(source), dtype=np.float).reshape(-1,1) / len(source)
        source = np.c_[source, pos_embed]
        inputseq = torch.from_numpy(source).unsqueeze(1).float()
        
        # train
        for e in range(iteration):
            # (seqlen, batch, feature)
            out = model(inputseq, inputseq)
            loss = criterion(out, inputseq)
            if loss.item() < loss_th:
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (e + 1) % 20 == 0:
                print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))
        self_loss[i] = loss.item()
        torch.save(model.state_dict(), modeldir+'msd%03d.pkl' % i)
    np.save(modeldir+'self_loss.pkl', self_loss)
    
    print('computing...')
    dm = np.zeros((traj_num, traj_num))
    for i in range(traj_num):
        print('-', i)
        source = trajs[i]
        pos_embed = np.arange(0, len(source), dtype=np.float).reshape(-1,1) / len(source)
        source = np.c_[source, pos_embed]
        input_i = torch.from_numpy(source).unsqueeze(1).float()
        
        for j in range(i+1, traj_num):
            if (j+1) % 50 == 0:
                print(j+1)
            target = trajs[j]
            pos_embed2 = np.arange(0, len(target), dtype=np.float).reshape(-1,1) / len(target)
            target = np.c_[target, pos_embed2]
            input_j = torch.from_numpy(target).unsqueeze(1).float()
            
            model.load_state_dict(torch.load(modeldir+'msd%03d.pkl' % i))
            model.train(False)
            out_j = model(input_j, input_j)
            loss_j = criterion(out_j, input_j)
            
            model.load_state_dict(torch.load(modeldir+'msd%03d.pkl' % j))
            model.train(False)
            out_i = model(input_i, input_i)
            loss_i = criterion(out_i, input_i)
            
            dm[i,j] = np.abs(loss_i.item() + loss_j.item() - self_loss[i] - self_loss[j])
    
    for i in range(traj_num):
        for j in range(traj_num):
            if i > j:
                dm[i,j] = dm[j,i]
    return dm
    

# one-way distance from cluster(batch) to traj
def aebdist(train_set, test_traj, feat_dim=3, 
            hidden_size=50, iteration=3000, loss_th=1e-5, learning_rate=0.001):
    
    # models
    model = Seq2SeqAE(hidden_size=hidden_size, feat_dim=feat_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # position embedding
    pos_embed1 = np.arange(0, len(train_set[0]), dtype=np.float).reshape(-1,1) / len(train_set[0])
    input_set = []
    for i in range(len(train_set)):
        traj = np.c_[train_set[i], pos_embed1]
        input_set.append(traj)
    source = np.array(input_set).transpose(1,0,2) # input shape: (time_steps, batch_size, feat_dim)
    
    pos_embed2 = np.arange(0, len(test_traj), dtype=np.float).reshape(-1,1) / len(test_traj)
    target = np.c_[test_traj, pos_embed2]
    
    # data processing
    input_seqs = torch.from_numpy(source).float()
    input_eval = torch.from_numpy(target).unsqueeze(1).float()
    
    # train
    for e in range(iteration):
        # (seqlen, batch, feature)
        out = model(input_seqs, input_seqs)
        loss = criterion(out, input_seqs)
        if loss.item() < loss_th:
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 50 == 0:
            print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))
    
    # evaluate
    model.train(False)
    out_eval = model(input_eval, input_eval)
    loss_eval = criterion(out_eval, input_eval)

    return np.abs(loss_eval.item() - loss.item())



def batch_test(sample0, sample1):
    
    # configs
    hidden_size = 50
    feat_dim = 2
    iteration = 100
    learning_rate = 0.01
    
    # models
    model = Seq2SeqAE(hidden_size=hidden_size, feat_dim=feat_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # data
    # input shape: (time_steps, batch_size, feat_dim)
    source = np.array([sample0, sample1]).transpose(1,0,2)
    target = sample1
    inputseq = torch.from_numpy(source).float()
    input_eval = torch.from_numpy(target).unsqueeze(1).float()
    
    # train
    for e in range(iteration):
        # (seqlen, batch, feature)
        out = model(inputseq, inputseq)
        loss = criterion(out, inputseq)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 20 == 0:
            print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))
    self_loss = loss.item()
    
    # evaluate
    model.train(False)
    out_eval = model(input_eval, input_eval)
    loss = criterion(out_eval, input_eval)

    print(loss.item() - self_loss)
    


if __name__ == '__main__':
    
    # prepare dataset
    from datasets import read_cross_dataset
    train_set, train_labels, test_set, true_labels = read_cross_dataset()
    sample0 = train_set[0]
    sample1 = train_set[7]
    sample2 = train_set[600]
    
#    d0 = aeowd(sample0, sample0)
#    d1 = aeowd(sample0, sample1)
#    d2 = aeowd(sample0, sample2)
#    d3 = aeowd(sample1, sample2)
#    print(d0,d1,d2,d3)

    dc = aebdist([sample0, sample1], sample2, loss_th=1e-4)
    print(dc)
