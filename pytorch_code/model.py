# #!/usr/bin/env python36
# # -*- coding: utf-8 -*-
# """
# Created on July, 2018

# @author: Tangrizzly
# """

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        self.dropout = nn.Dropout(0.2)

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.len_max = opt.len_max
        self.pos_emb = Parameter(torch.Tensor(self.len_max+1, self.hidden_size))
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_mean_a = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.linear_mean_b = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.linear_transform = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.linear_transform2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.positional_encoding = PositionalEncoding(self.hidden_size, 0.1, self.len_max)
        
        self.transformer_encoder1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4), num_layers=3)
        self.transformer_encoder2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4), num_layers=3)
    
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3,8], gamma=opt.lr_dc)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 2, eta_min = 1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, w):

        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        
        w = torch.tensor(w[-mask.shape[0]:]).unsqueeze(-1)
        
        # hta = hidden[torch.arange(mask.shape[0]).long(), :]*w 
      
        hidden = hidden + self.pos_emb[:mask.shape[1]] ## add 1
        # htfm = self.transformer_encoder1(hidden)[:, -1]
        c = torch.sum(hidden * w * mask.view(mask.shape[0], -1, 1).float(), 1)
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # q3 = self.linear_weight(c).view(ht.shape[0], 1, ht.shape[1])
        
        
        alpha = F.softmax(self.linear_three(torch.sigmoid(q1 + q2)) + (1-mask).unsqueeze(-1)*(-9999), dim=1) ## add 2
        a1 = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        # theta = F.softmax(self.linear_t(torch.sigmoid(q2 + q3)) + (1-mask).unsqueeze(-1)*(-9999), dim=1)
        # d = torch.sum(theta * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        # c = torch.sum(hidden * w * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = F.normalize(self.linear_transform(torch.cat([a1, c, ht], dim = -1)))
    
        b = F.normalize(self.embedding.weight[1:], dim = -1 ) # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))*16
        return scores
    

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        # hidden = self.gnn(A, hidden)
        hidden = self.dropout(F.normalize(hidden, dim=-1))
        hidden = self.dropout(self.gnn(A, hidden))
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets, time_weights = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    time_weights = trans_to_cuda(torch.Tensor(time_weights).float())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
        
    return targets, model.compute_scores(seq_hidden, mask, time_weights)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.detach()
        if j%100 == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    count = 0
    for i in slices:
        count += 1
        if count%1000==0:
            print('[%d/%d]' % (count, len(slices)))
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

# import datetime
# import math
# import numpy as np
# import torch
# from torch import nn
# from torch.nn import Module, Parameter
# import torch.nn.functional as F


# class GNN(Module):
#     def __init__(self, hidden_size, step=1):
#         super(GNN, self).__init__()
#         self.step = step
#         self.hidden_size = hidden_size
#         self.input_size = hidden_size * 2
#         self.gate_size = 3 * hidden_size
#         self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
#         self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
#         self.b_ih = Parameter(torch.Tensor(self.gate_size))
#         self.b_hh = Parameter(torch.Tensor(self.gate_size))
#         self.b_iah = Parameter(torch.Tensor(self.hidden_size))
#         self.b_oah = Parameter(torch.Tensor(self.hidden_size))

#         self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

#     def GNNCell(self, A, hidden):
#         input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
#         input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
#         inputs = torch.cat([input_in, input_out], 2)
#         gi = F.linear(inputs, self.w_ih, self.b_ih)
#         gh = F.linear(hidden, self.w_hh, self.b_hh)
#         i_r, i_i, i_n = gi.chunk(3, 2)
#         h_r, h_i, h_n = gh.chunk(3, 2)
#         resetgate = torch.sigmoid(i_r + h_r)
#         inputgate = torch.sigmoid(i_i + h_i)
#         newgate = torch.tanh(i_n + resetgate * h_n)
#         hy = newgate + inputgate * (hidden - newgate)
#         return hy

#     def forward(self, A, hidden):
#         for i in range(self.step):
#             hidden = self.GNNCell(A, hidden)
#         return hidden


# class SessionGraph(Module):
#     def __init__(self, opt, n_node):
#         super(SessionGraph, self).__init__()
#         self.hidden_size = opt.hiddenSize
#         self.n_node = n_node
#         self.batch_size = opt.batchSize
#         self.nonhybrid = opt.nonhybrid
#         self.embedding = nn.Embedding(self.n_node, self.hidden_size)
#         self.gnn = GNN(self.hidden_size, step=opt.step)
#         self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_mean = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
#         self.linear_transform = nn.Linear(self.hidden_size *3, self.hidden_size, bias=True)
#         self.loss_function = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

#     # def compute_scores(self, hidden, mask):
#     #     ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
#     #     q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
#     #     q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
#     #     alpha = self.linear_three(torch.sigmoid(q1 + q2))
#     #     hta = torch.mean(self.linear_mean(hidden[torch.arange(mask.shape[0]).long(), :]), dim = 1)
#     #     # hta = torch.mean(hidden[torch.arange(mask.shape[0]).long(), :], dim = 1)
#     #     a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
#     #     if not self.nonhybrid:
#     #         a = self.linear_transform(torch.cat([a, ht, hta], 1))
#     #     b = self.embedding.weight[1:]  # n_nodes x latent_size
#     #     scores = torch.matmul(a, b.transpose(1, 0))
#     #     return scores
#     # def compute_scores(self, hidden, mask):
#     #     ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
#     #     q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
#     #     q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
#     #     alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (b,s,1)
#     #     alpha = F.softmax(alpha, 1) # B,S,1
#     #     a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # (b,d)
#     #     if not self.nonhybrid:
#     #         a = self.linear_transform(torch.cat([a, ht], 1))
#     #     b = self.embedding.weight[1:]  # n_nodes x latent_size
#     #     hidden = hidden * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
#     #     qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
#     #     beta = F.softmax(b @ qt.transpose(1,2), -1)  # batch_size x n_nodes x seq_length
#     #     target = beta @ hidden  # batch_size x n_nodes x latent_size
#     #     a = a.view(ht.shape[0], 1, ht.shape[1])  # b,1,d
#     #     a = a + target  # b,n,d
#     #     scores = torch.sum(a * b, -1)  # b,n
#     #     return scores

#     def forward(self, inputs, A):
#         hidden = self.embedding(inputs)
#         hidden = self.gnn(A, hidden)
#         return hidden


# def trans_to_cuda(variable):
#     if torch.cuda.is_available():
#         return variable.cuda()
#     else:
#         return variable


# def trans_to_cpu(variable):
#     if torch.cuda.is_available():
#         return variable.cpu()
#     else:
#         return variable


# def forward(model, i, data):
#     alias_inputs, A, items, mask, targets = data.get_slice(i)
#     alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
#     items = trans_to_cuda(torch.Tensor(items).long())
#     A = trans_to_cuda(torch.Tensor(A).float())
#     mask = trans_to_cuda(torch.Tensor(mask).long())
#     hidden = model(items, A)
#     get = lambda i: hidden[i][alias_inputs[i]]
#     seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
#     return targets, model.compute_scores(seq_hidden, mask)


# def train_test(model, train_data, test_data):
#     model.scheduler.step()
#     print('start training: ', datetime.datetime.now())
#     model.train()
#     total_loss = 0.0
#     slices = train_data.generate_batch(model.batch_size)
#     for i, j in zip(slices, np.arange(len(slices))):
#         model.optimizer.zero_grad()
#         targets, scores = forward(model, i, train_data)
#         targets = trans_to_cuda(torch.Tensor(targets).long())
#         loss = model.loss_function(scores, targets - 1)
#         loss.backward()
#         model.optimizer.step()
#         total_loss += loss.detach()
#         # if j % int(len(slices) / 5 + 1) == 0:
#         #     print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
#         if j%10 == 0:
#             print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
#     print('\tLoss:\t%.3f' % total_loss)

#     print('start predicting: ', datetime.datetime.now())
#     model.eval()
#     hit, mrr = [], []
#     slices = test_data.generate_batch(model.batch_size)
#     for i in slices:
#         targets, scores = forward(model, i, test_data)
#         sub_scores = scores.topk(20)[1]
#         sub_scores = trans_to_cpu(sub_scores).detach().numpy()
#         for score, target, mask in zip(sub_scores, targets, test_data.mask):
#             hit.append(np.isin(target - 1, score))
#             if len(np.where(score == target - 1)[0]) == 0:
#                 mrr.append(0)
#             else:
#                 mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
#     hit = np.mean(hit) * 100
#     mrr = np.mean(mrr) * 100
#     return hit, mrr