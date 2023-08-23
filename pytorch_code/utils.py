#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np


def build_graph(train_data):
    print('ccccccccccccccccccccccc')
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, timestart, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_times = [upois + [upois[-1]] * (len_max - le) for upois, le in zip(timestart, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_times, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        weights = data[2]
        inputs, weights, mask, len_max = data_masks(inputs, weights, [0])
        
        self.weights = np.asarray(weights)
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
        

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.weights = self.weights[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, weights, mask, targets = self.inputs[i], self.weights[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs, time_weights = [], [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for idx, u_input in enumerate(inputs):
            w = []
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            u_w = np.zeros((max_n_node, max_n_node))
            max_session = max([weights[idx][k + 1] - weights[idx][k] for k in range(len(weights[idx]) - 1)])
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    w.append(1)
                    w += [0] * (len(u_input) - len(w))
                    time_weights.append(w)
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                if (weights[idx][i + 1] - weights[idx][-1]) == 0:
                    w.append(1)
                else:
                    if u_A[u][v] == 0:
                        w.append((weights[idx][i + 1] - weights[idx][i])/((weights[idx][-1] - weights[idx][i])))
                        u_w[u][v] = weights[idx][-1] - weights[idx][i]
                    else:
                        w.append((weights[idx][i + 1] - weights[idx][i])/((weights[idx][-1] - weights[idx][i])))
                
                u_A[u][v] = 1
            
                    
            if len(w) < len(u_input):
                print('No noooooooooo')
                print('U input', u_input[-5:])
                print('U input shape', len(u_input))
                w += [1]
                w += [0] * (len(u_input) - len(w))
                time_weights.append(w)
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            
        return alias_inputs, A, items, mask, targets, time_weights


# #!/usr/bin/env python36
# # -*- coding: utf-8 -*-
# """
# Created on July, 2018

# @author: Tangrizzly
# """

# import networkx as nx
# import numpy as np


# def build_graph(train_data):
#     print('ccccccccccccccccccccccc')
#     graph = nx.DiGraph()
#     for seq in train_data:
#         for i in range(len(seq) - 1):
#             if graph.get_edge_data(seq[i], seq[i + 1]) is None:
#                 weight = 1
#             else:
#                 weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
#             graph.add_edge(seq[i], seq[i + 1], weight=weight)
#     for node in graph.nodes:
#         sum = 0
#         for j, i in graph.in_edges(node):
#             sum += graph.get_edge_data(j, i)['weight']
#         if sum != 0:
#             for j, i in graph.in_edges(i):
#                 graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
#     return graph


# def data_masks(all_usr_pois, timestart, item_tail):
#     us_lens = [len(upois) for upois in all_usr_pois]
#     len_max = max(us_lens)
#     us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
#     us_times = [upois + [upois[-1]] * (len_max - le) for upois, le in zip(timestart, us_lens)]
#     us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
#     return us_pois, us_times, us_msks, len_max


# def split_validation(train_set, valid_portion):
#     train_set_x, train_set_y = train_set
#     n_samples = len(train_set_x)
#     sidx = np.arange(n_samples, dtype='int32')
#     np.random.shuffle(sidx)
#     n_train = int(np.round(n_samples * (1. - valid_portion)))
#     valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
#     valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
#     train_set_x = [train_set_x[s] for s in sidx[:n_train]]
#     train_set_y = [train_set_y[s] for s in sidx[:n_train]]

#     return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


# class Data():
#     def __init__(self, data, shuffle=False, graph=None):
#         inputs = data[0]
#         weights = data[2]
#         inputs, weights, mask, len_max = data_masks(inputs, weights, [0])
        
#         self.weights = np.asarray(weights)
#         self.inputs = np.asarray(inputs)
#         self.mask = np.asarray(mask)
#         self.len_max = len_max
#         self.targets = np.asarray(data[1])
#         self.length = len(inputs)
#         self.shuffle = shuffle
#         self.graph = graph
        

#     def generate_batch(self, batch_size):
#         if self.shuffle:
#             shuffled_arg = np.arange(self.length)
#             np.random.shuffle(shuffled_arg)
#             self.inputs = self.inputs[shuffled_arg]
#             self.weights = self.weights[shuffled_arg]
#             self.mask = self.mask[shuffled_arg]
#             self.targets = self.targets[shuffled_arg]
#         n_batch = int(self.length / batch_size)
#         if self.length % batch_size != 0:
#             n_batch += 1
#         slices = np.split(np.arange(n_batch * batch_size), n_batch)
#         slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
#         return slices

#     def get_slice(self, i):
#         inputs, weights, mask, targets = self.inputs[i], self.weights[i], self.mask[i], self.targets[i]
#         items, n_node, A, alias_inputs, time_weights = [], [], [], [], []
#         for u_input in inputs:
#             n_node.append(len(np.unique(u_input)))
#         max_n_node = np.max(n_node)
#         for idx, u_input in enumerate(inputs):
#             w = []
#             node = np.unique(u_input)
#             items.append(node.tolist() + (max_n_node - len(node)) * [0])
#             u_A = np.zeros((max_n_node, max_n_node))
#             max_session = max([weights[idx][k + 1] - weights[idx][k] for k in range(len(weights[idx]) - 1)])
#             start = 0
#             for i in np.arange(len(u_input) - 1):
#                 if u_input[i + 1] == 0:
#                     w.append(1)
#                     w += [0] * (len(u_input) - len(w))
#                     time_weights.append(w)
#                     break
#                 u = np.where(node == u_input[i])[0][0]
#                 v = np.where(node == u_input[i + 1])[0][0]

#                 if (weights[idx][i + 1] - weights[idx][-1]) == 0:
#                     w.append(1)
#                     u_A[u][v] = 1
#                 else:
#                     val = (weights[idx][i + 1] - weights[idx][i])/((weights[idx][-1] - weights[idx][i]))
#                     if val >= 0.1:
#                         u_A[start][u] = 1
#                         if i > 0 :
#                           start = u
#                         w.append(val)
#                     else:
#                         w.append(0)
                           
#             if len(w) < len(u_input):
#                 print('No noooooooooo')
#                 print('U input', u_input[-5:])
#                 print('U input shape', len(u_input))
#                 w += [1]
#                 w += [0] * (len(u_input) - len(w))
#                 time_weights.append(w)
#             u_sum_in = np.sum(u_A, 0)
#             u_sum_in[np.where(u_sum_in == 0)] = 1
#             u_A_in = np.divide(u_A, u_sum_in)
#             u_sum_out = np.sum(u_A, 1)
#             u_sum_out[np.where(u_sum_out == 0)] = 1
#             u_A_out = np.divide(u_A.transpose(), u_sum_out)
#             u_A = np.concatenate([u_A_in, u_A_out]).transpose()
#             A.append(u_A)
#             alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            
#         return alias_inputs, A, items, mask, targets, time_weights

