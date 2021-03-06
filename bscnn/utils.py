#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:55:33 2018

@author: zfq
"""

import scipy.io as sio 
import numpy as np


def Get_neighs_order():
    neigh_orders_10242 = get_neighs_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat.mat',
                                          '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order.mat')
    neigh_orders_2562 = get_neighs_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_2562.mat',
                                          '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_2562.mat')
    neigh_orders_642 = get_neighs_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_642.mat',
                                          '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_642.mat')
    neigh_orders_162 = get_neighs_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_162.mat',
                                          '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_162.mat')
    neigh_orders_42 = get_neighs_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_42.mat',
                                          '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_42.mat')
    neigh_orders_12 = get_neighs_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_12.mat',
                                          '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_12.mat')
    
    return neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12
  
    
def get_neighs_order(mat_path, order_path):
    adj_mat = sio.loadmat(mat_path)
    adj_order = sio.loadmat(order_path)
    adj_mat = adj_mat[mat_path.split('/')[-1][0:-4]]
    adj_order = adj_order[order_path.split('/')[-1][0:-4]]
    neigh_orders = np.zeros(len(adj_mat) * 7).astype(np.int64) - 1
    for i in range(len(adj_mat)):
        raw_neigh_order = list(np.nonzero(adj_mat[i]))[0]
        if len(raw_neigh_order) == 5:
            order = (adj_order[i][0:-1] - 1).astype(np.int64)
            correct_neigh_order = raw_neigh_order[order]
            correct_neigh_order = np.append(correct_neigh_order, i)
        else:
            order = (adj_order[i] - 1).astype(np.int64)
            correct_neigh_order = raw_neigh_order[order]
        assert(len(correct_neigh_order) == 6)
        correct_neigh_order = np.append(correct_neigh_order, i)
        neigh_orders[i*7: (i+1)*7] = correct_neigh_order
        
    return neigh_orders


def Get_upsample_neighs_order():
    
    upsample_neighs_10242 = get_upsample_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat.mat',
                                               '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order.mat')
    upsample_neighs_2562 = get_upsample_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_2562.mat',
                                               '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_2562.mat')
    upsample_neighs_642 = get_upsample_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_642.mat',
                                               '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_642.mat')
    upsample_neighs_162 = get_upsample_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_162.mat',
                                               '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_162.mat')
    upsample_neighs_42 = get_upsample_order('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_42.mat',
                                               '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_42.mat')
    
    return upsample_neighs_10242, upsample_neighs_2562, upsample_neighs_642, upsample_neighs_162, upsample_neighs_42
    

def get_upsample_order(mat_path, order_path):
    adj_mat = sio.loadmat(mat_path)
    adj_order = sio.loadmat(order_path)
    adj_mat = adj_mat[mat_path.split('/')[-1][0:-4]]
    adj_order = adj_order[order_path.split('/')[-1][0:-4]]
    nodes = len(adj_mat)
    next_nodes = int((len(adj_mat)+6)/4)
    upsample_neighs_order = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = list(np.nonzero(adj_mat[i]))[0]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        upsample_neighs_order[(i-next_nodes)*2:(i-next_nodes)*2+2]= parent_nodes      
    
    return upsample_neighs_order  
        



def Get_upconv_index():
    
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat.mat',
                                                                       '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_2562.mat',
                                                                     '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_2562.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_642.mat',
                                                                   '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_642.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat_162.mat',
                                                                   '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_order_162.mat')
    return upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 

def get_upconv_index(mat_path, order_path):  
    
    adj_mat = sio.loadmat(mat_path)
    adj_order = sio.loadmat(order_path)
    adj_mat = adj_mat[mat_path.split('/')[-1][0:-4]]
    adj_order = adj_order[order_path.split('/')[-1][0:-4]]
    nodes = len(adj_mat)
    next_nodes = int((len(adj_mat)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = list(np.nonzero(adj_mat[i]))[0]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(len(parent_nodes)):
            parent_neigh = list(np.nonzero(adj_mat[parent_nodes[j]]))[0]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index