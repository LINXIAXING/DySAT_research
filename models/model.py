# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss

from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
from utils.utilities import fixed_unigram_candidate_sampler

class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length):
        torch.nn.BCELoss()
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        # args.window为时间自注意力的窗口, 一次取一个窗口长度(未设置则取全部)做self-attention
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        # 节点的特征维度
        self.num_features = num_features
        # 结构的多头信息
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))  # [16, 8, 8]
        # 结构layer层信息
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        # 时序多头信息
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))  # [16]
        # 时序layer信息
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))  # [128]
        # 空间时间的droupout层
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        # 空间时间的自注意力模型
        self.structural_attn, self.temporal_attn = self.build_model()
        # 定义loss
        self.bceloss = BCEWithLogitsLoss()

    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps):
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        
        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_padded)
        
        return temporal_out

    def build_model(self):
        input_dim = self.num_features

        # 1: Structural Attention Layers
        # 添加结构注意力层
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):  # 遍历每层结构信息
            layer = StructuralAttentionLayer(input_dim=input_dim,  # 特征长度(维度)
                                             output_dim=self.structural_layer_config[i],  # output维度
                                             n_heads=self.structural_head_config[i],  # 多头参数
                                             attn_drop=self.spatial_drop,  # droupdout
                                             ffd_drop=self.spatial_drop,  # 同上
                                             residual=self.args.residual)  # 残差连接
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i] #层级之间output与input维度匹配
        
        # 2: Temporal Attention Layers
        # 添加时序注意力层
        input_dim = self.structural_layer_config[-1]  # 维度与注意力层最后一级output一致
        temporal_attention_layers = nn.Sequential()
        # 同结构注意力层
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers

    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn
        final_emb = self.forward(graphs) # [N, T, F]
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            emb_t = final_emb[:, t, :].squeeze() #[N, F]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten()
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.args.neg_weight*neg_loss
            self.graph_loss += graphloss
        return self.graph_loss

            




