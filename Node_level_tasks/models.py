import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

import torch.nn as nn
import dgl
import dgl.sparse as dglsp
import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import os.path
import numpy as np

import dgl.function as fn

"""
	Util functions
"""

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-10, 10))}

    return func


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}

    return func


# Improving implicit attention scores with explicit edge features, if available
def scaling(field, scale_constant):
    def func(edges):
        return {field: (((edges.data[field])) / scale_constant)}

    return func


def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn]  + edges.data[explicit_edge])}

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


class Transformer_class_lastLayer(nn.Module):
    def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim, graph_name,
                 cp_filename):
        super().__init__()

        print(f'Loading Transformer_class_lastLayer {cp_filename}')
        self.model = torch.load(cp_filename)


        unfrezz_layers = "layers." + str(num_layers)
        unfrezz_layers_1 = "layers." + str(num_layers - 1)
        for name, para in self.model.named_parameters():
            if num_layers == 2:
                if unfrezz_layers in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                if unfrezz_layers in name or unfrezz_layers_1 in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False

        self.MLP = MLP(out_dim, n_classes)

    def forward(self, g, current_epoch):
        h = self.model.extract_features(g, current_epoch)

        h = self.MLP(h)
        h = F.softmax(h, dim=1)

        return h

class Transformer_cluster_lastLayer(nn.Module):
    def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim, graph_name,
                 cp_filename):
        super().__init__()

        print(f'Loading Transformer_cluster_lastLayer {cp_filename}')
        self.model = torch.load(cp_filename)

        unfrezz_layers = "layers." + str(num_layers)
        unfrezz_layers_1 = "layers." + str(num_layers - 1)
        for name, para in self.model.named_parameters():
            if num_layers == 2:
                if unfrezz_layers in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                if unfrezz_layers in name or unfrezz_layers_1 in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
                # para.requires_grad = False

        self.MLP = MLPReadout(out_dim, n_classes)


    def forward(self, g, current_epoch):
        h = self.model.extract_features(g, current_epoch)
        h = self.MLP(h)
        h = F.softmax(h, dim=1)

        return h
class Transformer_cluster(nn.Module):
    def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim, graph_name,
                 cp_filename):
        super().__init__()

        print(f'Loading Transformer_cluster {cp_filename}')
        self.model = torch.load(cp_filename, map_location= 'cuda:0')


        unfrezz_layers = "layers." + str(num_layers)
        unfrezz_layers_1 = "layers." + str(num_layers - 1)
        for name, para in self.model.named_parameters():
            if num_layers == 2:
                if unfrezz_layers in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                if unfrezz_layers in name or unfrezz_layers_1 in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
                # para.requires_grad = False

        self.MLP = MLPReadout(out_dim, n_classes)


    def forward(self, g, current_epoch):
        h = self.model.extract_features(g, current_epoch)
        h = self.MLP(h)
        h = F.softmax(h, dim=1)

        return h


class Transformer_class(nn.Module):
    def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim, graph_name,
                 cp_filename):
        super().__init__()

        print(f'Loading Transformer_class {cp_filename}')
        self.model = torch.load(cp_filename, map_location= 'cuda:0')

        unfrezz_layers = "layers." + str(num_layers)
        unfrezz_layers_1 = "layers." + str(num_layers - 1)
        for name, para in self.model.named_parameters():
            if num_layers == 2:
                if unfrezz_layers in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                if unfrezz_layers in name or unfrezz_layers_1 in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
                # para.requires_grad = False

        self.MLP = MLP(out_dim, n_classes)

    def forward(self, g, current_epoch):
        h = self.model.extract_features(g, current_epoch)

        h = self.MLP(h)
        h = F.softmax(h, dim=1)

        return h

class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim,
                 k_transition):
        super().__init__()
        self.h = None
        self.embedding_h = nn.Linear(in_dim, hidden_dim, bias=False)  # node feat is an integer
        self.pos_enc_size = pos_enc_size
        self.D_dim = D_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.k_transition = k_transition

        self.lap_pos_enc = nn.Linear(pos_enc_size, hidden_dim)

        self.embedding_d = nn.Linear(D_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, D_dim) for _ in range(num_layers)])

        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, D_dim))

        self.embedding_de = nn.Linear(D_dim, hidden_dim)
        self.embedding_m = nn.Linear(8, hidden_dim)

        self.MLP_layer_x = Reconstruct_X(out_dim, in_dim)


    def extract_features(self, g, current_epoch):
        lap = g.ndata['PE']
        X = g.ndata['x']
        transM = g.edata['m']
        dis_E = g.edata['de']
        I = g.ndata['I']

        h = self.embedding_h(X)      + self.lap_pos_enc(lap.float())

        m = self.embedding_m(transM.float())


        dis_E = self.embedding_de(dis_E.float())
        count = 1
        for layer in self.layers:
            h = layer(h, g, dis_E, m, I, current_epoch)

            count += 1

        return h

    def forward(self, g, k_transition, current_epoch):
        h = self.extract_features(g, current_epoch)

        # compute X
        x_hat = self.MLP_layer_x(h)
        self.h = h

        return h, x_hat


class GraphTransformerLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, in_dim, out_dim, num_heads, D_dim):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.d_dim = D_dim

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, D_dim)

        self.O = nn.Linear(out_dim, out_dim)

        self.batchnorm1 = nn.BatchNorm1d(out_dim)
        self.batchnorm2 = nn.BatchNorm1d(out_dim)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.proj_i = nn.Linear(self.d_dim, out_dim)

    def forward(self, h, g, de, m, I, current_epoch):
        h_in1 = h  # for first residual connection

        attn_out = self.attention(h, g, de, m, current_epoch)


        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, 0.5, training=self.training)


        h = h + self.proj_i(I.float())

        h = self.O(h)

        h = h_in1 + h  # residual connection

        h = self.layer_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)

        h = F.dropout(h, 0.5, training=self.training)
        h = self.FFN_layer2(h)
        h = h_in2 + h  # residual connection
        h = self.layer_norm2(h)

        return h



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, D_dim):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

        self.hidden_size = in_dim  # 80
        self.num_heads = num_heads  # 8
        self.head_dim = out_dim // num_heads  # 10
        self.d_dim = D_dim  # 4*k_hop +1
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)


        self.proj_d = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.proj_m = nn.Linear(in_dim, out_dim * num_heads, bias=True)

    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))

        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        g.apply_edges(imp_exp_attn('score', 'proj_d'))

        g.apply_edges(imp_exp_attn('score', 'proj_m'))

        # softmax
        g.apply_edges(exp('score'))


        g.apply_edges(scaling('score', 2))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, dgl.function.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))  # src_mul_edge
        g.send_and_recv(eids, dgl.function.copy_e('score', 'score'), fn.sum('score', 'z'))  # copy_edge


    def forward(self, h, g, de, m, current_epoch):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        proj_d = self.proj_d(de)
        proj_m = self.proj_m(m)


        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        g.edata['proj_d'] = proj_d.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_m'] = proj_m.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here


        return h_out


def add_d(d, attn_sp):
    attn_sp = attn_sp + d
    d = d.to_sparse_csr()
    return attn_sp

# [131, 131, 8]
def tensor_2_sparse(attn_d, attn_sp):
    row = attn_sp.row
    col = attn_sp.col

    vals = []
    for i in range(len(row)):
        vals.append(attn_d[row[i], col[i]])

    vals = torch.stack(vals)
    return dglsp.val_like(attn_sp, vals)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)

        return y


class Reconstruct_X(torch.nn.Module):
    def __init__(self, inp, outp, dims=128):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(inp, dims),
            torch.nn.SELU(),
            torch.nn.Linear(dims, outp))

    def forward(self, x):
        x = self.mlp(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, dims=16):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, dims),
            torch.nn.ReLU(),
            torch.nn.Linear(dims, num_classes))

    def forward(self, x):
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):

    def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim, graph_name):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, 32)
        self.conv2 = GCNConv(32, n_classes)
        self.mlp = torch.nn.Linear(in_dim, 32)

        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, g, A_k, D, Kindices, de, M, I):
        h = self.conv1(x, edge_index)
        h1 = self.mlp(x)

        x = F.relu(h) + h1

        x = self.conv2(x, edge_index)
        x = self.drop(x)

        return F.softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim, graph_name):
        super(GIN, self).__init__()

        nn1 = Sequential(Linear(in_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, n_classes))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(n_classes)

        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, g, A_k, D, Kindices, de, M, I):
        x = F.selu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.drop(x)
        x = F.selu(self.conv2(x, edge_index))
        x = self.bn2(x)
        return F.softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, dim=16, drop=0.5):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, dim)
        self.conv2 = SAGEConv(dim, num_classes)
        self.drop = torch.nn.Dropout(p=drop)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), x
