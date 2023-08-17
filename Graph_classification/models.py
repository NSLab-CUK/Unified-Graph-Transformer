import torch
import torch.nn.functional as F

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


def exp(field):
	def func(edges):
		return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

	return func
def src_dot_dst(src_field, dst_field, out_field):
	def func(edges):
		return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
	return func
def scaled_exp(field, scale_constant):
	def func(edges):
		return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
	return func
def scaling(field, scale_constant):
	def func(edges):
		return {field: (((edges.data[field])) / scale_constant)}

	return func


def imp_exp_attn(implicit_attn, explicit_edge):

	def func(edges):
		return {implicit_attn: (edges.data[implicit_attn] +  edges.data[explicit_edge])}

	return func

def out_edge_features(edge_feat):
	def func(edges):
		return {'e_out': edges.data[edge_feat]}

	return func


class Transformer_cluster(nn.Module):
	def __init__(self, in_dim, out_dim, pos_enc_size, n_classes, hidden_dim, num_layers, num_heads, D_dim, graph_name,
				 cp_filename):
		super().__init__()

		print(f'Loading Transformer_cluster {cp_filename}')
		self.model = torch.load(cp_filename)

		for p in self.model.parameters():
			p.requires_grad = False

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
		self.model = torch.load(cp_filename)

		for p in self.model.parameters():
			p.requires_grad = False

		self.MLP = MLP(out_dim, n_classes)

	def forward(self, g, current_epoch):
		h = self.model.extract_features(g, current_epoch)

		h = self.MLP(h)
		h = F.softmax(h, dim=1)

		return h

class Transformer_Graph_class(nn.Module):
	def __init__(self, in_dim, out_dim, pos_enc_size,  hidden_dim, num_layers, num_heads, D_dim, k_transition, num_classes, cp_filename):
		super().__init__()
		print(f'Loading Transformer_Graph_class {cp_filename}')
		self.model = torch.load(cp_filename)

		for p in self.model.parameters():
			p.requires_grad = True

		
		self.MLP_layer = MLPReadout(out_dim, num_classes)
	def forward(self, batch_g, batch_x, batch_PE, batch_de, batch_m, batch_I ):
		h = self.model.extract_features(batch_g, batch_x, batch_PE, batch_de, batch_m, batch_I )

		batch_g.ndata['h'] = h
		
		self.h = h
	
	 
		hg = dgl.mean_nodes(batch_g, 'h')  # default readout is mean nodes
		hg = self.MLP_layer(hg)
		return F.softmax(hg, dim =1 )
	
	def loss(self, pred, label):
			criterion = nn.CrossEntropyLoss()
			loss = criterion(pred.to(torch.float32), label.squeeze(dim=-1))
			return loss

class Transformer(nn.Module):
	def __init__(self, in_dim, out_dim, pos_enc_size,  hidden_dim, num_layers, num_heads, D_dim, k_transition, num_classes):
		super().__init__()
		self.h = None
		self.embedding_h = nn.Linear(in_dim, hidden_dim, bias=False)  # node feat is an integer
		self.pos_enc_size = pos_enc_size
		self.D_dim = D_dim
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.k_transition = k_transition
		self.num_classes = num_classes

		self.lap_pos_enc = nn.Linear(pos_enc_size, hidden_dim)

		self.embedding_d = nn.Linear(D_dim, hidden_dim)

		self.layers = nn.ModuleList(
			[GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, D_dim) for _ in range(num_layers)])

		self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, D_dim))

		self.embedding_de = nn.Linear(D_dim, hidden_dim)
		self.embedding_m = nn.Linear(8, hidden_dim)
		
		self.MLP_layer_x = Reconstruct_X(out_dim, in_dim)

		self.MLP_layer = MLPReadout(out_dim, num_classes)

	def extract_features(self, g, batch_x, batch_PE, batch_de, batch_m, batch_I ):

		lap = batch_PE
		X = batch_x
		transM = batch_m
		dis_E = batch_de
		I = batch_I

		h = self.embedding_h(X)       + self.lap_pos_enc(lap.float())

		m = self.embedding_m(transM.float())


		dis_E = self.embedding_de(dis_E.float())
		count = 1
		for layer in self.layers:
			h = layer(h, g, dis_E, m, I)
			
			count+= 1

		return h

	def forward(self, batch_g, batch_x, batch_PE, batch_de, batch_m, batch_I ):
		h = self.extract_features(batch_g, batch_x, batch_PE, batch_de, batch_m, batch_I )

		batch_g.ndata['h'] = h
		
		self.h = h
		x_hat = self.MLP_layer_x(h)

		return h,  x_hat
	 
	
	def loss(self, pred, label):
			criterion = nn.CrossEntropyLoss()
			loss = criterion(pred.to(torch.float32), label.squeeze(dim=-1))
			return loss


	def compute_l1_loss(self, w):
		return torch.abs(w).sum()
	def compute_l2_loss(self, w):
		return torch.square(w).sum()

class GraphTransformerLayer(nn.Module):

	def __init__(self, in_dim, out_dim, num_heads, D_dim):
		super().__init__()

		self.in_channels = in_dim
		self.out_channels = out_dim
		self.num_heads = num_heads
		self.d_dim = D_dim

		self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, D_dim)

		self.O = nn.Linear(out_dim, out_dim)

		self.batch_norm1 = nn.BatchNorm1d(out_dim)
		self.batch_norm2 = nn.BatchNorm1d(out_dim)
		self.layer_norm1 = nn.LayerNorm(out_dim)
		self.layer_norm2 = nn.LayerNorm(out_dim)
		self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
		self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

		self.proj_i = nn.Linear(self.d_dim, out_dim)

	def forward(self, h, g, de, m, I ):

		h_in1 = h  # for first residual connection

		attn_out = self.attention(h, g, de, m)

		h = attn_out.view(-1, self.out_channels)

		h = F.dropout(h, 0.5, training=self.training)


		h = h + self.proj_i(I.float()) 

		
 
		h = self.O(h)

		h = h_in1 + h  # residual connection

		h = self.layer_norm1(h)
		h = self.batch_norm1(h)

		h_in2 = h  # for second residual connection
		
		# FFN
		h = self.FFN_layer1(h)
		h = F.relu(h)

		h = F.dropout(h, 0.5, training=self.training)
		h = self.FFN_layer2(h)
		h = h_in2 + h  # residual connection
		h = self.layer_norm2(h)
		h = self.batch_norm2(h)   


		return h


class MultiHeadAttentionLayer(nn.Module):
	# in_dim, out_dim, num_heads
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
		# Compute attention score
		g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  

		# scaling scale
		g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
	   
		g.apply_edges(imp_exp_attn('score', 'proj_d'))

		g.apply_edges(imp_exp_attn('score', 'proj_m'))

		g.apply_edges(exp('score'))

		eids = g.edges()
		g.send_and_recv(eids, dgl.function.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))  # src_mul_edge
		g.send_and_recv(eids, dgl.function.copy_e('score', 'score'), fn.sum('score', 'z'))  # copy_edge

	def forward(self, h, g, de, m):
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

		h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-10)) 

		return h_out

class Reconstruct_X(torch.nn.Module):
	def __init__(self, inp, outp, dims=16):
		super().__init__()
		
		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(inp, dims),
			torch.nn.SELU(),
			torch.nn.Linear(dims, outp))

	def forward(self, x):
		x = self.mlp(x)
		return x


def add_d(d, attn_sp):
	attn_sp = attn_sp + d
	d = d.to_sparse_csr()
	return attn_sp

def tensor_2_sparse(attn_d, attn_sp):
	row = attn_sp.row
	col = attn_sp.col

	vals = []
	for i in range(len(row)):
		vals.append(attn_d[row[i], col[i]])

	vals = torch.stack(vals)
	return dglsp.val_like(attn_sp, vals)


class MLPReadout(nn.Module):

	def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
		super().__init__()
		list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
		list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
		self.FC_layers = nn.ModuleList(list_FC_layers)
		self.L = L
		
	def forward(self, x):
		y = x
		for l in range(self.L):
			y = self.FC_layers[l](y)
			y = F.relu(y)
		y = self.FC_layers[self.L](y)
		return y
	

class MLP(torch.nn.Module):
	def __init__(self, num_features, num_classes, dims=16):
		super(MLP, self).__init__()
		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(num_features, dims),
			torch.nn.SELU(),
			torch.nn.Linear(dims, num_classes))

	def forward(self, x):
		x = self.mlp(x)
		return F.log_softmax(x, dim=1)