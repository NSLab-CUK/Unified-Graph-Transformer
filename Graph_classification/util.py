"""
	Utility functions for training one epoch 
	and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
import numpy as np
import argparse
import copy
import logging
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.sparse as sp
import os
import os.path
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset

import numpy as np
import torch
from tqdm import tqdm

import dgl
from dgl import LaplacianPE
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from collections import defaultdict 
import itertools

from collections import deque
from scipy import sparse

from metrics import accuracy_SBM as accuracy
np.seterr(divide = 'ignore') 

kth_step = defaultdict(list)


def train_epoch(model, optimizer, device, data_loader, epoch):

	model.train()
	epoch_loss = 0
	epoch_train_acc = 0
	nb_data = 0
	gpu_mem = 0
	for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
		batch_graphs = batch_graphs.to(device)
		batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
		batch_e = batch_graphs.edata['feat'].to(device)
		batch_labels = batch_labels.to(device)
		optimizer.zero_grad()
		try:
			batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
			sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
			sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
			batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
		except:
			batch_lap_pos_enc = None
			
		try:
			batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
		except:
			batch_wl_pos_enc = None

		batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
	
		loss = model.loss(batch_scores, batch_labels)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.detach().item()
		epoch_train_acc += accuracy(batch_scores, batch_labels)
	epoch_loss /= (iter + 1)
	epoch_train_acc /= (iter + 1)
	
	return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch):
	
	model.eval()
	epoch_test_loss = 0
	epoch_test_acc = 0
	nb_data = 0
	with torch.no_grad():
		for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
			batch_graphs = batch_graphs.to(device)
			batch_x = batch_graphs.ndata['feat'].to(device)
			batch_e = batch_graphs.edata['feat'].to(device)
			batch_labels = batch_labels.to(device)
			try:
				batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
			except:
				batch_lap_pos_enc = None
			
			try:
				batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
			except:
				batch_wl_pos_enc = None
				
			batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
			loss = model.loss(batch_scores, batch_labels) 
			epoch_test_loss += loss.detach().item()
			epoch_test_acc += accuracy(batch_scores, batch_labels)
		epoch_test_loss /= (iter + 1)
		epoch_test_acc /= (iter + 1)
		
	return epoch_test_loss, epoch_test_acc


def addPE(og_data, PE):
	edge_color = og_data.edge_color
	edge_index = og_data.edge_index
	edge_weight = og_data.edge_weight

	x = og_data.x
	
	
	print(f'cheking input datax: {np.shape(x)}')

	x_degree = og_data.x_degree
	y = og_data.y
	
	data = Data(x=x, x_degree=x_degree, edge_index=torch.LongTensor(edge_index),
						edge_weight=torch.FloatTensor(edge_weight),edge_color=torch.LongTensor(edge_color),PE=PE, y=y)
	return data

class BFS:
	def __init__(self, nx_g, source, k_hop):
		self.graph= nx_g
		self.num_nodes = len(self.graph.nodes)
		self.edges = self.graph.edges()
		self.source = source
		self.color=['W' for i in range(0,self.num_nodes)] # W for White
		self.queue = deque()
		self.k_hop = k_hop
		global kth_step
		self.bfs_traversal()

	def bfs_traversal(self):
		self.queue.append((self.source, 1))
		self.color[self.source] = 'B' # B for Black
		kth_step[0].append(self.source)


		while len(self.queue):
			u, level =  self.queue.popleft()
			if level > self.k_hop: # limit searching there
				return
			for v in self.graph.nodes:
				if self.graph.has_edge(u,v) and self.color[v]=='W':
					self.color[v]='B'
					kth_step[level].append(v)
					self.queue.append((v, level+1))
def getSI_D(G,target_node, value):
	list_nodes = list(value)
	list_degree = []
	length = len(value)
	min_d = 0
	max_d = 0
	mean_d = 0
	sigma_d = 0
	for dst_node in value:
		degree = G.degree(dst_node)
		list_degree.append(degree)
	  
	min_d = min(list_degree)
	max_d = max(list_degree)
	mean_d = np.round(sum(list_degree) / length,3)
	sigma_d = np.round(np.std(list_degree),3) + 1e-6


	return min_d, max_d, mean_d,sigma_d


def getI(nx_g, k_hop):
	dim = 4*k_hop + 1 #Ddim
	num_nodes = nx_g.number_of_nodes()
	I = np.zeros((num_nodes, dim))
	global kth_step
	for target_node in nx_g.nodes():
		node_SI = np.zeros(dim)
		k_list = np.zeros(4)
		try:
			node_degree = nx_g.degree(target_node)
			#list_SI.append(node_degree)
			node_SI[0] = node_degree

			bfs = BFS(nx_g, target_node, k_hop)
			loc = 4
			for key, value in kth_step.items(): # key = 1,2,3,4
				if key == 0:
					continue
				k_list = getSI_D(nx_g,target_node, value)
				ind = loc*key - 3 
				node_SI[ind ]= k_list[0]
				node_SI[ind + 1]= k_list[1]
				node_SI[ind + 2]= k_list[2]
				node_SI[ind + 3]= k_list[3]
		except:
			print('nan I')
		I[target_node] = node_SI
		kth_step = defaultdict(list)
	s=np.isnan(I); I[s]=0.0

 
	return I

def getD_vsI(nx_g, num_nodes,k_hop):
	I = getI(nx_g,k_hop)

	D = np.zeros((num_nodes, num_nodes, 4*k_hop + 1))
	for i in range(len(I)):
		r_i = I[i]
		for j in range(len(I)):
			temp = (r_i - I[j])* (r_i - I[j])
			D[i][j] = np.round(1/(temp + 0.5),3)

	return D, I, 4*k_hop + 1

def checkingVirtualEdges(G_og, G_st, G_all, data):
	nodes = list(G_og.nodes)
	freq1 = []
	ind = 0
	for i in nodes:
		deg = G_og.degree[i]
		freq1.append(deg)

	freq_1 = []
	
	deg = [*set(freq1)]

	for item in deg:
		count = 0
		for i in freq1:
			if item == i:
				count += 1
		freq_1.append(count)

	freq2 = []
	for d in deg:
		check = []
		count = 0
		nodes_d = []
		for node in G_og.nodes:
			if G_og.degree[node] == d:
				nodes_d.append(node)
		for node in nodes_d:
			for e in G_st.edges():
				if e[0] == node:
					if node in check:
						continue
					check.append(node)
					count += 1
				if e[1] == node:
					if node in check:
						continue
					check.append(node)
					count += 1
		freq2.append(count)


	plt.subplot(1, 3, 1)  # row 1, col 2 index 1
	degrees = [val for (node, val) in G_og.degree()]
	plt.hist(degrees, bins=50)
	plt.xlabel("original degree distribution")

	plt.subplot(1, 3, 2)  # row 1, col 2 index 1
	degrees = [val for (node, val) in G_all.degree()]
	plt.hist(degrees, bins=50)
	plt.xlabel("after degree distribution")

	# plot lines
	plt.subplot(1, 3, 3)  # row 1, col 2 index 1
	plt.plot(deg, freq_1, label="Original")
	plt.plot(deg, freq2, label="virtual edges")
	signal_final = [freq_1[i] + freq2[i] for i in range(len(freq2))]
	plt.plot(deg, signal_final, label="final edges")
	plt.legend()
	plt.show()


def convert_sparse_matrix_to_sparse_tensor(matrix):
	matrix = matrix.tocoo().astype(np.float32)
	indices = torch.from_numpy(
		np.vstack((matrix.row, matrix.col)).astype(np.int64))
	tensor_matrix = torch.from_numpy(matrix.data)
	shape = torch.Size(matrix.shape)

	return torch.sparse.FloatTensor(indices, tensor_matrix, shape)

def normalizeTensor(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed
def g_dgl(data, PE,  Kindices, D,  D_dim, M, I):

	I = torch.from_numpy(I)  # N x (4k+1)
	I1 = copy.deepcopy(I)
	nx_g = to_networkx(data, to_undirected=True)
	edge_idx11 = []
	edge_idx21 = []
	for e in nx_g.edges:
		edge_idx11.append(e[0])
		edge_idx21.append(e[1])
	g1 = dgl.graph((edge_idx11, edge_idx21))
	g1 = dgl.to_bidirected(g1)
	PE = dgl.lap_pe(g1, 10, padding=True)

	edge_idx1 = Kindices[0]
	edge_idx2 = Kindices[1]
	m_vals = []
	d_vals = []
	k_edge_idx1 = []
	k_edge_idx2 = []
	count = 0
	for i in range(len(edge_idx1)):
		count+= 1

		n1 = edge_idx1[i]
		n2 = edge_idx2[i]

		k_edge_idx1.append(n1)
		k_edge_idx2.append(n2)

		d = np.asarray(D[n1][n2], dtype=float)
		m = np.asarray(M[n1][n2], dtype=float)

		m_vals.append(m)
		d_vals.append(d)

	d_vals = np.array(d_vals)
	m_vals = np.array(m_vals)

	d_vals = normalize(d_vals, axis=0, norm='max')
	m_vals = normalize(m_vals, axis=0, norm='max')

	PE += torch.full_like(PE, 1e-5); 


	PE = normalizeTensor(PE)

	
	
	I += torch.full_like(I, 1e-5)
	I = normalizeTensor(I)
	ex = torch.tensor(d_vals)
	ex_m = torch.tensor(m_vals)

	g = dgl.graph((k_edge_idx1, k_edge_idx2))
	g = dgl.to_bidirected(g)

	g.ndata["PE"] = torch.Tensor(PE)
	g.edata['de'] = ex
	g.edata['m'] = ex_m
	g.ndata['I'] = I
	g.ndata['x'] = data.x
	
	return g

def get_A_D(nx_g, num_nodes):

	num_edges= nx_g.number_of_edges()
	d= np.zeros((num_nodes))

	Adj = np.zeros((num_nodes, num_nodes))

	for src in nx_g.nodes():
		src_degree = nx_g.degree(src)
		d[src] = src_degree
		for dst in nx_g.nodes():
			if nx_g.has_edge(src, dst):
				Adj[src][dst] = 1

	return Adj, d, num_edges
def normalizeRows(M):
    row_sums = M.sum(axis=1)
    return M / row_sums
def getM_logM(nx_g, num_nodes, k_transition):
	tran_M = []
	Adj = np.zeros((num_nodes, num_nodes))
	zerro_count = 0
	for src in nx_g.nodes():
		src_degree = nx_g.degree(src)

		if src_degree <= 0:
			zerro_count+= 1
			for dst in nx_g.nodes():
				if nx_g.has_edge(src, dst):
					Adj[src][dst] = 0	
		else:
			for dst in nx_g.nodes():
				if nx_g.has_edge(src, dst):
					Adj[src][dst] = round(1/src_degree,3)


	Adj2 = np.dot(Adj, Adj);	Adj3 = np.dot(Adj2, Adj);	Adj4 = np.dot(Adj3, Adj);	Adj5 = np.dot(Adj4, Adj)
	Adj6 = np.dot(Adj5, Adj);	Adj7 = np.dot(Adj6, Adj);	Adj8 = np.dot(Adj7, Adj)
	
	tran_M.append(Adj); 	tran_M.append(Adj2);	tran_M.append(Adj3);	tran_M.append(Adj4)
	tran_M.append(Adj5); 	tran_M.append(Adj6);	tran_M.append(Adj7);	tran_M.append(Adj8)

	tran_logM = []
	Ak = np.matrix(np.identity(num_nodes))
	for i in range(8):
		Ak = np.dot(Ak, Adj)

		
		probTranMat = GetProbTranMat(Ak,num_nodes)
		tran_logM.append(probTranMat)


	M = np.zeros((num_nodes, num_nodes, 8))
	for src in nx_g.nodes():
		for dst in nx_g.nodes():
		#if src == 0 and dst == 0:
			trans = []
			trans.append( Adj[src][dst])
			trans.append(Adj2[src][dst])
			trans.append(Adj3[src][dst])
			trans.append(Adj4[src][dst])
			trans.append(Adj5[src][dst])
			trans.append(Adj6[src][dst])
			trans.append(Adj7[src][dst])
			trans.append(Adj8[src][dst])

			M[src][dst] = trans
	return M, tran_M, tran_logM

def GetProbTranMat(Ak,num_node):
	num_node, num_node2 = Ak.shape
	if (num_node != num_node2):
		print('M must be a square matrix!')
	Ak_sum = np.sum(Ak, axis=0).reshape(1,-1)
	Ak_sum = np.repeat(Ak_sum, num_node, axis=0)
	probTranMat = np.log(np.divide(Ak, Ak_sum)) - np.log(1./num_node)  
	probTranMat[probTranMat < 0] = 0;                   #set zero for negative and -inf elements
	probTranMat[np.isnan(probTranMat)] = 0;             #set zero for nan elements (the isolated nodes)
	return probTranMat

def buildPE_Kindices(data1,dim,trans_M,k_hop):
	nx_g = to_networkx(data1, to_undirected=True)
	edge_idx1 = []
	edge_idx2 = []

	for e in nx_g.edges:
		edge_idx1.append(e[0])
		edge_idx2.append(e[1])
	g = dgl.graph((edge_idx1, edge_idx2))
	g = dgl.to_bidirected(g)

	Eig = dgl.lap_pe(g, dim,padding=True)

	path = dict(nx.all_pairs_shortest_path(nx_g, k_hop))

	nodes_ids = list(path.keys())
	all_path = list(map(path.get, nodes_ids))

	
	src = []
	dst = []
	for s_idx, s_node in enumerate(nodes_ids):
		spd_from_idx = all_path[s_idx]
		for target_node, path in spd_from_idx.items():

			len_of_path = len(path)
			if len_of_path == 1:
				continue
			elif len_of_path == 2:

				src.append(s_node)
				dst.append(target_node)
			else:

				src.append(s_node)
				dst.append(target_node)

	Kindices = get_k_indicaces(src,dst)
	Eig = np.round(Eig, 4)

	return Eig,   Kindices

def get_k_indicaces(src,dst):
	map1={}
	ind = []
	for i in range(len(src)-1):
		v1 = str(src[i]) +"_" + str(dst[i])
		v2 = str(dst[i]) +"_" + str(src[i])
		if not map1.get(v1):
			map1[v1] = True
			map1[v2] = True
		if not map1.get(v2):
			map1[v1] = True
			map1[v2] = True
	sd =list(map1.keys())
	source = []
	destination= []
	for i in range(len(sd)):
		x = sd[i].split("_")
		source.append(int(x[0]))
		destination.append(int(x[1]))
	return np.stack((np.array(source), np.array(destination)))
