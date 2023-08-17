import argparse
import logging
import os
import pickle
import copy
from collections import Counter

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from struc_sim import graph
from struc_sim import struc2vec
import dgl
from dgl import LaplacianPE
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--edgelist_file', type=str)
	parser.add_argument('--output_file', type=str)
	parser.add_argument('--nodelabels_file', type=str)
	parser.add_argument('--until-layer', type=int, default=None)
	parser.add_argument('--workers', type=int, default=32,
						help='Number of parallel workers. Default is 32.')
	parser.add_argument('--OPT1', default=False, type=bool,
					  help='optimization 1')
	parser.add_argument('--OPT2', default=False, type=bool,
					  help='optimization 2')
	parser.add_argument('--OPT3', default=False, type=bool,
					  help='optimization 3')

	parser.add_argument('--disassortative', default=False, type=bool,
	                    help='is it disassortative dataset')
	parser.add_argument('--dataset', default="film", type=str, help='dataset name')
	return parser.parse_args()

def read_graph(edgelist_file):
	'''
	Reads the input network.
	'''
	logging.info(" - Loading graph...")
	G = graph.load_edgelist(edgelist_file,undirected=True)
	logging.info("Graph loaded.")
	return G

def build_struc_layers(G,opt1=True, opt2=True, opt3=True, until_layer=None,workers=64):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	if(opt3):
		until_layer = until_layer
	else:
		until_layer = None


	G = struc2vec.Graph(G, False, workers, untilLayer=5)

	if(opt1):
		G.preprocess_neighbors_with_bfs_compact()
	else:
		G.preprocess_neighbors_with_bfs()

	if(opt2):
		G.create_vectors()
		G.calc_distances(compactDegree=opt1)
	else:
		G.calc_distances_all_vertices(compactDegree=opt1)

	G.create_distances_network()
	G.preprocess_parameters_random_walk()
	return

def build_multigraph_from_layers(networkx_graph, y, x=None, edge_index_og= None):

	num_of_nodes = networkx_graph.number_of_nodes()
	num_of_edges = networkx_graph.number_of_edges()

	x_degree = torch.zeros(num_of_nodes, 1)
	
	for i in range(0, num_of_nodes):
		x_degree[i] = torch.Tensor([networkx_graph.degree(i)])

	inp = open("struc_sim/pickles/distances_nets_graphs.pickle", "rb")
	distances_nets_graphs = pickle.load(inp, encoding="bytes")
	src = []
	dst = []
	edge_weight = []
	edge_color = []
	for layer, layergraph in distances_nets_graphs.items():
		if layer >= 3:
			break
		#print("Number of nodes in layer "+ str(layer)+" is "+str(len(layergraph)))
		logging.info("Number of nodes in layer "+ str(layer)+" is "+str(len(layergraph)))
		filename = "struc_sim/pickles/distances_nets_weights-layer-" + str(layer) + ".pickle"
		inp = open(filename, "rb")
		distance_nets_weights_layergraph = pickle.load(inp, encoding="bytes")

		for node_id, nbd_ids in layergraph.items():
			s = list(np.repeat(node_id, len(nbd_ids)))
			d = nbd_ids
			src += s
			dst += d
			edge_weight += distance_nets_weights_layergraph[node_id]
			edge_color += list(np.repeat(layer, len(nbd_ids)))
		assert len(src) == len(dst) == len(edge_weight) == len(edge_color)

	edge_weight = np.array(edge_weight)
	edge_color = np.array(edge_color)

	edge_weight = edge_weight.tolist()
	edge_color = edge_color.tolist()

	try:
			
		for i in range(len(edge_weight)-1,-1,-1):
			node1 = src[i]
			node2 = dst[i]
			degree1 = networkx_graph.degree(node1)
			degree2 = networkx_graph.degree(node2)

			damp = np.sqrt(degree1 + degree2)

			w_new = edge_weight[i] + np.exp(-1/(damp))
			
			if w_new > 1:
				w_new = 1
		
			if w_new < 0.01:
				del src[i]
				del dst[i]
				del edge_weight[i]
				del edge_color[i]
				continue
			edge_weight[i] = w_new
	except:
		print("An exception occurred")

	ind = []
	count = 0

	for i in range(len(src)):
		if networkx_graph.has_edge(src[i], dst[i]):
			count+= 1
			ind.append(i)


	ind.sort()
	#ind = [*set(ind)]
	count = 0
	for ele in sorted(ind, reverse = True):
		del src[ele]
		del dst[ele]
		del edge_weight[ele]
		del edge_color[ele]
		count+= 1
	
	
	ind = []
	for i in range(len(src)):
		max_w = edge_weight[i]
		for j in range(i+1, len(src)):
			if src[i] == src[j]:
				if dst[i] == dst[j]:
					if max_w >= edge_weight[j]:
						ind.append(j)
					else:
						ind.append(i)
						max_w = edge_weight[j]

	ind = [*set(ind)]
	ind = sorted(ind, reverse = True)
	for ele in ind:
		del src[ele]
		del dst[ele]
		del edge_weight[ele]
		del edge_color[ele]

	num_of_edges_virtual = int(num_of_edges/5)

	ind = []
	vals = []
	edge_weight_cp = copy.deepcopy(edge_weight)

	edge_weight_cp.sort(reverse=True)
	if num_of_edges_virtual > len(edge_weight_cp):
		num_of_edges_virtual = len(edge_weight_cp)
	for i in range(0, num_of_edges_virtual):
		ind.append(i)
		w = edge_weight_cp[i]
		vals.append(w)
	dem = 0
	n1 = [0 for i in range(len(edge_weight))]
	for i in range(0, num_of_edges_virtual):
		for j in range(0, len(edge_weight)):
			if vals[i] == edge_weight[j]:
				if n1[j] == 0:
					ind[i] = j
					dem+= 1
					n1[j] = 1
					break

	len_w = len(edge_weight)
	for i in range(len_w - 1,-1,-1):
		if i not in ind:
			del edge_weight[i]
			del src[i]
			del dst[i]
			del edge_color[i]


	edge_index = np.stack((np.array(src), np.array(dst)))
	edge_color = np.asarray(edge_color)
	edge_weight =np.asarray(edge_weight)


	count = 0


	if x is None:
		data = Data(x=x_degree, edge_index=torch.LongTensor(edge_index), edge_weight=torch.FloatTensor(edge_weight),
				edge_color=torch.LongTensor(edge_color), y=y)
	else:
		data = Data(x=x, x_degree=x_degree, edge_index=torch.LongTensor(edge_index),
		            edge_weight=torch.FloatTensor(edge_weight),
		            edge_color=torch.LongTensor(edge_color), y=y)
		
		return data
count = 1
def build_pyg_struc_multigraph(pyg_data):
	global count

	count+=1

	logging.basicConfig(filename='struc2vec.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

	G = graph.from_pyg(pyg_data)
	networkx_graph = to_networkx(pyg_data)

	edge_index_og = pyg_data.edge_index
	
	build_struc_layers(G)
	data = build_multigraph_from_layers(networkx_graph, pyg_data.y, pyg_data.x, edge_index_og)
	if hasattr(pyg_data, 'train_mask'):
		data.train_mask = pyg_data.train_mask
		data.val_mask = pyg_data.val_mask
		data.test_mask = pyg_data.test_mask
	return data

def main(args):
	logging.basicConfig(filename='struc2vec.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
	G = read_graph(args.edgelist_file)
	build_struc_layers(G, args.OPT1, args.OPT2, args.OPT3, args.until_layer, args.workers)

	fin = open(args.nodelabels_file, 'r')
	if args.disassortative:
		tmp = fin.readlines()[1:]
		d = {}
		for l in tmp:
			n_id = int(l.split("\t")[0])
			n_f = list(map(int, l.split("\t")[1].split(",")))
			n_l = int(l.split("\t")[2].split("\n")[0])
			d[n_id] = (n_f, n_l)
		y = []
		nfs = []
		for n in sorted(d):
			if args.dataset == "film":
				# actually places where it is  # 932 is feature size for film dataset.
				features = np.zeros(932, dtype=np.float)
				features[d[n][0]] = 1.0
				nfs.append(features)
			else:
				# already in one hot format
				nfs.append(d[n][0])
			y.append(d[n][1])
		y = torch.LongTensor(y)
		x = torch.LongTensor(nfs)
		networkx_graph = nx.read_edgelist(args.edgelist_file, nodetype=int, comments="node", delimiter="\t")
	else:
		x = None
		tmp = fin.readlines()[0]
		y = tmp.strip('][').split(', ')
		y = list(map(int, y))
		y = torch.LongTensor(y)

		networkx_graph = nx.read_edgelist(args.edgelist_file, nodetype=int)

	data = build_multigraph_from_layers(networkx_graph, y, x)
	print(data)
	try:
		os.makedirs((os.path.dirname(args.output_file)))
	except OSError as e:
		pass
	torch.save(data, args.output_file)

if __name__ == "__main__":
	args = parse_args()
	main(args)
