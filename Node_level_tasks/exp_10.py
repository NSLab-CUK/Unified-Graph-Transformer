import argparse
import copy
import logging
import time
from pathlib import Path
import numpy as np

import os
import os.path

from torch_geometric.utils import to_networkx
import networkx as nx
import dgl.sparse as dglsp
import numpy as np
import torch

import networkx as nx
import numpy as np


from gnnutils_10 import make_masks, train, test, add_original_graph, load_webkb, load_planetoid, load_wiki, load_bgp, \
	load_film, load_airports, load_amazon, load_crocodile
from util import addPE, BFS, getSI_D, getI, getD_vsI, buildPE_Kindices, getM_logM, get_k_indicaces, g_dgl, get_A_D

from script_classification import run_node_classification, run_epoch_node_classification, update_evaluation_value, run_node_clustering

import warnings

warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")

from models import Transformer

MODEl_DICT = {"Transformer": Transformer}

db_name = 0


def run(data, num_features, out_size, pos_enc_size, num_classes, isbgp, g, A_k, D, D_dim, Kindices, M, I, trans_logM,trans_M, alfa, beta,adj, d, n_edges):
	graph_name = args.dataset
	best_loss = 100000000

	pat = 20
	best_epoch = 0

	if args.model in MODEl_DICT:
		if args.model == "GIN" or args.model == "GAT":
			model = MODEl_DICT[args.model](
				num_features, out_size, num_classes, dims=args.dims, drop=args.drop).to(device)
		else:
			model = MODEl_DICT[args.model](num_features, args.out_size, pos_enc_size, num_classes, hidden_dim=args.dims,
										   num_layers=args.num_layers, num_heads=args.num_heads, D_dim=D_dim, k_transition = args.k_transition).to(device)
	else:
		print("Model not supported")
		raise NotImplementedError

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

	if args.custom_masks:
		data = make_masks(data, val_test_ratio=0.0)
		train_mask = data.train_mask

		val_mask = data.val_mask
		test_mask = data.test_mask
	g.ndata["train_mask"] = train_mask
	g.ndata["val_mask"] = val_mask
	g.ndata["test_mask"] = test_mask
	de = g.edata['de']
	M = g.edata['m']

	best_model = model
	if args.task == 'pre_training':
		for epoch in range(1, args.epochs):
			try:
				lap_pos_enc = g.ndata["PE"]
				sign_flip = torch.rand(lap_pos_enc.size(1))
				sign_flip[sign_flip >= 0.5] = 1.0
				sign_flip[sign_flip < 0.5] = -1.0
				g.ndata["PE"] = lap_pos_enc * sign_flip.unsqueeze(0)
			except:
				lap_pos_enc = None

			train_loss = train(model, data, train_mask, optimizer, device, g=g, A_k=A_k, D=D,
							Kindices=Kindices, de=de, M=M, I=I, trans_logM=trans_logM, pre_train=1, trans_M=trans_M,
							k_transition=args.k_transition, current_epoch= epoch, alfa = alfa, beta = beta)

			if best_loss > train_loss:
				best_model = model
				best_epoch = epoch
				best_loss = train_loss

			if epoch - best_epoch > 200:
				break
			if epoch % 10 == 0:
				print('Epoch: {:02d}, Train Loss: {:0.4f}'.format(epoch, train_loss))
		# Testing
		print('saving model and embeddings')

		torch.save(best_model,
				'{}{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(
			args.output_path, args.dataset, args.lr, args.dims, args.k_hop,args.num_layers, args.k_transition, args.alfa, args.beta))

		print('output_path: {}, dataset: {}, lr: {}, dims: {}, k_hop: {}, numLayers: {}, k_transition: {}, best_epoch: {}.pt'.format(
			args.output_path, args.dataset, args.lr, args.dims, args.k_hop, args.num_layers, args.k_transition, best_epoch))
		print("pre-training done")
	if args.task == 'node_classification':
		run_node_classification(args.dataset, args.lr, args.dims, args.k_hop,args.num_layers, args.k_transition, args.alfa, args.beta, args.output_path, args.file_name, data, num_features, args.out_size,
								args.k_eigenvector,
								num_classes, isbgp, g, A_k, D, D_dim, Kindices, M, I, trans_logM, device,
								args.run_times_fine, args.percent,  current_epoch= 1)
		
		#time.sleep(1)
	if args.task == 'node_clustering':
		run_node_clustering(args.dataset, args.lr, args.dims, args.k_hop,args.num_layers, args.k_transition, args.alfa, args.beta, args.output_path, args.file_name, data, num_features, args.out_size, args.k_eigenvector, num_classes, isbgp, 
			  g, A_k, D, D_dim, Kindices, M, I, trans_logM, device, args.run_times_fine, adj, d, n_edges)
		time.sleep(1)
	return 0

from dgl.data.utils import save_graphs, load_graphs


def main():
	np.random.seed(0)

	timestr = time.strftime("%Y%m%d-%H%M%S")
	log_file = args.dataset + "-" + timestr + ".log"

	global db_name

	db_name = str(args.dataset)

	Path("./exp_logs").mkdir(parents=True, exist_ok=True)
	logging.basicConfig(filename="exp_logs/" + log_file, filemode="w", level=logging.INFO)
	logging.info("Starting on device: %s", device)
	logging.info("Config: %s ", args)

	isbgp = False
	st_data = None
	if args.dataset in ['cornell', 'texas', 'wisconsin']:
		assert args.custom_masks == True
		og_data, st_data = load_webkb(args.dataset)
	elif args.dataset in ['cora', 'citeseer', 'pubmed']:
		assert args.custom_masks == True
		og_data, st_data = load_planetoid(args.dataset)
	

	elif args.dataset in ["chameleon", "squirrel"]:
		assert args.custom_masks == True
		og_data, st_data = load_wiki(args.dataset)
	
	elif args.dataset in ["crocodile"]:
		assert args.custom_masks == True
		og_data, st_data = load_crocodile(args.dataset)

	elif args.dataset in ["bgp"]:
		assert args.custom_masks == True
		isbgp = True
		og_data, st_data = load_bgp(args.dataset)
	elif args.dataset in ["film"]:
		assert args.custom_masks == True
		og_data, st_data = load_film(args.dataset)

	elif args.dataset in ["brazil", "europe", "usa"]:
		assert args.custom_masks == True
		og_data, st_data = load_airports(args.dataset)
	

	og_data_O = copy.deepcopy(og_data)


	if args.original_edges == 1:
		print("Adding original graph edges with weight %f" % args.original_edges_weight)
		data_all = add_original_graph(og_data, st_data, weight=args.original_edges_weight)
	else:
		data_all = copy.deepcopy(og_data)


	G_nx_O = to_networkx(og_data_O, to_undirected=True)

	path = "pts/" + args.dataset
	A = nx.adjacency_matrix(G_nx_O)


	G_nx = to_networkx(data_all, to_undirected=True)


	num_classes = len(data_all.y.unique())
	num_features = data_all.x.shape[1]


	N_nx = G_nx.number_of_nodes()

	####lOADING pts
	adj, d, n_edges = get_A_D(G_nx_O, N_nx)
	####lOADING pts
	path = "pts/" 

	if args.pre_load == 1:
		print('saving M, trans_M, trans_logM Eig, Kedge Kindices A_k D I')
		M, trans_M, trans_logM = getM_logM(G_nx_O, N_nx, args.k_transition)
		M = torch.from_numpy(M).float()
		trans_M = torch.from_numpy(np.array(trans_M)).float()
		trans_logM = torch.from_numpy(np.array(trans_logM)).float()
		k_eigenvector = args.k_eigenvector
		k_hop = 1
		for i in range(2):
			print(f'saving k_hop: {k_hop}')
			Eig, Kedge = buildPE_Kindices(data_all, k_eigenvector, trans_M, k_hop)
			Kindices = torch.LongTensor(Kedge)
			A_k = dglsp.spmatrix(Kindices, shape=(N_nx, N_nx))
			D, I, _ = getD_vsI(G_nx_O, N_nx, k_hop)
			D_dim = 4 * k_hop + 1
			D = torch.from_numpy(D)
			I = torch.from_numpy(I)  # N x (4k+1)

			path = "pts/"
			torch.save({"M": M,"trans_M":trans_M,"trans_logM":trans_logM,"Eig": Eig, "Kedge": Kedge,"Kindices": Kindices, "D": D, "I": I}, path + args.dataset+ '_k_hop_'+ str(k_hop)+ '.pt')
			
			print(f'saved in : {path}{args.dataset}_k_hop_{k_hop}_.pt')

			if os.path.exists("pts/"+ args.dataset + "_k_hop_" + str(k_hop)+ ".bin") == False:
				g = g_dgl(og_data_O, Eig, Kindices, D, D_dim, M, I)
				save_graphs("pts/"+ args.dataset + "_k_hop_" + str(k_hop)+ ".bin", g)
			else:
				g = load_graphs("pts/"+ args.dataset + "_k_hop_" + str(k_hop)+ ".bin")
				g = g[0][0]			
			k_hop+= 1
		raise SystemExit()
	else:
		path = "pts/"

		path1 = path + args.dataset + "_k_hop_" + str(args.k_hop) + ".pt"

		print(f'path *.pt: {path1}')
		dic = torch.load(path1)
		M = dic['M']
		trans_M = dic['trans_M']
		trans_logM = dic['trans_logM']
		k_eigenvector = args.k_eigenvector
		k_hop = args.k_hop
		Eig, Kedge = dic['Eig'], dic['Kedge']
		Kindices = dic['Kindices']
		A_k = dglsp.spmatrix(Kindices, shape=(N_nx, N_nx))
		D_dim = 4 * k_hop + 1
		D, I = dic['D'], dic['I']
		

		g = load_graphs(path  + args.dataset + "_k_hop_" + str(k_hop)+ ".bin")
		g = g[0][0]
		
		N = g.num_nodes()
		if N != N_nx:
			print('Num of nodes is different from Nx and Dgl !!!!')

		_ = run( data_all, num_features, args.out_size, k_eigenvector, num_classes, isbgp, g, A_k, D, D_dim,
				Kindices, M, I, trans_logM, trans_M, args.alfa, args.beta, adj, d, n_edges)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="WRGAT/WRGCN (structure + proximity) Experiments")

	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	print("Using device: ", device)

	parser.add_argument("--dataset", required=True,default="cora", help="Dataset")
	parser.add_argument("--model", default="Transformer", help="Transformer  Model")
	parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

	# common hyper parameters
	parser.add_argument("--original_edges", type=int, default=1)
	parser.add_argument("--original_edges_weight", type=float, default=1.0)
	parser.add_argument("--k_eigenvector", type=int, default=10)
	parser.add_argument("--drop", type=float, default=0.5, help="dropout")
	parser.add_argument("--run_times", type=int, default=1)
	parser.add_argument("--num_heads", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=2000)

	# specific params

	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--dims", type=int, default=16, help="hidden dims")
	parser.add_argument("--out_size", type=int, default=16, help="outsize dims")
	parser.add_argument("--k_hop", type=int, default=1)
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--k_transition", type=int, default=6)
	parser.add_argument("--run_times_fine", type=int, default=500)
	parser.add_argument("--percent", type=int, default=10, help="percent dims")

	parser.add_argument("--output_path", default="outputs/", help="outputs  Model")
	parser.add_argument("--file_name", default="brazil_10.xlsx", help="file_name  Model")
	parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
	parser.add_argument("--device", default="cuda:0", help="device  Model")
	parser.add_argument("--pre_load", type=int, default=1)
	parser.add_argument("--alfa", type=float, default=0.5)
	parser.add_argument("--beta", type=float, default=0.5)
	parser.add_argument("--task", default="pre_training", help="node_classification/node_clustering/pre_training")

	args = parser.parse_args()
	print(args)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print("Using device: ", device)
	main()