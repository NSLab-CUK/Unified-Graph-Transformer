import argparse
import copy
import logging
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import os
import os.path
from torch_geometric.utils import to_networkx

import dgl.sparse as dglsp
import numpy as np
import torch
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import normalize


from gnnutils_10 import make_masks, train_finetuning_class, test, add_original_graph, load_webkb, load_planetoid, \
	load_wiki, load_bgp, load_film, load_airports, train_finetuning_cluster, test_cluster
from util import addPE, BFS, getSI_D, getI, getD_vsI, buildPE_Kindices, getM_logM, get_k_indicaces, g_dgl

import warnings

warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")

from models import Transformer_class, Transformer_cluster, GIN, GCN

MODEl_DICT = {"Transformer_class": Transformer_class, "Transformer_cluster": Transformer_cluster, "GCN": GCN,
			  "GIN": GIN}

db_name = 0


def update_evaluation_value(file_path, colume, row, value):
	try:
		df = pd.read_excel(file_path)

		df[colume][row] = value

		df.to_excel(file_path, sheet_name='data', index=False)

		return
	except:
		print("Error when saving results! Save again!")
		time.sleep(3)


def generate_D_I_M_G():
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
	elif args.dataset in ["bgp"]:
		# for bgp only one split given from original paper is used.
		assert args.custom_masks == True
		isbgp = True
		og_data, st_data = load_bgp(args.dataset)
	elif args.dataset in ["film"]:
		assert args.custom_masks == True
		og_data, st_data = load_film(args.dataset)
	elif args.dataset in ["brazil", "europe", "usa"]:
		assert args.custom_masks == True
		og_data, st_data = load_airports(args.dataset)

	print(f'Number of edges st_data before filtering: {st_data.num_edges}')

	og_data_O = copy.deepcopy(og_data)
	print(f'number original edges: {np.shape(og_data.edge_index)}')

	if args.original_edges:
		print("Adding original graph edges with weight %f" % args.original_edges_weight)
		data_all = add_original_graph(og_data, st_data, weight=args.original_edges_weight)
	else:
		data_all = og_data

	print(f'number all edges: {np.shape(data_all.edge_index)}')

	G_nx_O = to_networkx(og_data_O, to_undirected=True)

	G_nx = to_networkx(data_all, to_undirected=True)
	print("Virtual edges reconstructing DONE...")
	num_classes = len(data_all.y.unique())
	num_features = data_all.x.shape[1]
	# num_relations = len(data.edge_color.unique())
	out_size = 80
	N_nx = G_nx.number_of_nodes()
	M, trans_M, trans_logM = getM_logM(G_nx_O, N_nx)

	# change data	og_data_O
	k_eigenvector = args.k_eigenvector
	k_hop = args.k_hop

	Eig, Kedge = buildPE_Kindices(data_all, k_eigenvector, trans_M, k_hop)

	Kindices = torch.LongTensor(Kedge)
	A_k = dglsp.spmatrix(Kindices, shape=(N_nx, N_nx))

	D, I, D_dim = getD_vsI(G_nx_O, N_nx, k_hop)
	D = torch.from_numpy(D)
	I = torch.from_numpy(I)  # N x (4k+1)

	# D = torch.LongTensor(D)


	M = torch.from_numpy(M)

	# Construct dgl graph:
	g = g_dgl(og_data_O, Eig, Kindices, D, D_dim, M)

	# indices = torch.stack(g.edges())
	N = g.num_nodes()
	if N != N_nx:
		print('Num of nodes is different from Nx and Dgl !!!!')

	return data_all, num_features, out_size, k_eigenvector, num_classes, isbgp, g, A_k, D, D_dim, Kindices, M, I, trans_logM


def main():
	data = pd.read_excel(args.file_path + args.file_name)

	for index_excel, row in data.iterrows():
		if row["Mean"] != -1:
			acc = data['Mean'][index_excel]
			print(f'already main done in file, index: {index_excel}, mean: {acc}')
		else:
			args.dataset = data['dataset'][index_excel]
			args.lr = data['lr'][index_excel]
			args.dims = data['dims'][index_excel]
			args.out_size = data['out_size'][index_excel]
			args.k_hop = data['k_hop'][index_excel]
			args.num_layers = data['num_layers'][index_excel]
			# local_epochs = data['epochs'][index_excel]
			args.k_transition = data['k_transition'][index_excel]
			print(f"processing training - {index_excel}")
			cp_filename = args.file_path + f'{args.dataset}_{args.lr}_{args.dims}_{args.k_hop}_{args.num_layers}_{args.k_transition}.pt'
			# args.dataset, args.lr, args.dims, args.k_hop, args.num_layers, args.k_transition)
			if os.path.isfile(cp_filename) == False:
				print(f"no file {cp_filename}")
				continue

			data_all, num_features, out_size, k_eigenvector, num_classes, isbgp, g, A_k, D, D_dim, Kindices, M, I, trans_logM = generate_D_I_M_G()

			runs_acc = []
			for i in tqdm(range(args.run_times)):
				# print(f'run time: {i}')
				acc = run_epoch_node_classification(i, data_all, num_features, out_size, k_eigenvector, num_classes,
													isbgp, g, A_k, D, D_dim, Kindices, M, I, trans_logM, cp_filename)
				runs_acc.append(acc)
				time.sleep(5)
			runs_acc = np.array(runs_acc) * 100
			update_evaluation_value(args.file_path + args.file_name, 'Mean', index_excel, runs_acc.mean())
			update_evaluation_value(args.file_path + args.file_name, 'Variant', index_excel, runs_acc.std())
			final_msg = "Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
			print(final_msg)


# Node classification

def run_node_classification(dataset, lr, dims, k_hop,num_layers, k_transition, alfa, beta, output_path, file_name, data_all, num_features, out_size, k_eigenvector,
							num_classes,isbgp, g, A_k, D, D_dim, Kindices, M, I, trans_logM, device, num_epochs, percent, current_epoch):

	cp_filename = output_path + f'{dataset}_{lr}_{dims}_{k_hop}_{num_layers}_{k_transition}_{alfa}_{beta}.pt'
	if os.path.isfile(cp_filename) == False:
		return None

	runs_acc = []
	for i in tqdm(range(1)):
		acc = run_epoch_node_classification(i, data_all, num_features, out_size, k_eigenvector, num_classes, isbgp,
											g, A_k, D, D_dim, Kindices, M, I, trans_logM, cp_filename, dims,
											num_layers, lr, device, num_epochs, current_epoch)
		runs_acc.append(acc)
		time.sleep(1)
	runs_acc = np.array(runs_acc) * 100

	final_msg = "Node Classification: Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())

def run_epoch_node_classification(i, data, num_features, out_size, pos_enc_size, num_classes, isbgp, g, A_k, D, D_dim,
								  Kindices, M, I, trans_logM, cp_filename,
								  dims, num_layers, lr, device, num_epochs, current_epoch):
	graph_name = ""
	best_val_acc = 0
	best_model = None
	pat = 20
	best_epoch = 0

	# fine tuning & testing

	model = Transformer_class(num_features, out_size, pos_enc_size, num_classes, hidden_dim=dims,
							  num_layers=num_layers, num_heads= 4, D_dim=D_dim, graph_name=graph_name,
							  cp_filename=cp_filename).to(device)

	best_model = model

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

	data = make_masks(data, val_test_ratio = 0.1)
	train_mask = data.train_mask
	val_mask = data.val_mask
	test_mask = data.test_mask

	# save dataload.
	g.ndata["train_mask"] = train_mask
	g.ndata["val_mask"] = val_mask
	g.ndata["test_mask"] = test_mask
	de = g.edata['de']
	M = g.edata['m']

	for epoch in range(1, num_epochs):
		# lap_pos_enc = g.ndata["PE"]
		# sign_flip = torch.rand(lap_pos_enc.size(1))
		# sign_flip[sign_flip>=0.5] = 1.0
		# sign_flip[sign_flip<0.5] = -1.0
		# g.ndata["PE"] = lap_pos_enc * sign_flip.unsqueeze(0)
		train_loss, train_acc = train_finetuning_class(model, data, train_mask, optimizer, device, g=g, A_k=A_k, D=D,
													   Kindices=Kindices, de=de, M=M, I=I, trans_logM=trans_logM,
													   pre_train=0, current_epoch= current_epoch)

		if epoch % 10 == 0:
			# valid
			valid_acc, valid_f1 = test(model, data, val_mask, device, g=g, A_k=A_k, D=D, Kindices=Kindices, de=de, M=M,I=I, current_epoch = current_epoch)

			if valid_acc > best_val_acc:
				best_val_acc = valid_acc
				best_model = model
				best_epoch = epoch
				pat = (pat + 1) if (pat < 5) else pat
			else:
				pat -= 1
			print('Epoch: {:02d}, Train Loss: {:0.4f}, Train Acc: {:0.4f}, Val Acc: {:0.4f} '.format(epoch, train_loss,
																									 train_acc,
																									 valid_acc))
			logging.info(
				'Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:0.4f}, Val Acc: {:0.4f} '.format(epoch, train_loss,
																								  train_acc, valid_acc))
			if pat < -100:
				print("validation patience reached ... finish training")
				logging.info("validation patience reached ... finish training")
				break
	# Testing
	test_acc, test_f1 = test(best_model, data, test_mask, device, g=g, A_k=A_k, D=D,Kindices=Kindices, de=de, M=M, I=I, current_epoch=current_epoch)
	print('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f}, F1_test: {:0.4f}'.format(
		best_epoch, best_val_acc, test_acc, test_f1))
	logging.info('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f} '.format(
		best_epoch, best_val_acc, test_acc))
	return test_acc, test_f1


def run_node_clustering(dataset, lr, dims, k_hop,num_layers, k_transition, alfa, beta, output_path, file_name, data_all, num_features, out_size, k_eigenvector,
							num_classes,isbgp, g, A_k, D, D_dim, Kindices, M, I, trans_logM, device, num_epochs, adj, d, n_edges):
	print("running run_node_clustering")
	
	cp_filename = output_path + f'{dataset}_{lr}_{dims}_{k_hop}_{num_layers}_{k_transition}_{alfa}_{beta}.pt'
	if os.path.isfile(cp_filename) == False:
		print(f"run_node_classification: no file {cp_filename}")
		return None


	for i in tqdm(range(1)):
		print(f'run time: {i}')
		q, c = run_epoch_node_clustering(i, data_all, num_features, out_size, k_eigenvector, num_classes, isbgp,
											g, A_k, D, D_dim, Kindices, M, I, trans_logM, cp_filename, dims,
											num_layers, lr, device, num_epochs, adj, d, n_edges)
		time.sleep(1)

	print('Q: {:0.4f}, , C: {:0.4f}'.format(q, c))

def run_epoch_node_clustering(i, data, num_features, out_size, pos_enc_size, num_classes, isbgp, g, A_k, D, D_dim,
								  Kindices, M, I, trans_logM, cp_filename,
								  dims, num_layers, lr, device, num_epochs, adj, d, n_edges):
	graph_name = ""
	best_val_acc = 0
	best_model = None
	pat = 20
	best_epoch = 0

	# fine tuning & testing
	print('fine tuning run_epoch_node_clustering...')

	model = Transformer_cluster(num_features, out_size, pos_enc_size, num_classes, hidden_dim=dims,
							  num_layers=num_layers, num_heads=4, D_dim=D_dim, graph_name=graph_name,
							  cp_filename=cp_filename).to(device)

	best_model = model

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

	print("creating  random mask")
	data = make_masks(data, val_test_ratio = 0.0)
	train_mask = data.train_mask
	val_mask = data.val_mask
	test_mask = data.test_mask

	# save dataload.
	g.ndata["train_mask"] = train_mask
	g.ndata["val_mask"] = val_mask
	g.ndata["test_mask"] = test_mask
	de = g.edata['de']
	M = g.edata['m']

	for epoch in range(1, num_epochs):
		train_loss = train_finetuning_cluster(model, data, train_mask, optimizer, device,  g=g, A_k=A_k, D = D,
					Kindices =Kindices, de = de, M = M, I= I,trans_logM =trans_logM, pre_train=0, adj=adj, d=d, n_edges= n_edges)
		
		if epoch % 10 == 0:
			print('Epoch: {:02d}, Train Loss: {:0.4f}'.format(epoch, train_loss))

	# Testing
	q, c = test_cluster(best_model, data, train_mask, device, g=g, A_k=A_k, D=D,Kindices=Kindices, de=de, M=M, I=I)

	print(' Q: {:0.4f},  C: {:0.4f}'.format( q, c))
	return q, c

#</end node clustering>
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="WRGAT/WRGCN (structure + proximity) Experiments")

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Using device: ", device)

	parser.add_argument("--dataset", required=True, help="Dataset")
	parser.add_argument("--model", default="Transformer_class", help="Transformer  Model")
	parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

	# common hyper parameters
	parser.add_argument("--original_edges", default=True, action='store_true')
	parser.add_argument("--original_edges_weight", type=float, default=1.0)
	parser.add_argument("--k_eigenvector", type=int, default=10)

	parser.add_argument("--drop", type=float, default=0.5, help="dropout")
	parser.add_argument("--run_times", type=int, default=10)
	parser.add_argument("--num_heads", type=int, default=4)

	# fixed params
	parser.add_argument("--dims", type=int, default=16, help="hidden dims")
	parser.add_argument("--out_size", type=int, default=16, help="outsize dims")
	parser.add_argument("--k_hop", type=int, default=1)
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--k_transition", type=int, default=2)

	# changing./ not changing, run directly fine tuning.
	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--epochs", type=int, default=1000)

	parser.add_argument("--file_path", default="outputs/brazil_10/", help="outputs  Model")
	parser.add_argument("--file_name", default="brazil_10.xlsx", help="file_name  Model")

	args = parser.parse_args()
	print(args)
	main()
