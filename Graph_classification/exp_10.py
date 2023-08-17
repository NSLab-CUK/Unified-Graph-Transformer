import argparse
import copy
import logging
import time
from pathlib import Path
import numpy as np
import os
import os.path
from torch_geometric.utils import to_networkx
import dgl.sparse as dglsp
import numpy as np
import torch
from tqdm import tqdm

import dgl
import numpy as np


from gnnutils_10 import add_original_graph
from util import  getD_vsI, buildPE_Kindices, getM_logM,  g_dgl

from torch.utils.data import DataLoader

import warnings, random

warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")

from models import Transformer, Transformer_Graph_class
from torch_geometric.datasets import TUDataset
from dgl.data.utils import save_graphs, load_graphs

MODEl_DICT = {"Transformer": Transformer, "Transformer_Graph_class": Transformer_Graph_class}

db_name = 0

def collate(self, samples):
	graphs, labels = map(list, zip(*samples))
	labels = torch.tensor(np.array(labels)).unsqueeze(1)
	batched_graph = dgl.batch(graphs)       
	
	return batched_graph, labels

def run(i,  dataset_full, num_features, pos_enc_size, D_dim, num_classes,  k_transition,  alfa, beta, run_times_fine ):

	if args.model in MODEl_DICT:
		model = MODEl_DICT[args.model](num_features, args.out_size, pos_enc_size,  hidden_dim=args.dims,num_layers=args.num_layers, 
				 num_heads=args.num_heads,D_dim = D_dim, k_transition = args.k_transition, num_classes = args.num_classes ).to(device)
	else:

		print("Model not supported")
		raise NotImplementedError

	
	best_model =model
	trainset, valset, testset = dataset_full.train, dataset_full.val, dataset_full.test
	print("Training Graphs: ", len(trainset))
	print("Validation Graphs: ", len(valset))
	print("Test Graphs: ", len(testset))


	train_loader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn = dataset_full.collate )
	val_loader = DataLoader(valset, batch_size=64, shuffle=False, collate_fn = dataset_full.collate)
	test_loader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn = dataset_full.collate)

	if args.task == 'graph_classification':
		file_name = args.output_path + f'{args.dataset}_{args.lr}_{args.dims}_{args.k_hop}_{args.num_layers}_{args.k_transition}_{args.alfa}_{args.beta}.pt'
		torch.save(best_model,'{}{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.output_path, args.dataset, args.lr, args.dims, args.k_hop,
												args.num_layers, args.k_transition, args.alfa, args.beta))

	time.sleep(1)
	runs_acc = []
	for i in tqdm(range(1)):
		print(f'run time: {i}')
		acc = run_epoch_graph_classification(model, train_loader, val_loader,test_loader,num_features, pos_enc_size, D_dim, file_name, run_times_fine )
		runs_acc.append(acc)
		time.sleep(1)
	
	runs_acc = np.array(runs_acc) * 100

	final_msg = "Graph classification: Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
	print(final_msg)

	return 0
def run_epoch_graph_classification(model, train_loader, val_loader,test_loader,num_features, pos_enc_size, D_dim ,file_name, run_times_fine):

	model = Transformer_Graph_class(num_features, args.out_size, pos_enc_size,  hidden_dim=args.dims,num_layers=args.num_layers, 
				 num_heads=args.num_heads,D_dim = D_dim, k_transition = args.k_transition, num_classes = args.num_classes, cp_filename =file_name ).to(device)
	best_model = model
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
	best_loss = 100000000

	for epoch in range(1, args.epochs):
		epoch_train_loss, epoch_train_mae, optimizer = train_epoch_graph_classification(model, optimizer, device, train_loader, epoch)
			
		epoch_val_loss, epoch_val_mae = evaluate_network(model, device, val_loader, epoch)
	
		if best_loss >= epoch_train_loss:
			best_model = model
			best_epoch = epoch
			best_loss = epoch_train_loss
		if epoch - best_epoch > 100:
			break
		if epoch % 2 == 0:
			print(f'Epoch: {epoch}: Train_loss: {epoch_train_loss}, Val_loss: {epoch_val_loss} ,Train_acc: {epoch_train_mae}, Val_acc: {epoch_val_mae} ')

	_, test_acc = evaluate_network(best_model, device, test_loader, epoch)

	return test_acc

class GraphClassificationDataset:
	def __init__(self):
		self.graph_lists = []  # A list of DGLGraph objects
		self.graph_labels = []
		self.trans_logMs = []
	
	def add(self, g):

		self.graph_lists.append(g)

	def __len__(self):
		return len(self.graphs)

	def __getitem__(self, i):
		# Get the i^th sample and label
		return self.graphs[i], self.labels[i], self.trans_logM[i]


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


	graph_lists = []
	graph_labels = []
	st_data_lists =None
	
	if args.dataset in ["ENZYMES", "PROTEINS", "NCI1", "NCI109"]:
		dataset = TUDataset(root='original_datasets/' + args.dataset,use_node_attr =True, name=args.dataset)
		args.num_classes = dataset.num_classes
		args.num_features = dataset.num_features
		if args.pre_load == 1:			
			#st_data_lists = TUDataset(root="datasets_py_geom_format_10/"+ args.dataset, name = args.dataset,use_node_attr =True,   pre_transform= build_pyg_struc_multigraph)
			k_hop = 1
			D_dim = 4 * k_hop + 1
			for hop in range(1):
				print(f' computing k_hop: {k_hop}')
				graph_ds  = generate_graphs(dataset, st_data_lists, args.k_eigenvector, k_hop)
				save_graphs("pts/"+ args.dataset + "_k_hop_" + str(k_hop) + ".bin" , graph_ds.graph_lists, graph_ds.graph_labels)
				
				k_hop += 1 
			
			raise SystemExit()
		else:
			if args.dataset == "PROTEINS":
				file_name = "pts/"+ args.dataset + "_k_hop_" + str(args.k_hop) + ".bin" 
			if args.dataset == "NCI109":
				file_name = "pts/"+ args.dataset + "_k_hop_" + str(args.k_hop) + ".bin"
			if args.dataset == "NCI1":
				file_name = "pts/"+ args.dataset + "_k_hop_" + str(args.k_hop) +".bin" 
			if args.dataset == "ENZYMES":
				file_name = "pts/"+ args.dataset + "_k_hop_" + str(args.k_hop) + ".bin"

			file_name2 = "pts/"+ args.dataset + "_M.pt"
			file_name = "pts/"+ args.dataset + "_k_hop_" + str(args.k_hop) + ".bin"
			graph_lists, graph_labels = load_graphs(file_name)
			dic  = torch.load(file_name2)
			trans_logMs = dic['trans_logMs']

			print(f'Loaded gdl graph classification {file_name}')
			print(f'graph_lists: {np.shape(graph_lists)}, graph_labels: {np.shape(graph_labels["glabel"])} ')
	else:
		print('error loading dataset')
	num_features = args.num_features
	print(f'num_features: {num_features}, args.num_classes: {args.num_classes} ')
	samples_all = []
	for i in range(len(graph_lists)):
		
		current_graph = graph_lists[i]
		current_label = graph_labels['glabel'][i]
		current_trans_logM =   trans_logMs[i]
		
		pair = (current_graph, current_label, current_trans_logM)
		samples_all.append(pair)
	random.shuffle(samples_all)
	
	dataset_full = LoadData(samples_all, args.dataset)

	runs_acc = []
	D_dim = 4 * args.k_hop + 1
	for i in tqdm(range(args.run_times)):
		acc = run(i, dataset_full, num_features, args.k_eigenvector, D_dim, args.num_classes , args.k_transition,  args.alfa, args.beta, args.run_times_fine)
		runs_acc.append(acc)

	runs_acc = np.array(runs_acc) * 100


def generate_graphs(dataset, st_data_lists, pos_enc_size, k_hop):
	graph_ds = GraphClassificationDataset()
	graph_labels = []
	graph_lists = []
	trans_logMs = []
	for i in range(len(dataset)):
		if i %10 ==0:
			print(f' processing graph: {i}')
			time.sleep(1)
		og_data = dataset[i]
		if st_data_lists ==None:
			og_data_O = copy.deepcopy(og_data)
			data_all = copy.deepcopy(og_data)
		else:
			st_data = st_data_lists[i]
			og_data_O = copy.deepcopy(og_data)
			if args.original_edges == 1:
				data_all = add_original_graph(og_data, st_data, weight=args.original_edges_weight)
			else:
				data_all = copy.deepcopy(og_data)
		G_nx_O = to_networkx(og_data_O, to_undirected=True)
		G_nx = to_networkx(data_all, to_undirected=True)

		N_nx = G_nx.number_of_nodes()
		#### LOADING pts
		M, trans_M, trans_logM = getM_logM(G_nx_O, N_nx, args.k_transition)
		M = torch.from_numpy(M).float()
		trans_M = torch.from_numpy(np.array(trans_M)).float()
		trans_logM = torch.from_numpy(np.array(trans_logM)).float()
		
		Eig, Kedge = buildPE_Kindices(data_all, pos_enc_size, trans_M, k_hop)

		Kindices = torch.LongTensor(Kedge)
		A_k = dglsp.spmatrix(Kindices, shape=(N_nx, N_nx))
		D, I, _ = getD_vsI(G_nx_O, N_nx, k_hop)
		D_dim = 5 #4 * k_hop + 1
		D = torch.from_numpy(D); 
		
		try:
			g = g_dgl(og_data_O, Eig, Kindices, D, D_dim, M, I)
			graph_ds.graph_lists.append(g)
			trans_logMs.append(trans_logM)
			graph_lists.append(g)
			N = g.num_nodes()
			if N != N_nx:
				print('Num of nodes is different from Nx and Dgl !!!!')
			graph_labels.append(og_data.y)
			graph_ds.graph_labels = {"glabel": torch.tensor(graph_labels)}
			
		except:
			print(f'Error loading dgl graph: {i}')
			# if k_hop == 2:
			# 	raise SystemExit()	
	if os.path.exists("pts/"+ args.dataset + "_10p_M.pt") == False:
		torch.save({"trans_logMs": trans_logMs},"pts/"+ args.dataset + "_M.pt")

	return graph_ds

from molecules import MoleculeDataset

def LoadData(samples_all, DATASET_NAME):

	return MoleculeDataset(samples_all, DATASET_NAME)


from train_molecules_graph_regression import train_epoch, evaluate_network, train_epoch_graph_classification
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="WRGAT/WRGCN (structure + proximity) Experiments")

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Using device: ", device)

	parser.add_argument("--dataset", required=True, help="Dataset")
	parser.add_argument("--model", default="Transformer", help="Transformer  Model")
	parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

	# common hyper parameters
	parser.add_argument("--original_edges", type=int, default=1)
	parser.add_argument("--original_edges_weight", type=float, default=1.0)
	parser.add_argument("--k_eigenvector", type=int, default=10)
	parser.add_argument("--drop", type=float, default=0.5, help="dropout")
	parser.add_argument("--run_times", type=int, default=1)
	parser.add_argument("--num_heads", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=500)

	# specific params

	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--dims", type=int, default=16, help="hidden dims")
	parser.add_argument("--out_size", type=int, default=16, help="outsize dims")
	parser.add_argument("--k_hop", type=int, default=1)
	parser.add_argument("--num_layers", type=int, default=4)
	parser.add_argument("--k_transition", type=int, default=6)
	parser.add_argument("--run_times_fine", type=int, default=10)
	parser.add_argument("--num_features", type=int, default=-1, help="num_features dims")
	
	parser.add_argument("--output_path", default="outputs/", help="outputs  Model")
	parser.add_argument("--file_name", default="brazil_10.xlsx", help="file_name  Model")
	parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
	parser.add_argument("--device", default="cuda:0", help="device  Model")
	parser.add_argument("--pre_load", type=int, default=1) 
	parser.add_argument("--alfa", type=float, default=0.5)
	parser.add_argument("--beta", type=float, default=0.5)
	parser.add_argument("--task", default="graph_classification")

	parser.add_argument("--num_classes", type=int, default=-1, help="num_classes")
	args = parser.parse_args()
	print(args)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print("Using device: ", device)
	main()