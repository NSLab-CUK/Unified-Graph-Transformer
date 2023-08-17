import argparse
import copy
import logging
import time
from pathlib import Path
import numpy as np

from torch_geometric.utils import to_networkx
import numpy as np
import torch
from tqdm import tqdm
import dgl
import numpy as np



from util import getD_vsI, buildPE_Kindices, getM_logM,  g_dgl

from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")

from models import Transformer
from molecules import MoleculeDataset
from libs.utils import SRDataset, Grapg8cDataset


MODEl_DICT = {"Transformer": Transformer}

db_name = 0

def collate(self, samples):
	# The input samples is a list of pairs (graph, label).
	graphs, labels = map(list, zip(*samples))
	labels = torch.tensor(np.array(labels)).unsqueeze(1)
	batched_graph = dgl.batch(graphs)       
	
	return batched_graph, labels

def run(i,  dataset_full, num_features, pos_enc_size, D_dim, num_classes ):

	if args.model in MODEl_DICT:
		if args.model =="Transformer":
			model = MODEl_DICT[args.model](num_features, args.out_size, pos_enc_size,  hidden_dim=args.dims,num_layers=args.num_layers, \
				  num_heads=args.num_heads,D_dim = D_dim, k_transition = args.k_transition, num_classes = args.num_classes ).to(device)

	else:
		print("Model not supported")
		raise NotImplementedError

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

	data_all = dataset_full.data_all

	pre_train_loader = DataLoader(data_all, batch_size=32, collate_fn = dataset_full.collate )

	for epoch in range(1, 50):
		train_epoch_clas(args.model, model, optimizer, device, pre_train_loader, epoch)

	return 0
class GraphClassificationDataset:
	def __init__(self):
		self.graph_lists = [] 
		self.graph_labels = []
	
	def add(self, g):

		self.graph_lists.append(g)

	def __len__(self):
		return len(self.graphs)

	def __getitem__(self, i):
		return self.graphs[i], self.labels[i]


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

	st_data = None

	st_data =None
	k_hop = 20; D_dim = 4 * k_hop + 1
	pos_enc_size = 5
	sr = [ "sr16622"  ,  "sr251256" ,   "sr261034" ,   "sr281264"  ,"sr291467"  ,  "sr351668"   , "sr351899"  ,  "sr361446"  ,  "sr401224"]
	if args.dataset in sr:
		og_data = SRDataset(root="original_datasets/sr25/", name=args.dataset)
		args.num_classes = 10; 
		num_features =  og_data.num_features
		num_classes = args.num_classes
		print(f'num_classes: {num_classes}, num_features: {num_features}')
		if args.pre_load == 1:
			print(f'processing SRDataset')
			
			print(f' computing k_hop: {k_hop}')
	elif args.dataset in ["graph8c"]:
		og_data = Grapg8cDataset(root="original_datasets/graph8c/", name=args.dataset)
		args.num_classes = 10; num_classes = 10; num_features = og_data.num_features; 
		print(f'num_classes: {num_classes}, num_features: {num_features}')
		if args.pre_load == 1:
			print(f' computing k_hop: {k_hop}')
		
	
	org_datas, dgl_graphs = generate_graphs(og_data, st_data, pos_enc_size, k_hop)


	print(f'num_features: {num_features}, args.num_classes: {num_classes} ')
	print(len(dgl_graphs))
	samples_all = []
	for i in range(len(dgl_graphs)):
		current_data = org_datas[i]
		current_dgl = dgl_graphs[i]
		pair = (current_data, current_dgl)
		samples_all.append(pair)
	
	dataset_full = LoadData(samples_all, args.dataset)
	runs_acc = []
	args.k_hop =k_hop
	for i in tqdm(range(args.run_times)):
		acc = run(i, dataset_full, num_features, pos_enc_size, D_dim, args.num_classes )
		runs_acc.append(acc)

	runs_acc = np.array(runs_acc) * 100

import time
def generate_graphs(og_datas, st_datas, pos_enc_size, k_hop):

	print(f'number of graphs og_datas: {len(og_datas)}')

	graph_lists = []

	for i in range(len(og_datas)):
		og_data = og_datas[i]
		
		if i%100 == 0:
			print(f'processing graph: {i}')
			time.sleep(1)
		if st_datas ==None:
			data_all = og_data
			og_data_O = copy.deepcopy(og_data)
		else:
			st_data = copy.deepcopy(st_datas[i])
			og_data_O = copy.deepcopy(og_data)
			if args.original_edges == 1:
				st_data = copy.deepcopy(st_data)
				e_i = torch.cat((og_data.edge_index, st_data.edge_index), dim=1)

				st_data.edge_color = st_data.edge_color + 1
				e_c = torch.cat((torch.zeros(og_data.edge_index.shape[1], dtype=torch.long), st_data.edge_color), dim=0)
				e_w = torch.cat((torch.ones(og_data.edge_index.shape[1], dtype=torch.float), st_data.edge_weight), dim=0)
				st_data.edge_index = e_i
				st_data.edge_color = e_c
				st_data.edge_weight = e_w
			
				data_all = st_data
			else:
				data_all = og_data

		# num edge before and after.
		G_nx_O = to_networkx(og_data_O, to_undirected=True)
		G_nx = to_networkx(data_all, to_undirected=True)
		N_nx = G_nx.number_of_nodes()
		M, trans_M, trans_logM = getM_logM(G_nx_O, N_nx, args.k_transition)
		M = torch.from_numpy(M).float()
		trans_M = torch.from_numpy(np.array(trans_M)).float()
		trans_logM = torch.from_numpy(np.array(trans_logM)).float()
		
		Eig, Kedge = buildPE_Kindices(data_all, pos_enc_size, trans_M, k_hop)

		Kindices = torch.LongTensor(Kedge)
		D, I, _ = getD_vsI(G_nx_O, N_nx, k_hop)
		D_dim = 4 * k_hop + 1
		D = torch.from_numpy(D)
		
		#try:
		g = g_dgl(og_data_O, Eig, Kindices, D, D_dim, M, I, pos_enc_size)
		graph_lists.append(g)
		N = g.num_nodes()
		if N != N_nx:
			print('Num of nodes is different from Nx and Dgl !!!!')
	
	return og_datas, graph_lists


def LoadData(samples_all, DATASET_NAME):
   
	return MoleculeDataset(samples_all, DATASET_NAME)


from train_molecules_graph_regression import train_epoch_clas, evaluate_network
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="WRGAT/WRGCN (structure + proximity) Experiments")

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Using device: ", device)

	parser.add_argument("--dataset",default="graph8c",  required=False, help="Dataset")
	parser.add_argument("--model", default="Transformer", help="Transformer  Model")
	parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

	# common hyper parameters
	parser.add_argument("--original_edges", type=int, default= 0)
	parser.add_argument("--original_edges_weight", type=float, default=1.0)
	parser.add_argument("--k_eigenvector", type=int, default=10)
	parser.add_argument("--drop", type=float, default=0.5, help="dropout")
	parser.add_argument("--run_times", type=int, default=1)
	parser.add_argument("--num_heads", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=3000)

	# specific params

	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--dims", type=int, default=16, help="hidden dims")
	parser.add_argument("--out_size", type=int, default=16, help="outsize dims")
	parser.add_argument("--k_hop", type=int, default=1)
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--k_transition", type=int, default=6)
	parser.add_argument("--run_times_fine", type=int, default=2000)

	parser.add_argument("--output_path", default="outputs/brazil_10/", help="outputs  Model")
	parser.add_argument("--file_name", default="brazil_10.xlsx", help="file_name  Model")
	parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
	parser.add_argument("--device", default="cuda:0", help="device  Model")
	parser.add_argument("--pre_load", type=int, default=1) 
	parser.add_argument("--alfa", type=float, default=0.5)
	parser.add_argument("--beta", type=float, default=0.5)
	parser.add_argument("--task", default="iso_test")

	parser.add_argument("--num_classes", type=int, default=-1, help="num_classes")
	args = parser.parse_args()
	print(args)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print("Using device: ", device)
	main()