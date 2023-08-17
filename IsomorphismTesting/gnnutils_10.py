import copy
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
# Import KMeans module
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn.metrics.cluster import contingency_matrix
from scipy import sparse
from sklearn.metrics.cluster import normalized_mutual_info_score
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn import preprocessing

import networkx as nx
import numpy as np
import scipy.sparse as sparse
import torch_geometric as tg
import torch
import torch.nn.functional as F
from networkx.utils import dict_to_numpy_array
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid,  Amazon, WikipediaNetwork
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm
import math
from metrics import accuracy_score, precision, modularity, conductance, recall
import networkx as nx
from torch_geometric.utils import to_networkx
from build_multigraph import build_pyg_struc_multigraph
from datasets import WebKB, WikipediaNetwork, FilmNetwork, BGP, Airports
from torch_geometric.datasets import TUDataset

def load_airports(dataset):
    assert dataset in ["brazil", "europe", "usa"]
    og = Airports("original_datasets/airports_dataset/" + dataset, dataset_name=dataset)[0]
    st = Airports(root="datasets_py_geom_format_10/airports_dataset/" + dataset, dataset_name=dataset,
                  pre_transform=build_pyg_struc_multigraph)[0]
    return og, st


def load_bgp(dataset):
    assert dataset in ["bgp"]
    og = BGP(root="original_datasets/bgp_dataset")[0]
    st = BGP(root="datasets_py_geom_format_10/bgp_dataset", pre_transform=build_pyg_struc_multigraph)[0]
    return og, st


def load_film(dataset):
    assert dataset in ["film"]
    og = FilmNetwork(root="original_datasets/film", name=dataset)[0]
    st = FilmNetwork(root="datasets_py_geom_format_10/film", name=dataset,
                     pre_transform=build_pyg_struc_multigraph)[0]
    return og, st


def load_wiki(dataset):
    assert dataset in ["chameleon", "squirrel"]
    og = WikipediaNetwork(root="original_datasets/wiki", name=dataset)[0]
    st = WikipediaNetwork(root="datasets_py_geom_format_10/wiki", name=dataset,
                          pre_transform=build_pyg_struc_multigraph)[0]

    return og, st


def load_ENZYMES(dataset):

    assert dataset in ["ENZYMES"]
    for i in range(len(dataset)):    
        og = TUDataset(root="original_datasets/ENZYMES", name='ENZYMES')[i]
        st = TUDataset(root="datasets_py_geom_format_10/ENZYMES", name = dataset , pre_transform=build_pyg_struc_multigraph)[i]
        
    return og, st

def load_crocodile(dataset):
    assert dataset in ["crocodile"]
    og = tg.datasets.WikipediaNetwork(root="original_datasets/wiki", name=dataset, geom_gcn_preprocess =False)[0]
    st = tg.datasets.WikipediaNetwork(root="datasets_py_geom_format_10/wiki", name=dataset,geom_gcn_preprocess=False, pre_transform=build_pyg_struc_multigraph)[0]

    return og, st

def load_amazon(dataset):
    assert dataset in ["Computers", "Photo"]
    og = Amazon(root="original_datasets/amazon", name=dataset)[0]
    st = Amazon(root="datasets_py_geom_format_10/amazon", name=dataset,pre_transform=build_pyg_struc_multigraph)[0]

    return og, st


def load_webkb(dataset):
    assert dataset in ["cornell", "texas", "wisconsin"]
    og = WebKB(root="original_datasets/webkb", name=dataset)[0]
    st = WebKB(root="datasets_py_geom_format_10/webkb", name=dataset,
               pre_transform=build_pyg_struc_multigraph)[0]

    return og, st


def load_planetoid(dataset):
    assert dataset in ["cora", "citeseer", "pubmed"]
    og = Planetoid(root="original_datasets/planetoid", name=dataset, split="public")[0]

    st = Planetoid(root="datasets_py_geom_format_10/planetoid", name=dataset, split="public", pre_transform=build_pyg_struc_multigraph)[0]

    return og, st


    return og, st
def load_karate(dataset):
    from torch_geometric.datasets import KarateClub
    og = KarateClub()[0]
    st = KarateClub(transform=build_pyg_struc_multigraph)[0]
def structure_edge_weight_threshold(data, threshold):
    data = copy.deepcopy(data)
    mask = data.edge_weight > threshold
    data.edge_weight = data.edge_weight[mask]
    data.edge_index = data.edge_index[:, mask]
    data.edge_color = data.edge_color[mask]
    return data


def add_original_graph(og_data, st_data, weight=1.0):
    st_data = copy.deepcopy(st_data)
    e_i = torch.cat((og_data.edge_index, st_data.edge_index), dim=1)
    st_data.edge_color = st_data.edge_color + 1
    e_c = torch.cat((torch.zeros(og_data.edge_index.shape[1], dtype=torch.long), st_data.edge_color), dim=0)
    e_w = torch.cat((torch.ones(og_data.edge_index.shape[1], dtype=torch.float) * weight, st_data.edge_weight), dim=0)
    st_data.edge_index = e_i
    st_data.edge_color = e_c
    st_data.edge_weight = e_w

    return st_data


def create_target_matrix(A):

    A_hat = sparse.coo_matrix(A_hat.dot(A))

    scores = np.log(A_hat.data) - math.log(A.shape[0])
    rows = A_hat.row[scores < 0]
    cols = A_hat.col[scores < 0]
    scores = scores[scores < 0]

    target_matrix = sparse.coo_matrix((scores, (rows, cols)), shape=A.shape, dtype=np.float32)
    return target_matrix


def train_finetuning_class(model, train_data, mask, optimizer, device, g, A_k, D, Kindices, de, M, I, trans_logM,pre_train, current_epoch):
    # pre_train = 0
    model.train()
    optimizer.zero_grad()
    mask = mask.to(device)
    true_label = train_data.y.to(device)

    out = model( g.to(device),current_epoch )

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out[mask], true_label[mask])

    pred = out.max(1)[1].to(device)[mask]
    acc = pred.eq(train_data.y.to(device)[mask]).sum().item() / len(train_data.y.to(device)[mask])
    loss.backward()
    optimizer.step()
    return loss.item(), acc


@torch.no_grad()
def test_cluster(model, test_data, mask, device, g, A_k, D, Kindices, de, M, I):
    model.eval()
    mask = mask.to(device)
    logits = model( g.to(device), -1)

    pred = torch.argmax(logits, dim=1)
    pred = pred.to(device)[mask]

    pred = pred.cpu().numpy()
    y_true = test_data.y.to(device)[mask].cpu().numpy()
    f1 = f1_score(y_true, pred, average='micro')

    G_st = to_networkx(test_data, to_undirected=True)
    Adj = nx.adjacency_matrix(G_st)
    Adj = Adj.todense()
  

    acc = accuracy_score(y_true, pred)
    p = precision(y_true, pred)
    r = recall(y_true, pred)
    nmi= normalized_mutual_info_score(y_true, pred)
    q = modularity(Adj, pred)
    c = conductance(Adj, pred)

    return acc, p,r, nmi, q, c

def train_finetuning_cluster(model, train_data, mask, optimizer, device, g, A_k, D, Kindices, de, M, I, trans_logM, pre_train, adj, d, n_edges):

    model.train()
    optimizer.zero_grad()
    mask = mask.to(device)
    k = torch.numel(train_data.y.unique())
    C = model( g.to(device), -1 )
    n = g.number_of_nodes()
    adj = torch.FloatTensor(adj).to(device)

    d = torch.FloatTensor(d).unsqueeze(1).to(device)
    C_t = C.t()
    graph_pooled = torch.mm(C_t, adj)
    graph_pooled = torch.mm(graph_pooled, C)

    normalizer_left = torch.mm(C_t, d)

    normalizer_right = torch.mm(d.t(), C)

    normalizer = torch.mm(normalizer_left, normalizer_right) / 2 / n_edges

    spectral_loss = - torch.trace(graph_pooled - normalizer) / 2 / n_edges

    cluster_sizes = torch.sum(C, axis=0)  # Size [k]
    collapse_loss = (k / (2 * n)) * (torch.norm(cluster_sizes))

    Identity = torch.eye(g.number_of_nodes()).to(device)

    loss = spectral_loss + collapse_loss

    loss.backward()
    optimizer.step()
    return loss.item()



def NormalizeTensor(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))
def cosinSim(x_hat):
    x_norm = torch.norm(x_hat, p=2, dim=1)
    nume = torch.mm(x_hat, x_hat.t())
    deno = torch.ger(x_norm, x_norm)
    cosine_similarity = nume / deno
    return cosine_similarity


def train(model, train_data, mask, optimizer, device, g, A_k, D, Kindices, de, M, I, trans_logM, pre_train, trans_M, k_transition, current_epoch, alfa, beta ):
    model.train()
    optimizer.zero_grad()
    mask = mask.to(device)

    h, x_hat = model( g.to(device), k_transition, current_epoch)


    loss_M = 0
    for i in range(k_transition):
        loss_M += torch.sum(((cosinSim(h) - (torch.FloatTensor(trans_logM[i])).to(device)) ** 2))

    row_num, col_num = (torch.FloatTensor(trans_logM[i])).size()
    loss_M = loss_M / (k_transition* row_num * col_num)

    row_num, col_num = train_data.x.size()

    
    loss_X = F.mse_loss(x_hat, train_data.x.to(device) )

    loss_all = loss_M * alfa + loss_X * beta
    loss_all.backward(); optimizer.step()

    return loss_M.item()  # , loss_X.item()


@torch.no_grad()
def test(model, test_data, mask, device, g, A_k, D, Kindices, de, M, I, current_epoch):
    model.eval()
    mask = mask.to(device)
    # n x 4 size
    logits = model(g.to(device),current_epoch)

    pred = torch.argmax(logits, dim=1)
    pred = pred.to(device)[mask]

    acc = pred.eq(test_data.y.to(device)[mask]).sum().item() / len(test_data.y.to(device)[mask])

    pred = pred.cpu().numpy()
    y_true = test_data.y.to(device)[mask].cpu().numpy()
    f1 = f1_score(y_true, pred, average='micro')


    return acc, f1


def calculate_accuracy(predicted_labels, actual_labels):
    predictions = predicted_labels.max(1)[1].type_as(actual_labels)

    results = predictions.eq(actual_labels).double().sum()

    return results / len(actual_labels)


def filter_relations(data, num_relations, rel_last):
    if rel_last:
        l = data.edge_color.unique(sorted=True).tolist()
        mask_l = l[-num_relations:]
        mask = data.edge_color == mask_l[0]
        for c in mask_l[1:]:
            mask = mask | (data.edge_color == c)
    else:
        mask = data.edge_color < (num_relations + 1)

    data.edge_index = data.edge_index[:, mask]
    data.edge_weight = data.edge_weight[mask]
    data.edge_color = data.edge_color[mask]
    return data


def make_masks(data, val_test_ratio=0.2, stratify=False):
    data = copy.deepcopy(data)
    n_nodes = data.x.shape[0]
    all_nodes_idx = np.arange(n_nodes)
    all_y = data.y.numpy()
    if stratify:
        train, test_idx, y_train, _ = train_test_split(all_nodes_idx, all_y, test_size=0.2, stratify=all_y)

        train_idx, val_idx, _, _ = train_test_split(
            train, y_train, test_size=0.25, stratify=y_train)

    else:
        val_test_num = 2 * int(val_test_ratio * data.x.shape[0])
        val_test_idx = np.random.choice(n_nodes, (val_test_num,), replace=False)
        val_idx = val_test_idx[:int(val_test_num / 2)]
        test_idx = val_test_idx[int(val_test_num / 2):]

    val_mask = np.zeros(n_nodes)
    val_mask[val_idx] = 1
    test_mask = np.zeros(n_nodes)
    test_mask[test_idx] = 1
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    val_test_mask = val_mask | test_mask
    train_mask = ~ val_test_mask
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def create_self_loops(data):
    orig_relations = len(data.edge_color.unique())
    data.edge_index, data.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_weight,
                                                                 fill_value=1.0)
    row, col = data.edge_index[0], data.edge_index[1]
    mask = row == col
    tmp = torch.full(mask.nonzero().shape, orig_relations + 1, dtype=torch.long).squeeze()
    data.edge_color = torch.cat([data.edge_color, tmp], dim=0)
    return data


# create adjacency matrix and degree sequence
def createA(E, n, m, undir=True):
    if undir:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(list(E))
        A = nx.to_scipy_sparse_matrix(G)
    else:
        A = sparse.coo_matrix((np.ones(m), (E[:, 0], E[:, 1])), shape=(n, n)).tocsc()
    degree = np.array(A.sum(1)).flatten()

    return A, degree


def calculateRWRrange(A, degree, i, prs, n, trans=True, maxIter=1000):
    pr = prs[-1]
    D = sparse.diags(1. / degree, 0, format='csc')
    W = D * A
    diff = 1
    it = 1

    F = np.zeros(n)
    Fall = np.zeros((n, len(prs)))
    F[i] = 1
    Fall[i, :] = 1
    Fold = F.copy()
    T = F.copy()

    if trans:
        W = W.T

    oneminuspr = 1 - pr

    while diff > 1e-9:
        F = pr * W.dot(F)
        F[i] += oneminuspr
        Fall += np.outer((F - Fold), (prs / pr) ** it)
        T += (F - Fold) / ((it + 1) * (pr ** it))

        diff = np.sum((F - Fold) ** 2)
        it += 1
        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0
        Fold = F.copy()

    return Fall, T, it


def localAssortF(G, M, pr=np.arange(0., 1., 0.1), undir=True, missingValue=-1, edge_attribute=None):
    n = len(M)
    ncomp = (M != missingValue).sum()
    # m = len(E)
    m = G.number_of_edges()

    # A, degree = createA(E, n, m, undir)
    if edge_attribute is None:
        A = nx.to_scipy_sparse_matrix(G, weight=None)
    else:
        A = nx.to_scipy_sparse_matrix(G, weight=edge_attribute)

    degree = np.array(A.sum(1)).flatten()

    D = sparse.diags(1. / degree, 0, format='csc')
    W = D.dot(A)
    c = len(np.unique(M))
    if ncomp < n:
        c -= 1

    Z = np.zeros(n)
    Z[M == missingValue] = 1.
    Z = W.dot(Z) / degree

    values = np.ones(ncomp)
    yi = (M != missingValue).nonzero()[0]
    yj = M[M != missingValue]
    Y = sparse.coo_matrix((values, (yi, yj)), shape=(n, c)).tocsc()

    assortM = np.empty((n, len(pr)))
    assortT = np.empty(n)

    eij_glob = np.array(Y.T.dot(A.dot(Y)).todense())
    eij_glob /= np.sum(eij_glob)
    ab_glob = np.sum(eij_glob.sum(1) * eij_glob.sum(0))

    WY = W.dot(Y).tocsc()

    print("start iteration")

    for i in tqdm(range(n)):
        pis, ti, it = calculateRWRrange(A, degree, i, pr, n)
    
        YPI = sparse.coo_matrix((ti[M != missingValue], (M[M != missingValue],
                                                         np.arange(n)[M != missingValue])),
                                shape=(c, n)).tocsr()
        e_gh = YPI.dot(WY).toarray()
        Z[i] = np.sum(e_gh)
        e_gh /= np.sum(e_gh)
        trace_e = np.trace(e_gh)
        assortT[i] = trace_e

    assortT -= ab_glob
    assortT /= (1. - ab_glob + 1e-200)

    return assortM, assortT, Z


def mixing_dict(xy, normalized=True):
    d = {}
    psum = 0.0
    for x, y, w in xy:
        if x not in d:
            d[x] = {}
        if y not in d:
            d[y] = {}
        v = d[x].get(y, 0)
        d[x][y] = v + w
        psum += w

    if normalized:
        for k, jdict in d.items():
            for j in jdict:
                jdict[j] /= psum
    return d


def node_attribute_xy(G, attribute, edge_attribute=None, nodes=None):
    if nodes is None:
        nodes = set(G)
    else:
        nodes = set(nodes)
    Gnodes = G.nodes
    for u, nbrsdict in G.adjacency():
        if u not in nodes:
            continue
        uattr = Gnodes[u].get(attribute, None)
        if G.is_multigraph():
            raise NotImplementedError
        else:
            for v, eattr in nbrsdict.items():
                vattr = Gnodes[v].get(attribute, None)
                if edge_attribute is None:
                    yield (uattr, vattr, 1)
                else:
                    edge_data = G.get_edge_data(u, v)
                    yield (uattr, vattr, edge_data[edge_attribute])


def global_assortativity(networkx_graph, labels, weights=None):
    attr_dict = {}
    for i in networkx_graph.nodes():
        attr_dict[i] = labels[i]

    nx.set_node_attributes(networkx_graph, attr_dict, "label")
    if weights is None:
        xy_iter = node_attribute_xy(networkx_graph, "label", edge_attribute=None)
        d = mixing_dict(xy_iter)
        M = dict_to_numpy_array(d, mapping=None)
        s = (M @ M).sum()
        t = M.trace()
        r = (t - s) / (1 - s)
    else:
        edge_attr = {}
        for i, e in enumerate(networkx_graph.edges()):
            edge_attr[e] = weights[i]
        nx.set_edge_attributes(networkx_graph, edge_attr, "weight")

        xy_iter = node_attribute_xy(networkx_graph, "label", edge_attribute="weight")
        d = mixing_dict(xy_iter)
        M = dict_to_numpy_array(d, mapping=None)
        s = (M @ M).sum()
        t = M.trace()
        r = (t - s) / (1 - s)

    return r, M


def local_assortativity(networkx_graph, labels, weights=None):
    if weights is None:
        assort_m, assort_t, z = localAssortF(networkx_graph, np.array(labels))
    else:
        edge_attr = {}
        for i, e in enumerate(networkx_graph.edges()):
            edge_attr[e] = weights[i]
        nx.set_edge_attributes(networkx_graph, edge_attr, "weight")
        assort_m, assort_t, z = localAssortF(networkx_graph, np.array(labels), edge_attribute="weight")

    return assort_m, assort_t, z
