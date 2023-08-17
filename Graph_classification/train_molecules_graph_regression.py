
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
loss_func = nn.CrossEntropyLoss()
from metrics import accuracy_TU as accuracy
def cosinSim(x_hat):
    x_norm = torch.norm(x_hat, p=2, dim=1)
    nume = torch.mm(x_hat, x_hat.t())
    deno = torch.ger(x_norm, x_norm)
    cosine_similarity = nume / deno
    return cosine_similarity
def train_epoch(model, optimizer, device, data_loader, epoch, k_transition,  alfa, beta):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets, batch_trans_logM) in enumerate(data_loader):

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].to(device)  # num x feat
        batch_PE = batch_graphs.ndata['PE'].to(device)  # num x feat
        
        batch_I = batch_graphs.ndata['I'].to(device)  # num x feat

        batch_de = batch_graphs.edata['de'].to(device)              # num x feat
        batch_m = batch_graphs.edata['m'].to(device)  # num x feat
        optimizer.zero_grad()

        batch_targets = batch_targets.to(device)

        h,  x_hat = model.forward(batch_graphs, batch_x, batch_PE, batch_de, batch_m, batch_I )
        
        loss_M = 0
        cos_h = cosinSim(h) 
        for i in range(k_transition):
            m =  torch.FloatTensor(batch_trans_logM[0][i])

            loss_M += torch.sum((cos_h-  m.to(device)) ** 2)
        row_num, col_num = cos_h.size()
        loss_M = loss_M / (k_transition* row_num * col_num)

        row_num, col_num = x_hat.size()
        
        loss_X = F.mse_loss(x_hat, batch_x )

        loss_all = loss_M * alfa + loss_X * beta

        loss_all.backward()
        optimizer.step()
        epoch_loss += loss_all.detach().item()
       
    return loss_all, epoch_train_acc, optimizer


def train_epoch_graph_classification(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets, _) in enumerate(data_loader):

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].to(device)  # num x feat
        batch_PE = batch_graphs.ndata['PE'].to(device)  # num x feat

        batch_I = batch_graphs.ndata['I'].to(device)  # num x feat

        batch_de = batch_graphs.edata['de'].to(device)  # num x feat
        batch_m = batch_graphs.edata['m'].to(device)  # num x feat
        optimizer.zero_grad()

        try:
            sign_flip = torch.rand(batch_PE.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            batch_PE = batch_PE * sign_flip.unsqueeze(0)
            batch_PE = batch_PE.to(device)
        except:
            batch_PE = batch_graphs.ndata['PE'].to(device)  # num x feat
        

        batch_targets = batch_targets.to(device)

        batch_scores = model.forward(batch_graphs, batch_x, batch_PE, batch_de, batch_m, batch_I )

        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))
        loss = model.loss(batch_scores, batch_targets)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
       
        epoch_train_acc += accuracy(batch_scores, batch_targets)
       
        nb_data += batch_targets.size(0)

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer
def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, _) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            
            batch_x = batch_graphs.ndata['x'].to(device)    # num x feat
            batch_PE = batch_graphs.ndata['PE'].to(device)  # num x feat
            batch_de = batch_graphs.edata['de'].to(device)  # num x feat
            batch_m = batch_graphs.edata['m'].to(device)    # num x feat
            batch_I = batch_graphs.ndata['I'].to(device)    # num x feat
            
            batch_targets = batch_targets.to(device)
           
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_PE, batch_de, batch_m, batch_I )

            loss = model.loss(batch_scores, batch_targets)

            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data 
        
    return epoch_test_loss, epoch_test_acc

