"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import numpy as np
from metrics import MAE
loss_func = nn.CrossEntropyLoss()
from metrics import accuracy_TU as accuracy
M = 0
def train_epoch_iso(model_name,dataset, model, optimizer, device, data_loader, epoch, dataset_name):

    model.train()
    global M
    embeddings=[]
    model.eval()
    for data in data_loader:
        data =data.to(device)
        pre = model.forward(data )

        embeddings.append(pre)



    E = torch.cat(embeddings).cpu().detach().numpy()    
    M = M + 1*((np.abs(np.expand_dims(E,1)-np.expand_dims(E,0))).sum(2)>0.001)
    sm = ((M==0).sum()-M.shape[0])/2


    print('similar:',sm)
    file = open("Baselines/results.txt", "a")
    final_msg = "model_name: " + model_name + ", ds name: " + dataset_name+  " #graph: " +   str(len(dataset)) + " similarity: " +  str(sm)   +       "\n"
    file.write(final_msg)
    file.close()

def train_epoch_clas(model_name, model, optimizer, device, data_loader, epoch):
    print(f'current epoch: {epoch}')
    model.train()
    global M
    embeddings=[]
    model.eval()
    for iter, (batch_data, batch_dgls) in enumerate(data_loader):
        torch.manual_seed(iter)
        batch_graphs = batch_dgls.to(device)
        batch_x = batch_graphs.ndata['x'].to(device)  # num x feat
        
        batch_PE = batch_graphs.ndata['PE'].to(device)  # num x feat

        batch_I = batch_graphs.ndata['I'].to(device)  # num x feat

        batch_de = batch_graphs.edata['de'].to(device)  # num x feat
        batch_m = batch_graphs.edata['m'].to(device)  # num x feat

        if model_name == "Transformer":
            pre = model.forward(batch_graphs, batch_x, batch_PE, batch_de, batch_m, batch_I )
        if  model_name == "GraphT_baseline" or model_name == "SAN": #GT model
             pass

        embeddings.append(pre)

    E = torch.cat(embeddings).cpu().detach().numpy()    
    M = M+1*((np.abs(np.expand_dims(E,1)-np.expand_dims(E,0))).sum(2)   >   0.001)
    sm = ((M==0).sum()-M.shape[0])/2
    print('similar:',sm)

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
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

