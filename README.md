# Unified Graph Transformer
Unified Graph Transformer (UGT) is a novel Graph Transformer model specialised in preserving both local and global graph structures and developed by [NS Lab, CUK](https://nslab-cuk.github.io/) based on pure [PyTorch](https://github.com/pytorch/pytorch) backend. The paper is available on [on AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/29138).

<p align=center>
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.8.8-3776AB?logo=python&style=flat-square" alt="Python">
  </a>    
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.4-FF6F00?logo=pytorch&style=flat-square" alt="pytorch">
  </a>    
  <img src="https://custom-icon-badges.demolab.com/github/last-commit/NSLab-CUK/Unified-Graph-Transformer?logo=history&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/languages/code-size/NSLab-CUK/Unified-Graph-Transformer?logo=file-code&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/issues-pr-closed/NSLab-CUK/Unified-Graph-Transformer?color=purple&logo=git-pull-request&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/v/tag/NSLab-CUK/Unified-Graph-Transformer?logo=tag&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/stars/NSLab-CUK/Unified-Graph-Transformer?logo=star&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/issues-raw/NSLab-CUK/Unified-Graph-Transformer?logo=issue&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/license/NSLab-CUK/Unified-Graph-Transformer?logo=law&style=flat-square"/>
</p>

<br>


## 1. Overview

Over the past few years, graph neural networks and graph transformers have been successfully used to analyze graph-structured data, mainly focusing on node classification and link prediction tasks. However, the existing studies mostly only consider local connectivity while ignoring long-range connectivity and the roles of nodes. We propose Unified Graph Transformer Networks (**UGT**) that effectively integrate local and global structural information into fixed-length vector representations. **UGT** learns local structure by identifying the local substructures and aggregating features of the k-hop neighborhoods of each node. We construct virtual edges, bridging distant nodes with structural similarity to capture the long-range dependencies. **UGT** learns unified representations through self-attention, encoding structural distance and p-step transition probability between node pairs. Furthermore, we propose a self-supervised learning task that effectively learns transition probability to fuse local and global structural features, which could then be transferred to other downstream tasks. Experimental results on real-world benchmark datasets over various downstream tasks showed that **UGT** significantly outperformed baselines that consist of state-of-the-art models. In addition, **UGT** reaches the third-order Weisfeiler-Lehman power to distinguish non-isomorphic graph pairs.

<br>

<p align="center">
  <img src="./Figures/UGT.jpg" alt="Graph Transformer Architecture" width="800">
  <br>
  <b></b> The overall architecture of Unified Graph Transformer Networks.
</p>


## 2. Reproducibility

### Datasets and Tasks

The package ```Node_level_tasks``` contains the modules required for node clustering and node classification tasks. The package ```Graph_classification``` is for graph classification task. The package ```IsomorphismTesting``` is for Isomorphism testing. 

For the node-level tasks, we used eleven publicly available datasets, which are grouped into three different domains, including Air-traffic networks (e.g., Brazil, Europe, and USA), Webpage networks (e.g., Chameleon, Squirrel, Actor, Cornell, Texas, and Wisconsin), and Citation networks (e.g., Cora and Citeseer). We used four publicly available datasets for the graph classification task, including Enzymes, Proteins, NCI1, and NCI9 from TUDataset.  For isomorphism testing, we used Graph8c and five Strongly Regular Graphs datasets (graph8c, SR251256, SR261034, SR281264, SR401224), which contain 1d-WL and 3d-WL equivalent graph pairs, respectively, for isomorphism testing. The datasets are automatically downloaded.

### Requirements and Environment Setup

The source code developed in Python 3.8.8. UGT are built using Torch-geometric 2.3.1 and DGL 1.1.0. Please refers to official websites for installation and setup.
All the requirements are included in ```environment.yml``` file. 

```
# Conda installation

# Install python environment

conda env create -f environment.yml 
```
### Hyperparameters

Following Options can be passed to exp_10.py

```--dataset:``` The name of datasets. For example: ```--dataset cora```

```--lr:``` Learning rate for training the model. For example: ```--lr 0.001```

```--epochs:``` Number of epochs for training the model. For example: ```--epochs 1000```

```--layers:``` Number of layers for training the model. For example: ```--layers 4```

```--task:``` The specific task. For example: ```--task pre_training```.

```--pre_load:``` The processing mode, including pre-processing and training. For example: ```--pre_load 0``` for pre-processing graph.

```--dims:``` The dimmension of hidden vectors.  For example: ```--dims 16```.

```--k_transition:``` The number of transition step. For example: ```--k_transition 6```.

```--k_hop:``` The number of hop for sasmpling. For example: ```--k_hop 2```.

```--alpha:``` Hyperparameters for transition construction loss. For example: ```--alpha 0.4```.

```--beta:``` Hyperparameters for feature construction loss. For example: ```--beta 0.6```.

### Node-level Tasks

The the source code for node clustering and node classification are included in the ```Node_level_tasks``` folder. 
We do experiments on eleven public benchmark datasets, including cora, citeseer, brazil, europe, usa, chameleon, squirrel, film, cornell,  texas, and wisconsin. The datasets are automatically downloaded.
Note that one can test node classification and node clustering task followed by pre-training tasks. 
We sample k-hop neighbourhood and virtual edges in the pre-processing step and store the processed data in the ```pts``` and ```outputs``` folders. For example, one can first pre-train ***UGT*** on cora dataset and then test on downstream tasks, such as node classification and node clustering:

```
cd Node_level_tasks

# Pre-preprocessing data

python exp_10.py --dataset cora --pre_load 1

# Pre_training

python exp_10.py --dataset cora --task pre_training --pre_load 0

# Running Node classification task:

python exp_10.py --dataset cora --task node_classification --pre_load 0 --lr 0.001 --dims 16 --k_hop 1 --num_layers 2 --k_transition 6 --alfa 0.5 --beta 0.5

# Running Node clustering task:

python exp_10.py --dataset cora --task node_clustering --pre_load 0 --lr 0.001 --dims 16 --k_hop 1 --num_layers 2 --k_transition 6 --alfa 0.5 --beta 0.5
```

### Graph-level Classification Task

The the source code for Graph-level classification task is included in the ```Graph_classification``` folder. 
We do experiments on four public benchmark datasets, including Enzymes, Proteins, NCI1, and NCI9 from TUDataset. The datasets are automatically downloaded. For example, one can run UGT on Proteins dataset:
```
cd Graph_classification

# Pre-processing:

python exp_10.py --dataset PROTEINS --pre_load 1 --task graph_classification

# Graph classification task:

python exp_10.py --dataset PROTEINS --pre_load 0 --task graph_classification
```

### Isomorphism Testing

There are five graph datasets, including Graph8c and five Strongly Regular Graphs datasets (SR251256, SR261034, SR281264, SR401224), which contain 1d-WL and 3d-WL equivalent graph pairs, respectively. For example, one can test the power of UGT on sr16622 dataset: 

```
cd IsomorphismTesting

python exp_10.py --dataset sr16622 --task iso_test
```

## 3. Reference

:page_with_curl: Paper [on AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/index): 
* [![DOI](http://img.shields.io/:DOI-10.1609/aaai.v38i11.29138-blue?style=flat-square)](https://doi.org/10.1609/aaai.v38i11.29138) 

:page_with_curl: Paper [on arXiv](https://arxiv.org/): 
* [![arXiv](https://img.shields.io/badge/arXiv-2308.09517-b31b1b?style=flat-square&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2308.09517) 

:chart_with_upwards_trend: Experimental results [on Papers With Code](https://paperswithcode.com/): 
* [![PwC](https://custom-icon-badges.demolab.com/badge/Papers%20With%20Code-UGT-21CBCE?style=flat-square&logo=paperswithcode)](https://paperswithcode.com/paper/transitivity-preserving-graph-representation)

:pencil: Blog post [on Network Science Lab](https://nslab-cuk.github.io/2023/08/17/UGT/): 
* [![Web](https://img.shields.io/badge/NS@CUK-Post-0C2E86?style=flat-square&logo=jekyll&logoColor=FFFFFF)](https://nslab-cuk.github.io/2023/08/17/UGT/)


## 4. Citing UGT

Please cite our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29138) if you find *UGT* useful in your work:
```
@InProceedings{Hoang_2024,
  author    = {Hoang, Van Thuy and Lee, O-Joun},
  booktitle = {Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI 2024)},
  title     = {Transitivity-Preserving Graph Representation Learning for Bridging Local Connectivity and Role-Based Similarity},
  year      = {2024},
  address   = {Vancouver, Canada},
  editor    = {Michael Wooldridge and Jennifer Dy and Sriraam Natarajan},
  month     = feb,
  number    = {11},
  pages     = {12456--12465},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  volume    = {38},
  doi       = {10.1609/aaai.v38i11.29138},
  issn      = {2159-5399},
  url       = {https://doi.org/10.1609/aaai.v38i11.29138},
}
```

Please take a look at our community-aware graph transformer model, [**CGT**](https://github.com/NSLab-CUK/Community-aware-Graph-Transformer), which can mitigate degree bias problem of message passing mechanism, together. 

## 5. Contributors

<a href="https://github.com/NSLab-CUK/Unified-Graph-Transformer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NSLab-CUK/Unified-Graph-Transformer" />
</a>



<br>

***

<a href="https://nslab-cuk.github.io/"><img src="https://github.com/NSLab-CUK/NSLab-CUK/raw/main/Logo_Dual_Wide.png"/></a>

***




