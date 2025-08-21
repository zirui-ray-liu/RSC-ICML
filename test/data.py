import imp
from typing import Tuple
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (Yelp, Flickr, Reddit2, Reddit)
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from torch import Tensor
#from fastgnn import rabbit_reorder


def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def get_arxiv(root: str, reorder: bool) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB')
    data = dataset[0]
    # data.adj_t = data.adj_t.to_symmetric()
    data.edge_index = to_undirected(data.edge_index)
    data.x = data.x.contiguous()
    data.node_year = None
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    data.train_idx = split_idx['train']
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str, reorder: bool) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB')
    data = dataset[0]
    data.x = data.x.contiguous()
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes

def get_proteins(root: str, reorder: bool) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-proteins', f'{root}/OGB', transform=T.ToSparseTensor())
    data = dataset[0]
    #import pdb
    #pdb.set_trace()
    data.x = data.adj_t.mean(dim=1) # add node features from edge features
    data.adj_t.set_value_(None)
    #data.x = data.x.contiguous()
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, data.num_features, 112 # the number of node classes, from ogb examples
    

def get_yelp(root: str, reorder: bool) -> Tuple[Data, int, int]:
    dataset = Yelp(f'{root}/YELP')
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str, reorder: bool) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr')
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str, reorder: bool) -> Tuple[Data, int, int]:
    dataset = Reddit(f'{root}/Reddit')
    data = dataset[0]
    #if reorder:
    #    data = rabbit_reorder(data)
    return data, dataset.num_features, dataset.num_classes

def get_reddit2(root: str, reorder: bool) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2')
    data = dataset[0]
    #if reorder:
    #    data = rabbit_reorder(data)
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes

def get_data(root: str, name: str, reorder: bool=False) -> Tuple[Data, int, int]:
    if name.lower() == 'reddit':
        return get_reddit(root, reorder)
    elif name.lower() == 'reddit2':
        return get_reddit2(root, reorder)
    elif name.lower() == 'flickr':
        return get_flickr(root, reorder)
    elif name.lower() == 'yelp':
        return get_yelp(root, reorder)
    elif name.lower() == 'arxiv':
        return get_arxiv(root, reorder)
    elif name.lower() == 'products':
        return get_products(root, reorder)
    elif name.lower() == 'proteins':
        return get_proteins(root, reorder)
    else:
        raise NotImplementedError