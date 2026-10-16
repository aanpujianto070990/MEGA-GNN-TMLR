import json, itertools
import sys, os
import os.path as osp
import pandas as pd
import numpy as np
try:
    # Only needed to format the raw Jodie files; optional so that importing
    # MegaGNN (which imports every dataset module) works without datatable.
    import datatable as dt
    from datatable import f,join,sort
except ImportError:
    dt = None
from datetime import datetime
from collections import defaultdict
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import index_to_mask
from torch_geometric.data import download_url

from .temporal_dataset import TemporalDataset

from MegaGNN.cpp import ports_cpp
from MegaGNN.graphgym.config import cfg



def ports_with_cpp(graph, permute_ports=False):    
    edge_index = graph['node', 'to', 'node'].edge_index
    timestamp = graph['node', 'to', 'node'].timestamps

    if permute_ports:
        permutation = torch.randperm(timestamp.size(0))
        timestamp = timestamp[permutation]

    edges = torch.cat([edge_index.T, timestamp.reshape((-1,1))], dim=1).numpy().astype('int')
    ports_in, ports_out = ports_cpp.assign_ports(edges,edge_index.numpy().astype('int'), graph.num_nodes)
    
    return torch.from_numpy(ports_in), torch.from_numpy(ports_out)


def find_parallel_edges(edge_index, edge_types=None):
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    
    if edge_types is not None:
        # If edge types are provided, consider both edge and its type
        for idx, edge in enumerate(edge_index.T):
            tuple_edge_with_type = (*edge.tolist(), edge_types[idx].item() if torch.is_tensor(edge_types[idx]) else edge_types[idx])
            if tuple_edge_with_type not in simplified_edge_mapping:
                simplified_edge_mapping[tuple_edge_with_type] = i
                i += 1
            simplified_edge_batch.append(simplified_edge_mapping[tuple_edge_with_type])
    else:
        # Original behavior when edge types are not provided
        for edge in edge_index.T:
            tuple_edge = tuple(edge.tolist())
            if tuple_edge not in simplified_edge_mapping:
                simplified_edge_mapping[tuple_edge] = i
                i += 1
            simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
            
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)
    return simplified_edge_batch

class JodieDataset(TemporalDataset):
    url = 'http://snap.stanford.edu/jodie/{}.csv'
    names = ['reddit', 'wikipedia', 'mooc']
    def __init__(self, root: str, name: str, 
                 reverse_mp: bool = False,
                 add_ports: bool = False,
                 multi_edge_agg: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name 
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        self.multi_edge_agg = multi_edge_agg
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0])
        
        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                del self.data_dict[split]['node', 'rev_to', 'node']

        if add_ports:
            print("Original Ports are loaded!")
            self.ports_dict = torch.load(self.processed_paths[1])
            
            for split in ['train', 'val', 'test']:
                if cfg.dataset.fraudgt_ports:
                    self.data_dict[split] = self.add_ports_func_fraudgt(self.data_dict[split], self.ports_dict[split])
                else:
                    self.data_dict[split] = self.add_ports_func(self.data_dict[split], self.ports_dict[split])
        
        if multi_edge_agg:
            self.simp_edge_batch_dict = torch.load(self.processed_paths[2]) 
            for split in ['train', 'val', 'test']:
                for edge_type in self.data_dict[split].edge_types:
                    self.data_dict[split][edge_type].simp_edge_batch = self.simp_edge_batch_dict[split]
            
    def add_ports_func(self, data, ports):

        in_ports, out_ports = ports
        ports_arr = torch.stack([in_ports, out_ports], dim=1)
        data['node', 'to', 'node'].edge_attr = torch.cat([data['node', 'to', 'node'].edge_attr, ports_arr], dim=1)
        if self.reverse_mp:
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([data['node', 'rev_to', 'node'].edge_attr, ports_arr[:, [1, 0]]], dim=1)
        return data

    def add_ports_func_fraudgt(self, data, ports):
        if not self.reverse_mp:
            in_ports, out_ports = ports
            out_ports = [out_ports]
            data['node', 'to', 'node'].edge_attr = \
                torch.cat([data['node', 'to', 'node'].edge_attr, in_ports] + out_ports, dim=1)
        else:
            '''Adds port numberings to the edge features'''
            in_ports, out_ports = ports
            data['node', 'to', 'node'].edge_attr = torch.cat([data['node', 'to', 'node'].edge_attr, in_ports.reshape(-1,1)], dim=1)
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([data['node', 'rev_to', 'node'].edge_attr, out_ports.reshape(-1,1)], dim=1)
        return data

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test']
        # return [osp.join(self.names[self.name], f) for f in x]
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return ['data.pt', 'ports.pt', 'simp_edge_batch.pt']
    
    def download(self) -> None:
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):

        df = pd.read_csv(self.raw_paths[0], skiprows=1, header=None)
        df = df.sort_values(by=df.columns[2])
        
        src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
        dst += int(src.max()) + 1
        edge_index = torch.stack([src, dst], dim=0)

        timestamps = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
        y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
        edge_attr = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float)

        num_nodes = max(src.max(), dst.max()) + 1
        num_edges = edge_index.shape[1]

        x = torch.ones(num_nodes, 1).float()

        print(f"Positive label ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
        print(f"Number of nodes = {num_nodes}")
        print(f"Number of edges = {edge_index.shape[0]}")

        simp_edge_batch = find_parallel_edges(edge_index)

        splits = [0.6, 0.2, 0.2]

        train_end = int(num_edges * 0.6)
        val_end = int(num_edges * 0.8)

        train_inds = torch.arange(train_end)
        val_inds = torch.arange(train_end, val_end)
        test_inds = torch.arange(val_end, num_edges)
        
        e_train = train_inds
        e_val = torch.cat([train_inds, val_inds])
        e_test = torch.cat([train_inds, val_inds, test_inds])

        
        self.ports_dict = {}
        self.simp_edge_batch_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            e_mask = eval(f'e_{split}')

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr = edge_attr[e_mask]
            masked_y = y[e_mask]
            masked_timestamps = timestamps[e_mask]

            data = HeteroData()
            data['node'].x = x
            data['node'].num_nodes = int(x.shape[0])
            data['node', 'to', 'node'].edge_index = masked_edge_index
            data['node', 'to', 'node'].edge_attr = masked_edge_attr
            # We use "y" here so LinkNeighborLoader won't mess up the edge label
            data['node', 'to', 'node'].y = masked_y
            data['node', 'to', 'node'].timestamps = masked_timestamps
            data['node', 'to', 'node'].split_mask = index_to_mask(inds, size=masked_edge_index.shape[1])

            data['node', 'rev_to', 'node'].edge_index = masked_edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = masked_edge_attr
            
            # For megagnn
            masked_simp_edge_batch = simp_edge_batch[e_mask]
        
            in_ports, out_ports = ports_with_cpp(data)
            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data
            self.simp_edge_batch_dict[split] = masked_simp_edge_batch
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])
        torch.save(self.simp_edge_batch_dict, self.processed_paths[2])

    def __repr__(self) -> str:
        return f'JodieDataset(name={self.name})'