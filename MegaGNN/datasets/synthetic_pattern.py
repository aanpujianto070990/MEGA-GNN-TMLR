import os.path as osp
import pandas as pd
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
    timestamp = torch.ones(edge_index.shape[1]).float()

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

class SyntheticPatternDataset(TemporalDataset):

    '''
    tasks = ["deg-in","deg-out","fan-in","fan-out","C2","C3","C4","C5","C6","S-G","B-C"]
    in total there are 11 tasks.

    '''

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


        if cfg.dataset.random_edge_feats:
            for split in ['train', 'val', 'test']:
                # 16 is completely an arbitrary number, it is just for the sake of the experiment.
                
                self.data_dict[split]['node', 'to', 'node'].edge_attr = torch.randn(self.data_dict[split]['node', 'to', 'node'].edge_index.shape[1], 16).float()
                self.data_dict[split]['node', 'rev_to', 'node'].edge_attr = torch.randn(self.data_dict[split]['node', 'rev_to', 'node'].edge_index.shape[1], 16).float()
    
        if cfg.dataset.full_one_edge_feats:
            for split in ['train', 'val', 'test']:
                self.data_dict[split]['node', 'to', 'node'].edge_attr = torch.ones(self.data_dict[split]['node', 'to', 'node'].edge_index.shape[1], 1).float()
                self.data_dict[split]['node', 'rev_to', 'node'].edge_attr = torch.ones(self.data_dict[split]['node', 'rev_to', 'node'].edge_index.shape[1], 1).float()
            
        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                del self.data_dict[split]['node', 'rev_to', 'node']

        if add_ports:
            print("Original Ports are loaded!")
            self.ports_dict = torch.load(self.processed_paths[1])
            
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.add_ports_func(self.data_dict[split], self.ports_dict[split])
        
        if multi_edge_agg:
            self.simp_edge_batch_dict = torch.load(self.processed_paths[2]) 
            for split in ['train', 'val', 'test']:
                for edge_type in self.data_dict[split].edge_types:
                    self.data_dict[split][edge_type].simp_edge_batch = self.simp_edge_batch_dict[split]
        
            
    def add_ports_func(self, data, ports):

        in_ports, out_ports = ports
        ports_arr = torch.stack([in_ports, out_ports], dim=1).float()

        if hasattr(data['node', 'to', 'node'], 'edge_attr'):
            data['node', 'to', 'node'].edge_attr = torch.cat([data['node', 'to', 'node'].edge_attr, ports_arr], dim=1)
        else:
            data['node', 'to', 'node'].edge_attr = ports_arr

        if self.reverse_mp:
            if hasattr(data['node', 'rev_to', 'node'], 'edge_attr'):
                data['node', 'rev_to', 'node'].edge_attr = torch.cat([data['node', 'rev_to', 'node'].edge_attr, ports_arr[:, [1, 0]]], dim=1)
            else:
                data['node', 'rev_to', 'node'].edge_attr = ports_arr[:, [1, 0]]
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
        return ['train_fixed.pt', 'val_fixed.pt', 'test_fixed.pt']

    @property
    def processed_file_names(self) -> str:
        return ['data.pt', 'ports.pt', 'simp_edge_batch.pt']
    
    def process(self):
         
        train_graph = torch.load(self.raw_paths[0], weights_only=False, map_location="cpu")
        val_graph = torch.load(self.raw_paths[1], weights_only=False, map_location="cpu")
        test_graph = torch.load(self.raw_paths[2], weights_only=False, map_location="cpu")
                
        self.ports_dict = {}
        self.simp_edge_batch_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            graph = eval(f'{split}_graph')
            
            N = graph.y.shape[0] if getattr(graph, 'y', None) is not None else int(graph.edge_index.max()) + 1
            
            data = HeteroData()
            data['node'].num_nodes = N
            data['node'].x = torch.ones((N, 1), dtype=torch.float)
            data['node'].y = graph.y
            data['node'].split_mask = index_to_mask(torch.arange(N), size=N)

            data['node', 'to', 'node'].edge_index = graph.edge_index
            data['node', 'rev_to', 'node'].edge_index = graph.edge_index.flipud()
        
            # For megagnn
            simp_edge_batch = find_parallel_edges(graph.edge_index)
        
            in_ports, out_ports = ports_with_cpp(data)
            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data
            self.simp_edge_batch_dict[split] = simp_edge_batch
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])
        torch.save(self.simp_edge_batch_dict, self.processed_paths[2])

    def __repr__(self) -> str:
        return f'SyntheticPatternDataset(name={self.name})'