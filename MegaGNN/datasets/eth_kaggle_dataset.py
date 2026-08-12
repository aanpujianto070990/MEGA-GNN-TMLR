import os.path as osp
import pandas as pd
import numpy as np

from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import index_to_mask

from .utils import download_dataset
from .temporal_dataset import TemporalDataset
from MegaGNN.cpp import ports_cpp
from MegaGNN.graphgym.config import cfg


def collapse_multi_edges(df_edges, collapse_rate, seed=42):
    if collapse_rate == 0.0:
        return df_edges
    
    rng = np.random.default_rng(seed)
    
    # Find multi-edge pairs
    pair_counts = df_edges.groupby(['from_address', 'to_address']).size()
    multi_pairs = pair_counts[pair_counts >= 2].index.tolist()
    
    # Sample fraction to collapse
    n_collapse = int(len(multi_pairs) * collapse_rate)
    chosen = rng.choice(len(multi_pairs), size=n_collapse, replace=False)
    pairs_to_collapse = set(multi_pairs[i] for i in chosen)
    
    df_edges['_pair'] = list(zip(df_edges['from_address'], df_edges['to_address']))
    mask = df_edges['_pair'].isin(pairs_to_collapse)
    
    keep = df_edges[~mask].drop(columns=['_pair'])
    to_collapse = df_edges[mask].drop(columns=['_pair'])
    
    collapsed = (
        to_collapse
        .groupby(['from_address', 'to_address'], as_index=False)
        .agg(amount=('amount', 'mean'), timestamp=('timestamp', 'min'))
    )
    
    return pd.concat([keep, collapsed], ignore_index=True)

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def ports_with_cpp(graph, permute_ports=False):    
    edge_index = graph['node', 'to', 'node'].edge_index
    timestamp = graph['node', 'to', 'node'].timestamps

    if permute_ports:
        permutation = torch.randperm(timestamp.size(0))
        timestamp = timestamp[permutation]

    edges = torch.cat([edge_index.T, timestamp.reshape((-1,1))], dim=1).numpy().astype('int')
    ports_in, ports_out = ports_cpp.assign_ports(edges,edge_index.numpy().astype('int'), graph.num_nodes)
    
    return torch.from_numpy(ports_in), torch.from_numpy(ports_out)

def find_parallel_edges(edge_index):
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    for edge in edge_index.T:
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i
            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

    return simplified_edge_batch

class ETHKaggleDataset(TemporalDataset):

    def __init__(self, root: str, name: str,
                 reverse_mp: bool = False,
                 add_ports: bool = False, 
                 multi_edge_agg: bool = False,
                 collapse_rate: float = 0.0,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        self.multi_edge_agg = multi_edge_agg
        self.collapse_rate = collapse_rate
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0])
        # del self._data['node'].x
        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                del self.data_dict[split]['node', 'rev_to', 'node']


        if add_ports:
            if cfg.dataset.permute_ports:
                print("Permuted Ports are loaded!")
                self.ports_dict = torch.load(self.processed_paths[2])
            else:
                print("Original Ports are loaded!")
                self.ports_dict = torch.load(self.processed_paths[1])
            
            for split in ['train', 'val', 'test']:
                if cfg.dataset.fraudgt_ports:
                    self.data_dict[split] = self.add_ports_func_fraudgt(self.data_dict[split], self.ports_dict[split])
                else:
                    self.data_dict[split] = self.add_ports_func(self.data_dict[split], self.ports_dict[split])
        
        if multi_edge_agg:
            self.simp_edge_batch_dict = torch.load(self.processed_paths[3]) 
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
        r = self.collapse_rate
        if r == 0.0:
            return osp.join(self.root, self.name, 'processed')
        else:   
            return osp.join(self.root, self.name, 'processed', 'collapsed')

    @property
    def raw_file_names(self) -> List[str]:
        file_names = ['edges-kaggle.csv', 'nodes-kaggle.csv']
        return file_names

    # def processed_file_names(self) -> str:
    #     return ['data.pt', 'ports.pt', 'ports_permuted.pt', 'simp_edge_batch.pt']

    @property
    def processed_file_names(self) -> str:
        r = self.collapse_rate
        if r == 0.0:
            return ['data.pt', 'ports.pt', 'ports_permuted.pt', 'simp_edge_batch.pt']
        else:
            return [f'data_{r}.pt', f'ports_{r}.pt', f'ports_permuted_{r}.pt', f'simp_edge_batch_{r}.pt']


    def process(self):
        
        # eth_transactions_all.csv: contains ETH transactions
        df_edges = pd.read_csv(osp.join(self.raw_dir, 'edges-kaggle.csv')).drop('Unnamed: 0', axis=1)
        # Collapse a fraction of the parallel edges (no-op when collapse_rate == 0.0)
        df_edges = collapse_multi_edges(df_edges, self.collapse_rate)
        # node_labels.csv: contains label for each ETH account: 0 - no phishing, 1 - phishing
        df_nodes = pd.read_csv(osp.join(self.raw_dir, 'nodes-kaggle.csv')).drop('Unnamed: 0', axis=1)

        print(f'Available Edge Features: {df_edges.columns.tolist()}')
        print(f"Number of Nodes: {df_nodes.shape[0]}")
        print(f"Number of Edges: {df_edges.shape[0]}")

        df_nodes = df_nodes.sort_values(by='first_transaction').reset_index(drop=True)
        
        assign_dict = {}
        for row in df_nodes.itertuples():
            assign_dict[row[1]] = row[0]

        def assign_node_ids(node_id):
            return assign_dict[node_id] 
    
        df_edges['to_address'] = df_edges['to_address'].apply(assign_node_ids)
        df_edges['from_address'] = df_edges['from_address'].apply(assign_node_ids)
        df_nodes.drop(columns=['node'], inplace=True)

        edge_features = ['amount', 'timestamp']
        node_features = ['Feature']
        
        print(f'Edge features being used: {edge_features}')
        print(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

        max_n_id = df_nodes.shape[0]

        splits = [0.65, 0.15, 0.20]

        t1 = df_nodes.iloc[int(max_n_id * splits[0])]['first_transaction']
        t2 = df_nodes.iloc[int(max_n_id * (splits[0] + splits[1]))]['first_transaction']

        train_nodes = df_nodes.loc[df_nodes['first_transaction'] <= t1]
        val_nodes = df_nodes.loc[df_nodes['first_transaction'] <= t2]
        test_nodes = df_nodes

        tr_nodes_max_id = train_nodes.index[-1]
        val_nodes_max_id = val_nodes.index[-1]
        te_nodes_max_id = test_nodes.index[-1]

        train_inds = torch.arange(0, tr_nodes_max_id+1)
        val_inds = torch.arange(tr_nodes_max_id+1, val_nodes_max_id+1)
        test_inds = torch.arange(val_nodes_max_id+1, te_nodes_max_id+1)
        
        print(f"Total train samples: {train_nodes.shape[0] / df_nodes.shape[0] * 100 :.2f}% || IR: "
                f"{train_nodes['label'].mean() * 100 :.2f}%")
        print(f"Total validation samples: {val_inds.shape[0] / df_nodes.shape[0] * 100 :.2f}% || IR: "
                f"{val_nodes.loc[val_inds.numpy(),'label'].mean() * 100 :.2f}%")
        print(f"Total test samples: {test_inds.shape[0] / df_nodes.shape[0] * 100 :.2f}% || IR: "
                f"{test_nodes.loc[test_inds.numpy(), 'label'].mean() * 100 :.2f}%")


        tr_nodes_max_id = train_nodes.index[-1]
        val_nodes_max_id = val_nodes.index[-1]
        te_nodes_max_id = test_nodes.index[-1]

        split_name = []
        for row in df_edges.itertuples():
            '''
            row[0]: index
            row[1]: from_address
            row[2]: to_address
            row[3]: amount
            row[4]: timestamp
            '''
            if row[1] <= tr_nodes_max_id and row[2] <= tr_nodes_max_id:
                if row[4] <= t1:
                    split_name.append('train')
                elif row[4] > t1 and row[4] <= t2:
                    split_name.append('val')
                else:
                    split_name.append('test') 
                continue 
            elif row[1] <= val_nodes_max_id and row[2] <= val_nodes_max_id:
                if row[4] <= t2:
                    split_name.append('val')
                else:
                    split_name.append('test') 
                continue
            else:
                split_name.append('test')


        df_edges['split'] = split_name
        df_edges['timestamp'] = df_edges['timestamp'] - df_edges['timestamp'].min()
        
        train_edges = df_edges.loc[df_edges['split'] == 'train']
        val_edges = df_edges.loc[(df_edges['split'] == 'train') | (df_edges['split'] == 'val')]
        test_edges = df_edges 

        self.ports_dict = {}
        self.data_dict = {}
        self.ports_dict_permuted = {}
        self.simp_edge_batch_dict = {}
        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            split_edge_df = eval(f'{split}_edges')
            split_node_df = eval(f'{split}_nodes')


            x = torch.tensor(np.ones(split_node_df.shape[0])).float().view(-1, 1)
            edge_index = torch.LongTensor(split_edge_df.loc[:, ['from_address', 'to_address']].to_numpy().T)
            edge_attr = torch.tensor(split_edge_df.loc[:, edge_features].to_numpy()).float()
            timestamps = torch.Tensor(split_edge_df['timestamp'].to_numpy())
            y = torch.LongTensor(split_node_df['label'].to_numpy())

            data = HeteroData()
            data['node'].x = z_norm(x) #  will render all x be 0
            data['node'].y = y # masked_y
            data['node'].num_nodes = int(x.shape[0])
            data['node', 'to', 'node'].edge_index = edge_index
            data['node', 'to', 'node'].edge_attr = z_norm(edge_attr)
            data['node', 'to', 'node'].timestamps = timestamps


            data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = z_norm(edge_attr)

            simp_edge_batch = find_parallel_edges(data['node', 'to', 'node'].edge_index)

            data['node'].split_mask = index_to_mask(inds, size=data['node'].num_nodes)

            in_ports, out_ports = ports_with_cpp(data)
            in_ports_permuted, out_ports_permuted = ports_with_cpp(data, permute_ports=True)
            self.ports_dict[split] = [in_ports, out_ports]
            self.ports_dict_permuted[split] = [in_ports_permuted, out_ports_permuted]
            self.data_dict[split] = data
            self.simp_edge_batch_dict[split] = simp_edge_batch

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])
        torch.save(self.ports_dict_permuted, self.processed_paths[2])
        torch.save(self.simp_edge_batch_dict, self.processed_paths[3])

    def __repr__(self) -> str:
        return 'ETH_Kaggle_Dataset()'
    


