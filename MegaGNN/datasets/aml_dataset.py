import json, itertools
import sys, os
import os.path as osp
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import index_to_mask

from .temporal_dataset import TemporalDataset

from MegaGNN.cpp import ports_cpp
from MegaGNN.graphgym.config import cfg


def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def format_dataset(inPath):
    r'''
    Turn text attributed dataset into a dataset only contains numbers.

    Vectorized pandas port of the original row-by-row datatable loop, so the
    dataset can be built without datatable (no wheels for recent Pythons) and
    in seconds instead of tens of minutes. Output matches the loop:
    - currencies, payment formats and accounts get integer ids in order of
      first appearance, scanning each row's fields in the loop's order
      (receiving before payment currency, sender before receiver account);
    - Timestamp is seconds since 10 s before midnight of the first row's day;
    - amounts are rounded to 6 decimals, as the loop's '%f' formatting did;
    - rows are stably sorted by Timestamp, as datatable's sort did.
    '''
    outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

    raw = pd.read_csv(inPath, dtype=str)

    def first_seen_ids(*cols):
        # Interleave the columns row by row (row0.col0, row0.col1, row1.col0, ...)
        # so first-appearance order matches the original per-row loop.
        order = pd.unique(np.column_stack([c.to_numpy() for c in cols]).ravel())
        return {v: i for i, v in enumerate(order)}

    currency = first_seen_ids(raw["Receiving Currency"], raw["Payment Currency"])
    paymentFormat = first_seen_ids(raw["Payment Format"])

    # Columns 2 and 4 are the two "Account" columns (sender, receiver).
    from_acc = raw["From Bank"] + raw.iloc[:, 2]
    to_acc = raw["To Bank"] + raw.iloc[:, 4]
    account = first_seen_ids(from_acc, to_acc)

    timestamps = pd.to_datetime(raw["Timestamp"], format='%Y/%m/%d %H:%M')
    firstTs = timestamps.iloc[0].normalize() - pd.Timedelta(seconds=10)
    ts = (timestamps - firstTs) // pd.Timedelta(seconds=1)

    formatted = pd.DataFrame({
        "EdgeID": np.arange(len(raw)),
        "from_id": from_acc.map(account),
        "to_id": to_acc.map(account),
        "Timestamp": ts.astype(np.int64),
        "Amount Sent": raw["Amount Paid"].astype(float).round(6),
        "Sent Currency": raw["Payment Currency"].map(currency),
        "Amount Received": raw["Amount Received"].astype(float).round(6),
        "Received Currency": raw["Receiving Currency"].map(currency),
        "Payment Format": raw["Payment Format"].map(paymentFormat),
        "Is Laundering": raw["Is Laundering"].astype(int),
    })
    formatted = formatted.sort_values(by="Timestamp", kind="stable")

    formatted.to_csv(outPath, index=False)


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

class AMLDataset(TemporalDataset):
    dataset_sizes = ['Small', 'Medium', 'Large']
    dataset_rates = ['LI', 'HI']
    csv_names = {
        'Small-LI': 'LI-Small_Trans.csv',
        'Small-HI': 'HI-Small_Trans.csv',
        'Medium-LI': 'LI-Medium_Trans.csv',
        'Medium-HI': 'HI-Medium_Trans.csv',
        'Large-LI': 'LI-Large_Trans.csv',
        'Large-HI': 'HI-Large_Trans.csv',
        'Small-HI-cleaned': 'HI-Small_Trans.csv',
        'Small-LI-cleaned': 'LI-Small_Trans.csv',
    }

    def __init__(self, root: str, name: str, 
                 reverse_mp: bool = False,
                 add_ports: bool = False,
                 multi_edge_agg: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name # Small-LI
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        self.multi_edge_agg = multi_edge_agg
        assert self.name.split('-')[0] in self.dataset_sizes
        assert self.name.split('-')[1] in self.dataset_rates
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0], weights_only=False)
        
        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                del self.data_dict[split]['node', 'rev_to', 'node']

        if add_ports:
            
            if cfg.dataset.permute_ports:
                print("Permuted Ports are loaded!")
                self.ports_dict = torch.load(self.processed_paths[2], weights_only=False)
            else:
                print("Original Ports are loaded!")
                self.ports_dict = torch.load(self.processed_paths[1], weights_only=False)
            
            for split in ['train', 'val', 'test']:
                if cfg.dataset.fraudgt_ports:
                    self.data_dict[split] = self.add_ports_func_fraudgt(self.data_dict[split], self.ports_dict[split])
                else:
                    self.data_dict[split] = self.add_ports_func(self.data_dict[split], self.ports_dict[split])
        
        if multi_edge_agg:
            self.simp_edge_batch_dict = torch.load(self.processed_paths[3], weights_only=False) 
            for split in ['train', 'val', 'test']:
                for edge_type in self.data_dict[split].edge_types:
                    self.data_dict[split][edge_type].simp_edge_batch = self.simp_edge_batch_dict[split]
        
        if cfg.gnn.layer_type in ['RGCN', 'RGCNE']:
            self.currency_type_dict = torch.load(self.processed_paths[5], weights_only=False)
            for split in ['train', 'val', 'test']:
                for edge_type in self.data_dict[split].edge_types:
                    self.data_dict[split][edge_type].currency_type = self.currency_type_dict[split]

            if multi_edge_agg:
                # If multi_edge_agg is True, we need to load the simp_edge_batch_with_edge_type_dict since 
                # simp_edge_batch is computed with edge_types.
                self.simp_edge_batch_with_edge_type_dict = torch.load(self.processed_paths[4], weights_only=False)
                for split in ['train', 'val', 'test']:
                    for edge_type in self.data_dict[split].edge_types:
                        self.data_dict[split][edge_type].simp_edge_batch = self.simp_edge_batch_with_edge_type_dict[split]
    
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
        return []

    @property
    def processed_file_names(self) -> str:
        return ['data.pt', 'ports.pt', 'ports_permuted.pt', 'simp_edge_batch.pt', 'simp_edge_batch_with_currency_type.pt', 'currency_type.pt']

    def process(self):

        transaction_file = osp.join(self.raw_dir, "formatted_transactions.csv")
        
        # Check if the formatted dataset already exists to avoid reprocessing
        if not osp.exists(transaction_file):
            format_dataset(osp.join(self.raw_dir, self.csv_names[self.name]))
        
        df_edges = pd.read_csv(transaction_file)
        df_edges = df_edges.sort_values(by='Timestamp')

        print(f'Available Edge Features: {df_edges.columns.tolist()}')

        df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

        max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
        df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
        timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
        y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

        print(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
        print(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
        print(f"Number of transactions = {df_edges.shape[0]}")

        edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
        node_features = ['Feature']

        print(f'Edge features being used: {edge_features}')
        print(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

        x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
        edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
        edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

        # Store the currency info separately
        currency_type = df_edges['Received Currency'].to_numpy()
        currency_type = torch.LongTensor(currency_type)

        simp_edge_batch = find_parallel_edges(edge_index)
        simp_edge_batch_with_edge_type = find_parallel_edges(edge_index, edge_types=currency_type)

        n_days = int(timestamps.max() / (3600 * 24) + 1)
        n_samples = y.shape[0]
        print(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

        #data splitting
        daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = illicit ratios, inds = indices, trans = transactions
        for day in range(n_days):
            l = day * 24 * 3600
            r = (day + 1) * 24 * 3600
            day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
            daily_irs.append(y[day_inds].float().mean())
            weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
            daily_inds.append(day_inds)
            daily_trans.append(day_inds.shape[0])

        split_per = [0.6, 0.2, 0.2]
        daily_totals = np.array(daily_trans)
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i,j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v/split_totals_sum for v in split_totals]
                split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
                score = max(split_error) #- (split_totals_sum/total) + 1
                split_scores[(i,j)] = score
            else:
                continue

        i,j = min(split_scores, key=split_scores.get)
        #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
        print(f'Calculate split: {split}')

        #Now, we seperate the transactions based on their indices in the timestamp array
        split_inds = {k: [] for k in range(3)}
        for i in range(3):
            for day in split[i]:
                split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately
                
        train_inds = torch.cat(split_inds[0])
        val_inds = torch.cat(split_inds[1])
        test_inds = torch.cat(split_inds[2])
        e_train = train_inds
        e_val = torch.cat([train_inds, val_inds])
        e_test = torch.cat([train_inds, val_inds, test_inds])

        
        self.ports_dict = {}
        self.ports_dict_permuted = {}
        self.currency_type_dict = {}
        self.simp_edge_batch_dict = {}
        self.simp_edge_batch_with_edge_type_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            e_mask = eval(f'e_{split}')

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr = z_norm(edge_attr[e_mask])
            masked_y = y[e_mask]
            masked_timestamps = timestamps[e_mask]

            data = HeteroData()
            data['node'].x = z_norm(x) # will render all x be 0
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
            masked_simp_edge_batch_with_edge_type = simp_edge_batch_with_edge_type[e_mask]
            masked_currency_type = currency_type[e_mask]
        

            in_ports, out_ports = ports_with_cpp(data)
            in_ports_permuted, out_ports_permuted = ports_with_cpp(data, permute_ports=True)
            self.ports_dict[split] = [in_ports, out_ports]
            self.ports_dict_permuted[split] = [in_ports_permuted, out_ports_permuted]
            self.data_dict[split] = data
            self.simp_edge_batch_dict[split] = masked_simp_edge_batch
            self.simp_edge_batch_with_edge_type_dict[split] = masked_simp_edge_batch_with_edge_type
            self.currency_type_dict[split] = masked_currency_type
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])
        torch.save(self.ports_dict_permuted, self.processed_paths[2])
        torch.save(self.simp_edge_batch_dict, self.processed_paths[3])
        torch.save(self.simp_edge_batch_with_edge_type_dict, self.processed_paths[4])
        torch.save(self.currency_type_dict, self.processed_paths[5])

    def __repr__(self) -> str:
        return f'AML_Dataset(name={self.name})'