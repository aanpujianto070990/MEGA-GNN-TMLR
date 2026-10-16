from typing import Callable, List, Optional, Union
import time 
import copy
import logging

import torch
from torch_geometric.data import (Data, HeteroData)
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.loader import  NeighborLoader, LinkNeighborLoader

from MegaGNN.datasets.temporal_dataset import TemporalDataset
from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.register import register_sampler


class InPlaceTransform(BaseTransform):
    r"""BaseTransform that works with both old and new PyG.

    Newer PyG makes `forward()` abstract and has `__call__()` call it on a
    shallow copy; older PyG makes `__call__()` abstract. Subclasses implement
    `forward()`, and `__call__()` applies it directly to the batch, as the
    original `__call__`-based transforms did.
    """
    def __call__(self, data):
        return self.forward(data)


class AddEgoIdsForLinkNeighbor(InPlaceTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def forward(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data
    
    
class AddEgoIdsForNeighbor(InPlaceTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def forward(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.arange(data.batch_size, device=device)
        else:
            nodes = torch.arange(data['node'].batch_size, device=device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

class AddIdentityMatrixEgoIdsForNeighbor(InPlaceTransform):
    r"""Add identity matrix to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def forward(self, data: Union[HeteroData]):
        x = data['node'].x
        device = x.device
        batch_size = data['node'].batch_size

        ego = torch.zeros((x.shape[0], batch_size), device=device)
        idx = torch.arange(batch_size, device=device)
        ego[idx, idx] = 1.0

        data['node'].x = torch.cat([x, ego], dim=1)
        
        return data


class LoaderWrapper:
    '''
    A wrapper for dataloaders that:
    - For training: Limits the number of steps to a fixed number for consistency across different graph sizes
    - For validation/test: Always traverses the entire graph
    '''
    def __init__(self, dataloader, n_step=-1, split='train'):
        self.step = n_step if n_step > 0 else len(dataloader)
        self.idx = 0
        self.loader = dataloader
        self.split = split
        self.iter_loader = iter(dataloader)
    
    def __iter__(self):
        return self

    def __len__(self):
        if self.split == 'train':
            return min(self.step, len(self.loader))
        else:
            return len(self.loader)

    def __next__(self):
        if self.idx == self.__len__():
            self.idx = 0
            if self.split in ['val', 'test']:
                # Make sure we are always using the same set of data for evaluation, 
                # so always restart the iterator
                self.iter_loader = iter(self.loader)
            raise StopIteration
        else:
            self.idx += 1
        try: 
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue until the desired number of steps are reached
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)
        
    def set_step(self, n_step):
        self.step = n_step


@register_sampler('link_neighbor')
def get_LinkNeighborLoader(dataset, batch_size, shuffle=True, split='train'):
    task = cfg.dataset.task_entity
    data = dataset[split]
    mask = data[task].split_mask
    edge_label_index = data[task].edge_index[:, mask]
    edge_label = data[task].y[mask]
    
        
    loader_train = \
        LoaderWrapper(
            LinkNeighborLoader(
                data=data,
                num_neighbors=cfg.train.neighbor_sizes,
                edge_label_index=(task, edge_label_index),
                edge_label=edge_label,
                batch_size=batch_size,
                num_workers=cfg.num_workers,
                shuffle=shuffle,
                transform=AddEgoIdsForLinkNeighbor() if cfg.train.add_ego_id else None
            ),
            getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
            split=split
        )
    
    return loader_train


@register_sampler('hetero_neighbor')
def get_NeighborLoader(dataset, batch_size, shuffle=True, split='train'):
    task = cfg.dataset.task_entity
   
    data = dataset[split]
    mask = data[task].split_mask
    input_nodes = (task, mask)
    sample_sizes = {key: cfg.train.neighbor_sizes for key in data.edge_types}
    
    
    transform = None
    if cfg.train.add_ego_id:
        if cfg.dataset.name == "pattern":
            transform = AddIdentityMatrixEgoIdsForNeighbor()
        else:
            transform = AddEgoIdsForNeighbor()

    start = time.time()
    loader_train = \
         LoaderWrapper(
             NeighborLoader(
                data,
                num_neighbors=sample_sizes,
                input_nodes=input_nodes,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=cfg.num_workers,
                persistent_workers=cfg.train.persistent_workers,
                pin_memory=cfg.train.pin_memory,
                transform=transform
            ),
            getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
            split=split
        )
    
    end = time.time()
    print(f'Data {split} loader initialization took:', round(end - start, 3), 'seconds.')
    
    return loader_train


@register_sampler('full_batch_link')
def get_FullBatchLinkLoader(dataset, batch_size, shuffle=True, split='train'):
    """
    Full batch loader for link prediction tasks. 
    Loads the entire graph at once without neighbor sampling.
    
    Args:
        dataset: The dataset containing the graph
        batch_size: Ignored in full batch setup, but kept for API consistency
        shuffle: Whether to shuffle the edge indices (still relevant for training)
        split: Which data split to use ('train', 'val', 'test')
        
    Returns:
        A loader that yields the full graph with target edges for training
    """
    task = cfg.dataset.task_entity
    data = dataset[split]
    mask = data[task].split_mask
    edge_label_index = data[task].edge_index[:, mask]
    edge_label = data[task].y[mask]
    
    # Create a copy of the data to avoid modifying the original
    full_batch_data = copy.deepcopy(data)
    
    # Add the edge label indices and labels directly to the data object
    full_batch_data[task].edge_label_index = edge_label_index
    full_batch_data[task].edge_label = edge_label
    
    logging.info(f"Full batch mode: Loading entire graph with {full_batch_data['node'].num_nodes} nodes")
    logging.info(f"Target edges for {split}: {edge_label_index.size(1)}")
    
    # Create a simple loader that just yields the full graph once per epoch
    class FullBatchLoader:
        def __init__(self, data, shuffle):
            self.data = data
            self.shuffle = shuffle
            self.edge_label_index = edge_label_index
            self.edge_label = edge_label
            
        def __iter__(self):
            # Create a fresh copy of the data each time to ensure we don't have
            # any leftover computation graphs from previous iterations
            batch = copy.deepcopy(self.data)
            
            # If shuffle is True, shuffle the edge labels but keep the correspondence
            # between edge_label_index and edge_label
            if self.shuffle:
                idx = torch.randperm(self.edge_label_index.size(1))
                batch[task].edge_label_index = self.edge_label_index[:, idx]
                batch[task].edge_label = self.edge_label[idx]
            
            # Return the entire batch
            yield batch
            
        def __len__(self):
            # Only one batch per epoch
            return 1
    
    # Create our loader and wrap it
    loader = FullBatchLoader(full_batch_data, shuffle)
    
    # Wrap with LoaderWrapper to maintain API consistency
    # Set iter_per_epoch to 1 since we're using full batch
    loader_train = LoaderWrapper(
        loader,
        1,  # Only one iteration per epoch
        split=split
    )
    
    return loader_train

