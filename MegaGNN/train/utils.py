import torch 
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index

from MegaGNN.graphgym.config import cfg

def add_missing_rev_edges(batch, dataset):
    """Add missing reverse edges to a heterogeneous graph batch.
    
    In heterogeneous graph sampling, when forward edges are sampled, their corresponding
    reverse edges might not be included in the batch. This function identifies and adds
    these missing reverse edges to maintain the bidirectional nature of the graph.
    
    Args:
        batch (HeteroData): The current batch containing sampled edges and nodes
        dataset (HeteroData): The full dataset containing all edges and nodes
    
    Returns:
        HeteroData: The batch with added missing reverse edges and a mask indicating
                   which reverse edges correspond to the original forward edges
    
    Process:
    1. Identify forward edges in the batch that are part of the task
    2. Find which of these forward edges are missing their reverse counterparts
    3. Add the missing reverse edges from the full dataset to the batch
    4. Create a mask to track which reverse edges correspond to the original forward edges
    """
    
    # Get the task edge type and its corresponding reverse edge type
    task = cfg.dataset.task_entity
    reverse_task = (task[0], f'rev_{task[1]}', task[2])

    # Get indices of edges that are part of the task in the full dataset
    inds = mask_to_index(dataset[task].split_mask)
    
    # Identify which edges in the batch are the seed edges
    mask = torch.isin(batch[task].e_id, 
                     inds[batch[task].input_id]) 
    
    # Get the global edge IDs of the seed edges that are in the batch
    # Note: Not all seed edges are necessarily in the batch
    batch_edge_ids = batch[task].e_id[mask]
    
    # Find which forward edges are missing their reverse counterparts
    missing_rev = ~torch.isin(batch_edge_ids,
                            batch[reverse_task].e_id)

    # If there are missing reverse edges, add them to the batch
    if missing_rev.sum() != 0:
        # Get the global edge IDs of the missing reverse edges
        missing_ids = batch_edge_ids[missing_rev].int()

        # Get the global node IDs in the current batch for mapping
        n_ids = batch[task[0]].n_id
        # Get the edge indices and attributes of missing reverse edges from the full dataset
        add_edge_index = dataset[reverse_task].edge_index[:, missing_ids].detach().clone()
        # Create a mapping from global node IDs to local batch node IDs
        node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
        # Map the global edge indices to local batch indices
        add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
        add_edge_attr = dataset[reverse_task].edge_attr[missing_ids, :].detach().clone()
        add_e_ids = missing_ids

        # Add the missing reverse edges to the batch
        batch[reverse_task].edge_index = torch.cat((batch[reverse_task].edge_index, add_edge_index), dim=1)
        batch[reverse_task].edge_attr = torch.cat((batch[reverse_task].edge_attr, add_edge_attr), dim=0)
        batch[reverse_task].e_id = torch.cat((batch[reverse_task].e_id, add_e_ids), dim=0)

        # If using multi-edge aggregation, also add the corresponding simplified edge batches
        if cfg.gnn.multi_edge_agg:
            add_simp_edge_batch = dataset[reverse_task].simp_edge_batch[missing_ids].detach().clone()
            batch[reverse_task].simp_edge_batch = torch.cat((batch[reverse_task].simp_edge_batch, add_simp_edge_batch), dim=0)
        
        # Create a mask indicating which reverse edges correspond to the original forward edges
        rev_mask = torch.isin(batch[reverse_task].e_id, batch_edge_ids)

    # Store the reverse edge mask in the batch for later use
    batch[reverse_task].rev_mask = rev_mask
    return batch