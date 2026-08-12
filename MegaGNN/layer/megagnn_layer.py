import torch
import torch.nn as nn
from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder

from MegaGNN.graphgym import register as register
from MegaGNN.graphgym.config import cfg

from MegaGNN.layer.rgcn_conv import FastRGCNConv
from MegaGNN.layer.hetero_multiedgeagg import HeteroMultiEdgeAggregator

from torch_geometric.nn import (HeteroConv, PNAConv, GINEConv)
from torch_geometric.utils import degree
from torch_geometric.data import HeteroData



class MEGAGNNLayer(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        if self.is_hetero:
            self.metadata = dataset[0].metadata()

        self.data = dataset[0]
                
        # Initialize feature processing flags
        self.edge_updates = cfg.gnn.edge_updates
        self.multi_edge_agg = cfg.gnn.multi_edge_agg
        self.multi_edge_agg_type = cfg.gnn.multi_edge_agg_type
        self.layer_type = cfg.gnn.layer_type
        
        conv_dict = {}
        for edge_type in self.metadata[1]:
            conv_dict[edge_type] = self._get_gnn_layer()

        self.conv = HeteroConv(conv_dict, aggr='mean')

        if self.edge_updates:
            self.emlp = nn.ModuleDict()
            for edge_type in self.metadata[1]:
                edge_type = "__".join(edge_type)
                self.emlp[edge_type] = nn.Sequential(
                        nn.Linear(3 * cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                        nn.ReLU(),
                        nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                    )
        
        if self.multi_edge_agg:
            self.edge_aggr = HeteroMultiEdgeAggregator(
                    n_hidden=cfg.gnn.dim_inner,
                    agg_type=self.multi_edge_agg_type,
                    data=self.data
                    )


    def _get_gnn_layer(self):
        if cfg.gnn.layer_type == 'GINE':
            mlp = nn.Sequential(
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 
                    nn.ReLU(), 
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner)
                )
            conv = GINEConv(mlp, edge_dim=cfg.gnn.dim_inner)

        elif cfg.gnn.layer_type == 'GenAgg':
            mlp = nn.Sequential(
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 
                    nn.ReLU(), 
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner)
                )
            conv = GINEConv(mlp, edge_dim=cfg.gnn.dim_inner,
                            aggr=GenAgg(f=MLPAutoencoder, jit=False))
                            
        elif cfg.gnn.layer_type == 'PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            if not isinstance(self.data, HeteroData):
                d = degree(self.data.edge_index[1], dtype=torch.long)
            else:
                d = degree(self.data.to_homogeneous().edge_index[1], dtype=torch.long)
            deg = torch.bincount(d, minlength=1)

            conv = PNAConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner,
                        aggregators=aggregators, 
                        scalers=scalers, 
                        deg=deg,
                        edge_dim=cfg.gnn.dim_inner, 
                        towers=4, 
                        pre_layers=1, 
                        post_layers=1,
                        divide_input=False)
        
        elif cfg.gnn.layer_type == 'RGCN':
            conv = FastRGCNConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, num_relations=15,
                            use_edge_attr=False)
        
        elif cfg.gnn.layer_type == 'RGCNE':
            conv = FastRGCNConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, num_relations=15,
                            use_edge_attr=True)
                
        else:
            raise NotImplementedError(f"{cfg.gnn.layer_type} is not implemented!")
        
        return conv

    def forward(self, batch):

        x_dict = batch.collect('x')

        if self.multi_edge_agg:
            prev_edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}
            prev_edge_attr_dict = {edge_type: batch[edge_type].edge_attr for edge_type in batch.edge_types}

            if self.layer_type in ['RGCN' ,'RGCNE']:
                prev_currency_type_dict = {edge_type: batch[edge_type].currency_type for edge_type in batch.edge_types}

            batch = self.edge_aggr(batch) 

        if self.layer_type == 'RGCN' or self.layer_type == 'RGCNE': 
            h_dict = self.conv(x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.currency_type_dict)
        else:
            h_dict = self.conv(x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        

        if self.edge_updates:
            if self.multi_edge_agg:
                for edge_type in batch.edge_types:
                    prev_edge_index = prev_edge_index_dict[edge_type]
                    prev_edge_attr = prev_edge_attr_dict[edge_type]
                    src, dst = prev_edge_index

                    remapped_edge_attr = torch.index_select(batch[edge_type].edge_attr, 0, batch[edge_type].inverse_indices)
                    batch[edge_type].edge_attr = (
                                        self.emlp["__".join(edge_type)](
                                            torch.cat(
                                                [h_dict[edge_type[0]][src], remapped_edge_attr, prev_edge_attr],
                                                dim=-1
                                            ))
                                        )
                    # Update edge index so that in the next layer, the edge index is the same as the previous layer
                    batch[edge_type].edge_index = prev_edge_index
                    if self.layer_type == 'RGCN' or self.layer_type == 'RGCNE': 
                        # Update currency type so that in the next layer, the currency type is the same as the previous layer
                        batch[edge_type].currency_type = prev_currency_type_dict[edge_type]
            else:
                for edge_type in batch.edge_types:
                    src, dst = batch[edge_type].edge_index
                    batch[edge_type].edge_attr = (
                                        self.emlp["__".join(edge_type)](
                                            torch.cat(
                                                [h_dict[edge_type[0]][src], h_dict[edge_type[0]][dst], batch[edge_type].edge_attr],
                                                dim=-1
                                            ))
                                        )

        for node_type, x in h_dict.items():
            batch[node_type].x = x
        

        return batch
