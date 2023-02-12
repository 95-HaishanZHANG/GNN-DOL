import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch.nn.modules.distance import CosineSimilarity

from net.mlp import MLP


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()
        self.cos_sim = CosineSimilarity()
        self.distance = nn.PairwiseDistance()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x1, x2, edge_attr1, edge_attr2, matching_idx):
        """
        Does a single node and edge feature vectors update.
        Args:
            x1, x2: node features matrix
            edge_attr1, edge_attr2: edge features matrix 
            matching_idx: all node index which construct the matching matrix

        Returns: Updated Node and Edge Feature matrices

        """
        src, end = matching_idx

        # Edge Update
        if self.edge_model is not None:
            e_cos = self.cos_sim(x1[src], x2[end])
            
            sim_w, argsim = scatter_max(e_cos, src)
            out_edge_attr1 = self.edge_model(edge_attr1, x1, x2[end[argsim]]) 
            
            sim_w, argsim = scatter_max(e_cos, end)
            out_edge_attr2 = self.edge_model(edge_attr2, x2, x1[src[argsim]]) 

        # Node Update
        if self.node_model is not None:
            e_dis = -self.distance(edge_attr1[src], edge_attr2[end])
            e_cos = self.cos_sim(edge_attr1[src], edge_attr2[end])
            
            dis_w, argdis = scatter_max(e_dis, src)
            sim_w, argsim = scatter_max(e_cos, src)
            out_x1 = self.node_model(x1, edge_attr1, edge_attr2[end[argdis]], dis_w.unsqueeze(-1), sim_w.unsqueeze(-1))

            dis_w, argdis = scatter_max(e_dis, end)
            sim_w, argsim = scatter_max(e_cos, end)
            out_x2 = self.node_model(x2, edge_attr2, edge_attr1[src[argdis]], dis_w.unsqueeze(-1), sim_w.unsqueeze(-1))


        return out_x1, out_x2, out_edge_attr1, out_edge_attr2

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)

class EdgeModel(nn.Module):
    """
    Class used to peform the edge update
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, e, x1, x2):
        out = torch.cat([e, x1, x2], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    """
    Class used to peform the node update
    """
    def __init__(self, node_mlp):
        super(NodeModel, self).__init__()
        self.node_mlp = node_mlp

    def forward(self, x, e1, e2, w1, w2):
        out = torch.cat([x, e1, e2, w1, w2], dim=1) 
        # print('out shape: ', out.shape) 
        return self.node_mlp(out)

class MPNet(nn.Module):

    def __init__(self, model_params, mode):
        """
        Defines all components of the model
        Args:
            model_params: dictionary contaning all model hyperparameters
        """
        super(MPNet, self).__init__()

        self.model_params = model_params
        self.mode = mode

        # define params of mpn
        self.mpn_net = self._build_core_MPNet(model_params=model_params)
        


    def _build_core_MPNet(self, model_params):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            model_params: dictionary contaning all model hyperparameters
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define all MLPs involved in the graph network

        encoder_feats_dict = model_params['model_input_params']
        edge_model_in_dim = encoder_feats_dict['edge_init_dim'] * 1 + encoder_feats_dict['node_init_dim'] * 2
        node_model_in_dim = encoder_feats_dict['node_init_dim'] * 1 + encoder_feats_dict['edge_init_dim'] * 2 + 2 

        # Define all MLPs used within the MPN
        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']

        edge_mlp = MLP(input_dim=edge_model_in_dim,
                       fc_dims=edge_model_feats_dict['fc_dims'],
                       dropout_p=edge_model_feats_dict['dropout_p'],
                       use_batchnorm=edge_model_feats_dict['use_batchnorm'])
        
        node_mlp = MLP(input_dim=node_model_in_dim,
                          fc_dims=node_model_feats_dict['fc_dims'],
                          dropout_p=node_model_feats_dict['dropout_p'],
                          use_batchnorm=node_model_feats_dict['use_batchnorm'])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp),
                         node_model=NodeModel(node_mlp = node_mlp))
        
    def _reshape_matching_idx(self, matching_idx, num_x1, num_x2):
        src, end = matching_idx.detach().cpu().numpy()
        num_x1 = num_x1.detach().cpu().numpy()
        num_x2 = num_x2.detach().cpu().numpy()
        num_pre = 0
        for j, i_idx in enumerate(num_x1):
            if j == 0:
                continue 
            num_pre = num_pre + (num_x1[j-1] * num_x2[j-1])
            src[num_pre:] = src[num_pre:] + num_x1[j-1]
            end[num_pre:] = end[num_pre:] + num_x2[j-1] 
            # print('src: ', num_pre, src, end)
        src = np.vstack((src, end))
        return torch.from_numpy(src).cuda()

    def forward(self, x1, x2, edge_attr1, edge_attr2, matching_idx, num_x1, num_x2):
        """ 
        Args:
            data: object containing attribues
              - x: node features matrix [num_nodes, num_feas]
              - edge_index: tensor with shape [2, num_edges]
              - edge_attr: edge features matrix [num_edges, num_feas]
              - x_e: edge features matrix ordered by node [num_nodes, num_feas] 

        Returns:
            latent_node_feats
            latent_edge_feats
        """

        if self.mode in ['train', 'val']:
            matching_idx = self._reshape_matching_idx(matching_idx.T, num_x1, num_x2)
        else:
            matching_idx = matching_idx.T

        latent_node_feats1, latent_node_feats2, latent_edge_feats1, latent_edge_feats2 = self.mpn_net(x1, x2, edge_attr1, edge_attr2, matching_idx)

        # normalize
        latent_node_feats1 = F.normalize(latent_node_feats1, dim=1)
        latent_node_feats2 = F.normalize(latent_node_feats2, dim=1)
        latent_edge_feats1 = F.normalize(latent_edge_feats1, dim=1)  
        latent_edge_feats2 = F.normalize(latent_edge_feats2, dim=1) 
        
        return latent_node_feats1, latent_node_feats2, latent_edge_feats1, latent_edge_feats2