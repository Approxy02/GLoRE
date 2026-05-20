import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class InGramRelationLayer(nn.Module):
    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias = True, num_head = 8):
        super(InGramRelationLayer, self).__init__()

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        self.attn_proj = nn.Linear(2*dim_in_rel, dim_out_rel, bias = bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias = bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope = 0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain = nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)
    
    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)
        
        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]

        device = emb_rel.device

        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim = -1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * \
                        self.attn_vec).sum(dim = -1, keepdim = True) + self.attn_bin[relation_triplets[...,2]]

        scatter_idx = head_idxs.unsqueeze(dim = -1).repeat(1, self.num_head).unsqueeze(dim = -1)

        attn_val_max = torch.zeros((num_rel, self.num_head, 1), device=device).scatter_reduce(dim = 0, \
                                                                    index = scatter_idx, \
                                                                    src = attn_val_raw, reduce = 'amax', \
                                                                    include_self = False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1), device=device).index_add(dim = 0, index = head_idxs, source = attn_val)

        beta = attn_val / (attn_sums[head_idxs]+1e-16)
        
        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel), device=device).index_add(dim = 0, \
                                                                                            index = head_idxs, 
                                                                                            source = beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))

        return output.flatten(1,-1)