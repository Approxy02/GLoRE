import torch
import torch.nn as nn
from transformer import *
import numpy as np

class LocalPairLayer(nn.Module):
    def __init__(
        self, hidden_dim, max_arity, local_heads, local_dropout,
        activation, remove_mask, bias, trans_layers, times=1,
        pair_projector=None, pair_layers: int = 1
        ):
        super().__init__()
        self.num_entity_roles = 3
        self.num_rel_roles    = 2

        self.hidden_dim = hidden_dim
        self.max_aux = max_arity - 2

        assert pair_projector is not None, "pair_projector must be provided for shared node->pair projection"
        self.pair_projector = pair_projector
        self.pair_layers = pair_layers
        assert hidden_dim % local_heads == 0, "hidden_dim must be divisible by local_heads for edge embeddings"


        self.W_e = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=True)
            for _ in range(self.num_entity_roles)
        ])
        self.P_e = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(self.num_entity_roles)
        ])

        self.W_r = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=True)
            for _ in range(self.num_entity_roles)
        ])
        self.P_r = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(self.num_entity_roles)
        ])

        self.ln_e = nn.LayerNorm(hidden_dim) 
        self.ln_r = nn.LayerNorm(hidden_dim)
        self.merge_act = nn.GELU()

        pairs = [(1,0), (1,2)] + [
            (3 + 2*i, 4 + 2*i)
            for i in range(self.max_aux)
        ]
        pair_pos = [(0,0), (0,1)] + [
            (1,2)
            for _ in range(self.max_aux)
        ]

        rel_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        ent_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        rel_roles = torch.tensor([pos[0] for pos in pair_pos], dtype=torch.long)
        ent_roles = torch.tensor([pos[1] for pos in pair_pos], dtype=torch.long)

        self.register_buffer('rel_idx',   rel_idx)
        self.register_buffer('ent_idx',   ent_idx)
        self.register_buffer('rel_roles', rel_roles)
        self.register_buffer('ent_roles', ent_roles)

        edge_labels = self.make_pair_edge_labels(max_arity)

        self.register_buffer('pair_edge_labels', torch.from_numpy(edge_labels))

        self.edge_query_embedding = nn.Embedding(5, hidden_dim // local_heads, padding_idx=0)
        self.edge_key_embedding = nn.Embedding(5, hidden_dim // local_heads, padding_idx=0)
        self.edge_value_embedding = nn.Embedding(5, hidden_dim // local_heads, padding_idx=0)

        self.transformers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, local_heads, local_dropout,
                activation, remove_mask,
                bias, times=times
            )
            for _ in range(trans_layers)
        ])
        self.param_init()
    
    def param_init(self):
        for layer in self.W_e:
            nn.init.xavier_normal_(layer.weight, gain = nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)
        for layer in self.W_r:
            nn.init.xavier_normal_(layer.weight, gain = nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)
        for seq_list in [self.P_e, self.P_r]:
            for seq in seq_list:
                for m in seq: # type: ignore
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                        nn.init.zeros_(m.bias)
        nn.init.ones_(self.ln_e.weight)
        nn.init.zeros_(self.ln_e.bias)
        nn.init.ones_(self.ln_r.weight)
        nn.init.zeros_(self.ln_r.bias)


    def make_pair_edge_labels(self, max_arity):
        max_aux = max_arity - 2
        pair_edge_labels = []
        pair_edge_labels.append([0, 1] + [2] * max_aux)
        pair_edge_labels.append([1, 0] + [3] * max_aux)
        for idx in range(max_aux):
            pair_edge_labels.append([2, 3] + [4] * idx + [0] + [4] * (max_aux - idx - 1))
        pair_edge_labels = np.asarray(pair_edge_labels).astype("int64")

        return pair_edge_labels
    


    def forward(self, x, input_mask, pair_layer_idx: int = 0):
        """
        x: (B, L, H)
        input_mask: (B, L, L)  ← outer(orig_input_mask, orig_input_mask)
        """
        B, L, H = x.size()
        device = x.device

        token_mask = input_mask.diagonal(dim1=1, dim2=2).bool()
        orig_mask = token_mask

        rel_idx   = self.rel_idx
        ent_idx   = self.ent_idx
        rel_roles = self.rel_roles
        ent_roles = self.ent_roles

        rel_repr = x[:, rel_idx, :]
        ent_repr = x[:, ent_idx, :]

        valid = (orig_mask[:, rel_idx] & orig_mask[:, ent_idx]).unsqueeze(-1)
        rel_repr = rel_repr * valid
        ent_repr = ent_repr * valid

        pair_layer_idx = int(pair_layer_idx) % max(1, self.pair_layers)
        pair_repr = self.pair_projector.project(ent_repr, rel_repr, ent_roles, layer_idx=pair_layer_idx)

        pair_graph = self.pair_edge_labels.unsqueeze(0)
        pair_graph = pair_graph.repeat(B, 1, 1).unsqueeze(1)
        pair_graph = pair_graph.repeat(1, 5, 1, 1)

        pair_mask = valid.squeeze(-1)
        pair_mask2d = pair_mask.unsqueeze(2) & pair_mask.unsqueeze(1)
        pair_graph = pair_graph * pair_mask2d.unsqueeze(1)

        P_size = pair_graph.size(-1)
        diag_idx = torch.arange(P_size, device=pair_graph.device)
        diag_vals = pair_graph[:, :, diag_idx, diag_idx]
        diag_mask = pair_mask.unsqueeze(1)
        pair_graph[:, :, diag_idx, diag_idx] = torch.where(
            diag_mask, torch.ones_like(diag_vals), diag_vals
        )

        edge_labels = self.pair_edge_labels.unsqueeze(0).repeat(B, 1, 1).clone()
        diag_vals_lbl = edge_labels[:, diag_idx, diag_idx]
        edge_labels[:, diag_idx, diag_idx] = torch.where(
            pair_mask, torch.ones_like(diag_vals_lbl), diag_vals_lbl
        )
        edge_q = self.edge_query_embedding(edge_labels)
        edge_k = self.edge_key_embedding(edge_labels)
        edge_v = self.edge_value_embedding(edge_labels)

        edge_q = edge_q * pair_mask2d.unsqueeze(-1)
        edge_k = edge_k * pair_mask2d.unsqueeze(-1)
        edge_v = edge_v * pair_mask2d.unsqueeze(-1)

        for transformer in self.transformers:
            pair_repr = transformer(pair_repr, pair_graph, edge_q, edge_k, edge_v, ent_roles)
            pair_repr = pair_repr * valid
        
        _, P, _ = pair_repr.size()

        msg_e = torch.zeros_like(pair_repr)
        for ent_role in range(self.num_entity_roles):
            m = (ent_roles == ent_role)
            if m.any():
                v_ = self.W_e[ent_role](pair_repr[:, m, :])
                msg_e[:, m, :] = self.P_e[ent_role](v_)
        agg_e = torch.zeros(B, L, H, device=device)
        agg_e.index_add_(1, self.ent_idx, msg_e) 

        count_e = torch.zeros(B, L, 1, device=device)
        ones = torch.ones(B, P, 1, device=device)
        count_e.index_add_(1, self.ent_idx, ones)
        count_e = count_e.clamp(min=1.0)

        mean_e = agg_e / count_e

        x = x + self.merge_act(mean_e)
        x = self.ln_e(x)

        msg_r = torch.zeros_like(pair_repr)
        for ent_role in range(self.num_entity_roles):
            m = (ent_roles == ent_role)
            if m.any():
                r_ = self.W_r[ent_role](pair_repr[:, m, :])
                msg_r[:, m, :] = self.P_r[ent_role](r_)
        agg_r = torch.zeros(B, L, H, device=device)
        agg_r.index_add_(1, self.rel_idx, msg_r)

        count_r = torch.zeros(B, L, 1, device=device)
        count_r.index_add_(1, self.rel_idx, ones)
        count_r = count_r.clamp(min=1.0)

        mean_r = agg_r / count_r

        x = x + self.merge_act(mean_r)
        x = self.ln_r(x)

        return x
