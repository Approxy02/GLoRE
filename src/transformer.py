import torch
import torch.nn as nn
import math

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, bias: bool) -> None:
        super().__init__()
        self.heads = heads     

        self.layer_subpair = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.layer_objpair = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.layer_valpair = nn.Linear(hidden_dim, hidden_dim, bias=bias)


    def forward(self, x : torch.Tensor, pair_roles: torch.Tensor):
        B, P, H = x.size()
        device = x.device

        out = torch.zeros_like(x)
        m_sub = (pair_roles == 0).to(device)
        m_obj = (pair_roles == 1).to(device)
        m_val = (pair_roles == 2).to(device)

        if m_sub.any():
            out[:, m_sub, :] = self.layer_subpair(x[:, m_sub, :])
        if m_obj.any():
            out[:, m_obj, :] = self.layer_objpair(x[:, m_obj, :])
        if m_val.any():
            out[:, m_val, :] = self.layer_valpair(x[:, m_val, :])

        return out.view(B, P, self.heads, H // self.heads)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, remove_mask: bool, bias: bool) -> None:
        super().__init__()
        assert hidden_dim % heads == 0
        self.dim = hidden_dim // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(hidden_dim, heads, bias)
        self.key = PrepareForMultiHeadAttention(hidden_dim, heads, bias)
        self.value = PrepareForMultiHeadAttention(hidden_dim, heads, True)

        self.pair_role_emb = nn.Embedding(3, self.dim)

        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.remove_mask = remove_mask
        self.scale = 1 / math.sqrt(self.dim)
        # trasformer-xl
        self.r_w_bias = nn.Parameter(torch.Tensor(heads, self.dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(heads, self.dim))

    def get_mask(self, graph: torch.Tensor):
        mask_2d = (graph > 0).any(dim=1)
        return mask_2d.unsqueeze(1).repeat(1, self.heads, 1, 1)

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor, pair_roles: torch.Tensor):
        shape = query.shape[:-1]
        query = self.query(query, pair_roles)
        key = self.key(key, pair_roles)
        value = self.value(value, pair_roles)

        role_bias = self.pair_role_emb(pair_roles)
        role_bias = role_bias.unsqueeze(0).unsqueeze(2)
        query = query + role_bias 

        seq_len = query.size(1)
        
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) + torch.einsum("bqhd,bqkd->bhqk", query, edge_key) + torch.einsum("bkqd,bkhd->bhqk", edge_query, key) + torch.einsum("bkqd,bqkd->bqk", edge_query, edge_key).unsqueeze(1)
        scores = scores * self.scale
        mask = self.get_mask(graph)
   
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum("bhqk,bkhd->bqhd", attn, value) + torch.einsum("bhqk,bqkd->bqhd", attn, edge_value)
        x = x.reshape(*shape, -1)

        return self.output(x)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation) -> None:
        super().__init__()

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        else:
            act = nn.Identity()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layer(x)

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, activation: str, remove_mask: bool, bias=True, times=2) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout_prob, remove_mask, bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, hidden_dim * times, hidden_dim, activation)
        self.param_init()

    def param_init(self):
        for name, param in self.named_parameters():
            if "norm" in name:
                continue
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name or "att" in name:
                nn.init.normal_(param, mean=0, std=0.02)
            elif "embedding" in name:
                nn.init.normal_(param, mean=0, std=0.02)
            else:
                raise TypeError("Invalid Parameters")

    def forward(self, x: torch.Tensor, graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor, pair_roles: torch.Tensor):
        attn = self.attention(query=x, key=x, value=x, graph=graph, edge_key=edge_key, edge_value=edge_value, edge_query=edge_query, pair_roles=pair_roles)
        x = self.norm_attention(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ff))
        return x
