import torch
import torch.nn as nn
from ingram import *
from pair_encoder import PairProjector
from global_module import Global_module
from localpairlayer import LocalPairLayer

class JointGlobalLocalLayer(nn.Module):
    def __init__(self, global_module, local_module, hidden_dim, sync_layers: int):
        super().__init__()
        self.global_module = global_module
        self.local_module = local_module
        self.gate_tok = nn.Linear(2 * hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.sync_layers = sync_layers
        self._h_cache = None

    def reset_state(self):
        self._pair_cursor = 0
        self._h_cache = None

    def forward(self, embedding, seq_emb, input_ids, input_mask, neighbor_graph, layer_idx: int):
        """
        One sync-layer step:
          - run Global(one layer) with shared Node→pair projector at `layer_idx`
          - run Local(one layer) with the same `layer_idx`
        No fusion here; caller (Model.forward) will fuse once per fusion layer after finishing all sync_layers.
        Returns:
          x_local: (B, L, H)
          x_global: (B, L, H)
          updated_emb: (V, H)
        """
        x_global, updated_emb, self._h_cache = self.global_module.forward_one_layer(
            embedding,
            input_ids,
            neighbor_graph["fact_rel_ids"],
            neighbor_graph["fact_ent_ids"],
            neighbor_graph["fact_entity_roles"],
            neighbor_graph["fact_rel_roles"],
            neighbor_graph["fact_pair_mask"],
            h_emb=self._h_cache,
            layer_idx=layer_idx,
        )

        x_local = self.local_module(seq_emb, input_mask, pair_layer_idx=layer_idx)

        return x_local, x_global, updated_emb

class GLoRE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_rel: int,
        num_ent: int,
        max_arity: int,
        sync_layers: int,
        fusion_layers: int,
        trans_layers: int,
        hidden_dim: int,
        local_heads: int,
        global_dropout: float,
        local_dropout: float,
        global_activation: str,
        decoder_activation: str,
        num_head: int,
        remove_mask: bool,
        dim_rel: int,
        hid_dim_ratio_rel: int,
        num_bin: int,
        rel_layers: int,
        use_relation_gnn: bool,
        use_global: bool,
        rel_triplets=None,
        bias: bool = True,
        times=2,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.parameter.Parameter(torch.Tensor(vocab_size, hidden_dim))

        self.rel_triplets = rel_triplets
        self.fusion_layers = fusion_layers
        self.sync_layers = sync_layers
    
        self.num_rel = num_rel
        self.num_ent = num_ent
        self.dim_rel = dim_rel
        self.max_arity = max_arity
        self.hidden_dim = hidden_dim

        self.use_relation_gnn = use_relation_gnn
        self.use_global = use_global

        self.pair_projector = PairProjector(hidden_dim, sync_layers, num_entity_roles=3)

        layer_dim_rel = hid_dim_ratio_rel * dim_rel
        self.rel_proj1 = nn.Linear(hidden_dim, layer_dim_rel, bias=bias)
        self.rel_proj2 = nn.Linear(layer_dim_rel, hidden_dim, bias=bias)

        self.layers_rel = nn.ModuleList([
            InGramRelationLayer(
                layer_dim_rel, layer_dim_rel, num_bin, bias=bias, num_head=num_head
            ) for _ in range(rel_layers)
        ])

        self.res_proj_rel = nn.ModuleList([
            nn.Linear(layer_dim_rel, layer_dim_rel, bias=bias)
            for _ in range(rel_layers)
        ])

        self.global_module = Global_module(
            hidden_dim, sync_layers, vocab_size, global_activation, global_dropout,
            pair_projector=self.pair_projector
        )

        self.local_module = LocalPairLayer(
                hidden_dim, max_arity, local_heads, local_dropout,
                decoder_activation, remove_mask,
                bias, trans_layers, times=times,
                pair_projector=self.pair_projector, pair_layers=sync_layers
            )
        
        self.joint_layer = JointGlobalLocalLayer(self.global_module, self.local_module, hidden_dim, sync_layers=sync_layers)
        
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(p=local_dropout)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.output_act = nn.GELU()
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.act = nn.ReLU()

        self.param_init()

    def param_init(self):
        nn.init.normal_(self.token_embedding, mean=0, std=0.02)
        with torch.no_grad():
            self.token_embedding[0].zero_()
    
    def forward(
        self,
        input_ids,
        input_mask,
        mask_position,
        mask_output,
        neighbor_graph
    ):
        B, L = input_ids.shape
        device = input_ids.device

        embedding = self.token_embedding  # (V, H)


        if self.use_relation_gnn:
            assert self.rel_triplets is not None, "rel_triplets must be provided when using relation GNN"
            emb = embedding.clone()
            emb_rel_init = embedding[2:2 + self.num_rel]
            layer_emb_rel = self.rel_proj1(emb_rel_init)
            for idx, layer in enumerate(self.layers_rel):
                layer_emb_rel = layer(layer_emb_rel, self.rel_triplets) + self.res_proj_rel[idx](layer_emb_rel)
                layer_emb_rel = self.act(layer_emb_rel)
            emb_rel = self.rel_proj2(layer_emb_rel)
            emb[2:2 + self.num_rel] = emb_rel
        else:
            emb = embedding
        emb = emb.to(device)

        x = torch.nn.functional.embedding(input_ids, emb)
        x = self.input_dropout(self.input_norm(x))

        if self.use_global:
            fused_seq = x
            for _ in range(self.fusion_layers):
                self.joint_layer.reset_state()
                x_local = fused_seq
                x_global = x_local
                for layer_idx in range(self.sync_layers):
                    x_local, x_global, emb = self.joint_layer(
                        emb, x_local, input_ids, input_mask, neighbor_graph, layer_idx=layer_idx
                    )
                gated_input = torch.cat([x_local, x_global], dim=-1)
                g = torch.sigmoid(self.joint_layer.gate_tok(gated_input))
                fused_seq = self.joint_layer.layer_norm(g * x_local + (1.0 - g) * x_global + fused_seq)
        else:
            fused_seq = x
            for layer_idx in range(self.sync_layers):
                fused_seq = self.local_module(fused_seq, input_mask)

        x_masked = fused_seq[torch.arange(B, device=device), mask_position]
        y = self.output_linear(x_masked)
        y = self.output_act(y)
        y = self.output_norm(y)

        logits = torch.matmul(y, emb.transpose(0,1)) + self.output_bias
        logits = logits.masked_fill(mask_output == 0, -1e5)

        return logits