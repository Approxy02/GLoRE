import torch
import torch.nn as nn
from pair_encoder import PairProjector

class Global_module(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_nodes: int,
        activation: str = "elu",
        global_dropout: float = 0.1,
        pair_projector: PairProjector | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        assert pair_projector is not None, "pair_projector must be provided for shared node->pair projection"
        self.pair_projector = pair_projector
        self.num_entity_roles = 3
        self.num_rel_roles    = 2 
        self.W_en_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_entity_roles)])
            for _ in range(num_layers)
        ])
        self.P_en_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                for _ in range(self.num_entity_roles)
            ])
            for _ in range(num_layers)
        ])
        self.W_rn_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_rel_roles)])
            for _ in range(num_layers)
        ])
        self.P_rn_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                for _ in range(self.num_rel_roles)
            ])
            for _ in range(num_layers)
        ])
        self.ln_e = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])  # for hyperedges
        self.ln_v = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])  # for entities
        self.ln_r = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])  # for relations
        self.act = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }[activation]
        self.drop_res = nn.Dropout(p=global_dropout)

        self.param_init()

    def param_init(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, 
                node_emb: torch.Tensor,
                input_ids: torch.LongTensor,
                fact_rel_ids: torch.LongTensor,
                fact_ent_ids: torch.LongTensor,
                fact_entity_roles: torch.LongTensor,
                fact_rel_roles: torch.LongTensor,
                fact_pair_mask: torch.Tensor,
                h_emb: torch.Tensor | None = None
        ):
        """
        Returns:
          x_global: (B, L, H)  # sequence-level updated representations via updated embedding table
          updated_embedding: (V, H)
          h_emb: (B*num_h, H)  # hyperedge embedding after global message passing

        Note:
          If h_emb is provided, it is reused/updated instead of zero-initialization.
        """
        device = node_emb.device
        dtype = node_emb.dtype
        B, L = input_ids.shape
        _, num_h, max_p = fact_rel_ids.shape

        node_emb = node_emb.clone()

        flat_rel = fact_rel_ids.view(-1, max_p)
        flat_ent = fact_ent_ids.view(-1, max_p)
        flat_eroles = fact_entity_roles.view(-1, max_p)
        flat_rroles = fact_rel_roles.view(-1, max_p)
        flat_mask = fact_pair_mask.view(-1, max_p)

        h_idx, pos = torch.where(flat_mask)
        ent_idx = flat_ent[h_idx, pos]
        rel_idx = flat_rel[h_idx, pos]
        eroles = flat_eroles[h_idx, pos]
        rroles = flat_rroles[h_idx, pos]

        counts_h = torch.bincount(h_idx, minlength=flat_rel.size(0)).unsqueeze(1)
        counts_v = torch.bincount(ent_idx, minlength=node_emb.size(0)).unsqueeze(1)
        counts_r = torch.bincount(rel_idx, minlength=node_emb.size(0)).unsqueeze(1)

        unique_ent = torch.unique(ent_idx)
        unique_rel = torch.unique(rel_idx)

        if (h_emb is None) or (h_emb.shape[0] != flat_rel.size(0)) or (h_emb.shape[1] != self.dim):
            h_emb = torch.zeros(flat_rel.size(0), self.dim, device=device, dtype=dtype)
        else:
            h_emb = h_emb.to(device=device, dtype=dtype)

        for l in range(self.num_layers):

            # Node -> Hyperedge
            v_prev = node_emb[ent_idx]
            r_prev = node_emb[rel_idx]
            E = h_idx.size(0)

            msgs = self.pair_projector.project(v_prev, r_prev, eroles, layer_idx=l)

            agg_h = torch.zeros(flat_rel.size(0), self.dim, device=device, dtype=dtype)
            agg_h.index_add_(0, h_idx, msgs)
            agg_h = agg_h / counts_h.clamp(min=1)
            agg_h = self.act(agg_h)
            h_emb = self.ln_e[l](h_emb + self.drop_res(agg_h))

            # Hyperedge -> Entity
            h_prev = h_emb[h_idx]  # type: ignore
            msgs_ent = torch.zeros(E, self.dim, device=device, dtype=dtype)
            for er in range(self.num_entity_roles):
                mask = (eroles == er)
                if not mask.any():
                    continue
                msgs_ent[mask] = self.P_en_layers[l][er](self.W_en_layers[l][er](h_prev[mask]))  # type: ignore

            agg_v = torch.zeros(node_emb.size(0), self.dim, device=device, dtype=dtype)
            agg_v.index_add_(0, ent_idx, msgs_ent)
            agg_v = agg_v / counts_v.clamp(min=1)
            agg_v = self.act(agg_v)

            upd_ent = self.ln_v[l](node_emb[unique_ent] + self.drop_res(agg_v[unique_ent]))
            node_emb = node_emb.scatter(
                0,
                unique_ent.unsqueeze(1).expand(-1, node_emb.size(1)),
                upd_ent
            )

            # Hyperedge -> Relation
            msgs_rel = torch.zeros(E, self.dim, device=device, dtype=dtype)
            for rr in range(self.num_rel_roles):
                mask = (rroles == rr)
                if not mask.any():
                    continue
                msgs_rel[mask] = self.P_rn_layers[l][rr](self.W_rn_layers[l][rr](h_prev[mask]))  # type: ignore

            agg_r = torch.zeros(node_emb.size(0), self.dim, device=device, dtype=dtype)
            agg_r.index_add_(0, rel_idx, msgs_rel)
            agg_r = agg_r / counts_r.clamp(min=1)
            agg_r = self.act(agg_r)

            upd_rel = self.ln_r[l](node_emb[unique_rel] + self.drop_res(agg_r[unique_rel]))
            node_emb = node_emb.scatter(
                0,
                unique_rel.unsqueeze(1).expand(-1, node_emb.size(1)),
                upd_rel
            )

        x_global = torch.nn.functional.embedding(input_ids, node_emb)
        return x_global, node_emb, h_emb

    def forward_one_layer(self, 
                node_emb: torch.Tensor,
                input_ids: torch.LongTensor,
                fact_rel_ids: torch.LongTensor,
                fact_ent_ids: torch.LongTensor,
                fact_entity_roles: torch.LongTensor,
                fact_rel_roles: torch.LongTensor,
                fact_pair_mask: torch.Tensor,
                h_emb: torch.Tensor | None,
                layer_idx: int
        ):
        """
        Run exactly ONE global layer (layer_idx) and return updated x_global, node_emb, h_emb.
        This method is designed for step-synchronous fusion with LocalPairLayer so that both
        modules use the same PairProjector layer (W/U/P) per fusion step.
        """
        device = node_emb.device
        dtype = node_emb.dtype
        B, L = input_ids.shape
        _, num_h, max_p = fact_rel_ids.shape

        node_emb = node_emb.clone()

        flat_rel = fact_rel_ids.view(-1, max_p)
        flat_ent = fact_ent_ids.view(-1, max_p)
        flat_eroles = fact_entity_roles.view(-1, max_p)
        flat_rroles = fact_rel_roles.view(-1, max_p)
        flat_mask = fact_pair_mask.view(-1, max_p)

        h_idx, pos = torch.where(flat_mask)
        ent_idx = flat_ent[h_idx, pos]
        rel_idx = flat_rel[h_idx, pos]
        eroles = flat_eroles[h_idx, pos]
        rroles = flat_rroles[h_idx, pos]
        E = h_idx.size(0)

        counts_h = torch.bincount(h_idx, minlength=flat_rel.size(0)).unsqueeze(1)
        counts_v = torch.bincount(ent_idx, minlength=node_emb.size(0)).unsqueeze(1)
        counts_r = torch.bincount(rel_idx, minlength=node_emb.size(0)).unsqueeze(1)

        unique_ent = torch.unique(ent_idx)
        unique_rel = torch.unique(rel_idx)

        if (h_emb is None) or (h_emb.shape[0] != flat_rel.size(0)) or (h_emb.shape[1] != self.dim):
            h_emb = torch.zeros(flat_rel.size(0), self.dim, device=device, dtype=dtype)
        else:
            h_emb = h_emb.to(device=device, dtype=dtype)

        l = int(layer_idx)

        # Node -> Hyperedge  (use shared projector at layer l)
        v_prev = node_emb[ent_idx]
        r_prev = node_emb[rel_idx]
        msgs = self.pair_projector.project(v_prev, r_prev, eroles, layer_idx=l)

        agg_h = torch.zeros(flat_rel.size(0), self.dim, device=device, dtype=dtype)
        agg_h.index_add_(0, h_idx, msgs)
        agg_h = agg_h / counts_h.clamp(min=1)
        agg_h = self.act(agg_h)
        h_emb = self.ln_e[l](h_emb + self.drop_res(agg_h))


        # Hyperedge -> Entity
        h_prev = h_emb[h_idx] # type: ignore
        msgs_ent = torch.zeros(E, self.dim, device=device, dtype=dtype)
        for er in range(self.num_entity_roles):
            mask = (eroles == er)
            if mask.any():
                msgs_ent[mask] = self.P_en_layers[l][er](self.W_en_layers[l][er](h_prev[mask]))  # type: ignore

        agg_v = torch.zeros(node_emb.size(0), self.dim, device=device, dtype=dtype)
        agg_v.index_add_(0, ent_idx, msgs_ent)
        agg_v = agg_v / counts_v.clamp(min=1)
        agg_v = self.act(agg_v)

        upd_ent = self.ln_v[l](node_emb[unique_ent] + self.drop_res(agg_v[unique_ent]))
        node_emb = node_emb.scatter(
            0,
            unique_ent.unsqueeze(1).expand(-1, node_emb.size(1)),
            upd_ent
        )

        # Hyperedge -> Relation
        msgs_rel = torch.zeros(E, self.dim, device=device, dtype=dtype)
        for rr in range(self.num_rel_roles):
            mask = (rroles == rr)
            if mask.any():
                msgs_rel[mask] = self.P_rn_layers[l][rr](self.W_rn_layers[l][rr](h_prev[mask]))  # type: ignore

        agg_r = torch.zeros(node_emb.size(0), self.dim, device=device, dtype=dtype)
        agg_r.index_add_(0, rel_idx, msgs_rel)
        agg_r = agg_r / counts_r.clamp(min=1)
        agg_r = self.act(agg_r)

        upd_rel = self.ln_r[l](node_emb[unique_rel] + self.drop_res(agg_r[unique_rel]))
        node_emb = node_emb.scatter(
            0,
            unique_rel.unsqueeze(1).expand(-1, node_emb.size(1)),
            upd_rel
        )

        x_global = torch.nn.functional.embedding(input_ids, node_emb)
        return x_global, node_emb, h_emb