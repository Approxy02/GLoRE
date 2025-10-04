import torch
import torch.nn as nn

class PairProjector(nn.Module):
    """
    """
    def __init__(self, dim: int, num_layers: int, num_entity_roles: int = 3):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_entity_roles = num_entity_roles

        self.W_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(num_entity_roles)])
            for _ in range(num_layers)
        ])
        self.U_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(num_entity_roles)])
            for _ in range(num_layers)
        ])
        self.P_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                for _ in range(num_entity_roles)
            ])
            for _ in range(num_layers)
        ])
        self._init_params()

    def _init_params(self):
        for l in range(self.num_layers):
            for role in range(self.num_entity_roles):
                nn.init.xavier_normal_(self.W_layers[l][role].weight, gain=nn.init.calculate_gain('relu')) # pyright: ignore[reportIndexIssue]
                nn.init.zeros_(self.W_layers[l][role].bias) # type: ignore
                nn.init.xavier_normal_(self.U_layers[l][role].weight, gain=nn.init.calculate_gain('relu')) # type: ignore
                nn.init.zeros_(self.U_layers[l][role].bias) # type: ignore
                for m in self.P_layers[l][role]: # type: ignore
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                        nn.init.zeros_(m.bias)

    @torch.no_grad()
    def num_pair_layers(self) -> int:
        return self.num_layers

    def project(self, v_repr: torch.Tensor, r_repr: torch.Tensor, ent_roles: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        v_repr: (E, H) or (B, P, H)
        r_repr: (E, H) or (B, P, H)
        ent_roles: (E,) or (P,)
        layer_idx: which layer's W/U/P to use
        returns: pair_repr with same shape as v_repr/r_repr
        """
        if v_repr.dim() == 2:
            E, H = v_repr.size()
            out = torch.zeros_like(v_repr)
            for role in range(self.num_entity_roles):
                mask = (ent_roles == role)
                if mask.any():
                    v_p = self.W_layers[layer_idx][role](v_repr[mask]) # type: ignore
                    r_p = self.U_layers[layer_idx][role](r_repr[mask]) # type: ignore
                    out[mask] = self.P_layers[layer_idx][role](v_p * r_p) # type: ignore
            return out
        elif v_repr.dim() == 3:
            B, P, H = v_repr.size()
            out = torch.zeros_like(v_repr)
            for role in range(self.num_entity_roles):
                mask = (ent_roles == role)
                if mask.any():
                    v_p = self.W_layers[layer_idx][role](v_repr[:, mask, :]) # type: ignore
                    r_p = self.U_layers[layer_idx][role](r_repr[:, mask, :]) # type: ignore
                    out[:, mask, :] = self.P_layers[layer_idx][role](v_p * r_p) # type: ignore
            return out
        else:
            raise ValueError("Unsupported tensor rank for project(): expected 2 or 3 dims.")