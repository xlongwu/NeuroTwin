"""Profile 生成的个体算子，以及仅由 BOLD tokens 推断的病理增量。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def low_rank_energy(left: torch.Tensor, right: torch.Tensor,
                    coefficients: torch.Tensor) -> torch.Tensor:
    """逐样本 U diag(g) Vᵀ 的平方 Frobenius 范数，不构造完整算子。"""
    gram = (left.T @ left) * (right.T @ right)
    return torch.einsum('bi,ij,bj->b', coefficients, gram, coefficients)


class IndividualDynamicsGenerator(nn.Module):
    def __init__(self, features: int, dim: int, profile_dim: int,
                 hidden_dim: int = 128, rank: int = 8, graph_rank: int = 8,
                 residual_scale: float = 0.05, regional_diffusion: bool = True,
                 regional_decay: bool = False, personalized_graph: bool = True):
        super().__init__()
        if min(features, dim, profile_dim, hidden_dim, rank, graph_rank) < 1:
            raise ValueError('IDG 的维度和秩必须为正')
        self.features = features
        self.rank = rank
        self.graph_rank = graph_rank
        self.scale = residual_scale
        self.regional_diffusion = regional_diffusion
        self.regional_decay = regional_decay
        self.personalized_graph = personalized_graph
        self.generator = nn.Sequential(nn.Linear(profile_dim, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, 2 * features + rank + graph_rank))
        # 系数清零、基矩阵非零，保证起点兼容且新增分支能够获得梯度。
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        self.left = nn.Parameter(torch.randn(dim, rank) / dim ** 0.5)
        self.right = nn.Parameter(torch.randn(dim, rank) / dim ** 0.5)
        if personalized_graph:
            self.graph_left = nn.Parameter(torch.randn(features, graph_rank) / features ** 0.5)
            self.graph_right = nn.Parameter(torch.randn(features, graph_rank) / features ** 0.5)

    def forward(self, profile: torch.Tensor) -> dict:
        # 与 ODE 同用 fp32；整段 rollout 只生成一次。
        with torch.autocast(device_type=profile.device.type, enabled=False):
            k, decay, g, a = self.generator(profile.float()).split(
                [self.features, self.features, self.rank, self.graph_rank], dim=-1)
            delta = None
            if self.personalized_graph:
                raw = (self.graph_left.unsqueeze(0) * a.unsqueeze(1)) @ self.graph_right.T
                delta = 0.5 * (raw + raw.transpose(-1, -2))
            return {'kappa_delta': k if self.regional_diffusion else torch.zeros_like(k),
                    'gamma': F.softplus(decay - 10.0) if self.regional_decay else torch.zeros_like(decay),
                    'coefficients': g, 'graph_coefficients': a, 'graph_delta': delta}

    def residual(self, normalized_z: torch.Tensor, parameters: dict) -> torch.Tensor:
        return self.scale * ((F.gelu(normalized_z) @ self.right)
                             * parameters['coefficients'].unsqueeze(1)) @ self.left.T

    def regularization(self, parameters: dict, kappa_raw: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=kappa_raw.device.type, enabled=False):
            energy = low_rank_energy(self.left, self.right, parameters['coefficients'])
            if parameters['graph_delta'] is not None:
                energy = energy + parameters['graph_delta'].square().sum((-1, -2))
            diff = F.softplus(kappa_raw + parameters['kappa_delta']) - F.softplus(kappa_raw)
            return (energy + diff.square().sum(-1)).mean()


class PathologyDynamics(nn.Module):
    """不接收疾病标签或临床评分；标签仅由训练损失使用。"""
    def __init__(self, dim: int, pathology_dim: int = 16, rank: int = 4,
                 residual_scale: float = 0.05):
        super().__init__()
        if min(dim, pathology_dim, rank) < 1:
            raise ValueError('PDD 的维度和秩必须为正')
        self.scale = residual_scale
        self.encoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, pathology_dim), nn.Tanh())
        self.coefficients = nn.Linear(pathology_dim, rank)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)
        self.left = nn.Parameter(torch.randn(dim, rank) / dim ** 0.5)
        self.right = nn.Parameter(torch.randn(dim, rank) / dim ** 0.5)
        self.disease_head = nn.Linear(pathology_dim, 1)
        self.clinical_head = nn.Linear(pathology_dim, 1)

    def forward(self, bold_tokens: torch.Tensor) -> dict:
        with torch.autocast(device_type=bold_tokens.device.type, enabled=False):
            d = self.encoder(bold_tokens.float().mean(dim=(1, 2)))
            g = self.coefficients(d)
            return {'embedding': d, 'coefficients': g,
                    'disease_logit': self.disease_head(d).squeeze(-1),
                    'clinical_prediction': self.clinical_head(d).squeeze(-1),
                    'operator_energy': low_rank_energy(self.left, self.right, g)}

    def residual(self, normalized_z: torch.Tensor, parameters: dict) -> torch.Tensor:
        return self.scale * ((F.gelu(normalized_z) @ self.right)
                             * parameters['coefficients'].unsqueeze(1)) @ self.left.T
