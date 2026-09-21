# coding=utf-8
"""不确定性加权多任务混合损失。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeightedHybridLoss(nn.Module):
    """同方差不确定性加权多任务损失。

    同时优化 PCC、MAE、一阶差分与窗口内标准差四项，
    权重由可学习的 log-variance 参数自动平衡。
    """

    def __init__(
        self,
        eps=1e-8, init_log_var_pcc=0.0, init_log_var_mae=-1.5,
        init_log_var_diff=-2.0, init_log_var_std=-2.0,
        clamp_log_vars=True, log_var_min=-6.0, log_var_max=6.0,
    ):
        super().__init__()
        self.eps = eps
        self.clamp_log_vars = clamp_log_vars
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.log_var_pcc  = nn.Parameter(torch.tensor(float(init_log_var_pcc)))
        self.log_var_mae  = nn.Parameter(torch.tensor(float(init_log_var_mae)))
        self.log_var_diff = nn.Parameter(torch.tensor(float(init_log_var_diff)))
        self.log_var_std  = nn.Parameter(torch.tensor(float(init_log_var_std)))

    def _bounded(self, x):
        return torch.clamp(x, self.log_var_min, self.log_var_max) if self.clamp_log_vars else x

    def pearson(self, pred, target):
        b, f = pred.shape[0], pred.shape[1]
        pf = pred.reshape(b, f, -1)
        tf = target.reshape(b, f, -1)
        pc = pf - pf.mean(-1, keepdim=True)
        tc = tf - tf.mean(-1, keepdim=True)
        return (pc * tc).sum(-1) / ((pc**2).sum(-1).sqrt() * (tc**2).sum(-1).sqrt() + self.eps)

    def forward(self, pred, target):
        pf = pred.reshape(pred.shape[0], pred.shape[1], -1)
        tf = target.reshape(target.shape[0], target.shape[1], -1)
        pcc      = self.pearson(pred, target)
        loss_pcc  = 1.0 - pcc.mean()
        loss_mae  = F.l1_loss(pred, target)
        loss_diff = F.l1_loss(torch.diff(pf, dim=-1), torch.diff(tf, dim=-1))
        loss_std  = F.l1_loss(pf.std(-1, unbiased=False), tf.std(-1, unbiased=False))

        lv_pcc, lv_mae = self._bounded(self.log_var_pcc), self._bounded(self.log_var_mae)
        lv_diff, lv_std = self._bounded(self.log_var_diff), self._bounded(self.log_var_std)

        total = (torch.exp(-lv_pcc) * loss_pcc  + lv_pcc
               + torch.exp(-lv_mae) * loss_mae  + lv_mae
               + torch.exp(-lv_diff)* loss_diff + lv_diff
               + torch.exp(-lv_std) * loss_std  + lv_std)
        stats = {
            'loss_total': total.detach(), 'loss_pcc': loss_pcc.detach(),
            'loss_mae': loss_mae.detach(), 'loss_diff': loss_diff.detach(),
            'loss_std': loss_std.detach(),
            'weighted_pcc': (torch.exp(-lv_pcc)*loss_pcc+lv_pcc).detach(),
            'weighted_mae': (torch.exp(-lv_mae)*loss_mae+lv_mae).detach(),
            'weighted_diff': (torch.exp(-lv_diff)*loss_diff+lv_diff).detach(),
            'weighted_std': (torch.exp(-lv_std)*loss_std+lv_std).detach(),
            'pcc': pcc.mean().detach(),
            'log_var_pcc': lv_pcc.detach(), 'log_var_mae': lv_mae.detach(),
            'log_var_diff': lv_diff.detach(), 'log_var_std': lv_std.detach(),
            'precision_pcc':  torch.exp(-lv_pcc).detach(),
            'precision_mae':  torch.exp(-lv_mae).detach(),
            'precision_diff': torch.exp(-lv_diff).detach(),
            'precision_std':  torch.exp(-lv_std).detach(),
        }
        return total, stats
