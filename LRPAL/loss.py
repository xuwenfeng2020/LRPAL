# # ============================================
# # losses.py
# # 定义多种损失：交叉熵 + 低秩 + 不确定性感知损失
# # ============================================
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # ============================================
# # 带权重的分类损失（交叉熵）
# # ============================================
# def weighted_classification_loss(logits, labels, class_weights=None):
#     """
#     带类别权重的分类损失
#     logits: [B, C]
#     labels: [B]
#     class_weights: [C] 类别权重，可以为None
#     """
#     if class_weights is not None:
#         # 确保权重在正确的设备上
#         class_weights = class_weights.to(logits.device)
#         loss_fn = nn.CrossEntropyLoss(weight=class_weights)
#     else:
#         loss_fn = nn.CrossEntropyLoss()
#
#     return loss_fn(logits, labels)
#
#
# # ============================================
# # 低秩重构损失
# # ============================================
# def lowrank_loss(fused, recon, coeff, lambda_reg=1e-3):
#     """
#     fused: 原始融合特征
#     recon: 低秩重构特征
#     coeff: 样本对原型的系数 [B, K]
#     lambda_reg: 稀疏正则
#     """
#     # 重构误差
#     recon_error = F.mse_loss(recon, fused)
#
#     # 系数稀疏约束（鼓励低秩）
#     coeff_reg = torch.mean(torch.sum(torch.abs(coeff), dim=1))
#
#     total_loss = recon_error + lambda_reg * coeff_reg
#     return total_loss
#
#
# # ============================================
# # 不确定性感知损失
# # ============================================
# def uncertainty_aware_loss(logits, labels, uncertainties=None,
#                            uncertainty_weight=0.1, focal_alpha=0.25, focal_gamma=2.0):
#     """
#     不确定性感知损失函数
#     对高不确定性样本给予更高权重或使用Focal Loss
#
#     Args:
#         logits: 分类logits [B, C]
#         labels: 真实标签 [B]
#         uncertainties: 不确定性分数 [B]
#         uncertainty_weight: 不确定性权重
#         focal_alpha, focal_gamma: Focal Loss参数
#     """
#     if uncertainties is not None:
#         # 基于不确定性的样本权重
#         # 高不确定性样本获得更高权重
#         sample_weights = 1.0 + uncertainty_weight * uncertainties
#         loss = F.cross_entropy(logits, labels, reduction='none')
#         loss = (loss * sample_weights).mean()
#     else:
#         # 使用Focal Loss处理不平衡数据
#         ce_loss = F.cross_entropy(logits, labels, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = focal_alpha * (1 - pt) ** focal_gamma * ce_loss
#         loss = focal_loss.mean()
#
#     return loss
#
#
# # ============================================
# # 证据损失（用于不确定性量化）
# # ============================================
# def evidential_loss(logits, labels, evidence, num_classes, annealing_coeff=1.0):
#     """
#     证据学习损失（狄利克雷分布）
#
#     Args:
#         logits: 分类logits [B, C]
#         labels: 真实标签 [B]
#         evidence: 证据值 [B, C]
#         num_classes: 类别数
#         annealing_coeff: 退火系数（用于KL散度）
#     """
#     # 狄利克雷分布参数
#     alpha = evidence + 1.0  # [B, C]
#
#     # 总浓度参数
#     S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]
#
#     # 分类损失（负对数似然）
#     one_hot = F.one_hot(labels, num_classes=num_classes).float()  # [B, C]
#     loss_classification = torch.sum(
#         one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1
#     ).mean()
#
#     # KL散度正则化（鼓励先验）
#     kl_div = torch.sum(torch.lgamma(alpha) - (alpha - 1.0) * torch.digamma(alpha), dim=1).mean()
#
#     # 总损失
#     total_loss = loss_classification + annealing_coeff * kl_div
#
#     return total_loss, loss_classification, kl_div
#
#
# # ============================================
# # 综合损失封装（更新为支持不确定性）
# # ============================================
# class TotalLoss(nn.Module):
#     """
#     total = CE + alpha * lowrank_recon_error + beta * uncertainty_regularizer
#
#     更新：支持不确定性感知损失
#     """
#
#     def __init__(self, alpha=1, beta=0.3, coeff_reg=1e-3,
#                  use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
#                  use_uncertainty_loss=False, uncertainty_weight=0.1,
#                  use_evidential_loss=False, evidential_annealing=1.0,
#                  class_weight=None, reduction='mean'):
#         super().__init__()
#
#         self.alpha = alpha
#         self.beta = beta
#         self.coeff_reg = coeff_reg
#         self.use_focal_loss = use_focal_loss
#         self.use_uncertainty_loss = use_uncertainty_loss
#         self.use_evidential_loss = use_evidential_loss
#         self.uncertainty_weight = uncertainty_weight
#         self.evidential_annealing = evidential_annealing
#         self.reduction = reduction
#
#         # 初始化类别权重
#         if class_weight is not None:
#             self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float32))
#         else:
#             self.class_weight = None
#
#         # 选择损失函数
#         if use_focal_loss:
#             self.classification_loss = nn.CrossEntropyLoss(reduction='none')
#             self.focal_alpha = focal_alpha
#             self.focal_gamma = focal_gamma
#         else:
#             if class_weight is not None:
#                 self.classification_loss = nn.CrossEntropyLoss(
#                     weight=self.class_weight,
#                     reduction=reduction
#                 )
#             else:
#                 self.classification_loss = nn.CrossEntropyLoss(reduction=reduction)
#
#         # 用于存储损失历史
#         self.loss_history = {
#             'ce': [],
#             'lrr': [],
#             'unc': [],
#             'evidential': [],
#             'total': []
#         }
#
#     def forward(self, logits, labels, fused=None, recon=None, coeff=None,
#                 uncertainties=None, evidence=None):
#         # 分类损失
#         if self.use_focal_loss:
#             ce_loss_raw = self.classification_loss(logits, labels)
#             pt = torch.exp(-ce_loss_raw)
#             focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss_raw
#             loss_ce = focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
#         elif self.use_uncertainty_loss and uncertainties is not None:
#             loss_ce = uncertainty_aware_loss(
#                 logits, labels, uncertainties,
#                 uncertainty_weight=self.uncertainty_weight
#             )
#         else:
#             loss_ce = self.classification_loss(logits, labels)
#
#         # 证据损失（如果使用）
#         loss_evidential = torch.tensor(0.0, device=logits.device)
#         if self.use_evidential_loss and evidence is not None:
#             loss_evidential, _, _ = evidential_loss(
#                 logits, labels, evidence,
#                 num_classes=logits.shape[1],
#                 annealing_coeff=self.evidential_annealing
#             )
#             self.loss_history['evidential'].append(loss_evidential.item())
#
#         # 低秩重构损失
#         loss_lrr = torch.tensor(0.0, device=logits.device)
#         if (fused is not None) and (recon is not None):
#             loss_lrr = F.mse_loss(recon, fused)
#             if (coeff is not None) and (self.coeff_reg > 0):
#                 loss_lrr = loss_lrr + self.coeff_reg * torch.mean(torch.sum(torch.abs(coeff), dim=1))
#
#         # 不确定性正则项
#         loss_unc = torch.tensor(0.0, device=logits.device)
#         if self.beta != 0:
#             probs = F.softmax(logits, dim=1)
#             ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
#             if self.reduction == 'mean':
#                 loss_unc = ent.mean()
#             elif self.reduction == 'sum':
#                 loss_unc = ent.sum()
#             else:
#                 loss_unc = ent
#
#         # 总损失
#         if self.use_evidential_loss and evidence is not None:
#             total = loss_evidential + self.alpha * loss_lrr + self.beta * loss_unc
#         else:
#             total = loss_ce + self.alpha * loss_lrr + self.beta * loss_unc
#
#         # 记录损失历史
#         self.loss_history['ce'].append(loss_ce.item())
#         self.loss_history['lrr'].append(loss_lrr.item())
#         self.loss_history['unc'].append(loss_unc.item())
#         self.loss_history['total'].append(total.item())
#
#         return total, {
#             "ce": loss_ce.item(),
#             "evidential": loss_evidential.item() if self.use_evidential_loss else 0.0,
#             "lrr": loss_lrr.item(),
#             "unc": loss_unc.item(),
#             "total": total.item()
#         }
#
#     def get_loss_stats(self):
#         """获取损失统计信息"""
#         stats = {}
#         for key, values in self.loss_history.items():
#             if values:
#                 stats[f'{key}_mean'] = sum(values) / len(values)
#                 stats[f'{key}_std'] = torch.std(torch.tensor(values)).item() if len(values) > 1 else 0.0
#         return stats
#
#     def clear_history(self):
#         """清空损失历史"""
#         for key in self.loss_history:
#             self.loss_history[key] = []

# ============================================
# losses.py
# 定义多种损失：交叉熵 + 低秩 + 不确定性感知损失
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 带权重的分类损失（交叉熵）
# ============================================
def weighted_classification_loss(logits, labels, class_weights=None):
    """
    带类别权重的分类损失
    logits: [B, C]
    labels: [B]
    class_weights: [C] 类别权重，可以为None
    """
    if class_weights is not None:
        # 确保权重在正确的设备上
        class_weights = class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    return loss_fn(logits, labels)


# ============================================
# 低秩重构损失
# ============================================
def lowrank_loss(fused, recon, coeff, lambda_reg=1e-3):
    """
    fused: 原始融合特征
    recon: 低秩重构特征
    coeff: 样本对原型的系数 [B, K]
    lambda_reg: 稀疏正则
    """
    # 重构误差
    recon_error = F.mse_loss(recon, fused)

    # 系数稀疏约束（鼓励低秩）
    coeff_reg = torch.mean(torch.sum(torch.abs(coeff), dim=1))

    total_loss = recon_error + lambda_reg * coeff_reg
    return total_loss


# ============================================
# 不确定性感知损失
# ============================================
def uncertainty_aware_loss(logits, labels, uncertainties=None,
                           uncertainty_weight=0.1, focal_alpha=0.25, focal_gamma=2.0):
    """
    不确定性感知损失函数
    对高不确定性样本给予更高权重或使用Focal Loss

    Args:
        logits: 分类logits [B, C]
        labels: 真实标签 [B]
        uncertainties: 不确定性分数 [B]
        uncertainty_weight: 不确定性权重
        focal_alpha, focal_gamma: Focal Loss参数
    """
    if uncertainties is not None:
        # 基于不确定性的样本权重
        # 高不确定性样本获得更高权重
        sample_weights = 1.0 + uncertainty_weight * uncertainties
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss * sample_weights).mean()
    else:
        # 使用Focal Loss处理不平衡数据
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = focal_alpha * (1 - pt) ** focal_gamma * ce_loss
        loss = focal_loss.mean()

    return loss


# ============================================
# 证据损失（用于不确定性量化）
# ============================================
def evidential_loss(logits, labels, evidence, num_classes, annealing_coeff=1.0):
    """
    证据学习损失（狄利克雷分布）

    Args:
        logits: 分类logits [B, C]
        labels: 真实标签 [B]
        evidence: 证据值 [B, C]
        num_classes: 类别数
        annealing_coeff: 退火系数（用于KL散度）
    """
    # 狄利克雷分布参数
    alpha = evidence + 1.0  # [B, C]

    # 总浓度参数
    S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]

    # 分类损失（负对数似然）
    one_hot = F.one_hot(labels, num_classes=num_classes).float()  # [B, C]
    loss_classification = torch.sum(
        one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1
    ).mean()

    # KL散度正则化（鼓励先验）
    kl_div = torch.sum(torch.lgamma(alpha) - (alpha - 1.0) * torch.digamma(alpha), dim=1).mean()

    # 总损失
    total_loss = loss_classification + annealing_coeff * kl_div

    return total_loss, loss_classification, kl_div


# ============================================
# 综合损失封装（更新为支持不确定性）
# ============================================
class TotalLoss(nn.Module):
    """
    total = CE + alpha * lowrank_recon_error + beta * uncertainty_regularizer

    更新：支持不确定性感知损失
    """

    def __init__(self, alpha=1, beta=0.3, coeff_reg=1e-3,
                 use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
                 use_uncertainty_loss=False, uncertainty_weight=0.1,
                 use_evidential_loss=False, evidential_annealing=1.0,
                 class_weight=None, reduction='mean'):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.coeff_reg = coeff_reg
        self.use_focal_loss = use_focal_loss
        self.use_uncertainty_loss = use_uncertainty_loss
        self.use_evidential_loss = use_evidential_loss
        self.uncertainty_weight = uncertainty_weight
        self.evidential_annealing = evidential_annealing
        self.reduction = reduction

        # 初始化类别权重
        if class_weight is not None:
            self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None

        # 选择损失函数
        if use_focal_loss:
            self.classification_loss = nn.CrossEntropyLoss(reduction='none')
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma
        else:
            if class_weight is not None:
                self.classification_loss = nn.CrossEntropyLoss(
                    weight=self.class_weight,
                    reduction=reduction
                )
            else:
                self.classification_loss = nn.CrossEntropyLoss(reduction=reduction)

        # 用于存储损失历史
        self.loss_history = {
            'ce': [],
            'lrr': [],
            'unc': [],
            'evidential': [],
            'total': []
        }

    def forward(self, logits, labels, fused=None, recon=None, coeff=None,
                uncertainties=None, evidence=None):
        # 分类损失
        if self.use_focal_loss:
            ce_loss_raw = self.classification_loss(logits, labels)
            pt = torch.exp(-ce_loss_raw)
            focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss_raw
            loss_ce = focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
        elif self.use_uncertainty_loss and uncertainties is not None:
            loss_ce = uncertainty_aware_loss(
                logits, labels, uncertainties,
                uncertainty_weight=self.uncertainty_weight
            )
        else:
            loss_ce = self.classification_loss(logits, labels)

        # 证据损失（如果使用）
        loss_evidential = torch.tensor(0.0, device=logits.device)
        if self.use_evidential_loss and evidence is not None:
            loss_evidential, _, _ = evidential_loss(
                logits, labels, evidence,
                num_classes=logits.shape[1],
                annealing_coeff=self.evidential_annealing
            )
            self.loss_history['evidential'].append(loss_evidential.item())

        # 低秩重构损失
        loss_lrr = torch.tensor(0.0, device=logits.device)
        if (fused is not None) and (recon is not None):
            loss_lrr = F.mse_loss(recon, fused)
            if (coeff is not None) and (self.coeff_reg > 0):
                loss_lrr = loss_lrr + self.coeff_reg * torch.mean(torch.sum(torch.abs(coeff), dim=1))

        # 不确定性正则项
        loss_unc = torch.tensor(0.0, device=logits.device)
        if self.beta != 0:
            probs = F.softmax(logits, dim=1)
            ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            if self.reduction == 'mean':
                loss_unc = ent.mean()
            elif self.reduction == 'sum':
                loss_unc = ent.sum()
            else:
                loss_unc = ent

        # 总损失
        if self.use_evidential_loss and evidence is not None:
            total = loss_evidential + self.alpha * loss_lrr + self.beta * loss_unc
        else:
            total = loss_ce + self.alpha * loss_lrr + self.beta * loss_unc

        # 记录损失历史
        self.loss_history['ce'].append(loss_ce.item())
        self.loss_history['lrr'].append(loss_lrr.item())
        self.loss_history['unc'].append(loss_unc.item())
        self.loss_history['total'].append(total.item())

        return total, {
            "ce": loss_ce.item(),
            "evidential": loss_evidential.item() if self.use_evidential_loss else 0.0,
            "lrr": loss_lrr.item(),
            "unc": loss_unc.item(),
            "total": total.item()
        }

    def get_loss_stats(self):
        """获取损失统计信息"""
        stats = {}
        for key, values in self.loss_history.items():
            if values:
                stats[f'{key}_mean'] = sum(values) / len(values)
                stats[f'{key}_std'] = torch.std(torch.tensor(values)).item() if len(values) > 1 else 0.0
        return stats

    def clear_history(self):
        """清空损失历史"""
        for key in self.loss_history:
            self.loss_history[key] = []