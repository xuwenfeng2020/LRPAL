import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from config import get_config

cfg = get_config()


class GraphEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, out_dim=128):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        g_emb = global_mean_pool(x, batch)  # [B, 128]
        return g_emb


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, out_dim=128):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(),
            nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(),
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base_dim * 4, out_dim)

    def forward(self, img):
        assert img.ndim == 4, f"ImageEncoder期望4D输入，实际是{img.ndim}D"
        assert img.shape[1] == 3, f"ImageEncoder期望3通道输入，实际是{img.shape[1]}通道"
        x = self.conv(img)
        x = x.view(x.size(0), -1)  # [B, base_dim*4]
        x = self.fc(x)  # [B, 128]
        return x


class LowRankModule(nn.Module):
    def __init__(self, feature_dim, num_prototypes=8):
        super(LowRankModule, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))

    def forward(self, features):
        sim = torch.matmul(features, self.prototypes.T)  # [B, K]
        coeff = F.softmax(sim, dim=1)
        recon = torch.matmul(coeff, self.prototypes)  # [B, D]
        return recon, coeff


class UncertaintyAwareClassifier(nn.Module):
    """
    不确定性感知深度分类器
    同时输出分类概率、认知不确定性（模型不确定性）和任意不确定性（数据不确定性）
    参考：Uncertainty-Aware Deep Neural Network Training for Imbalanced Geochemical Data Distributions
    """

    def __init__(self, input_dim, num_classes, ensemble_size=5, dropout_rate=0.3,
                 uncertainty_beta=1.0, use_mc_dropout=True):
        super(UncertaintyAwareClassifier, self).__init__()

        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        self.uncertainty_beta = uncertainty_beta
        self.use_mc_dropout = use_mc_dropout

        # 主分类网络
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 分类头
        self.class_head = nn.Linear(128, num_classes)

        # 不确定性头（用于估计任意不确定性）
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保输出为正，表示方差
        )

        # 证据头（用于狄利克雷分布参数估计，可选）
        self.evidence_head = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softplus()  # 证据值必须为正
        )

        # 初始化
        nn.init.kaiming_normal_(self.class_head.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.uncertainty_head[0].weight, mode='fan_out')
        nn.init.kaiming_normal_(self.evidence_head[0].weight, mode='fan_out')

    def forward(self, x, return_uncertainty=True, mc_samples=None):
        """
        前向传播

        Args:
            x: 输入特征 [B, D]
            return_uncertainty: 是否返回不确定性估计
            mc_samples: Monte Carlo采样次数（如果使用MC Dropout）

        Returns:
            字典包含：logits, prob, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty
        """
        batch_size = x.shape[0]

        if mc_samples is None:
            mc_samples = self.ensemble_size

        # 获取共享特征
        shared_features = self.shared_encoder(x)

        # 获取分类logits和概率
        logits = self.class_head(shared_features)
        prob = F.softmax(logits, dim=1)

        if not return_uncertainty:
            return {
                "logits": logits,
                "prob": prob,
                "epistemic_uncertainty": torch.zeros(batch_size, device=x.device),
                "aleatoric_uncertainty": torch.zeros(batch_size, device=x.device),
                "total_uncertainty": torch.zeros(batch_size, device=x.device)
            }

        # ===== 估计认知不确定性（模型不确定性） =====
        epistemic_samples = []

        if self.use_mc_dropout and self.training:
            # 训练时：使用MC Dropout进行多次前向传播
            for _ in range(mc_samples):
                # 启用Dropout
                self.shared_encoder.train()
                self.class_head.train()

                # 前向传播（Dropout会随机激活）
                features_sample = self.shared_encoder(x)
                logits_sample = self.class_head(features_sample)
                prob_sample = F.softmax(logits_sample, dim=1)
                epistemic_samples.append(prob_sample)
        else:
            # 推理时或使用集成方法
            self.shared_encoder.eval()
            self.class_head.eval()

            with torch.no_grad():
                for _ in range(mc_samples):
                    # 轻微扰动权重来模拟模型不确定性
                    features_sample = self.shared_encoder(x)
                    logits_sample = self.class_head(features_sample)
                    prob_sample = F.softmax(logits_sample, dim=1)
                    epistemic_samples.append(prob_sample)

        # 计算认知不确定性（预测分布的方差）
        epistemic_samples = torch.stack(epistemic_samples, dim=0)  # [S, B, C]
        epistemic_variance = torch.var(epistemic_samples, dim=0)  # [B, C]
        epistemic_uncertainty = epistemic_variance.mean(dim=1)  # [B]

        # ===== 估计任意不确定性（数据不确定性） =====
        # 使用不确定性头预测方差
        aleatoric_variance = self.uncertainty_head(shared_features).squeeze()  # [B]

        # 对任意不确定性进行变换（对数方差 -> 标准差）
        aleatoric_uncertainty = torch.log(1.0 + aleatoric_variance)

        # ===== 计算总不确定性 =====
        total_uncertainty = epistemic_uncertainty + self.uncertainty_beta * aleatoric_uncertainty

        # ===== 证据学习（可选） =====
        # 使用狄利克雷分布参数估计不确定性
        evidence = self.evidence_head(shared_features) + 1.0  # [B, C]，加1确保为正
        alpha = evidence + 1.0  # 狄利克雷分布参数
        S = torch.sum(alpha, dim=1, keepdim=True)  # 总浓度参数
        prob_evidential = alpha / S

        # 证据不确定性：u = C / S，其中C为类别数
        evidential_uncertainty = self.num_classes / S.squeeze()

        return {
            "logits": logits,
            "prob": prob,
            "prob_evidential": prob_evidential,
            "evidence": evidence,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "evidential_uncertainty": evidential_uncertainty,
            "total_uncertainty": total_uncertainty,
            "shared_features": shared_features
        }

    def compute_uncertainty_loss(self, logits, targets, evidence=None, lambda_reg=0.01):
        """
        计算不确定性感知损失

        Args:
            logits: 分类logits [B, C]
            targets: 真实标签 [B]
            evidence: 证据值 [B, C]（如果使用证据学习）
            lambda_reg: 正则化系数

        Returns:
            total_loss, loss_dict
        """
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(logits, targets)

        # 不确定性正则化项（鼓励模型对困难样本表达不确定性）
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # 对高熵（高不确定性）预测给予惩罚
        uncertainty_reg = torch.mean(entropy)

        # 总损失
        total_loss = ce_loss + lambda_reg * uncertainty_reg

        loss_dict = {
            "ce_loss": ce_loss.item(),
            "uncertainty_reg": uncertainty_reg.item(),
            "total_loss": total_loss.item()
        }

        # 如果使用证据学习，添加证据损失
        if evidence is not None:
            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)

            # 狄利克雷损失
            one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            loss_evidential = torch.sum(
                one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1
            ).mean()

            # KL散度正则化
            kl_reg = torch.sum(torch.lgamma(alpha) - (alpha - 1.0) * torch.digamma(alpha), dim=1).mean()

            total_loss = total_loss + loss_evidential + 0.1 * kl_reg

            loss_dict["loss_evidential"] = loss_evidential.item()
            loss_dict["kl_reg"] = kl_reg.item()

        return total_loss, loss_dict


class AL_LRR_Model(nn.Module):
    def __init__(self, g_in=3, img_in=3, hidden=64, out_dim=128, num_classes=2,
                 use_uncertainty_classifier=True,
                 ensemble_size=5,
                 dropout_rate=0.3,
                 uncertainty_beta=1.0):
        super(AL_LRR_Model, self).__init__()

        self.graph_encoder = GraphEncoder(in_channels=g_in,
                                          hidden_dim=hidden,
                                          out_dim=out_dim)

        self.image_encoder = ImageEncoder(in_channels=img_in,
                                          out_dim=out_dim)

        self.lowrank = LowRankModule(feature_dim=out_dim * 2)

        self.use_uncertainty_classifier = use_uncertainty_classifier

        if use_uncertainty_classifier:
            # 使用不确定性感知分类器
            self.classifier = UncertaintyAwareClassifier(
                input_dim=out_dim * 2,
                num_classes=num_classes,
                ensemble_size=ensemble_size,
                dropout_rate=dropout_rate,
                uncertainty_beta=uncertainty_beta
            )
        else:
            # 保持原有的线性分类器
            self.classifier = nn.Sequential(
                nn.Linear(out_dim * 2, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, num_classes)
            )

        # 特征投影头
        self.feature_projection = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim * 2)
        )

        # 用于存储不确定性统计
        self.uncertainty_stats = {
            'epistemic_mean': 0.0,
            'aleatoric_mean': 0.0,
            'total_mean': 0.0
        }

    def forward(self, graph_data, image, labels=None, return_uncertainty=True):
        # 图编码
        g_feat = self.graph_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.batch
        )  # [B, 128]

        # 图像编码
        img_feat = self.image_encoder(image)  # [B, 128]

        # 验证Batch和通道维度
        assert g_feat.shape[0] == img_feat.shape[0], "Batch大小不匹配"

        # 融合
        fused = torch.cat([g_feat, img_feat], dim=-1)  # [B, 256]

        # 特征投影
        fused_projected = self.feature_projection(fused)

        # 低秩重构
        recon, coeff = self.lowrank(fused_projected)

        # 分类
        if self.use_uncertainty_classifier:
            classifier_output = self.classifier(
                fused_projected,
                return_uncertainty=return_uncertainty
            )

            logits = classifier_output["logits"]

            # 更新不确定性统计
            if return_uncertainty:
                self.uncertainty_stats['epistemic_mean'] = classifier_output["epistemic_uncertainty"].mean().item()
                self.uncertainty_stats['aleatoric_mean'] = classifier_output["aleatoric_uncertainty"].mean().item()
                self.uncertainty_stats['total_mean'] = classifier_output["total_uncertainty"].mean().item()

            # 返回所有信息
            output_dict = {
                "logits": logits,
                "fused": fused,
                "recon": recon,
                "coeff": coeff,
                "prob": classifier_output.get("prob", F.softmax(logits, dim=1)),
                "epistemic_uncertainty": classifier_output.get("epistemic_uncertainty", torch.zeros_like(logits[:, 0])),
                "aleatoric_uncertainty": classifier_output.get("aleatoric_uncertainty", torch.zeros_like(logits[:, 0])),
                "total_uncertainty": classifier_output.get("total_uncertainty", torch.zeros_like(logits[:, 0])),
                "evidence": classifier_output.get("evidence", None),
                "shared_features": classifier_output.get("shared_features", None)
            }

            # 如果提供了标签，可以计算不确定性损失
            if labels is not None and self.training:
                uncertainty_loss, loss_dict = self.classifier.compute_uncertainty_loss(
                    logits, labels, classifier_output.get("evidence", None)
                )
                output_dict["uncertainty_loss"] = uncertainty_loss
                output_dict["uncertainty_loss_dict"] = loss_dict

            return output_dict
        else:
            logits = self.classifier(fused)
            prob = F.softmax(logits, dim=1)

            return {
                "logits": logits,
                "fused": fused,
                "recon": recon,
                "coeff": coeff,
                "prob": prob,
                "epistemic_uncertainty": torch.zeros(fused.shape[0], device=fused.device),
                "aleatoric_uncertainty": torch.zeros(fused.shape[0], device=fused.device),
                "total_uncertainty": torch.zeros(fused.shape[0], device=fused.device)
            }

    def get_uncertainty_stats(self):
        """获取不确定性统计信息"""
        return self.uncertainty_stats

    def reset_uncertainty_stats(self):
        """重置不确定性统计"""
        self.uncertainty_stats = {
            'epistemic_mean': 0.0,
            'aleatoric_mean': 0.0,
            'total_mean': 0.0
        }


# 不确定性感知模型的变体（带边际）
class AL_LRR_Model_Uncertainty_Margin(nn.Module):
    """不确定性感知模型，结合边际损失"""

    def __init__(self, g_in=3, img_in=3, hidden=64, out_dim=128, num_classes=2,
                 ensemble_size=5, margin=0.3, scale=30.0):
        super(AL_LRR_Model_Uncertainty_Margin, self).__init__()

        self.graph_encoder = GraphEncoder(in_channels=g_in,
                                          hidden_dim=hidden,
                                          out_dim=out_dim)

        self.image_encoder = ImageEncoder(in_channels=img_in,
                                          out_dim=out_dim)

        self.lowrank = LowRankModule(feature_dim=out_dim * 2)

        # 使用不确定性感知分类器
        self.classifier = UncertaintyAwareClassifier(
            input_dim=out_dim * 2,
            num_classes=num_classes,
            ensemble_size=ensemble_size
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim * 2)
        )

        self.margin = margin
        self.scale = scale

    def forward(self, graph_data, image, labels=None):
        # 图编码
        g_feat = self.graph_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.batch
        )  # [B, 128]

        # 图像编码
        img_feat = self.image_encoder(image)  # [B, 128]

        # 验证Batch和通道维度
        assert g_feat.shape[0] == img_feat.shape[0], "Batch大小不匹配"

        # 融合
        fused = torch.cat([g_feat, img_feat], dim=-1)  # [B, 256]

        # 特征投影
        fused_projected = self.feature_projection(fused)

        # 低秩重构
        recon, coeff = self.lowrank(fused_projected)

        # 分类
        classifier_output = self.classifier(fused_projected, return_uncertainty=True)
        logits = classifier_output["logits"]

        # 应用边际损失（如果提供了标签）
        if labels is not None:
            # 对logits应用边际
            one_hot = F.one_hot(labels, num_classes=self.classifier.num_classes).float()

            # 计算带边际的logits
            logits_margin = logits.clone()
            logits_margin = logits_margin - self.margin * one_hot
            logits_margin = self.scale * logits_margin

            classifier_output["logits_margin"] = logits_margin

        # 返回结果
        output_dict = {
            "logits": logits,
            "fused": fused,
            "recon": recon,
            "coeff": coeff,
            "prob": classifier_output["prob"],
            "epistemic_uncertainty": classifier_output["epistemic_uncertainty"],
            "aleatoric_uncertainty": classifier_output["aleatoric_uncertainty"],
            "total_uncertainty": classifier_output["total_uncertainty"]
        }

        if labels is not None:
            output_dict["logits_margin"] = classifier_output.get("logits_margin", logits)

        return output_dict