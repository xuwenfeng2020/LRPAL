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


    def __init__(self, input_dim, num_classes, ensemble_size=5, dropout_rate=0.3,
                 uncertainty_beta=1.0, use_mc_dropout=True):
        super(UncertaintyAwareClassifier, self).__init__()

        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        self.uncertainty_beta = uncertainty_beta
        self.use_mc_dropout = use_mc_dropout


        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )


        self.class_head = nn.Linear(128, num_classes)

        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  
        )

  
        self.evidence_head = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softplus() 
        )


        nn.init.kaiming_normal_(self.class_head.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.uncertainty_head[0].weight, mode='fan_out')
        nn.init.kaiming_normal_(self.evidence_head[0].weight, mode='fan_out')

    def forward(self, x, return_uncertainty=True, mc_samples=None):
   
        batch_size = x.shape[0]

        if mc_samples is None:
            mc_samples = self.ensemble_size


        shared_features = self.shared_encoder(x)

  
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


        epistemic_samples = []

        if self.use_mc_dropout and self.training:

            for _ in range(mc_samples):

                self.shared_encoder.train()
                self.class_head.train()

    
                features_sample = self.shared_encoder(x)
                logits_sample = self.class_head(features_sample)
                prob_sample = F.softmax(logits_sample, dim=1)
                epistemic_samples.append(prob_sample)
        else:
 
            self.shared_encoder.eval()
            self.class_head.eval()

            with torch.no_grad():
                for _ in range(mc_samples):
                 
                    features_sample = self.shared_encoder(x)
                    logits_sample = self.class_head(features_sample)
                    prob_sample = F.softmax(logits_sample, dim=1)
                    epistemic_samples.append(prob_sample)


        epistemic_samples = torch.stack(epistemic_samples, dim=0)  # [S, B, C]
        epistemic_variance = torch.var(epistemic_samples, dim=0)  # [B, C]
        epistemic_uncertainty = epistemic_variance.mean(dim=1)  # [B]

      
        aleatoric_variance = self.uncertainty_head(shared_features).squeeze()  # [B]


        aleatoric_uncertainty = torch.log(1.0 + aleatoric_variance)

        total_uncertainty = epistemic_uncertainty + self.uncertainty_beta * aleatoric_uncertainty


        evidence = self.evidence_head(shared_features) + 1.0  
        alpha = evidence + 1.0  
        S = torch.sum(alpha, dim=1, keepdim=True)  
        prob_evidential = alpha / S

   
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
        
        ce_loss = F.cross_entropy(logits, targets)


        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


        uncertainty_reg = torch.mean(entropy)


        total_loss = ce_loss + lambda_reg * uncertainty_reg

        loss_dict = {
            "ce_loss": ce_loss.item(),
            "uncertainty_reg": uncertainty_reg.item(),
            "total_loss": total_loss.item()
        }


        if evidence is not None:
            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)


            one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            loss_evidential = torch.sum(
                one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1
            ).mean()


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

            self.classifier = UncertaintyAwareClassifier(
                input_dim=out_dim * 2,
                num_classes=num_classes,
                ensemble_size=ensemble_size,
                dropout_rate=dropout_rate,
                uncertainty_beta=uncertainty_beta
            )
        else:

            self.classifier = nn.Sequential(
                nn.Linear(out_dim * 2, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, num_classes)
            )


        self.feature_projection = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim * 2)
        )


        self.uncertainty_stats = {
            'epistemic_mean': 0.0,
            'aleatoric_mean': 0.0,
            'total_mean': 0.0
        }

    def forward(self, graph_data, image, labels=None, return_uncertainty=True):

        g_feat = self.graph_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.batch
        )  # [B, 128]


        img_feat = self.image_encoder(image)  # [B, 128]


        assert g_feat.shape[0] == img_feat.shape[0], "Batch大小不匹配"


        fused = torch.cat([g_feat, img_feat], dim=-1)  # [B, 256]


        fused_projected = self.feature_projection(fused)


        recon, coeff = self.lowrank(fused_projected)


        if self.use_uncertainty_classifier:
            classifier_output = self.classifier(
                fused_projected,
                return_uncertainty=return_uncertainty
            )

            logits = classifier_output["logits"]


            if return_uncertainty:
                self.uncertainty_stats['epistemic_mean'] = classifier_output["epistemic_uncertainty"].mean().item()
                self.uncertainty_stats['aleatoric_mean'] = classifier_output["aleatoric_uncertainty"].mean().item()
                self.uncertainty_stats['total_mean'] = classifier_output["total_uncertainty"].mean().item()


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

        return self.uncertainty_stats

    def reset_uncertainty_stats(self):

        self.uncertainty_stats = {
            'epistemic_mean': 0.0,
            'aleatoric_mean': 0.0,
            'total_mean': 0.0
        }



class AL_LRR_Model_Uncertainty_Margin(nn.Module):


    def __init__(self, g_in=3, img_in=3, hidden=64, out_dim=128, num_classes=2,
                 ensemble_size=5, margin=0.3, scale=30.0):
        super(AL_LRR_Model_Uncertainty_Margin, self).__init__()

        self.graph_encoder = GraphEncoder(in_channels=g_in,
                                          hidden_dim=hidden,
                                          out_dim=out_dim)

        self.image_encoder = ImageEncoder(in_channels=img_in,
                                          out_dim=out_dim)

        self.lowrank = LowRankModule(feature_dim=out_dim * 2)


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

        g_feat = self.graph_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.batch
        )  # [B, 128]


        img_feat = self.image_encoder(image)  # [B, 128]


        assert g_feat.shape[0] == img_feat.shape[0], "Batch大小不匹配"

 
        fused = torch.cat([g_feat, img_feat], dim=-1)  # [B, 256]

        fused_projected = self.feature_projection(fused)


        recon, coeff = self.lowrank(fused_projected)


        classifier_output = self.classifier(fused_projected, return_uncertainty=True)
        logits = classifier_output["logits"]

        if labels is not None:
   
            one_hot = F.one_hot(labels, num_classes=self.classifier.num_classes).float()


            logits_margin = logits.clone()
            logits_margin = logits_margin - self.margin * one_hot
            logits_margin = self.scale * logits_margin

            classifier_output["logits_margin"] = logits_margin


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
