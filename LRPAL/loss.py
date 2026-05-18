import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_classification_loss(logits, labels, class_weights=None):

    if class_weights is not None:
  
        class_weights = class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    return loss_fn(logits, labels)


def lowrank_loss(fused, recon, coeff, lambda_reg=1e-3):

    recon_error = F.mse_loss(recon, fused)


    coeff_reg = torch.mean(torch.sum(torch.abs(coeff), dim=1))

    total_loss = recon_error + lambda_reg * coeff_reg
    return total_loss


def uncertainty_aware_loss(logits, labels, uncertainties=None,
                           uncertainty_weight=0.1, focal_alpha=0.25, focal_gamma=2.0):

    if uncertainties is not None:
     
        sample_weights = 1.0 + uncertainty_weight * uncertainties
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss * sample_weights).mean()
    else:
      
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = focal_alpha * (1 - pt) ** focal_gamma * ce_loss
        loss = focal_loss.mean()

    return loss



def evidential_loss(logits, labels, evidence, num_classes, annealing_coeff=1.0):
  
    alpha = evidence + 1.0  # [B, C]

  
    S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]

   
    one_hot = F.one_hot(labels, num_classes=num_classes).float()  # [B, C]
    loss_classification = torch.sum(
        one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1
    ).mean()

   
    kl_div = torch.sum(torch.lgamma(alpha) - (alpha - 1.0) * torch.digamma(alpha), dim=1).mean()

 
    total_loss = loss_classification + annealing_coeff * kl_div

    return total_loss, loss_classification, kl_div



class TotalLoss(nn.Module):
 

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

     
        if class_weight is not None:
            self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None

     
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

       
        self.loss_history = {
            'ce': [],
            'lrr': [],
            'unc': [],
            'evidential': [],
            'total': []
        }

    def forward(self, logits, labels, fused=None, recon=None, coeff=None,
                uncertainties=None, evidence=None):

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

        loss_evidential = torch.tensor(0.0, device=logits.device)
        if self.use_evidential_loss and evidence is not None:
            loss_evidential, _, _ = evidential_loss(
                logits, labels, evidence,
                num_classes=logits.shape[1],
                annealing_coeff=self.evidential_annealing
            )
            self.loss_history['evidential'].append(loss_evidential.item())

 
        loss_lrr = torch.tensor(0.0, device=logits.device)
        if (fused is not None) and (recon is not None):
            loss_lrr = F.mse_loss(recon, fused)
            if (coeff is not None) and (self.coeff_reg > 0):
                loss_lrr = loss_lrr + self.coeff_reg * torch.mean(torch.sum(torch.abs(coeff), dim=1))

   
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


        if self.use_evidential_loss and evidence is not None:
            total = loss_evidential + self.alpha * loss_lrr + self.beta * loss_unc
        else:
            total = loss_ce + self.alpha * loss_lrr + self.beta * loss_unc

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
  
        stats = {}
        for key, values in self.loss_history.items():
            if values:
                stats[f'{key}_mean'] = sum(values) / len(values)
                stats[f'{key}_std'] = torch.std(torch.tensor(values)).item() if len(values) > 1 else 0.0
        return stats

    def clear_history(self):

        for key in self.loss_history:
            self.loss_history[key] = []
