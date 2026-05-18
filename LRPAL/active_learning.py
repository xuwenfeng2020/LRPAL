# ============================================
# active_learning.py（改进稳定版）
# ============================================

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import pairwise_distances


# ============================================
# 1. 不确定性评分（Entropy）
# ============================================
def compute_uncertainty_score(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.cpu().numpy()


# ============================================
# 2. 原型代表性得分（避免挑噪声）
# ============================================
def compute_represent_score(features, prototypes):
    """
    features: [B, D]
    prototypes: [K, D]
    返回：样本与最近原型的相似度（越大越代表性好）
    """
    sim = F.cosine_similarity(
        features.unsqueeze(1),
        prototypes.unsqueeze(0),
        dim=-1
    )  # [B, K]

    max_sim = torch.max(sim, dim=1)[0]     # 越大越接近某个原型簇
    rep_score = max_sim.cpu().numpy()

    return rep_score


# ============================================
# 3. 离群点检测（过滤噪声样本）
# ============================================
def detect_outliers(features, ratio=0.05):
    """
    使用 L2 距离 + IQR 方法检测离群样本
    返回：不是离群点的位置（True 表示保留）
    """
    feat = features.cpu().numpy()
    center = np.mean(feat, axis=0)
    d = np.linalg.norm(feat - center, axis=1)

    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    keep = d < threshold
    return keep  # boolean mask


# ============================================
# 4. k-center 多样性采样（核心）
# ============================================
def kcenter_greedy(features, pool_idx, select_count):
    """
    features: numpy [N, D]
    pool_idx: 还未选中的样本索引
    """
    X = features[pool_idx]
    n = X.shape[0]

    # 随机选一个做第一个中心
    first = np.random.randint(0, n)
    centers = [first]

    dist = pairwise_distances(X, X[first:first+1]).reshape(-1)

    for _ in range(select_count - 1):
        idx = np.argmax(dist)
        centers.append(idx)

        d_new = pairwise_distances(X, X[idx:idx+1]).reshape(-1)
        dist = np.minimum(dist, d_new)

    selected = pool_idx[centers]
    return selected


# ============================================
# 5. 综合选样策略（不确定性 + 多样性 + 代表性）
# ============================================
def select_samples(model, dataloader, device, select_ratio=0.05):
    model.eval()

    all_feats = []
    all_logits = []

    with torch.no_grad():
        for graph, imgs, _ in dataloader:
            graph = graph.to(device)
            imgs = imgs.to(device)

            outputs, _, _, C = model(graph, imgs)

            all_feats.append(C.cpu())      # 🔥 用Z
            all_logits.append(outputs.cpu())

    feats = torch.cat(all_feats, dim=0)
    logits = torch.cat(all_logits, dim=0)

    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    feats_np = feats.numpy()

    N = len(entropy)
    num_select = int(select_ratio * N)

    # Top uncertain
    idx = np.argsort(entropy.numpy())[-3 * num_select:]

    # K-center
    selected = kcenter_greedy(feats_np, idx, num_select)

    return selected
