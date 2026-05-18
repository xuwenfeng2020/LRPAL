import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from loss import TotalLoss
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.metrics import precision_recall_curve


# ======================================
# 改进的阈值寻找函数
# ======================================
def find_optimal_threshold(y_true, y_score, method='balanced'):
    """
    改进的阈值寻找方法

    Args:
        method:
            'balanced' - 基于平衡准确率
            'f1' - 基于F1分数
            'gmean' - 基于几何平均数
            'youden' - Youden's J统计量
            'pr' - 精确率-召回率曲线
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_value = -1

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        if method == 'balanced':
            # 平衡准确率，对不平衡数据更友好
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = (sensitivity + specificity) / 2

        elif method == 'f1':
            value = f1_score(y_true, y_pred, zero_division=0)

        elif method == 'gmean':
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = np.sqrt(sensitivity * specificity) if sensitivity > 0 and specificity > 0 else 0

        elif method == 'youden':
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = sensitivity + specificity - 1

        elif method == 'pr':
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            if len(f1_scores) > 0:
                best_idx = np.argmax(f1_scores)
                return thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.5

        if value > best_value:
            best_value = value
            best_threshold = threshold

    return best_threshold


# ======================================
# 改进的指标计算
# ======================================
def compute_comprehensive_metrics(y_true, y_score, threshold=0.5, uncertainties=None):
    """
    计算更全面的指标，特别关注不平衡数据集

    Args:
        y_true: 真实标签
        y_score: 预测概率分数
        threshold: 分类阈值
        uncertainties: 不确定性分数（可选）

    Returns:
        包含所有指标的字典
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # 使用阈值进行预测
    y_pred = (y_score >= threshold).astype(int)

    metrics = {}

    # 基础指标
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)

    # 精确率、召回率、F1（两种平均方式）
    try:
        # 二元分类的精确率、召回率、F1
        metrics["precision"] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    except Exception as e:
        print(f"计算二元分类指标时出错: {e}")
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

    # 宏平均（对不平衡数据更有意义）
    try:
        metrics["precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"计算宏平均指标时出错: {e}")
        metrics["precision_macro"] = 0.0
        metrics["recall_macro"] = 0.0
        metrics["f1_macro"] = 0.0

    # 计算每个类别的精确率和召回率
    unique_classes = np.unique(y_true)
    for cls in unique_classes:
        # 对于二分类问题，计算正类和负类的指标
        if cls == 1:
            # 正类的精确率、召回率
            try:
                pos_precision = precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
                pos_recall = recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
                metrics[f"class_{cls}_precision"] = pos_precision
                metrics[f"class_{cls}_recall"] = pos_recall
            except:
                metrics[f"class_{cls}_precision"] = 0.0
                metrics[f"class_{cls}_recall"] = 0.0
        elif cls == 0:
            # 负类的精确率、召回率
            try:
                neg_precision = precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
                neg_recall = recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
                metrics[f"class_{cls}_precision"] = neg_precision
                metrics[f"class_{cls}_recall"] = neg_recall
            except:
                metrics[f"class_{cls}_precision"] = 0.0
                metrics[f"class_{cls}_recall"] = 0.0

    # AUC指标
    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except:
        metrics["auc"] = 0.0

    try:
        metrics["auprc"] = average_precision_score(y_true, y_score)
    except:
        metrics["auprc"] = 0.0

    # 马修斯相关系数（对不平衡数据稳定）
    try:
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    except:
        metrics["mcc"] = 0.0

    # 计算混淆矩阵元素
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["tp"] = tp
    metrics["fn"] = fn

    # 计算特异性（真阴性率）
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["specificity"] = specificity

    # 计算阳性预测值（PPV）和阴性预测值（NPV）
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics["ppv"] = ppv  # 阳性预测值（Positive Predictive Value）
    metrics["npv"] = npv  # 阴性预测值（Negative Predictive Value）

    # 几何平均数
    sensitivity = metrics["recall"]
    metrics["gmean"] = np.sqrt(sensitivity * specificity) if sensitivity > 0 and specificity > 0 else 0

    # F-beta分数（可调整召回率和精确率的权重）
    # F2分数（更重视召回率）
    try:
        metrics["f2"] = (1 + 2 ** 2) * (metrics["precision"] * metrics["recall"]) / (
                    (2 ** 2 * metrics["precision"]) + metrics["recall"]) if (metrics["precision"] + metrics[
            "recall"]) > 0 else 0
    except:
        metrics["f2"] = 0.0

    # F0.5分数（更重视精确率）
    try:
        metrics["f05"] = (1 + 0.5 ** 2) * (metrics["precision"] * metrics["recall"]) / (
                    (0.5 ** 2 * metrics["precision"]) + metrics["recall"]) if (metrics["precision"] + metrics[
            "recall"]) > 0 else 0
    except:
        metrics["f05"] = 0.0

    # 存储阈值
    metrics["threshold"] = threshold

    # 类别分布信息
    n_samples = len(y_true)
    metrics["n_samples"] = n_samples
    metrics["positive_ratio"] = np.sum(y_true == 1) / n_samples if n_samples > 0 else 0
    metrics["predicted_positive_ratio"] = np.sum(y_pred == 1) / n_samples if n_samples > 0 else 0

    # 不确定性统计（如果提供了不确定性）
    if uncertainties is not None:
        uncertainties = np.asarray(uncertainties)
        if len(uncertainties) > 0:
            metrics["uncertainty_mean"] = uncertainties.mean()
            metrics["uncertainty_std"] = uncertainties.std()
            metrics["uncertainty_min"] = uncertainties.min()
            metrics["uncertainty_max"] = uncertainties.max()

            # 正确和错误预测的不确定性比较
            correct_mask = (y_pred == y_true)
            if np.any(correct_mask) and np.any(~correct_mask):
                metrics["uncertainty_correct_mean"] = uncertainties[correct_mask].mean()
                metrics["uncertainty_incorrect_mean"] = uncertainties[~correct_mask].mean()
                metrics["uncertainty_ratio"] = metrics["uncertainty_incorrect_mean"] / (
                            metrics["uncertainty_correct_mean"] + 1e-8)
            else:
                metrics["uncertainty_correct_mean"] = 0.0
                metrics["uncertainty_incorrect_mean"] = 0.0
                metrics["uncertainty_ratio"] = 0.0

    # 计算精确率-召回率曲线下的面积
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        # 使用梯形法则计算PR曲线下面积
        metrics["pr_auc"] = np.trapz(precision_curve, recall_curve)
    except:
        metrics["pr_auc"] = 0.0

    return metrics


# ======================================
# 带类别权重的训练函数（适配不确定性感知分类器）
# ======================================
def train_one_epoch_with_weights(model, dataloader, optimizer, device,
                                 class_weights=None, loss_alpha=1.0, loss_beta=0.0,
                                 uncertainty_lambda=0.01):
    """
    带类别权重的训练函数，适配不确定性感知分类器

    Args:
        uncertainty_lambda: 不确定性正则化权重
    """
    model.train()

    # 如果提供了类别权重，创建带权重的损失函数
    if class_weights is not None:
        # 将numpy权重转换为tensor
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = TotalLoss(alpha=loss_alpha, beta=loss_beta, class_weight=weight_tensor)
    else:
        criterion = TotalLoss(alpha=loss_alpha, beta=loss_beta)

    total_loss = 0.0
    total_uncertainty_loss = 0.0
    all_preds, all_labels, all_scores, all_uncertainties = [], [], [], []

    for batch_graph, batch_images in tqdm(dataloader, desc="Train", leave=False):
        batch_graph = batch_graph.to(device)
        batch_images = batch_images.to(device)

        optimizer.zero_grad()

        # 模型返回字典（包含不确定性信息）
        model_output = model(batch_graph, batch_images, labels=batch_graph.y, return_uncertainty=True)

        logits = model_output["logits"]
        fused = model_output["fused"]
        recon = model_output["recon"]
        coeff = model_output["coeff"]

        # 基础损失（分类损失 + 低秩重构损失）
        base_loss, loss_dict = criterion(logits, batch_graph.y, fused, recon, coeff)

        # 不确定性损失（如果模型计算了）
        uncertainty_loss = 0.0
        if "uncertainty_loss" in model_output:
            uncertainty_loss = model_output["uncertainty_loss"]
            total_loss_value = base_loss + uncertainty_lambda * uncertainty_loss
        else:
            total_loss_value = base_loss

        total_loss_value.backward()
        optimizer.step()

        total_loss += base_loss.item() * batch_graph.num_graphs
        total_uncertainty_loss += uncertainty_loss.item() * batch_graph.num_graphs if uncertainty_loss != 0.0 else 0.0

        # 保存指标
        probs = model_output["prob"][:, 1].detach().cpu().numpy()
        uncertainties = model_output["total_uncertainty"].detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labs = batch_graph.y.detach().cpu().numpy()

        all_scores.extend(probs)
        all_uncertainties.extend(uncertainties)
        all_preds.extend(preds)
        all_labels.extend(labs)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_uncertainty_loss = total_uncertainty_loss / len(dataloader.dataset) if total_uncertainty_loss > 0 else 0.0

    metrics = compute_comprehensive_metrics(all_labels, all_scores, threshold=0.5, uncertainties=all_uncertainties)
    metrics["loss"] = avg_loss
    metrics["uncertainty_loss"] = avg_uncertainty_loss
    metrics["total_loss"] = avg_loss + uncertainty_lambda * avg_uncertainty_loss

    return metrics


# ======================================
# 评估函数（适配不确定性感知分类器）
# ======================================
def evaluate_with_optimal_threshold(model, dataloader, device,
                                    loss_alpha=1.0, loss_beta=0.0,
                                    threshold_method='balanced',
                                    return_uncertainty=True):
    """
    使用最优阈值的评估函数，适配不确定性感知分类器
    """
    model.eval()
    criterion = TotalLoss(alpha=loss_alpha, beta=loss_beta)

    total_loss = 0.0
    all_scores, all_labels, all_uncertainties = [], [], []

    with torch.no_grad():
        for batch_graph, batch_images in tqdm(dataloader, desc="Eval", leave=False):
            batch_graph = batch_graph.to(device)
            batch_images = batch_images.to(device)

            # 模型返回字典
            model_output = model(batch_graph, batch_images, return_uncertainty=return_uncertainty)

            logits = model_output["logits"]
            fused = model_output["fused"]
            recon = model_output["recon"]
            coeff = model_output["coeff"]

            loss, _ = criterion(logits, batch_graph.y, fused, recon, coeff)

            total_loss += loss.item() * batch_graph.num_graphs

            probs = model_output["prob"][:, 1].cpu().numpy()
            uncertainties = model_output["total_uncertainty"].cpu().numpy() if return_uncertainty else np.zeros_like(
                probs)
            labs = batch_graph.y.cpu().numpy()

            all_scores.extend(probs)
            all_uncertainties.extend(uncertainties)
            all_labels.extend(labs)

    # 寻找最优阈值
    optimal_threshold = find_optimal_threshold(all_labels, all_scores, method=threshold_method)

    # 使用最优阈值计算指标
    metrics = compute_comprehensive_metrics(all_labels, all_scores, optimal_threshold, uncertainties=all_uncertainties)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    metrics["optimal_threshold"] = optimal_threshold

    # 存储原始分数和标签用于分析
    metrics["all_scores"] = all_scores
    metrics["all_labels"] = all_labels
    metrics["all_uncertainties"] = all_uncertainties

    return metrics


# ======================================
# 不确定性感知的主动学习样本选择
# ======================================
def select_samples_with_uncertainty(model, unlabeled_dataloader, device,
                                    alpha=0.7, k=10, uncertainty_type='total'):
    """
    基于不确定性感知的主动学习样本选择

    Args:
        model: 不确定性感知模型
        unlabeled_dataloader: 未标注数据加载器
        device: 设备
        alpha: 不确定性权重 (1-alpha为多样性权重)
        k: 选择样本数量
        uncertainty_type: 不确定性类型 'total', 'epistemic', 'aleatoric', 'evidential'
    """
    model.eval()
    uncertainties = []
    features = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (batch_graph, batch_images) in enumerate(unlabeled_dataloader):
            batch_graph = batch_graph.to(device)
            batch_images = batch_images.to(device)

            # 获取模型输出（包含不确定性）
            model_output = model(batch_graph, batch_images, return_uncertainty=True)

            # 根据指定类型选择不确定性
            if uncertainty_type == 'total':
                batch_uncertainties = model_output["total_uncertainty"]
            elif uncertainty_type == 'epistemic':
                batch_uncertainties = model_output["epistemic_uncertainty"]
            elif uncertainty_type == 'aleatoric':
                batch_uncertainties = model_output["aleatoric_uncertainty"]
            elif uncertainty_type == 'evidential' and "evidential_uncertainty" in model_output:
                batch_uncertainties = model_output["evidential_uncertainty"]
            else:
                batch_uncertainties = model_output["total_uncertainty"]

            # 获取特征用于多样性计算
            batch_features = model_output.get("shared_features", model_output["fused"])

            uncertainties.extend(batch_uncertainties.cpu().numpy())
            features.extend(batch_features.cpu().numpy())

            batch_size = batch_graph.num_graphs
            all_indices.extend(range(batch_idx * batch_size, batch_idx * batch_size + batch_size))

    if len(uncertainties) == 0:
        return None, None

    uncertainties = np.array(uncertainties)
    features = np.array(features)

    # 计算多样性得分（与已选择样本的多样性）
    diversity_scores = np.zeros(len(uncertainties))
    if len(features) > 1:
        # 使用特征空间的覆盖度作为多样性度量
        from sklearn.metrics.pairwise import cosine_distances
        if len(features) > 100:  # 对于大数据集，使用近似计算
            # 随机采样计算多样性
            sample_size = min(100, len(features))
            random_indices = np.random.choice(len(features), sample_size, replace=False)
            sample_features = features[random_indices]

            # 计算每个样本与随机样本的平均距离
            for i in range(len(features)):
                dists = np.linalg.norm(sample_features - features[i:i + 1], axis=1)
                diversity_scores[i] = np.mean(dists)
        else:
            # 计算所有样本对的距离
            dist_matrix = cosine_distances(features)
            diversity_scores = np.mean(dist_matrix, axis=1)

    # 归一化得分
    uncertainty_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-8)
    diversity_norm = (diversity_scores - diversity_scores.min()) / (
                diversity_scores.max() - diversity_scores.min() + 1e-8)

    # 综合得分
    combined_scores = alpha * uncertainty_norm + (1 - alpha) * diversity_norm

    # 选择得分最高的k个样本
    if k > len(combined_scores):
        k = len(combined_scores)

    selected_local_idx = np.argsort(combined_scores)[-k:][::-1]
    selected_global_idx = [all_indices[i] for i in selected_local_idx]

    return combined_scores, selected_global_idx


# ======================================
# 不确定性分析函数
# ======================================
def analyze_uncertainty_distribution(model, dataloader, device):
    """
    分析模型的不确定性分布
    """
    model.eval()

    epistemic_uncertainties = []
    aleatoric_uncertainties = []
    total_uncertainties = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_graph, batch_images in tqdm(dataloader, desc="Analyzing uncertainty", leave=False):
            batch_graph = batch_graph.to(device)
            batch_images = batch_images.to(device)

            model_output = model(batch_graph, batch_images, return_uncertainty=True)

            epistemic_uncertainties.extend(model_output["epistemic_uncertainty"].cpu().numpy())
            aleatoric_uncertainties.extend(model_output["aleatoric_uncertainty"].cpu().numpy())
            total_uncertainties.extend(model_output["total_uncertainty"].cpu().numpy())
            all_probs.extend(model_output["prob"][:, 1].cpu().numpy())
            all_labels.extend(batch_graph.y.cpu().numpy())

    if len(epistemic_uncertainties) == 0:
        return None

    analysis = {
        'epistemic_mean': np.mean(epistemic_uncertainties),
        'epistemic_std': np.std(epistemic_uncertainties),
        'aleatoric_mean': np.mean(aleatoric_uncertainties),
        'aleatoric_std': np.std(aleatoric_uncertainties),
        'total_mean': np.mean(total_uncertainties),
        'total_std': np.std(total_uncertainties),
        'correlation_epistemic_aleatoric': np.corrcoef(epistemic_uncertainties, aleatoric_uncertainties)[0, 1],
        'uncertainty_by_class': {}
    }

    # 按类别分析不确定性
    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)

    for cls in unique_classes:
        mask = (all_labels == cls)
        if np.any(mask):
            analysis['uncertainty_by_class'][f'class_{cls}_epistemic_mean'] = np.mean(
                np.array(epistemic_uncertainties)[mask])
            analysis['uncertainty_by_class'][f'class_{cls}_aleatoric_mean'] = np.mean(
                np.array(aleatoric_uncertainties)[mask])
            analysis['uncertainty_by_class'][f'class_{cls}_total_mean'] = np.mean(np.array(total_uncertainties)[mask])
            analysis['uncertainty_by_class'][f'class_{cls}_count'] = np.sum(mask)

    return analysis


# ======================================
# 校准分析函数
# ======================================
def analyze_calibration(model, dataloader, device, n_bins=10):
    """
    分析模型预测的校准性（预测概率与实际准确率的关系）
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_uncertainties = []

    with torch.no_grad():
        for batch_graph, batch_images in dataloader:
            batch_graph = batch_graph.to(device)
            batch_images = batch_images.to(device)

            model_output = model(batch_graph, batch_images, return_uncertainty=True)

            probs = model_output["prob"][:, 1].cpu().numpy()
            uncertainties = model_output["total_uncertainty"].cpu().numpy()
            labels = batch_graph.y.cpu().numpy()

            all_probs.extend(probs)
            all_uncertainties.extend(uncertainties)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_uncertainties = np.array(all_uncertainties)

    # 计算ECE（Expected Calibration Error）
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    calibration_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (all_probs >= bin_lower) & (all_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(all_labels[in_bin])
            avg_confidence_in_bin = np.mean(all_probs[in_bin])

            calibration_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': np.sum(in_bin),
                'avg_uncertainty': np.mean(all_uncertainties[in_bin]) if np.any(in_bin) else 0.0
            })

            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    calibration_summary = {
        'ece': ece,
        'bins': calibration_data,
        'avg_uncertainty': np.mean(all_uncertainties),
        'correlation_prob_uncertainty': np.corrcoef(all_probs, all_uncertainties)[0, 1]
    }

    return calibration_summary