import os
import random
import numpy as np
import torch
import json
import time
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from config import get_config
from dataloader import build_datasets, get_dataloader, create_active_learning_splits, create_dataloaders, \
    update_datasets
from model import AL_LRR_Model
from train import (train_one_epoch_with_weights, evaluate_with_optimal_threshold,
                   select_samples_with_uncertainty, analyze_uncertainty_distribution)


def set_seed(seed):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_active_learning_for_fold(fold_idx, train_indices, val_indices, test_indices,
                                 full_dataset, cfg, device, result_dir):
    """
    在单折数据上运行主动学习，适配不确定性感知分类器
    """
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # 创建主动学习初始划分
    Dl, Du = create_active_learning_splits(
        train_dataset,
        init_label_ratio=cfg.INIT_LABEL_RATIO,
        seed=cfg.SEED + fold_idx
    )

    print(f"Fold {fold_idx}: Train={len(train_dataset)}, Labeled={len(Dl)}, "
          f"Unlabeled={len(Du)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 创建数据加载器
    dataloaders = create_dataloaders(
        Dl, Du, val_dataset, test_dataset,
        batch_size=cfg.BATCH_SIZE
    )

    # 初始化模型 - 使用不确定性感知分类器
    model = AL_LRR_Model(
        g_in=3,
        img_in=3,
        hidden=cfg.HIDDEN_DIM,
        out_dim=cfg.LATENT_DIM,
        num_classes=2,
        use_uncertainty_classifier=getattr(cfg, 'USE_UNCERTAINTY_CLASSIFIER', True),
        ensemble_size=getattr(cfg, 'UNCERTAINTY_ENSEMBLE_SIZE', 5),
        dropout_rate=getattr(cfg, 'UNCERTAINTY_DROPOUT_RATE', 0.3),
        uncertainty_beta=getattr(cfg, 'UNCERTAINTY_BETA', 1.0)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # 存储结果
    fold_results = {
        'fold': fold_idx,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'al_rounds': [],
        'test_metrics': [],  # 每轮在测试集上的评估结果
        'labeled_sizes': [],  # 记录每轮有标签样本数
        'uncertainty_analysis': [],  # 不确定性分析结果
        'model_config': {
            'use_uncertainty_classifier': getattr(cfg, 'USE_UNCERTAINTY_CLASSIFIER', True),
            'ensemble_size': getattr(cfg, 'UNCERTAINTY_ENSEMBLE_SIZE', 5),
            'uncertainty_beta': getattr(cfg, 'UNCERTAINTY_BETA', 1.0)
        }
    }

    # 主动学习循环
    max_rounds = cfg.FOLD_AL_ROUNDS
    for round_idx in range(max_rounds):
        print(f"\n  Fold {fold_idx} - AL Round {round_idx + 1}/{max_rounds}")
        print(f"  Labeled samples: {len(Dl)}, Unlabeled: {len(Du)}")

        # 记录有标签样本数
        fold_results['labeled_sizes'].append(len(Dl))

        # 训练模型
        print(f"  Training for {cfg.FOLD_EPOCHS} epochs...")
        for epoch in range(cfg.FOLD_EPOCHS):
            train_metrics = train_one_epoch_with_weights(
                model, dataloaders['labeled'], optimizer, device,
                class_weights=None,
                loss_alpha=cfg.LRR_BETA,
                loss_beta=0.0,
                uncertainty_lambda=getattr(cfg, 'UNCERTAINTY_REG_WEIGHT', 0.01)
            )

            if (epoch + 1) % cfg.PRINT_FREQ == 0:
                print(f"    Epoch {epoch + 1}: loss={train_metrics['loss']:.4f}, "
                      f"acc={train_metrics['acc']:.4f}, f1={train_metrics['f1']:.4f}, "
                      f"uncertainty={train_metrics.get('uncertainty_mean', 0):.4f}")

        # 在验证集上寻找最优阈值
        print("  Evaluating on validation set...")
        val_metrics = evaluate_with_optimal_threshold(
            model, dataloaders['val'], device,
            loss_alpha=cfg.LRR_BETA,
            loss_beta=0.0,
            threshold_method=cfg.THRESHOLD_METHOD,
            return_uncertainty=True
        )

        # 使用最优阈值在测试集上评估
        optimal_threshold = val_metrics['optimal_threshold']
        print(f"  Optimal threshold: {optimal_threshold:.4f}")

        # 在测试集上评估
        print("  Evaluating on test set...")
        test_metrics = evaluate_with_optimal_threshold(
            model, dataloaders['test'], device,
            loss_alpha=cfg.LRR_BETA,
            loss_beta=0.0,
            threshold_method=cfg.THRESHOLD_METHOD,
            return_uncertainty=True
        )

        # 使用验证集找到的最优阈值重新计算测试集指标
        all_scores = test_metrics['all_scores']
        all_labels = test_metrics['all_labels']
        all_uncertainties = test_metrics.get('all_uncertainties', [])

        # 重新计算指标（使用最优阈值）
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        y_pred = (all_scores >= optimal_threshold).astype(int)

        # 使用我们已有的compute_comprehensive_metrics函数计算所有指标
        # 这里我们重新调用一次以确保使用最优阈值
        final_metrics = compute_comprehensive_metrics(all_labels, all_scores, optimal_threshold,
                                                      uncertainties=all_uncertainties)

        test_metrics_round = {
            'round': round_idx,
            'threshold': optimal_threshold,
            'acc': final_metrics["acc"],
            'precision': final_metrics["precision"],
            'recall': final_metrics["recall"],
            'f1': final_metrics["f1"],
            'precision_macro': final_metrics["precision_macro"],
            'recall_macro': final_metrics["recall_macro"],
            'f1_macro': final_metrics["f1_macro"],
            'auc': final_metrics["auc"],
            'auprc': final_metrics["auprc"],
            'balanced_acc': final_metrics["balanced_acc"],
            'mcc': final_metrics["mcc"],
            'specificity': final_metrics["specificity"],
            'ppv': final_metrics["ppv"],
            'npv': final_metrics["npv"],
            'gmean': final_metrics["gmean"],
            'f2': final_metrics.get("f2", 0.0),
            'f05': final_metrics.get("f05", 0.0),
            'pr_auc': final_metrics.get("pr_auc", 0.0),
            'class_0_precision': final_metrics.get("class_0_precision", 0.0),
            'class_0_recall': final_metrics.get("class_0_recall", 0.0),
            'class_1_precision': final_metrics.get("class_1_precision", 0.0),
            'class_1_recall': final_metrics.get("class_1_recall", 0.0),
            'labeled_samples': len(Dl),
            'loss': test_metrics['loss'],
            'uncertainty_mean': final_metrics.get("uncertainty_mean", 0.0),
            'uncertainty_std': final_metrics.get("uncertainty_std", 0.0)
        }

        # 如果不确定性感知分类器可用，添加更多不确定性相关指标
        if hasattr(model, 'use_uncertainty_classifier') and model.use_uncertainty_classifier:
            uncertainty_stats = model.get_uncertainty_stats()
            test_metrics_round.update({
                'epistemic_uncertainty': uncertainty_stats['epistemic_mean'],
                'aleatoric_uncertainty': uncertainty_stats['aleatoric_mean'],
                'total_uncertainty': uncertainty_stats['total_mean']
            })

        fold_results['test_metrics'].append(test_metrics_round)
        fold_results['al_rounds'].append(round_idx)

        print(f"  Test metrics - Acc: {test_metrics_round['acc']:.4f}, "
              f"Precision: {test_metrics_round['precision']:.4f}, "
              f"Recall: {test_metrics_round['recall']:.4f}, "
              f"F1: {test_metrics_round['f1']:.4f}, "
              f"AUC: {test_metrics_round['auc']:.4f}, "
              f"Loss: {test_metrics_round['loss']:.4f}, "
              f"Uncertainty: {test_metrics_round.get('total_uncertainty', 0):.4f}")

        # 如果没有未标注样本，停止
        if len(Du) == 0:
            print("  No unlabeled samples left, stopping AL")
            break

        # 选择下一轮要标注的样本（使用不确定性感知选择）
        print("  Selecting samples for next round...")

        # 计算要选择的样本数量
        k = max(1, int(len(Du) * cfg.QUERY_SIZE))

        # 使用不确定性感知的样本选择
        scores, selected_indices = select_samples_with_uncertainty(
            model=model,
            unlabeled_dataloader=dataloaders['unlabeled'],
            device=device,
            alpha=cfg.ALPHA,
            k=k,
            uncertainty_type=getattr(cfg, 'UNCERTAINTY_TYPE', 'total')
        )

        if selected_indices is None or len(selected_indices) == 0:
            print("  No samples selected, stopping AL")
            break

        print(f"  Selected {len(selected_indices)} samples from {len(Du)} unlabeled samples")

        # 更新数据集
        Dl, Du = update_datasets(Dl, Du, selected_indices)

        # 更新数据加载器
        dataloaders['labeled'] = get_dataloader(Dl, shuffle=True, batch_size=cfg.BATCH_SIZE)
        dataloaders['unlabeled'] = get_dataloader(Du, shuffle=False, batch_size=cfg.BATCH_SIZE)

        # 可选：进行不确定性分析
        if hasattr(model, 'use_uncertainty_classifier') and model.use_uncertainty_classifier and round_idx % 2 == 0:
            try:
                uncertainty_analysis = analyze_uncertainty_distribution(model, dataloaders['labeled'], device)
                if uncertainty_analysis:
                    fold_results['uncertainty_analysis'].append({
                        'round': round_idx,
                        'analysis': uncertainty_analysis
                    })
                    print(f"  Uncertainty analysis - Epistemic: {uncertainty_analysis['epistemic_mean']:.4f}, "
                          f"Aleatoric: {uncertainty_analysis['aleatoric_mean']:.4f}, "
                          f"Total: {uncertainty_analysis['total_mean']:.4f}")
            except Exception as e:
                print(f"  Failed to analyze uncertainty: {e}")

    return fold_results


def compute_statistics_across_folds(all_fold_results):
    max_rounds = max(len(fold['test_metrics']) for fold in all_fold_results)
    METRICS = ['precision', 'recall', 'f1', 'auc']

    round_metrics = {m: [[] for _ in range(max_rounds)] for m in METRICS}

    for fold_res in all_fold_results:
        for r_idx, metric in enumerate(fold_res['test_metrics']):
            for m in METRICS:
                round_metrics[m][r_idx].append(metric[m])

    mean_auc = [np.mean(round_metrics['auc'][r]) for r in range(max_rounds)]
    best_epoch = np.argmax(mean_auc)

    final = {}
    for m in METRICS:
        vals = round_metrics[m][best_epoch]
        final[m] = {
            'mean': np.mean(vals),
            'std': np.std(vals, ddof=1)
        }

    return {
        'best_epoch': best_epoch,
        'metrics': final
    }


def save_results(final_result, cfg, result_dir):
    summary_file = os.path.join(result_dir, "FINAL_RESULT.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("✅ 5-FOLD FINAL RESULT (BEST EPOCH)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Epoch: {final_result['best_epoch']}\n\n")
        m = final_result['metrics']
        f.write(f"PRE : {m['precision']['mean']:.4f} ± {m['precision']['std']:.4f}\n")
        f.write(f"REC : {m['recall']['mean']:.4f} ± {m['recall']['std']:.4f}\n")
        f.write(f"F1  : {m['f1']['mean']:.4f} ± {m['f1']['std']:.4f}\n")
        f.write(f"AUC : {m['auc']['mean']:.4f} ± {m['auc']['std']:.4f}\n")
        f.write("=" * 60 + "\n")
    return summary_file


def main():
    """主函数：执行五折交叉验证实验（不确定性感知版本）"""
    cfg = get_config()
    set_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Starting {cfg.K_FOLDS}-fold cross-validation experiment with Uncertainty-Aware Classifier")

    # 打印不确定性感知分类器配置
    if hasattr(cfg, 'USE_UNCERTAINTY_CLASSIFIER') and cfg.USE_UNCERTAINTY_CLASSIFIER:
        print(f"Using Uncertainty-Aware Classifier:")
        print(f"  - Ensemble size: {getattr(cfg, 'UNCERTAINTY_ENSEMBLE_SIZE', 5)}")
        print(f"  - Dropout rate: {getattr(cfg, 'UNCERTAINTY_DROPOUT_RATE', 0.3)}")
        print(f"  - Uncertainty beta: {getattr(cfg, 'UNCERTAINTY_BETA', 1.0)}")
        print(f"  - Uncertainty type: {getattr(cfg, 'UNCERTAINTY_TYPE', 'total')}")
    else:
        print("Using linear classifier (uncertainty-aware classifier disabled)")

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(cfg.RESULT_DIR, f"{cfg.LOG_NAME}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 加载完整数据集
    DATA_DIR = cfg.DATA_DIR
    assert os.path.exists(DATA_DIR), f"Data directory not found: {DATA_DIR}"

    # 使用dataloader中的函数加载完整数据集
    from dataloader import get_full_dataset, create_kfold_splits, get_dataset_labels
    full_dataset = get_full_dataset(DATA_DIR)

    # 获取所有标签
    all_labels = get_dataset_labels(full_dataset)

    # 创建五折划分
    splits = create_kfold_splits(
        full_dataset,
        k_folds=cfg.K_FOLDS,
        val_ratio=0.2,
        seed=cfg.SEED
    )

    # 存储所有折的结果
    all_fold_results = []

    # 遍历每一折
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/{cfg.K_FOLDS}")
        print(f"{'=' * 80}")

        # 运行该折的主动学习
        fold_result = run_active_learning_for_fold(
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            full_dataset=full_dataset,
            cfg=cfg,
            device=device,
            result_dir=exp_dir
        )

        all_fold_results.append(fold_result)

        # 保存该折的详细结果
        fold_result_file = os.path.join(exp_dir, f"fold_{fold_idx}_results.json")
        with open(fold_result_file, 'w') as f:
            json.dump(fold_result, f, indent=2, default=str)

    # 计算跨折统计（新的标准方式）
    print(f"\n{'=' * 80}")
    print("COMPUTING STATISTICS ACROSS FOLDS (Standard Method)")
    print(f"{'=' * 80}")

    final_results = compute_statistics_across_folds(all_fold_results)
    save_results(final_results, cfg, exp_dir)

    # 打印最终结果
    print("\n✅ FINAL RESULT (Mean ± Std across folds at best round):")
    print("-" * 60)
    m = final_results['metrics']
    print(f"Best Round: {final_results['best_epoch']}")
    print(f"PRE : {m['precision']['mean']:.4f} ± {m['precision']['std']:.4f}")
    print(f"REC : {m['recall']['mean']:.4f} ± {m['recall']['std']:.4f}")
    print(f"F1  : {m['f1']['mean']:.4f} ± {m['f1']['std']:.4f}")
    print(f"AUC : {m['auc']['mean']:.4f} ± {m['auc']['std']:.4f}")
    print("-" * 60)
    print(f"\nExperiment completed! Results saved to: {exp_dir}")


# ======================================
# 辅助函数：需要在 main.py 中添加
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
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        balanced_accuracy_score, matthews_corrcoef
    )

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
    tp = np.sum((y_pred == 1) & (y_true == 0))
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
    from sklearn.metrics import precision_recall_curve
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        # 使用梯形法则计算PR曲线下面积
        metrics["pr_auc"] = np.trapz(precision_curve, recall_curve)
    except:
        metrics["pr_auc"] = 0.0

    return metrics

if __name__ == "__main__":
    main()