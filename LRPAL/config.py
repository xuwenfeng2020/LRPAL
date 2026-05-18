# ============================================
# config.py
# 配置文件：路径、参数、模型与训练超参数
# 更新为支持不确定性感知分类器
# ============================================

import os


class Config:
    # ========== 基本路径配置 ==========
    ROOT_DIR = r"E:\AL-LRR\data\ntu_graph_pt"  # 项目根目录
    DATA_DIR = r"E:\AL-LRR\data\ntu_graph_pt"
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_DIR = os.path.join(ROOT_DIR, "checkpoints")
    RESULT_DIR = os.path.join(ROOT_DIR, "results")

    # ========== 数据集配置 ==========
    DATASET_NAME = "AnomalyDetection"  # 数据集名称，可自定义
    TRAIN_SPLIT = 0.7  # 训练/验证划分比例
    SEED = 42  # 随机种子
    IMG_SIZE = 224  # 图像大小（如果有图像输入）
    FEATURE_DIM = 512  # 特征维度（低秩输入）

    # ========== 五折实验配置 ==========
    K_FOLDS = 5  # 交叉验证折数
    USE_KFOLD = True  # 是否使用K折交叉验证

    # ========== 阈值相关配置 ==========
    USE_OPTIMAL_THRESHOLD = True  # 是否使用最优阈值
    THRESHOLD_METHOD = 'balanced'  # 寻找最优阈值时使用的指标: 'f1', 'gmean', 'youden', 'balanced', 'pr'
    THRESHOLD_EVAL_METHOD = 'f1'  # 评估时计算F1使用的阈值方法

    # ========== 模型类型选择 ==========
    MODEL_TYPE = 'uncertainty'  # 'linear', 'cosine', 'prototype', 'uncertainty', 'uncertainty_margin'
    USE_UNCERTAINTY_CLASSIFIER = True  # 是否使用不确定性感知分类器

    # ========== 不确定性感知分类器参数 ==========
    UNCERTAINTY_ENSEMBLE_SIZE = 5  # 集成模型数量（用于估计认知不确定性）
    UNCERTAINTY_DROPOUT_RATE = 0.3  # Dropout率（用于MC Dropout）
    UNCERTAINTY_BETA = 1.0  # 任意不确定性权重（总不确定性 = 认知不确定性 + beta * 任意不确定性）
    UNCERTAINTY_TYPE = 'total'  # 不确定性类型: 'total', 'epistemic', 'aleatoric', 'evidential'
    USE_MC_DROPOUT = True  # 是否使用MC Dropout估计不确定性
    EVIDENTIAL_LEARNING = False  # 是否使用证据学习（狄利克雷分布）

    # ========== Cosine Classifier相关参数（如果使用） ==========
    COSINE_SCALE = 30.0  # Cosine分类器的缩放因子（温度参数）
    COSINE_MARGIN = 0.3  # Margin Cosine Classifier的margin值
    LEARNABLE_SCALE = True  # 是否让缩放因子可学习

    # ========== 模型参数 ==========
    HIDDEN_DIM = 384  # 隐藏层维度
    LATENT_DIM = 192  # 低秩嵌入维度
    RANK_LAMBDA = 0.1  # 低秩约束权重
    CONTRASTIVE_LAMBDA = 0.5  # 对比损失权重

    # ========== 主动学习配置 ==========
    INIT_LABEL_RATIO = 0.05  # 初始标注比例
    AL_ROUNDS = 20  # 主动学习循环次数
    QUERY_SIZE = 0.2  # 每轮选择未标注样本比例
    UNCERTAINTY_METHOD = "entropy"  # 选样策略：'entropy', 'margin', 'least_confidence', 'uncertainty'
    ALPHA = 0.7  # 不确定性权重 (1-alpha为多样性/代表性权重)，对于不确定性感知分类器更适用
    UNCERTAINTY_SELECTION_TYPE = 'total'  # 主动学习时使用的不确定性类型

    # ========== 训练参数 ==========
    EPOCHS = 30
    BATCH_SIZE = 128
    LR = 3e-4
    WEIGHT_DECAY = 5e-5
    DEVICE = "cuda"  # 'cuda' or 'cpu'

    # ========== 五折实验训练参数 ==========
    FOLD_EPOCHS = 50  # 每折训练的epoch数
    FOLD_AL_ROUNDS = 20 # 每折内的主动学习轮数

    # ========== 日志 & 保存 ==========
    SAVE_FREQ = 1  # 每多少轮保存模型
    PRINT_FREQ = 10
    LOG_NAME = "AL_LRR_Uncertainty_KFold"  # 更新日志名以反映使用不确定性感知分类器

    # ========== 低秩学习模块参数 ==========
    LRR_ALPHA = 0.2  # LRR稀疏惩罚系数
    LRR_BETA = 0.3  # LRR重构损失权重

    # ========== 损失函数参数 ==========
    USE_FOCAL_LOSS = False  # 是否使用Focal Loss
    FOCAL_ALPHA = 0.25  # Focal Loss的alpha参数
    FOCAL_GAMMA = 2.0  # Focal Loss的gamma参数
    UNCERTAINTY_REG_WEIGHT = 0.01  # 不确定性正则化权重

    # ========== 评估指标 ==========
    METRICS_TO_TRACK = ['acc', 'precision', 'recall', 'f1', 'auc', 'auprc', 'balanced_acc', 'mcc', 'uncertainty_mean']

    # ========== 其他 ==========
    EARLY_STOPPING = True
    PATIENCE = 30

    # ========== 不确定性分析 ==========
    ANALYZE_UNCERTAINTY = True  # 是否进行不确定性分析
    UNCERTAINTY_ANALYSIS_FREQ = 2  # 不确定性分析频率（每多少轮分析一次）

    # ========== 校准分析 ==========
    ANALYZE_CALIBRATION = True  # 是否进行校准分析
    CALIBRATION_BINS = 10  # 校准分析的分箱数


def get_config():
    """返回配置实例"""
    config = Config()

    # 根据MODEL_TYPE自动设置USE_UNCERTAINTY_CLASSIFIER
    if 'uncertainty' in config.MODEL_TYPE:
        config.USE_UNCERTAINTY_CLASSIFIER = True
    else:
        config.USE_UNCERTAINTY_CLASSIFIER = False

    return config




