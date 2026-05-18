

import os


class Config:

    ROOT_DIR = r"E:\AL-LRR\data\ntu_graph_pt" 
    DATA_DIR = r"E:\AL-LRR\data\ntu_graph_pt"
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_DIR = os.path.join(ROOT_DIR, "checkpoints")
    RESULT_DIR = os.path.join(ROOT_DIR, "results")

 
    DATASET_NAME = "AnomalyDetection" 
    TRAIN_SPLIT = 0.7  
    SEED = 42  
    IMG_SIZE = 224  
    FEATURE_DIM = 512  

   
    K_FOLDS = 5  
    USE_KFOLD = True  

 
    USE_OPTIMAL_THRESHOLD = True  
    THRESHOLD_METHOD = 'balanced'  
    THRESHOLD_EVAL_METHOD = 'f1'  


    MODEL_TYPE = 'uncertainty'  
    USE_UNCERTAINTY_CLASSIFIER = True  


    UNCERTAINTY_ENSEMBLE_SIZE = 5  
    UNCERTAINTY_DROPOUT_RATE = 0.3  
    UNCERTAINTY_BETA = 1.0 
    UNCERTAINTY_TYPE = 'total'  
    USE_MC_DROPOUT = True  
    EVIDENTIAL_LEARNING = False  

  
    COSINE_SCALE = 30.0  
    COSINE_MARGIN = 0.3  
    LEARNABLE_SCALE = True  

 
    HIDDEN_DIM = 384  
    LATENT_DIM = 192 
    RANK_LAMBDA = 0.1  
    CONTRASTIVE_LAMBDA = 0.5 

   
    INIT_LABEL_RATIO = 0.05  
    AL_ROUNDS = 20 
    QUERY_SIZE = 0.2  
    UNCERTAINTY_METHOD = "entropy"  
    ALPHA = 0.7  
    UNCERTAINTY_SELECTION_TYPE = 'total'  


    EPOCHS = 30
    BATCH_SIZE = 128
    LR = 3e-4
    WEIGHT_DECAY = 5e-5
    DEVICE = "cuda"  # 'cuda' or 'cpu'


    FOLD_EPOCHS = 50  
    FOLD_AL_ROUNDS = 20 


    SAVE_FREQ = 1  
    PRINT_FREQ = 10
    LOG_NAME = "AL_LRR_Uncertainty_KFold"  

   
    LRR_ALPHA = 0.2 
    LRR_BETA = 0.3  

 
    USE_FOCAL_LOSS = False  
    FOCAL_ALPHA = 0.25  
    FOCAL_GAMMA = 2.0  
    UNCERTAINTY_REG_WEIGHT = 0.01  

  
    METRICS_TO_TRACK = ['acc', 'precision', 'recall', 'f1', 'auc', 'auprc', 'balanced_acc', 'mcc', 'uncertainty_mean']

 
    EARLY_STOPPING = True
    PATIENCE = 30


    ANALYZE_UNCERTAINTY = True  
    UNCERTAINTY_ANALYSIS_FREQ = 2  

  
    ANALYZE_CALIBRATION = True 
    CALIBRATION_BINS = 10 


def get_config():

    config = Config()


    if 'uncertainty' in config.MODEL_TYPE:
        config.USE_UNCERTAINTY_CLASSIFIER = True
    else:
        config.USE_UNCERTAINTY_CLASSIFIER = False

    return config




