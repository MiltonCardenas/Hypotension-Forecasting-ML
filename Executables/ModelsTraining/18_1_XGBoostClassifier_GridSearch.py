"""
# ------------------------------------------------------------------------------------------------
# I Previously found a reduced set of features for model training via logistic regression
# (17_1 & 17_2 scripts). These predictors will be used to train a XGBoost classifier model. 
# This binary classification will serve as the risk forecaster to know whether a 
# patient will develop a hypotensive state the next 40 minutes of dialysis treatment. 
# This script develops the training workflow for the base uncalibrated XGBoost model. 
# v.1.0 author: Milton Dario Cardenas 


 /$$   /$$ /$$$$$$ /$$$$$$$                              /$$            /$$$$$$ /$$                          /$$ /$$$$$$ /$$                  
| $$  / $$/$$__  $| $$__  $$                            | $$           /$$__  $| $$                         |__//$$__  $|__/                  
|  $$/ $$| $$  \__| $$  \ $$ /$$$$$$  /$$$$$$  /$$$$$$$/$$$$$$        | $$  \__| $$ /$$$$$$  /$$$$$$$/$$$$$$$/$| $$  \__//$$ /$$$$$$  /$$$$$$ 
 \  $$$$/| $$ /$$$| $$$$$$$ /$$__  $$/$$__  $$/$$_____|_  $$_/        | $$     | $$|____  $$/$$_____/$$_____| $| $$$$   | $$/$$__  $$/$$__  $$
  >$$  $$| $$|_  $| $$__  $| $$  \ $| $$  \ $|  $$$$$$  | $$          | $$     | $$ /$$$$$$|  $$$$$|  $$$$$$| $| $$_/   | $| $$$$$$$| $$  \__/
 /$$/\  $| $$  \ $| $$  \ $| $$  | $| $$  | $$\____  $$ | $$ /$$      | $$    $| $$/$$__  $$\____  $\____  $| $| $$     | $| $$_____| $$      
| $$  \ $|  $$$$$$| $$$$$$$|  $$$$$$|  $$$$$$//$$$$$$$/ |  $$$$/      |  $$$$$$| $|  $$$$$$$/$$$$$$$/$$$$$$$| $| $$     | $|  $$$$$$| $$      
|__/  |__/\______/|_______/ \______/ \______/|_______/   \___/         \______/|__/\_______|_______|_______/|__|__/     |__/\_______|__/      
                                                                                                                                              
"""

# %% =============================================================================================
# ------------------------------------ Libraries and Config ------------------------------------
import os, gc
import sys
import socket
import configparser
import shap
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
import h2o
from scipy.stats import t
from h2o.automl import H2OAutoML
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OXGBoostEstimator

#: Custom Libraries -------------
DIR_CURRENT = r'/ihome/rparker/mdc147/PhD_Project';  os.chdir(DIR_CURRENT)
sys.path.append(os.path.join(DIR_CURRENT, 'Auxiliar_codes'))
from classes_repository import PathManager, GlobalParameters, ProcessingAuxiliar
from machine_learning_auxiliar import (
    FeaturesManagerV2, 
    sampling_methods, 
    get_training_validation_sets,
    check_division
)

#: Instances -------------------
sm = sampling_methods()
recall_at_fpr = sm.recall_at_fpr
precision_at_fpr = sm.precision_at_fpr
threshold_at_fpr = sm.threshold_at_fpr
pm = PathManager().paths
pa = ProcessingAuxiliar()
gp = GlobalParameters()
fm = FeaturesManagerV2()

#:   =============================== INPUT - OUTPUT PATHS ===========================================
#: INPUT PATHS -------------------
DIR_PROCESSED       = "Data/Processed_datasets/Summary/Merged_M24H15"
PATH_DYN            = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Signals.pkl")
PATH_SSUM           = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Summary.pkl") 
PATH_SPLIT          = os.path.join(DIR_CURRENT, "Models_storage/DynamicPredictionModel/StratifiedDivision/stratified_division_summary.pkl")


#: OUTPUT PATHS ------------------
DIR_OUTPUT              = "Models_storage/DynamicPredictionModel/XGBoostEstimator"
PATH_GRID_RESULTS       = os.path.join(DIR_OUTPUT, "XGBoost_All_GridMembers_Params.csv")
PATH_TOP10_STATISTICS   = os.path.join(DIR_OUTPUT, "XGBoost_Top10Metrics.csv")
PATH_BRIEFTXT_SUMMARY   = os.path.join(DIR_OUTPUT, "XGBoost_HoldOut_FinalPerformance.txt")
PATH_GRID_MODELS        = os.path.join(DIR_OUTPUT, "XGBoost_All_GridMembers")
PATH_OOF_PREDICTIONS    = os.path.join(DIR_OUTPUT, "XGBoost_oof_predictions.csv")
PATH_NANs_SUMMARY       = os.path.join(DIR_OUTPUT, "NaNs_summary_at_XGBOOST_training.csv")
PATH_TOP_MODEL          = os.path.join(DIR_OUTPUT, "Top1Model_and_KModels")
os.makedirs(DIR_OUTPUT, exist_ok=True)

#: OTHER PARAMS ----------------------
max_time            = int(3600/8) # max time for automl
reset               = gp.reset
random_seed         = 2000
mem                 = '64G' # mem size for H2O server  
MODEL_SEARCH        = 'XGBoost'   # AutoML / XGBoost
GPU                 = True          # If GPU available  
SCORING_METRIC      = "logloss"    # aucpr/logloss
DF                  = 4

#: GPU CONFIG (IF ANY AVAILABLE) ---
if not GPU:
    gpu_params = {}
else:
    import subprocess
    # This just checks if the GPU is available and prints the name of the GPU
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        text=True
    ).strip().splitlines()
    for line in out:
        idx, name = [x.strip() for x in line.split(",", 1)]
        print(f"GPU {idx}: {name}")
    # These parameters are going to be passed to the XGBoost solvers
    os.environ["H2O_XGBOOST_GPU"] = "1"
    gpu_params = {
        "backend":        "gpu",          # This just tells the solver to use gpu is available
        "tree_method":    "hist",
        "gpu_id":         0,              # GPU ID
    }

#: DATA LOADING -------------------
df_dyn      = pd.read_pickle(PATH_DYN)
df_summary  = pd.read_pickle(PATH_SSUM)
split_info  = pd.read_pickle(PATH_SPLIT)

#: FEATURES LOADING --------------
TARGET_FEATURES     = fm.get_top_predictors(selection_criteria='min_set_predictors_logloss') # Best features from ElasticNet
FEATURES_ORGANIZED  = fm.identify_predictors(df_dyn.columns.tolist(), by_family=True)
L6S_FEATURES        = [ x for x in FEATURES_ORGANIZED["L6S"]  if x in TARGET_FEATURES]
LIDHO_FEATURES      = [ x for x in FEATURES_ORGANIZED["LIDHO"] if x in TARGET_FEATURES]
W_FEATURES          = [ x for x in FEATURES_ORGANIZED["W"] if x in TARGET_FEATURES]
CW_FEATURES         = [ x for x in FEATURES_ORGANIZED["CW"] if x in TARGET_FEATURES]
LS_FEATURES         = [ x for x in FEATURES_ORGANIZED["LS"] if x in TARGET_FEATURES]
HOT_ENCODED_VAR     = [ x for x in fm.hot_encoded if x in TARGET_FEATURES]

df_training, df_validation = get_training_validation_sets( #df_validation is just used to check the split in this script
                        df_dyn, 
                        df_summary, 
                        split_info, 
                        up_to_IDH=True, 
                        fold_col='fold'
                    )

check_division(df_training, df_validation)
del df_validation; gc.collect()

# %% #* ========================= INFORMATION CHECK/CLASS WEIGHTS ===================================
#: This is the same weight as the one used for the feature selection process.
#: This aims to fined a pos:neg ration of 1/4 and the reason why i set the same
#: weight is to make the final log-loss comparable between the logistic regresion and the
#: XGBoost model.
neg     = df_training[df_training['Ground_truth'] == 0].shape[0]
pos     = df_training[df_training['Ground_truth'] == 1].shape[0]
weights = (neg/pos) / DF

#: NaNs Info ------------------------------------------
nans    = df_training[TARGET_FEATURES].isna().sum(axis = 0)
total   = pd.Series({"TRAINING OBSERVATIONS, TO COMPARE THE VALUES WITH": len(df_training)})
nans    = pd.concat([total, nans])
nans.to_csv(PATH_NANs_SUMMARY)
# IMPORTANT NOTE: Most predictors (columns) have complete values.
# For those with missing data, the XGBoost model will handle them if present
# during training. 

# %% #* ================================== MODEL TRAINING ==========================================
#: H2O INIT -------------------------------------------------
def get_free_port():
    """Find an available port on the system dynamically."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to any free port
    port = sock.getsockname()[1]
    sock.close()
    return port
# We dont want to oversaturate the server 
if h2o.connection():
    print("Shutting down existing H2O server...")
    h2o.cluster().shutdown()
random_port = get_free_port()
# H2o Initialization and some h2o Pre-processing Steps
h2o.init(port=random_port, max_mem_size=mem, verbose=False)
print(f"H2O started on port {random_port}")


#: H2O DATA INPUT ---------------------------------
response        = "Ground_truth"
predictors      = TARGET_FEATURES
train_h2o       = h2o.H2OFrame(df_training)
train_h2o[predictors]           = train_h2o[predictors].asnumeric() 
train_h2o["Ground_truth"]       = train_h2o["Ground_truth"].asfactor()
train_h2o['fold']               = train_h2o['fold'].asnumeric()


#: TIME COUNTER -----------------------------------------
t_start = pd.Timestamp.now()
current_date = t_start.strftime("%Y-%m-%d - %H:%M")
print(f"Initial Loop Time : {current_date}")
if MODEL_SEARCH  == 'AutoML':
    aml = H2OAutoML(
        max_runtime_secs        = max_time,
        seed                    = random_seed,
        stopping_metric         = SCORING_METRIC,
        sort_metric             = SCORING_METRIC,
        keep_cross_validation_models        = True,
        keep_cross_validation_predictions   = True,
        exploitation_ratio      = 0.25,  # This is the exploitation ratio
        exclude_algos           =["GLM", "DeepLearning", "StackedEnsemble"]
    )
    aml.train(
        x               = predictors,
        y               = 'Ground_truth',
        training_frame  =  train_h2o,  
        fold_column     = "fold",        
        weights_column  = "weights"
    )
    # Leaderboard --------------------------------------------
    performance_storage = {}
    lb_df      = aml.leaderboard.as_data_frame(use_multi_thread=True, use_pandas=True)
    top_10     = lb_df.head(5)
    model_ids  = top_10["model_id"].tolist()
    
elif MODEL_SEARCH =='XGBoost':

    # Hyper-parameters space 
    hyper_params = {
        
        # ---------------------------------------------------------------
        # The parameter history is stored in XGBoost_Params_runX.csv
        # ---------------------------------------------------------------
        # n_trees / learning rate:
        # (run 1): Models consistently reached 750 trees with a learning rate of 0.05 without early stopping.  
        #          I will tighten the stopping tolerance and reduce n_trees from 750 to 300 to prevent overfitting and improve speed.
        # (run 1.5): same hyperparameters.
        "ntrees":        [300],
        "learn_rate":    [0.05],

        # max depth:
        # (run 1): Best CV average was reached with lower depth, so I’ll modify the grid [4, 6] → [3, 4].
        # (run 1.5): [3, 4] → [4, 5]
        "max_depth":     [4, 5],

        # min_rows:
        # (run 1): There is no clear tendency, so I’ll leave them unchanged.
        # (run 1.5): same hyperparameters.
        "min_rows":      [1, 5],

        # L2 regularization (Ridge):
        # (run 1): Initial values were [5.0, 25.0, 50.0].
        # (run 1.5): Larger L2 improved cross-validation AUPRC; changed [5.0, 25.0, 50.0] → [25.0, 50.0, 75.0].
        "reg_lambda":    [25.0, 50.0, 75.0],       

        # L1 regularization (Lasso):
        # (run 1): Not present in the original model; added to the grid.
        # (run 1.5): Lower values performed better; changed [0.0, 0.2] → [0.0, 0.1].
        "reg_alpha":     [0.0, 0.1],       

        # gamma:
        # (run 1): Not present in the original model; added to the grid.
        # (run 1.5): same hyperparameters.
        "gamma":         [0, 0.05],

        # sample_rate:
        # (run 1): Lower values performed slightly better; changed [0.85, 1.0] → [0.85, 0.95].
        # (run 1.5): same hyperparameters.
        "sample_rate":   [0.85, 0.95],

        # col_sample_rate_per_tree:
        # (run 1): Not present in run 1; added for run 2.
        # (run 1.5): same hyperparameters.
        "col_sample_rate_per_tree": [0.9],

        # scale_pos_weight: 1 pos : 4 neg ratio. 
        "scale_pos_weight":            [weights]    
        }
    
    
    search_criteria = {
        # strategy:
        # (Run 1): Used random search with 40 models, but analyzing hyperparameter effects was difficult.  
        # Switching to Cartesian search to ensure all proposed models are evaluated.
        # (run 1.5): same setting.
        "strategy":        "Cartesian",
    }

    # Base model configuration
    xgb_base = H2OXGBoostEstimator(
        distribution                = "bernoulli",   
        fold_column                 = "fold",
        stopping_metric             = SCORING_METRIC,
        seed                        = random_seed,
        keep_cross_validation_models       = True,
        keep_cross_validation_predictions  = True,
        # Score interval/tolerance/stopping_roundsL
        # (run 1): scored every 20, 0.005 tolerance, Stopping rounds disabled
        #          I will reduce the score interval to 10 and enable stopping rounds
        score_tree_interval                = 10,            # how often the metrics are calculated (trees)
        stopping_rounds                    = 2,             # how many score intervals to consider to calculate the stopping metric
        stopping_tolerance                 = 0.0085,         # tolerance for the stopping metric using the last n score intervals
        **gpu_params
    )
    # GRID
    xgb_grid = H2OGridSearch(
        model               = xgb_base,
        hyper_params        = hyper_params,
        #hyper_params        = debug_hyperparams,
        search_criteria     = search_criteria,
        parallelism         = 0 # adaptive parallelism
    )
    xgb_grid.train(
        x                   = predictors,
        y                   = "Ground_truth",
        training_frame      = train_h2o
    )
    
    #: GRID INFORMATION STORAGE --------------------------
    # This saves extensive information about the training process. 
    # The resulting model is already trained on 100% of the training set as part of cross-validation, 
    # so there is not necessary any additional training steps after retrieving a model from the grid.
    saved_path = h2o.save_grid(
        grid_directory                          = PATH_GRID_MODELS,
        grid_id                                 = xgb_grid.grid_id,
        save_params_references                  = True,     # also save referenced frames
        export_cross_validation_predictions     = True,      # keep CV hold-out predictions 
    )
    print(f"Grid archived to {saved_path}")
    
    #: CSV FILE WITH A SUMMARY DIRECTLY FROM H2O -----------
    performance_storage = {}
    lb_df   = (
        xgb_grid.get_grid(sort_by="logloss", decreasing=False)
        .sorted_metric_table()  
    )
    # storage:
    if os.path.exists(PATH_GRID_RESULTS):
        lb_df.to_csv(
            PATH_GRID_RESULTS,
            mode='a',               # append
            header=False,
            index=False
        )  
    else:
        lb_df.to_csv(
            PATH_GRID_RESULTS,
            mode='w',               # write
            header=True,
            index=False
        )
    top_10    = lb_df.head(10)
    model_ids = top_10["model_ids"].tolist()
    
t_end = pd.Timestamp.now()
current_date2 = t_end.strftime("%Y-%m-%d - %H:%M")
elapsed_sec = (t_end - t_start).total_seconds()
print(f"Final Loop Time : {current_date2}")
print(f"Elapsed time: {elapsed_sec:.0f} seconds")

    
# %% #* ================================== Post Processing =========================================
# I’ll manually identify the best model based on the specified STOPPING_METRIC.
# H2O can provide this directly, but here I’ll replicate the process using scikit-learn.

#: HELPER FUNCTIONS ---------------------------
def stats(vals, ci):
    vals = np.asarray(vals)
    n     = len(vals)
    mean  = vals.mean()
    sd    = vals.std(ddof=1)  # sample standard deviation
    sem   = sd / np.sqrt(n)
    # Two tailed 95CI based on t-distribution             
    alpha   = 1.0 - ci/100.0            
    df      = n - 1
    t_val   = t.ppf(1 - alpha/2, df)   
    lo      = mean - t_val * sem
    hi      = mean + t_val * sem
    return mean, lo, hi, sd, sem

#: PER-FOLD METRICS SUMMARY -------------------
y_true = (
    train_h2o["Ground_truth"]
    .as_data_frame(use_multi_thread=True, use_pandas=True)
    ["Ground_truth"]           
    .astype(int)               
    .values
    .ravel()                  
)
folds = (
    train_h2o["fold"]
    .as_data_frame(use_multi_thread=True, use_pandas=True)["fold"]
    .astype(float)   
    .astype(int)    
    .values
)

# Iteration over top 10 models
results = []
oof_xgboost = []
for mid in model_ids:
    model   = h2o.get_model(mid)
    rows = []
    for k, pred_k in enumerate(model.cross_validation_predictions(), 1):
        
        #: TRUE AND PREDICTED VALUES AT EACH FOLD -----------
        mask = (folds==k) #This filters rows and orders the folds, 1, 2,3,4,5
        y_true_fold  = y_true[mask]
        y_pred_fold= (
            pred_k
            .as_data_frame(use_multi_thread=True, use_pandas=True)["p1"]
            .values[mask]
        )
        #: OOF METRICS STORAGE -----------------------------
        new = {
            "fold" : k,
            "auprc":            average_precision_score(y_true_fold, y_pred_fold),
            "logloss":          log_loss(y_true_fold,        y_pred_fold),
            "auroc":            roc_auc_score(y_true_fold, y_pred_fold),
            "recall_at_fpr":    recall_at_fpr(y_true_fold,y_pred_fold),
            "precision_at_fpr": precision_at_fpr(y_true_fold, y_pred_fold)
        }
        rows.append(new)
    
        #: OOF PREDICTIONS STORAGE -------------------------
        entries_fold = len(y_true_fold)
        fold_id_sto  = [f"{k}"] * entries_fold
        oof_true_k   = y_true_fold
        oof_prob_k   = y_pred_fold
        oof_xgboost.append({"model_id": mid,
                            "fold": fold_id_sto,
                            "y_true": oof_true_k,
                            "oof_predictions": oof_prob_k})
    # Results across all the folds
    df = pd.DataFrame(rows).set_index('fold')
    metrics = {}
    for col in df.columns:
        mean, lo, hi, sd, sem = stats(df[col], ci=95)
        metrics.update({
            f"cv_folds_mean_{col}" : mean,
            f"cv_folds_lower_{col}": lo,
            f"cv_folds_upper_{col}": hi,
            f"cv_folds_sd_{col}"   : sd,
            f"cv_folds_sem_{col}"  : sem
        })
        
    results.append({
        f"model": mid,
        **metrics
    })

#: TOP MODELS ---------------------------------------------
performance_df = pd.DataFrame(results).set_index("model")
df_oof_xgboost = pd.DataFrame(oof_xgboost)
df_oof_xgboost = (
    df_oof_xgboost
    .explode(["fold","y_true","oof_predictions"])
    .reset_index(drop=True)
)

performance_df['unique_identifer'] = random.random()
print("\n")
print(f"Latest Execution Date: {current_date}")

#: BEST MODEL (CV) ----------------------------------------
best_model_id       = performance_df['cv_folds_mean_logloss'].idxmin()
best_model          = h2o.get_model(best_model_id)
best_model_params   = best_model.actual_params
model_id            = best_model.model_id
df_oof_xgboost     = df_oof_xgboost[df_oof_xgboost["model_id"] == model_id]

#: STORAGE -------------------------------------------------
df_oof_xgboost.to_csv(PATH_OOF_PREDICTIONS)


#: FINAL MODEL -------------------------------------------------------------
BIN_DIR    = os.path.join(PATH_TOP_MODEL, "bin")                               
MOJO_DIR   = os.path.join(PATH_TOP_MODEL, "MOJO")  
os.makedirs(BIN_DIR, exist_ok = True)    
os.makedirs(MOJO_DIR, exist_ok = True)                   
leader_bin = h2o.save_model(
    model = best_model,
    path  = (os.path.join(BIN_DIR,"MainModel")),
    force = True,
    export_cross_validation_predictions = True,   # keep OOF preds
)
print(f"Leader binary   → {leader_bin}")

path_lmojo = os.path.join(MOJO_DIR,"MainModel")
best_model.save_mojo(path_lmojo,force=True)
print(f"Leader MOJO     → {path_lmojo}")

#: CV MODELS --------------------------------------------------------------
for i, k_model in enumerate(best_model.cross_validation_models(), start=1):

    # -------- binary
    fold_bin = h2o.save_model(
        model = k_model,
        path  = os.path.join(BIN_DIR, f"Kfold_model{i}"),
        force = True,
    )
    print(f"  ⋅ CV-fold {i} binary → {fold_bin}")

    # -------- mojo
    fold_mojo_path = os.path.join(MOJO_DIR, f"Kfold_model{i}")
    k_model.save_mojo(fold_mojo_path,force=True)
    print(f"  ⋅ CV-fold {i} MOJO   → {fold_mojo_path}")
    
print("All models saved as binaries and MOJOs")
#! 'export_cross_validation_models' is important: if any RL model is trained on the
#! same data, WE GOTTA use these proxy models to approximate the overall model’s behavior
#! and prevent data leakage.

if os.path.exists(PATH_TOP10_STATISTICS):
    performance_df.to_csv(
        PATH_TOP10_STATISTICS,
        mode='a',               # append
        header=False,
    )  
else:
    performance_df.to_csv(
        PATH_TOP10_STATISTICS,
        mode='w',               # write
        header=True,
    )

# Training Summary Text File
with open(PATH_BRIEFTXT_SUMMARY, 'w') as f:
    f.write("====================================================================\n")
    f.write("Training Set Metrics (H2O Output Summary):\n")
    f.write("====================================================================\n")
    f.write(f"Model ID (GRID ID) : {model_id}\n")
    f.write(f"Model Performance (h2o summary):\n\n {best_model}\n\n")
    f.write("====================================================================\n")
    f.write(f"Model Performance (custom summary averaged over out-of-fold predictions)):\n")
    row = performance_df.loc[model_id]          # 1-row Series
    for metric in ["auprc", "auroc","logloss", "recall_at_fpr", "precision_at_fpr"]:
        mean = row[f"cv_folds_mean_{metric}"]
        lo   = row[f"cv_folds_lower_{metric}"]
        hi   = row[f"cv_folds_upper_{metric}"]
        f.write(f"{metric:<18} : {mean:.4f}  95%CI({lo:.4f} -- {hi:.4f})\n")
            