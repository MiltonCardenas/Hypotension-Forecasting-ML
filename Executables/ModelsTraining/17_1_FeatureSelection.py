"""
# --------------------------------------------------------------------------------------------------
# This script performs a logistic regression to find the most important features
# out of the variables gathered. 
# v.1.0 author: Milton Dario Cardenas 

# Inputs:
# (1) df_dyn      - DataFrame with all signals from the Mladi24 and hidenic15 datasets.
# (2) df_summary  - DataFrame summarizing all sessions in the Mladi24 and hidenic15 datasets.
# (3) sessions_df - DataFrame defining the train/validation/internal-fold split,
#                   produced by "Executables/16_2_Training_validation_split.py".
# (4) To consider: There is DataFrame read from "Other/Signals.xlsx" containing signal metadata,
#                   filtered via FeaturesManager.default_exclusion.

# Outputs = 
# (1) Creates and save a DataFrame (df_results) that includes:
#     • feature coefficients
#     • alpha and lambda values
#     • model performance metrics
#     Other Information related to the elastic NET Logistic Regression
#     Output path: Models_storage/DynamicPredictionModel/results_ElasticNet.csv
# (2) This saves a plaintext summary of the coefficients for the
#     smallest feature set whose AUPRC is within one SEM of the
#     maximal AUPRC.
#     Output path: Models_storage/DynamicPredictionModel/ElasticNET_summary.txt
# (3) coefficients vs AUPRC, AUROC, and smallest feature set names vs 
#     coefficients values, these are stored at Documents/Multimedia/ElasticNET_ (...)
# (4) Scaler Parameters stored at "Models_storage/DynamicPredictionModel/ElasticNET_InitialFeaturesAndScalerParams.csv"
# Author V1.0: Milton Cardenas

_/$$$$$$$$                    /$$                                          /$$$$$$            /$$                       /$$     /$$                    
| $$_____/                   | $$                                         /$$__  $$          | $$                      | $$    |__/                    
| $$     /$$$$$$   /$$$$$$  /$$$$$$   /$$   /$$  /$$$$$$   /$$$$$$       | $$  \__/  /$$$$$$ | $$  /$$$$$$   /$$$$$$$ /$$$$$$   /$$  /$$$$$$  /$$$$$$$ 
| $$$$$ /$$__  $$ |____  $$|_  $$_/  | $$  | $$ /$$__  $$ /$$__  $$      |  $$$$$$  /$$__  $$| $$ /$$__  $$ /$$_____/|_  $$_/  | $$ /$$__  $$| $$__  $$
| $$__/| $$$$$$$$  /$$$$$$$  | $$    | $$  | $$| $$  \__/| $$$$$$$$       \____  $$| $$$$$$$$| $$| $$$$$$$$| $$        | $$    | $$| $$  \ $$| $$  \ $$
| $$   | $$_____/ /$$__  $$  | $$ /$$| $$  | $$| $$      | $$_____/       /$$  \ $$| $$_____/| $$| $$_____/| $$        | $$ /$$| $$| $$  | $$| $$  | $$
| $$   |  $$$$$$$|  $$$$$$$  |  $$$$/|  $$$$$$/| $$      |  $$$$$$$      |  $$$$$$/|  $$$$$$$| $$|  $$$$$$$|  $$$$$$$  |  $$$$/| $$|  $$$$$$/| $$  | $$
|__/    \_______/ \_______/   \___/   \______/ |__/       \_______/       \______/  \_______/|__/ \_______/ \_______/   \___/  |__/ \______/ |__/  |__/
"""

# %% #* ============================= Libraries and Instances ======================================
import sys
import os
import math
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_curve, make_scorer
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import Parallel, delayed

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

#: Instances --------------------
pm = PathManager().paths
gp = GlobalParameters()
pa = ProcessingAuxiliar()
sm = sampling_methods()
fm = FeaturesManagerV2()


#: INPUT PATHS ------------------
DIR_PROCESSED       = "Data/Processed_datasets/Summary/Merged_M24H15"
PATH_DYN            = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Signals.pkl")
PATH_SSUM           = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Summary.pkl") 
PATH_SPLIT          = os.path.join(DIR_CURRENT, "Models_storage/DynamicPredictionModel/StratifiedDivision/stratified_division_summary.pkl")


#: OUTPUT PATHS --------------
DIR_OUTPUT = "Models_storage/DynamicPredictionModel/ElasticNET"
model_path         = os.path.join(DIR_OUTPUT, "ElasticNET_Model.pkl")
path_scaler        = os.path.join(DIR_OUTPUT, "ElasticNET_data_scaler")
scaler_params_path = os.path.join(DIR_OUTPUT, "ElasticNET_InitialFeaturesAndScalerParams.csv")
path_imputation_info = os.path.join(DIR_OUTPUT, "ElasticNET_ImputationInfo.csv")
path_results       = os.path.join(DIR_OUTPUT, "ElasticNET_results.csv")
path_txt_file      = os.path.join(DIR_OUTPUT, "ElasticNET_summary.txt")
path_coefficients  = os.path.join(DIR_OUTPUT, "ElasticNET_Coefficients.csv")

#: CONSTRAINED LOGISTIC REGRESSION PARAMETERS  ------
random_seed     = 2000
ALPHAS          = [1.0]
TOTAL_LAMBDAS   = 300   # lambdas for the grid
DF              = 4     # this number is used to get the weights  
N_WORKERS       = 10    # One per core

#: Data loading ------------------------------------
df_dyn      = pd.read_pickle(PATH_DYN)
df_summary  = pd.read_pickle(PATH_SSUM)
split_info = pd.read_pickle(PATH_SPLIT)

#:FEATURES LOADING (AVAILABLE FEATURES, INTERNAL CLASSIFICATION) -----
TARGET_FEATURES0    = fm.identify_predictors(df_dyn.columns.tolist())
FEATURES_ORGANIZED  = fm.identify_predictors(df_dyn.columns.tolist(), by_family=True)
L6S_FEATURES        = FEATURES_ORGANIZED["L6S"]
LIDHO_FEATURES      = FEATURES_ORGANIZED["LIDHO"]
W_FEATURES          = FEATURES_ORGANIZED["W"]
CW_FEATURES         = FEATURES_ORGANIZED["CW"]
LS_FEATURES         = FEATURES_ORGANIZED["LS"]
HOT_ENCODED_VAR     = fm.hot_encoded

#: labeled data (This is how the model know Internal folds)
df_training, df_validation = get_training_validation_sets(df_dyn, df_summary, split_info, up_to_IDH=True, fold_col='fold_CV_10')
df_training = df_training.rename(columns={'fold_CV_10':'fold'})
check_division(df_training, df_validation)


# %% #*============================== Data Pre - Processing =========================================
#: POSITIVE CLASS WEIGHTS ----------------- 
# This help us to give a higher priority to the positive class. This value is divided by an amount `DF`
# just because by pluggiing in the raw class imbalance ratio the solver becomes unstable. 
neg = df_training[df_training['Ground_truth'] == 0].shape[0]
pos = df_training[df_training['Ground_truth'] == 1].shape[0]
weights = (neg/pos) / DF
print(f"Division Factor {DF}")
print(f"Weight for the positive class: {weights}")

#: DATA CHECK -----------------------------

df_training[TARGET_FEATURES0] = df_training[TARGET_FEATURES0].astype(float)
df_validation[TARGET_FEATURES0] = df_validation[TARGET_FEATURES0].astype(float)

#: MISSING VALUES IMPUTATION --------------
# Some values are missing, This is not a problem for a Standard XgBoost
# However, ElasticNet Needs data to be (1) Complete and (2) normalized.
# Most of the missing features are information related to previous
# dialysis sessions or hypotensive onsets. Ill shut these signals
# down by assigning 0 values. A better approach is encouraged. 
nan = df_training[TARGET_FEATURES0].isna().sum(axis=0)
fill_values = {}
for col in TARGET_FEATURES0:
    if col in LIDHO_FEATURES:
        fill_values[col] = 0
    else:
        fill_values[col] = df_training[col].median()
df_filling_policy = (
    pd.DataFrame.from_dict(fill_values, orient='index', columns=['fill_value'])
      .reset_index()
      .rename(columns={'index':'feature'})
)
df_filling_policy.to_csv(path_imputation_info)
df_training[TARGET_FEATURES0]   = df_training[TARGET_FEATURES0].fillna(fill_values)
df_validation[TARGET_FEATURES0] = df_validation[TARGET_FEATURES0].fillna(fill_values)


#: NORMALIZATION: ------------------------
# This is better done at the CV level (using a different scaler for each fold)
# However, I'll keep it simple for now.
target_scaler = [var for var in TARGET_FEATURES0 if not var in HOT_ENCODED_VAR]
sc = StandardScaler().fit(df_training[target_scaler])
df_training[target_scaler] = sc.transform(df_training[target_scaler])
max_df = df_training[TARGET_FEATURES0].max()
min_df = df_training[TARGET_FEATURES0].min()
df_validation[target_scaler] = sc.transform(df_validation[target_scaler])
joblib.dump(sc, path_scaler)
#: scaler parameters -----------------------------------------------------
params_df = pd.DataFrame({
    "feature": target_scaler,  
    "mean":    sc.mean_,          
    "scale":   sc.scale_,         
    "var":     sc.var_            
})
params_df.to_csv(scaler_params_path)


# %% #* ======================= REGULARIZED LOGISTIC REGRESSION ====================================

#: TRAINING INTERNAL DIVISION: ----------------
FEATURES   = TARGET_FEATURES0
X_scaled   = df_training[FEATURES].to_numpy()
y          = df_training["Ground_truth"].to_numpy()
fold_array = df_training["fold"].to_numpy().astype(int)     # Fold nubers 0, 1, 2
ps         = PredefinedSplit(test_fold=fold_array)          # PredefinedSplit object

#: GRID SEARCH PARAMETERS (20% more denser for lambdas that lead to a fewer coefficients)
extra_factor  = 0.2
extra_n       = int(TOTAL_LAMBDAS * extra_factor)  # +20%  extra
bounds        = {
                    1.0: (250, 32_000, 500, 18_000) 
                }
LAMBDA_GRID = {}
for α in ALPHAS:
    full_min, full_max, sub_min, sub_max = bounds[α]
    base = np.logspace(np.log10(full_min),
                       np.log10(full_max),
                       TOTAL_LAMBDAS)
    extra = np.logspace(np.log10(sub_min),
                        np.log10(sub_max),
                        extra_n)
    grid = np.unique(np.concatenate([base, extra]))
    LAMBDA_GRID[α] = grid
class_weights  = {0: 1,  1: int(weights)}  
COEF_EPS       = 2e-5
rng_global      = np.random.default_rng(random_seed)


#: HELPER FUNCTIONS -------------------------
def stats(vals, ci):
    vals = np.asarray(vals)
    n     = len(vals)
    mean  = vals.mean()
    sd    = vals.std(ddof=1)            # sample standard deviation
    sem   = sd / np.sqrt(n)
    # Two tailed 95CI based on t-distribution             
    alpha   = 1.0 - ci/100.0            
    df      = n - 1
    t_val   = t.ppf(1 - alpha/2, df)   
    lo      = mean - t_val * sem
    hi      = mean + t_val * sem
    return mean, lo, hi, sd, sem

# Recall at FPR
def recall_at_fpr(y_true, y_proba, target_fpr=0.20):
    eps         = 1e-12
    p1          = np.clip(y_proba, eps, 1 - eps)
    fpr, tpr, _ = roc_curve(y_true, p1)
    valid       = fpr <= target_fpr
    return tpr[valid].max() if valid.any() else 0.0

# This is just a modification to use it as an input for cross_validate
def rec_fpr20(y_true, y_proba, **_):
        return recall_at_fpr(y_true, y_proba, target_fpr=0.20)
rec_scorer = make_scorer(
                rec_fpr20,
                needs_proba=True # returns the 1-D positive-class prob vector
            )

# This cut extra decimals to enhance reading
def round_sig(x, sig=2):
    if x == 0:
        return 0.0
    # find how many places to shift
    shift = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, shift)

# The loop might take a long time to run, so I'll save the results
# at each iteration
def save_row(row_dict:dict, save_path:str):
    df_row = pd.DataFrame([row_dict])
    lock_path      = save_path + ".lock"             # used by FileLock
    with FileLock(lock_path):
        header_written = os.path.exists(save_path)    # avoid rewriting header
        df_row.to_csv(save_path,
                      mode="a",
                      index=False,
                      header=not header_written)
        
#: MAIN GRID LOOP --------------------------------------------
# This function fits a linear logistic regressor based on the parameters 
# alpha and lambda defined above.
def evaluate(alpha, lam):
    C_val = 1.0 / lam
    lam = round_sig(lam, 2)
    C_val = round_sig(C_val, 2)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=alpha,
        C=C_val,
        class_weight=class_weights,
        max_iter=5_000,
        tol=5e-4,
        n_jobs=1,                 
        warm_start=True,
    )
    
    #: PERFORMANCE METRICS (SCORERS) -------
    # The following are the metrics I'll get from each of the folds in the training set.
    scoring = {
        "auprc": "average_precision",
        "auroc": "roc_auc",
        "recall@0.2fpr": rec_scorer,
        "neg_log_loss"    : "neg_log_loss"
    }
    
    #: INTERNAL FOLD CROSS VALIDATION -----
    # ps already contains information about the internal division
    cv_results = cross_validate(
        model,
        X_scaled, y,
        cv              = ps,
        scoring         = scoring,
        return_train_score  = False,
        return_estimator    = True,     # Ill get the total iterations from here to know about the convergence
        n_jobs              = 1                
    )
    
    #: RESULTS ON EACH FOLD PREDICTIONS ---------
    # These are vectors with k=10 predictions 
    auprc_scores            = cv_results['test_auprc']
    auroc_scores            = cv_results['test_auroc']
    rec_scores              = cv_results['test_recall@0.2fpr']
    log_loss                = -cv_results["test_neg_log_loss"]
    estimators              = cv_results['estimator']   
             
    #: KEY COEFFICIENTS FINDING -----------
    n_iters    = [est.n_iter_.max() for est in estimators]
    intercepts = [est.intercept_[0] for est in estimators]
    coef_values = []
    for i, est in enumerate(estimators, 1):
        coefs = est.coef_.ravel()
        mask = np.abs(coefs) > COEF_EPS
        feats = np.array(FEATURES)[mask]
        target_coefs = coefs[mask]
        # Now we can define the pairs NAME:coeff in a dictionary
        pairs = {str(f): float(c)
                 for f, c in zip(feats, target_coefs)}
        # If features are zero, Ill set a placeholder
        if np.size(target_coefs) == 0:
            pholder = {'ZeroCoeff':0}
            pairs = {**pairs, **pholder}
        # Storage
        coef_values.append(pairs)
        
    #: AVERAGE COEFFICIENTS AND 95% CI ------
    # When a variable is not present in a model I'll include its 
    # coefficient as a 0.
    df      = pd.DataFrame(coef_values)
    n_feat  = df.notna().sum()       # Non zero coefficients
    df      = df.fillna(0)
    coeff_metrics = {}
    for col in df.columns.tolist():
        values = df[col].values
        results = stats(values, ci= 95)
        mean, low_th, hi_th, sd, sem = results 
        # This is the format of the names
        new_entry = {
            f"{col}_mean": mean,
            f"{col}_low": low_th,
            f"{col}_hi":  hi_th,
            f"{col}_sd":  sd,
            f"{col}_sem": sem,
        }
        # This overwrites if any key is repeated
        # It is an operation made in place, if i assign coeff_metrics == coeff_metrics.update()
        # the coeff_metrics will return "none"
        coeff_metrics.update({col:new_entry}) 
    # He Ill get the frequency of each variable across the different models
    nonzero_freq = {
        col: (df[col] != 0).sum() / df.shape[0]
        for col in df.columns
    }
    # I will select any feature present in at least 6/10 folds. I'will use
    # this amount of variables to select (1) the number of features per iteration
    # and (2) the names of the key features at each alpha and lambda pair
    key_freq = {
        key:val 
        for key, val in nonzero_freq.items() 
        if val >= 0.6       # Features present in at leat 6 out of 10 folds
    }
    # I will get a subset of the Coefficients' metrics for the key metrics
    if not coeff_metrics:
        key_coeff_metrics = {}
    else:
        key_coeff_metrics = {
            key:val
            for key, val in coeff_metrics.items()
            if key in key_freq.keys()
        } 
    all_names = list(nonzero_freq.keys())
    total_var =  len(all_names) if 'ZeroCoeff' not in all_names else len(all_names)-1
    key_names = list(key_freq.keys())
    total_key =  len(key_names) if 'ZeroCoeff' not in key_names else len(key_names)-1
    # ----------------------------------------------------------------------------
    logloss_cv_m, logloss_cv_lo, logloss_cv_hi, logloss_cv_sd, logloss_cv_sem = stats(
        log_loss,
        ci=95
    )
    auprc_cv_m, auprc_cv_lo, auprc_cv_hi, auprc_cv_sd, auprc_cv_sem = stats(auprc_scores, ci= 95)
    auroc_cv_m, auroc_cv_lo, auroc_cv_hi, auroc_cv_sd, auroc_cv_sem = stats(auroc_scores, ci= 95)
    rec_cv_m,   rec_cv_lo,   rec_cv_hi,   rec_cv_sd,   rec_cv_sem   = stats(rec_scores,   ci= 95)
    inter_m,    inter_lo,    inter_hi,   inter_sd, inter_sem = stats(intercepts, ci=95)
    # ---------------------------------------------------------------------------
    row_dict = {
        # grid & sparsity info -------------------------------------------------
        "alpha":                alpha,
        "lambda":               lam,
        "C":                    C_val,
        "n_iter_per_fold":      [int(x) for x in n_iters],
        "n_nonzero":            total_var,
        "nonzero_names":        all_names,
        "nonzero_freq":         nonzero_freq,     # dict: feature → [0–1]
        "nonzero_coeff_metrics": coeff_metrics,
        "n_key_nonzero":        total_key,
        "key_names":            key_names,
        "key_freq":             key_freq,
        "key_coeff_metrics":    key_coeff_metrics,
        # - Intercept stats
        "intercept":            inter_m,
        "intercept_lo" :        inter_lo,
        "intercept_hi" :        inter_hi,
        "intercept_sd":         inter_sd,
        "intercept_sem":        inter_sem,
        
        # --- LOGLOSS:
        "logloss"     : logloss_cv_m,
        "logloss_lo"  : logloss_cv_lo,
        "logloss_hi"  : logloss_cv_hi,
        "logloss_sd"  : logloss_cv_sd,
        "logloss_sem" : logloss_cv_sem,
        
        # - auprc stats
        "auprc"        : auprc_cv_m,
        "auprc_lo"     : auprc_cv_lo,
        "auprc_hi"     : auprc_cv_hi,
        "auprc_sd"     : auprc_cv_sd,
        "auprc_sem"    : auprc_cv_sem,
        # - auroc stats
        "auroc"        : auroc_cv_m,
        "auroc_lo"     : auroc_cv_lo,
        "auroc_hi"     : auroc_cv_hi,
        "auroc_sd"     : auroc_cv_sd,
        "auroc_sem"    : auroc_cv_sem,
        # - recall stats
        "recall_fpr20" : rec_cv_m,
        "recall_lo"    : rec_cv_lo,
        "recall_hi"    : rec_cv_hi,
        "recall_sd"    : rec_cv_sd,
        "recall_sem"   : rec_cv_sem,
    }
    save_row(row_dict, os.path.join(DIR_CURRENT,path_results))  
    return row_dict
    
# Parallel EXECUTION ------------------------------------------------------------
# Here I built the grid once
param_grid = [(a, l) for a in ALPHAS for l in LAMBDA_GRID[a]]
# Outer loop in parallel
results = Parallel(
    n_jobs=N_WORKERS,
    backend="loky",            
    verbose=5                  
)(delayed(evaluate)(a, l) for a, l in param_grid)
# Storage ------------------------------------------------------------------------
results_df = (
    pd.DataFrame(results)
      .sort_values(["alpha", "lambda"], ascending=[True, False])
      .reset_index(drop=True)
)
results_df.to_csv(os.path.join(DIR_CURRENT, path_results),
    index=False)




