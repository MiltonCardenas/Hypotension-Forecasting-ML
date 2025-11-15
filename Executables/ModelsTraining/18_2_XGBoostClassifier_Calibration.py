"""
# -------------------------------------------------------------------------------------------------
# This script calibrates the XGBoost model by adding an extra processing step. 
# I evaluated two methods, beta calibration and Platt calibration, and the former 
# achieved better log-loss and ECE* performance. 
# The script then saves the final model (XGBoost + beta calibration) as a Java object 
# for efficient real-time deployment, downloads the required H2O and XGBoost JAR dependencies, 
# and compiles the wrapper.
# v.1.0 author: Milton Dario Cardenas 

  /$$$$$$            /$$ /$$ /$$                            /$$     /$$                    
 /$$__  $$          | $$|__/| $$                           | $$    |__/                    
| $$  \__/  /$$$$$$ | $$ /$$| $$$$$$$   /$$$$$$  /$$$$$$  /$$$$$$   /$$  /$$$$$$  /$$$$$$$ 
| $$       |____  $$| $$| $$| $$__  $$ /$$__  $$|____  $$|_  $$_/  | $$ /$$__  $$| $$__  $$
| $$        /$$$$$$$| $$| $$| $$  \ $$| $$  \__/ /$$$$$$$  | $$    | $$| $$  \ $$| $$  \ $$
| $$    $$ /$$__  $$| $$| $$| $$  | $$| $$      /$$__  $$  | $$ /$$| $$| $$  | $$| $$  | $$
|  $$$$$$/|  $$$$$$$| $$| $$| $$$$$$$/| $$     |  $$$$$$$  |  $$$$/| $$|  $$$$$$/| $$  | $$
 \______/  \_______/|__/|__/|_______/ |__/      \_______/   \___/  |__/ \______/ |__/  |__/
                                                                                           
        

  /$$$            /$$$$$$   /$$                                                     
 /$$ $$          /$$__  $$ | $$                                                     
|  $$$          | $$  \__//$$$$$$    /$$$$$$   /$$$$$$  /$$$$$$   /$$$$$$   /$$$$$$ 
 /$$ $$/$$      |  $$$$$$|_  $$_/   /$$__  $$ /$$__  $$|____  $$ /$$__  $$ /$$__  $$
| $$  $$_/       \____  $$ | $$    | $$  \ $$| $$  \__/ /$$$$$$$| $$  \ $$| $$$$$$$$
| $$\  $$        /$$  \ $$ | $$ /$$| $$  | $$| $$      /$$__  $$| $$  | $$| $$_____/
|  $$$$/$$      |  $$$$$$/ |  $$$$/|  $$$$$$/| $$     |  $$$$$$$|  $$$$$$$|  $$$$$$$
 \____/\_/       \______/   \___/   \______/ |__/      \_______/ \____  $$ \_______/
                                                                 /$$  \ $$          
                                                                |  $$$$$$/          
                                                                 \______/                                                                                                                                               
"""

# %% #* =========================== MODULES AND INPUT - OUTPUT PATHS =============================
#: STANDARD LIBRARIES --------
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import urllib.request
import glob 

#: OTHER LIBRARIES ---------
import h2o
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import FormatStrFormatter
from IPython.display import display

#: LOCAL TOOLS -------------
sys.path.append("Auxiliar_codes")
from characterization import StandardFig
from machine_learning_auxiliar import (
    FeaturesManagerV2,
    get_training_validation_sets,
    sampling_methods,
)
from classes_repository import PathManager, StandardNames

#: WORKING DIRECTORY -------
DIR_CURRENT = r'/ihome/rparker/mdc147/PhD_Project';  os.chdir(DIR_CURRENT)
sys.path.append(os.path.join(DIR_CURRENT, 'Auxiliar_codes'))

#: INSTANCES ---------------
pm = PathManager().paths
sm = sampling_methods()
fm  = FeaturesManagerV2()
recall_at_fpr = sm.recall_at_fpr
precision_at_fpr = sm.precision_at_fpr
threshold_at_fpr = sm.threshold_at_fpr

#: INPUT: GRID SEARCH INFORMATION -----------
dir_results             = "Models_storage/DynamicPredictionModel/XGBoostEstimator"
path_grid               = os.path.join(dir_results, "XGBoost_All_GridMembers/Grid_XGBoost_py_3_sid_b590_model_python_1753055460352_1")
path_grid_params        = os.path.join(dir_results, "XGBoost_All_GridMembers_Params.csv")
path_top_10             = os.path.join(dir_results, "XGBoost_Top10Metrics.csv")
path_OOF_predictions    = os.path.join(dir_results, "XGBoost_oof_predictions.csv")

#: INPUT: DATA PATHS ------------------
DIR_PROCESSED       = "Data/Processed_datasets/Summary/Merged_M24H15"
PATH_DYN            = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Signals.pkl")
PATH_SSUM           = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Summary.pkl") 
PATH_SPLIT          = os.path.join(DIR_CURRENT, "Models_storage/DynamicPredictionModel/StratifiedDivision/stratified_division_summary.pkl")
PATH_NAMES          = "Other/Names_mapping.xlsx"
#: OUTPUT: BEST MODEL STORAGE ---------
path_out_model          = os.path.join(dir_results, "XGBoostFinal_MOJO")     # ✅

#: OTHER OUTPUT FILES -----------------
# Files related to the raw XGBoost Model
graphics_dir            = "Documents/Multimedia/XGBoost"
path_learning_curve     = os.path.join(graphics_dir,  "TrainingSetOOF_XGBoost_LearningRate.pdf") # ✅

# Files related to the calibration process
graphics_dir2           = "Documents/Multimedia/XGBoost_calibration"
path_cal_comparison     = os.path.join(graphics_dir2,  "TrainingSetOOF_CalibratedProbability_comparison.pdf")    # ✅
path_noncal_score_dis   = os.path.join(graphics_dir2,  "TrainingSetOOF_Uncalibrated_ScoreDistribution.pdf")      # ✅
path_cal_score_dis      = os.path.join(graphics_dir2,  "TrainingSetOOF_Calibrated_ScoreDistribution.pdf")        # ✅
path_cal_txt            = os.path.join(graphics_dir2,    "TrainingSetOOF_Calibration_summary.txt") # ✅
TXT_BINSINFORMATION     = os.path.join(graphics_dir2,    "TrainingSetOOF_CalibrationBins.txt")

#: SCRIPT PARAMETERS -------
SHOW_PLOTS  = True
RE_COMPILE  = True

# %% #* ==================================== DATA LOADING ==========================================
#: DATA LOADING -------------------
df_dyn      = pd.read_pickle(PATH_DYN)
df_summary  = pd.read_pickle(PATH_SSUM)
split_info = pd.read_pickle(PATH_SPLIT)
df_fm       = pd.read_csv(PATH_NAMES)

#: TARGET FEATURES ---------
TARGET_FEATURES     = fm.get_top_predictors(selection_criteria='min_set_predictors_logloss') # Best features from ElasticNet
FEATURES_ORGANIZED  = fm.identify_predictors(df_dyn.columns.tolist(), by_family=True)
L6S_FEATURES        = [ x for x in FEATURES_ORGANIZED["L6S"]  if x in TARGET_FEATURES]
LIDHO_FEATURES      = [ x for x in FEATURES_ORGANIZED["LIDHO"] if x in TARGET_FEATURES]
W_FEATURES          = [ x for x in FEATURES_ORGANIZED["W"] if x in TARGET_FEATURES]
CW_FEATURES         = [ x for x in FEATURES_ORGANIZED["CW"] if x in TARGET_FEATURES]
LS_FEATURES         = [ x for x in FEATURES_ORGANIZED["LS"] if x in TARGET_FEATURES]
HOT_ENCODED_VAR     = [ x for x in fm.hot_encoded if x in TARGET_FEATURES]

df_training, df_validation = get_training_validation_sets(df_dyn, df_summary, split_info, up_to_IDH=True, fold_col='fold')
pos = len(df_training[df_training["Ground_truth"]==1])
neg = len(df_training[df_training["Ground_truth"]==0])
weights = neg/pos / 4

# %% #* ================================= BEST MODEL LOADING =======================================
#: H2O SERVER --------------------------------
def get_free_port():
    """Find an available port on the system dynamically."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to any free port
    port = sock.getsockname()[1]
    sock.close()
    return port

if h2o.connection():
    print("Shutting down existing H2O server...")
    h2o.cluster().shutdown()
random_port = get_free_port()
h2o.init(port=random_port, max_mem_size='36G', verbose=False)
print(f"H2O started on port {random_port}")

#: BEST MODEL LOADING ------
grid = h2o.load_grid(path_grid) 
grid_params = (
    pd.read_csv(path_grid_params)
    .drop(columns=['Unnamed: 0'])
    .set_index("model_ids")
)
bidx = grid_params['logloss'].idxmin()
best_model = h2o.get_model(bidx)

#: PERFORMANCE CHECK --------
# I calculated manually the metrics for the top 10, the next is just a comparison:
top10 = pd.read_csv(path_top_10).set_index("model")
bidx2 = top10["cv_folds_mean_logloss"].idxmin()
print(f"The ID of best model from h2o grid is:   {best_model.model_id}")
print(f"The best model after replicating the performance in scikit-learn is:   {bidx2}")
# Both are the same so `best_model` is indeed the best model.

# %% #* ================================= MODEL CALIBRATION ========================================
#: THE NEXT FUNCTION EVALUATES THE CALIBRATION PERFORMANCE -----------------------------------------
print("Calibrating the model ...")
def calibration_performance(df_results, bins, strategy, column_name, display_=False) -> pd.DataFrame:
    print(f"\nBins summary for {column_name}") if display_ else ""
    N         = len(df_results)
    
    if  strategy == "uniform":
        bin_edges   = np.linspace(0.0, 1.0, bins + 1)
        df_results["bin"] = (pd.cut(
                                df_results[column_name], bin_edges, include_lowest=True)
                                .cat.remove_unused_categories()
                            )
        bin_summary = (
                    df_results.groupby("bin", observed=False)[["bin", column_name, "y_true"]].agg(
                            observations  = (column_name,   lambda x: len(x)),
                            positives     = ('y_true',  lambda x: np.sum(x)),
                            negatives      = ('y_true',  lambda x: len(x) - np.sum(x)),
                            positive_frac  = ('y_true',  lambda x: 1/len(x) * np.sum(x)),
                            positive_prob  = (column_name,   lambda x: 1/len(x) * np.sum(x)),
                        )
                    )
        
        EcE         = np.sum(bin_summary['observations'] / N
                        * np.abs(bin_summary['positive_frac'] - bin_summary['positive_prob']))

        display(bin_summary) if display_ else ""
        print(f"ECE for {strategy} strategy  = {EcE}") if display_ else ""
        
    elif strategy == "quantile":
        df_results["bin"]   = pd.qcut(df_results[column_name], bins, precision=8, duplicates='drop')
        bin_summary         = (
                                df_results.groupby("bin", observed=False)[["bin", column_name, "y_true"]].agg(
                                    observations    = (column_name,   lambda x: len(x)),
                                    positives       = ('y_true',  lambda x: np.sum(x)),
                                    negatives       = ('y_true',  lambda x: len(x) - np.sum(x)),
                                    positive_frac  = ('y_true',  lambda x: 1/len(x) * np.sum(x)),
                                    positive_prob  = (column_name,   lambda x: 1/len(x) * np.sum(x)),
                                )
                            )
        EcE         = np.sum(bin_summary['observations'] / N
                * np.abs(bin_summary['positive_frac'] - bin_summary['positive_prob']))
                
        display(bin_summary) if display_ else ""
        print(f"ECE for {strategy} strategy = {EcE}") if display_ else ""
        
    else:
        raise ValueError(f"The strategy {strategy} is not defined")
        
    return bin_summary, EcE

#: THIS IS THE BETA-CALIBRATOR OBJECT ---------------------------------------------------------------
class BetaCalibrator:
    """Fit p ↦ σ(a·log p + b·log (1-p) + c) via unpenalized logistic regression."""
    def __init__(self, eps: float = 1e-6, lmbd: float = 0):
        
        self.eps = eps
        if lmbd == 0:
            self.lr  = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=10_000
            )
        
        else:
            C_val = 1/lmbd
            self.lr  = LogisticRegression(
                penalty="l2",
                solver='lbfgs',
                C = C_val,
                max_iter=10_000
            )
        
    def fit(self, p: np.ndarray, y: np.ndarray):
        p = np.clip(p, self.eps, 1 - self.eps)
        X = np.vstack([np.log(p), np.log(1 - p)]).T
        self.lr.fit(X, y)
        return self
    
    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1 - self.eps)
        X = np.vstack([np.log(p), np.log(1 - p)]).T
        return self.lr.predict_proba(X)[:, 1]
    
    @property # @property decorator turns a method into an attribute-like object
    def alpha_(self) -> float:
        return self.lr.coef_[0,0]
    
    @property
    def beta_(self) -> float:
        return self.lr.coef_[0,1]
    
    @property
    def gamma_(self) -> float:
        return self.lr.intercept_[0]

#: THIS IS A FUNCTION TO CALCULATE THE 95%CI BY USING A T-STUDENT DISTRIBUTION ---------------------
def stats(vals, ci, series=False):
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
    if series:
        return pd.Series({
        "mean": mean,
        "lo":   lo,
        "hi":   hi,
        "sd":   sd,
        "sem":  sem
    })
    else:
        return mean, lo, hi, sd, sem

#: CALIBRATION USING THREE METHODS: PLATT, BETA-CALIBRATION AND ISOTONIC CALIBRATION ---------------
raw_risk = pd.read_csv(path_OOF_predictions).drop(columns=["Unnamed: 0"])
y_true  = raw_risk["y_true"].values
y_score = raw_risk["oof_predictions"].values
fold_id = raw_risk["fold"].values
strategy = "uniform"         
bins     = 10

#: AUXILIAR FUNCTION ---------------------------
def _ece_of(preds, name):
    df_tmp = pd.DataFrame(
        np.column_stack([y_true, preds]),
        columns=["y_true", name]
    )
    summ, ece = calibration_performance(
        df_tmp,
        bins=bins,
        strategy=strategy,
        column_name=name,
        display_=False
    )
    return summ, ece

summ_raw, EcE_raw = _ece_of(y_score,"UncalibratedScores")

#: 1)  BETA CALIBRATION  ------------------------
# This is a parametric method similar to Platt scaling, with three tunable parameters instead of two.
y_beta_cv = np.zeros_like(y_score, dtype=float)  
for fold in np.unique(fold_id):
    tr = fold_id != fold
    va = fold_id == fold

    beta_tmp = BetaCalibrator(eps=1e-6)
    beta_tmp.fit(y_score[tr], y_true[tr])
    y_beta_cv[va] = beta_tmp.predict(y_score[va])

summ_beta, EcE_beta = _ece_of(y_beta_cv, "Beta_tmp")
print(f"[β]   cross-validated   EcE = {EcE_beta:.4f}")
# final calibrator for deployment / inference
beta_final = BetaCalibrator(eps=1e-6)
beta_final.fit(y_score, y_true)

 
# -------

#: 2)  PLATT SCALING -----------------------------
# Here I will Introduce a regularization hyperparameter C into Platt scaling and evaluate its
# effect on calibration across the out-of-fold predictions. Candidate values are specified in C_grid.
best_EcE = np.inf
best_C   = None
y_platt_cv = np.zeros_like(y_score, dtype=float) 

for C_val in np.append(np.logspace(-2, 1, 100), 1/12):
    y_temp = np.zeros_like(y_score, dtype=float)

    for fold in np.unique(fold_id):
        tr = fold_id != fold
        va = fold_id == fold

        lr = LogisticRegression(solver="saga", penalty="l2", C=C_val)
        lr.fit(y_score[tr].reshape(-1, 1), y_true[tr])
        y_temp[va] = lr.predict_proba(y_score[va].reshape(-1, 1))[:, 1]

    _,EcE = _ece_of(y_temp, "Platt_tmp")

    if EcE < best_EcE:
        best_EcE   = EcE
        best_C     = C_val
        y_platt_cv = y_temp.copy()          

print(f"[Platt] best C = {best_C:.4g}  |  CV EcE = {best_EcE:.4f}")
lr_final = LogisticRegression(solver="saga", penalty="l2", C=best_C)
# final isotonic model for deployment
lr_final.fit(y_score.reshape(-1, 1), y_true)


#: 3)  ISOTONIC CALIBRATION -----------------------------
y_iso_cv = np.zeros_like(y_score, dtype=float)
for fold in np.unique(fold_id):
    tr = fold_id != fold
    va = fold_id == fold

    iso_tmp = IsotonicRegression(out_of_bounds="clip")
    iso_tmp.fit(y_score[tr], y_true[tr])
    y_iso_cv[va] = iso_tmp.predict(y_score[va])

summ_iso, EcE_iso = _ece_of(y_iso_cv, "Iso_tmp")
print(f"[Iso] cross-validated EcE = {EcE_iso:.4f}")
# final isotonic model for deployment
iso_final = IsotonicRegression(out_of_bounds="clip")
iso_final.fit(y_score, y_true)

#: 4. SUMMARY --------------------------------------------
print("\n=== Cross-validated Expected Calibration Error (lower is better) ===")
print(f"  Beta  : {EcE_beta:.4f}")
print(f"  Platt : {best_EcE:.4f}")
print(f"  Iso   : {EcE_iso:.4f}")

# %% #* ============================== CALIBRATION COMPARISON ======================================
# The comparison (Plots/ECE metrics) is done on OOF calibrated probabilities so it is fair to 
# compare the results. 
bins= 10
#: SUMMARY OF TRUE Y LABELS AND THE DIFFERENT CALIBRATED SIGNALS -----------
df_results = pd.DataFrame(
                np.column_stack([y_true, y_score, y_iso_cv, y_platt_cv, y_beta_cv]), 
                columns=["y_true", "XGBoost Output", "Isotonic", "Platt", "Beta-Calibration"]
            )

methods = {
    "Platt":    y_platt_cv, #Q we evaluate the results on unseen predictions
    #"Isotonic": y_iso_cv,
    "Beta":     y_beta_cv,
}

from sklearn.metrics import log_loss
logloss_Platt = log_loss(y_true, y_platt_cv)
logloss_Beta  = log_loss(y_true, y_beta_cv)
logloss_Iso   = log_loss(y_true, y_iso_cv)
logloss_original   = log_loss(y_true, y_score)
loglosses = {
    "Platt":    logloss_Platt, #Q we evaluate the results on unseen predictions
    #"Isotonic": y_iso_cv,
    "Beta":     logloss_Beta,
}

print(f"Log-loss Platt   : {logloss_Platt:.6f}")
print(f"Log-loss Beta    : {logloss_Beta:.6f}")
print(f"Log-loss Isotonic: {logloss_Iso:.6f}")
print(f"Log-loss XGBoost Raw Scores: (OOF) {logloss_Iso:.6f} ")


with open(TXT_BINSINFORMATION, 'w'):
    pass

#: COMPARISON BASED ON CALIBRATION PLOTS ----------------
fig, axes = StandardFig(        
                width   = "double",  
                height  = 0.5,                
                subplots = [1, 2],
            )

for ax, (name, preds) in zip(axes, methods.items()):
    frac_raw, mean_raw = calibration_curve(
        y_true, y_score,   n_bins=bins, strategy=strategy
    )
    frac_cal, mean_cal = calibration_curve(
        y_true, preds,     n_bins=bins, strategy=strategy
    )

    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = np.histogram(preds, bins=edges)[0]
    total = np.sum(counts >= 1)
    print(total)

    mean_cal = mean_cal[:total]
    frac_cal = frac_cal[:total]

    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.plot(mean_raw, frac_raw, marker="o", label="Uncalibrated", color = "black")
    ax.plot(mean_cal, frac_cal, marker="o", label=name, color = "green")

    #: ECE
    df_cal = pd.DataFrame(
        np.column_stack([y_true, preds]),
        columns=["y_true", name]
    )
    summ, ece = calibration_performance(
        df_cal,
        bins=bins,
        strategy=strategy,
        column_name=name,
        display_=True
    )
    
    # Storage of the summary:
    with open(TXT_BINSINFORMATION, 'a') as f:
        f.write("="*20 + "\n")
        f.write(f"Summary of bins = {bins} for {name} scaling")
        f.write(summ.to_string())
        f.write(f"\n\n")
    
    ax.text(
        0.05, 0.95,
        f"ECE {name}: {ece:.2e}\nLog-loss {name}: {loglosses[name]:.2e}\nLog-loss Uncalibrated: {logloss_original:.2e} ",
        transform=ax.transAxes,
        va="top"
    )

    ax.set_title(f"Uncalibrated Probabilities vs {name} Calibration \n (bins = {bins})(OOF)")
    ax.set_xlabel("Mean predicted probability")
    if name == "Platt":
        ax.set_ylabel("Observed frequency")
    ax.legend(loc="best")

fig.tight_layout()
fig.savefig(os.path.join(graphics_dir2, f"TrainingSetOOF_CalScores_k{bins}_comparison.pdf") , dpi=300)
plt.show() if SHOW_PLOTS else None


# %% #* ================ FINAL MODEL + CALIBRATOR OBJECT STORAGE AS A JAVA OBJECT ==================

if RE_COMPILE: # This should be triggered if some change was done down the workflow
    if os.path.exists(path_out_model):
        if os.path.isdir(path_out_model):
            shutil.rmtree(path_out_model)
        else:
            os.remove(path_out_model)
    os.makedirs(path_out_model, exist_ok=True)

    #:BEST MODEL STORAGE -----------------------------------------------------
    mojo_zip = os.path.join(path_out_model, "best_model.zip")
    best_model.save_mojo(mojo_zip, force=True)   

    #: THIS SECTION JUST MOVES THE BEST_MODEL ZIP file ONE FOLDER UP
    if os.path.isdir(mojo_zip):
        # Mojo Zip inside
        candidates = [f for f in os.listdir(mojo_zip) if f.endswith(".zip")]
        if candidates:
            real_name = candidates[0]
            src       = os.path.join(mojo_zip, real_name)
            tmp_zip   = os.path.join(path_out_model, "best_model_temp.zip")
            final_zip = os.path.join(path_out_model, "best_model.zip")
            # -----------------------
            shutil.move(src, tmp_zip)
            shutil.rmtree(mojo_zip, ignore_errors=True)
            os.replace(tmp_zip, final_zip)


    #: JAVA FILE THAT AUTOMATICALLY EMBEDS CALIBRATION ALONGSIDE THE XGBOOST MODEL
    fields_array = ", ".join(f'"{c}"' for c in TARGET_FEATURES)
    java_code = f"""
    /*  Auto-generated helper that wraps an H2O MOJO and applies β-calibration.
        Compile:
            javac -cp .:h2o-genmodel.jar:h2o-genmodel-ext-xgboost.jar CalibratedMojo.java
        Run:-
            java  -cp .:h2o-genmodel.jar:h2o-genmodel-ext-xgboost.jar CalibratedMojo
    */

    import hex.genmodel.MojoModel;
    import hex.genmodel.easy.EasyPredictModelWrapper;
    import hex.genmodel.easy.RowData;
    import hex.genmodel.easy.prediction.BinomialModelPrediction;

    public class CalibratedMojo {{

        /* β-calibration constants (filled from Python) */
        private static final double ALPHA = {beta_final.alpha_:.17g};
        private static final double BETA  = {beta_final.beta_:.17g};
        private static final double GAMMA = {beta_final.gamma_:.17g};
        private static final double EPS   = {beta_final.eps:.1e};

        /* Ordered list of input columns expected by the MOJO */
        private static final String[] FIELDS = new String[] {{ {fields_array} }};

        private static final String  MOJO_PATH = "{mojo_zip}";
        private static final EasyPredictModelWrapper MODEL;

        /* One-time MOJO load */
        static {{
            try {{
                MojoModel mojo = MojoModel.load(MOJO_PATH);
                MODEL = new EasyPredictModelWrapper(mojo);
            }} catch (Exception e) {{
                throw new RuntimeException("Could not load " + MOJO_PATH, e);
            }}
        }}

        /* ==========  PUBLIC API  ================================================= */

        /** Uncalibrated probability P(class=1). */
        public static double predictRaw(double[] row) throws Exception {{
            BinomialModelPrediction pr = scoreRow(row);
            return pr.classProbabilities[1];
        }}

        /** β-calibrated probability P(class=1). */
        public static double predictCalibrated(double[] row) throws Exception {{
            double p = predictRaw(row);
            return betaTransform(p);
        }}

        /* ==========  INTERNAL HELPERS  ========================================== */

        /** Core MOJO scoring (shared by both public methods). */
        private static BinomialModelPrediction scoreRow(double[] row) throws Exception {{
            if (row.length != FIELDS.length)
                throw new IllegalArgumentException(
                    "Expected " + FIELDS.length + " features, got " + row.length);

            RowData rd = new RowData();
            for (int i = 0; i < FIELDS.length; i++)
                rd.put(FIELDS[i], Double.toString(row[i]));

            return MODEL.predictBinomial(rd);
        }}

        /** β-calibration transform σ(a·log p + b·log(1−p) + c). */
        private static double betaTransform(double p) {{
            p = Math.max(EPS, Math.min(1.0 - EPS, p));
            double logit = ALPHA * Math.log(p) + BETA * Math.log(1.0 - p) + GAMMA;
            return 1.0 / (1.0 + Math.exp(-logit));
        }}

        /* ==========  SMOKE-TEST  ================================================= */

        public static void main(String[] args) throws Exception {{
            double[] dummy = new double[FIELDS.length];          // all-zero vector
            System.out.printf("Raw P1  : %.6f%n", predictRaw(dummy));
            System.out.printf("β-cal P1: %.6f%n", predictCalibrated(dummy));
        }}
    }}
    """
    
    os.makedirs(path_out_model, exist_ok=True)
    with open(os.path.join(path_out_model, "CalibratedMojo.java"), "w") as f:
        f.write(textwrap.dedent(java_code))
    print("✅ Saved best_model.zip and updated CalibratedMojo.java")

    #: DEPENDENCIES LOADING ------------------------------------------------------
    # WE NEED SOME JAVA DEPENDENCIES TO RUN THE MODEL, HERE I GOT THEM FROM MAVEN.ORG
    h2o_version = h2o.__version__
    JAR_DEPS = {
        "h2o-genmodel.jar":
        f"https://repo1.maven.org/maven2/ai/h2o/h2o-genmodel/{h2o_version}/h2o-genmodel-{h2o_version}.jar",
        "h2o-algos.jar":
        f"https://repo1.maven.org/maven2/ai/h2o/h2o-algos/{h2o_version}/h2o-algos-{h2o_version}-all.jar",
        "h2o-genmodel-ext-xgboost.jar":
        f"https://repo1.maven.org/maven2/ai/h2o/h2o-genmodel-ext-xgboost/{h2o_version}/h2o-genmodel-ext-xgboost-{h2o_version}.jar",
        "xgboost-predictor.jar":
        f"https://repo1.maven.org/maven2/ai/h2o/xgboost-predictor/0.3.20/xgboost-predictor-0.3.20.jar",
    }
    for fname, url in JAR_DEPS.items():
        dst = os.path.join(path_out_model, fname)
        if not os.path.exists(dst):
            print(f"⟳ Downloading {fname} …")
            urllib.request.urlretrieve(url, dst)

    #: THIS IS THE COMPILATION COMMAND FOR THE JAVA WRAPPER:
    cp_jars = ":".join([fname for fname in JAR_DEPS] + ["./"])
    print("⟳ Compiling CalibratedMojo.java")
    subprocess.run(
        ["javac", "-cp", cp_jars, "CalibratedMojo.java"],
        cwd=path_out_model,
        check=True
    )

    #:BRIEF TEST ------------------------------------------
    print("⟳ Running smoke-test")
    proc = subprocess.run(
        ["java", "-cp", cp_jars, "CalibratedMojo"],
        cwd=path_out_model,
        capture_output=True,
        text=True
    )
    print(proc.stdout)
    
# %% #* ============================ HISTOGRAMS OF OOF SCORES ======================================

#: SCORES ------------------- --------------------------------------------
y_true  = df_results["y_true"].values
y_score = df_results["XGBoost Output"].values
y_score_calibrated = df_results["Beta-Calibration"].values

#: RAW PROBABILITIES HISTOGRAM FOR OOF PREDICTIONS -----------------------
fig, ax1 = StandardFig()
pos_oof_scores =  y_score[y_true==1]
neg_oof_scores =  y_score[y_true==0]

bins = 20
color = "black"
neg_patches = ax1.hist(
    neg_oof_scores,
    bins=bins,
    color=color,
    alpha=0.3,
    label='Negative'
)
ax1.set_ylabel('Count (Negative)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = "green"
pos_patches = ax2.hist(
    pos_oof_scores,
    bins=bins,
    color=color,
    alpha=0.2,
    label='Positive'
)
ax2.set_ylabel('Count (Positive)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.set_xlabel('OOF score')
ax1.set_title('Histogram of OOF Scores by Class (Uncalibrated)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
fig.tight_layout()
fig.savefig(path_noncal_score_dis, bbox_inches='tight',pad_inches=0.1) 

#: CALIBRATED PROBABILITIES HISTOGRAM FOR OOF PREDICTIONS -----------------------
fig, ax1 = StandardFig()
pos_oof_scores =  y_score_calibrated[y_true==1]
neg_oof_scores =  y_score_calibrated[y_true==0]
color = "black"
neg_patches = ax1.hist(
    neg_oof_scores,
    bins=bins,
    color=color,
    alpha=0.3,
    label='Negative'
)
ax1.set_ylabel('Count (Negative)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = "green"
pos_patches = ax2.hist(
    pos_oof_scores,
    bins=bins,
    color=color,
    alpha=0.2,
    label='Positive'
)
ax2.set_ylabel('Count (Positive)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.set_xlabel('OOF score')
ax1.set_title('Histogram of OOF Scores by Class (Calibrated)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
fig.tight_layout()
fig.savefig(path_cal_score_dis, bbox_inches='tight',pad_inches=0.1) 

# %% #* =================================== LEARNING CURVE =========================================

#: HERE I WILL CHANGE THE FORMAT OF THE BUILT-IN LEARNING CURVE WITHIN THE BEST MODEL OBJECT, SO IT FITS
#: THE STANDARD FORMAT PLOT FORMAT OF THE PROJECT
fig_dummy, ax_dummy = StandardFig(
    width   = "single",   
    height  = 0.85,        
    base_pt = 8.5,
    scale   = 1.00
)
plt.close(fig_dummy)

mm_per_in = 25.4
width_mm  = 90  
width_in  = width_mm / mm_per_in
height_in = width_in * 0.85

#: BUILT-IN CURVE --------------------------------------------------
res = best_model.learning_curve_plot(
    metric        ="AUTO",
    cv_ribbon     =True,
    cv_lines      =False,
    figsize       =(width_in, height_in),
    colormap      ="viridis",
    save_plot_path=None
)
fig = res.figure()
ax  = fig.axes[0]

#: CURVE PROCESSING -----------------------------------------------
for art in list(ax.get_lines()) + list(ax.collections):
    if art.get_label() == "Training":   
        # This is a signal with the log-loss on all the training data but im focused on CV models
        # instead
        art.remove()
        
for line in ax.get_lines():
    lbl = line.get_label()
    if lbl == "Training (CV Models)":
        line.set_color("black")
        line.set_label("Training Average (k-1) Folds")
    elif lbl == "Cross-validation":
        line.set_color("green")
        line.set_label("Cross-validation Average (k) Folds")
    x, y = line.get_data()
    if len(x) > 3: 
        # The first two points excessively expand the y-axis range,
        # making it difficult to see the differences between the data series at the highest tree counts.
        line.set_data(x[2:-1], y[2:-1])

# This just changes the color of the 95%CI lines (we modified the color of the averages above)
train_ci_rgba = (0, 0, 0, 0.20)       
cv_ci_rgba    = (0, 0.6, 0, 0.20)    
cols = ax.collections
if len(cols) >= 1:
    cols[0].set_facecolor(train_ci_rgba); cols[0].set_edgecolor("none")
if len(cols) >= 2:
    cols[1].set_facecolor(cv_ci_rgba);    cols[1].set_edgecolor("none")

# This just changes the style of the vertical line where the solver stopped training. 
selector_line = None
for ln in ax.get_lines():
    if ln.get_label() == "Selected\nnumber_of_trees":
        selector_line = ln
        break
selector_line.set_color("darkred")
selector_line.set_linestyle("dashed")
selector_line.set_linewidth(1.2)
selector_line.set_alpha(0.8)
selector_line.set_label("Early Stopping Criterium")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper right", frameon=True)

# Final aesthetics over here
ax.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel("Number of Trees")
ax.set_ylabel("logloss")
ax.set_title("Learning Curve vs Number of Trees")
ax.relim(); ax.autoscale_view()

fig.tight_layout(pad=1.0)
fig.savefig(path_learning_curve, bbox_inches='tight',pad_inches=0.1) 
plt.show() if SHOW_PLOTS else None

# %% #* ======================== OTHER PERFORMANCE METRICS ON THE TRAINING SET ====================
from scipy import stats
h2o.init()
model_paths = glob.glob("Models_storage/DynamicPredictionModel/XGBoostEstimator/Top1Model_and_KModels/bin/Kfold_model*/XGBoost_model_*_cv_*")
loglosses = []
aucs      = []
auprcs    = []
for path in model_paths:
    mdl = h2o.load_model(path)
    # get performance
    perf = mdl.model_performance(valid=True)
    loglosses.append(perf.logloss())
    aucs     .append(perf.auc())
    auprcs   .append(perf.aucpr())

# 4. Compute means and 95% CIs
def mean_ci(arr, alpha=0.05):
    a = np.array(arr, dtype=float)
    n = len(a)
    m = a.mean()
    s = a.std(ddof=1)
    t_val = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_val * s / np.sqrt(n)
    return m, (m - margin, m + margin)

mean_ll, (ll_lo, ll_hi)   = mean_ci(loglosses)
mean_auc, (auc_lo, auc_hi)= mean_ci(aucs)
mean_prc, (prc_lo, prc_hi)= mean_ci(auprcs)

print("Training Information: \n")
print(f"Log-loss : {mean_ll:.4f} (95% CI [{ll_lo:.4f}, {ll_hi:.4f}])")
print(f"AUC      : {mean_auc:.4f} (95% CI [{auc_lo:.4f}, {auc_hi:.4f}])")
print(f"AUPRC    : {mean_prc:.4f} (95% CI [{prc_lo:.4f}, {prc_hi:.4f}])")

# -----------------------------------
df_names                = pd.read_excel(PATH_NAMES)
df_names["has_alias"]   = df_names["name_polished"].notna()
df_names["new_name"]    = np.where(
                            df_names["has_alias"], 
                            df_names["name_polished"], 
                            df_names["Feature"]
                        )
df_names                = df_names.set_index("Feature")



rank_lists = []
for p in model_paths:
    m = h2o.load_model(p)
    imp = (m.varimp(use_pandas=True)
             .sort_values("relative_importance", ascending=False)
             .reset_index(drop=True))
    rank_lists.append({row["variable"]: i + 1 for i, row in imp.iterrows()})


all_feats = sorted({f for d in rank_lists for f in d})
rank_df = pd.DataFrame(
    [{f: d.get(f, np.nan) for f in all_feats} for d in rank_lists]
)

avg_ranks = rank_df.mean(axis=0, skipna=True)
ordered_feats = avg_ranks.sort_values().index.tolist()
rank_df = rank_df[ordered_feats]          # reorder columns


fig, ax = StandardFig(width = "double")

im = ax.imshow(rank_df.T, cmap="YlGnBu", aspect="auto",
               vmin=1, vmax=len(all_feats))  # yellow → blue

# Annotate each rectangle with its rank
for i in range(rank_df.shape[0]):          # rows (folds)
    for j in range(rank_df.shape[1]):      # cols (features)
        val = rank_df.iloc[i, j]
        if not np.isnan(val):
            ax.text(i, j, int(val), ha="center", va="center", fontsize=7)

# Axis labelling
ax.set_xticks(np.arange(rank_df.shape[0]))
ax.set_xticklabels([f"Fold {k+1}" for k in range(rank_df.shape[0])],
                   rotation=90)

ax.set_yticks(np.arange(len(ordered_feats)))
ax.set_yticklabels(ordered_feats)
ax.set_xlabel("CV Fold")
ax.set_ylabel("Feature (sorted by average rank)")
ax.set_title("Heat-map of Feature-Importance Ranks Across Folds")

# Color‑bar
cbar = fig.colorbar(im, ax=ax, orientation="vertical")
cbar.set_label("Rank position (1 = most important)")

plt.tight_layout()
plt.show()

# %% #* ======================== CALIBRATION SUMMARY AS TEXT FILE ==================================
beta_params = dict(
    alpha = beta_final.alpha_,
    beta  = beta_final.beta_,
    gamma = beta_final.gamma_,
)
beta_params = {k: f"{v:.7g}" for k, v in beta_params.items()}

def banner(text: str, width: int = 100, char: str = "="):
    pad = max(0, width - len(text) - 2)
    left  = char * (pad // 2)
    right = char * (pad - pad // 2)
    return f"{left} {text} {right}\n"

with open(path_cal_txt, "w", encoding="utf-8") as f:

    f.write(banner("CALIBRATION SUMMARY"))
    f.write("\n[Bins information — calibrated scores]\n\n")
    f.write(summ_beta.to_string(index=True, justify="right"))
    f.write("\n")

    f.write("\n[β-calibration parameters]\n\n")
    for k, v in beta_params.items():
        f.write(f"{k:>6s} : {v}\n")

    f.write("\n[Bins information — raw (uncalibrated) scores]\n\n")
    f.write(summ_raw.to_string(index=True, justify="right"))
    f.write("\n")

    f.write("\n[Calibration Quality]\n\n")
    f.write(f"EcE_raw  : {EcE_raw:.4f}\n")
    f.write(f"EcE_beta : {EcE_beta:.4f}\n")
    f.write(banner("", char="="))  
