
# %% #*=============================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import ast, re
import os
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import expit
from matplotlib.patches import Patch
# -------------------------------
DIR_CURRENT = r'/ihome/rparker/mdc147/PhD_Project';  os.chdir(DIR_CURRENT)
sys.path.append(os.path.join(DIR_CURRENT, 'Auxiliar_codes'))
# -------------------------------
from characterization import StandardFig
from classes_repository import PathManager
from machine_learning_auxiliar import  FeaturesManagerV2
pm = PathManager().paths
fm = FeaturesManagerV2()

# Paths -------------------------
PATH_RESULTS        = "Models_storage/DynamicPredictionModel/ElasticNET/ElasticNET_results.csv"
PATH_SPLIT          = os.path.join(DIR_CURRENT, "Models_storage/DynamicPredictionModel/StratifiedDivision/stratified_division_summary.pkl")
DIR_PROCESSED       = "Data/Processed_datasets/Summary/Merged_M24H15"
PATH_DYN            = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Signals.pkl")
PATH_SSUM           = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Summary.pkl") 
PATH_NAMES          = "Other/Names_mapping.xlsx"
# Output -----------------------
graphics_dir        = "Documents/Multimedia/ElasticNET"
model_path          = "Models_storage/DynamicPredictionModel/ElasticNET/ElasticNET_Model.pkl"
path_scaler         = "Models_storage/DynamicPredictionModel/ElasticNET/ElasticNET_data_scaler"
scaler_params_path  = "Models_storage/DynamicPredictionModel/ElasticNET/ElasticNET_InitialFeaturesAndScalerParams.csv"
path_results        = "Models_storage/DynamicPredictionModel/ElasticNET/ElasticNET_results.csv"
path_txt_file       = "Models_storage/DynamicPredictionModel/ElasticNET/ElasticNET_summary.txt"
path_coefficients   = "Models_storage/DynamicPredictionModel/ElasticNET/ElasticNET_Coefficients.csv"

# PARAMETERES ------------------
SHOW_PLOTS          = True   
AUPRC_BASED         = False  

# Data --------------------------
df_res          = pd.read_csv(PATH_RESULTS)
df_dyn          = pd.read_pickle(PATH_DYN)
df_summary      = pd.read_pickle(PATH_SSUM)
sessions_df     = pd.read_pickle(PATH_SPLIT)
df_names        = pd.read_excel(PATH_NAMES)

#: Other Processing steps
df_names["has_alias"]   = df_names["name_polished"].notna()
df_names["new_name"]    = np.where(
                            df_names["has_alias"], 
                            df_names["name_polished"], 
                            df_names["Feature"]
                        )
df_names                = df_names.set_index("Feature")


#:FEATURES LOADING (AVAILABLE FEATURES, INTERNAL CLASSIFICATION) -----
TARGET_FEATURES0    = fm.identify_predictors(df_dyn.columns.tolist())
FEATURES_ORGANIZED  = fm.identify_predictors(df_dyn.columns.tolist(), by_family=True)
L6S_FEATURES        = FEATURES_ORGANIZED["L6S"]
LIDHO_FEATURES      = FEATURES_ORGANIZED["LIDHO"]
W_FEATURES          = FEATURES_ORGANIZED["W"]
CW_FEATURES         = FEATURES_ORGANIZED["CW"]
LS_FEATURES         = FEATURES_ORGANIZED["LS"]
HOT_ENCODED_VAR     = fm.hot_encoded


# %% #* ============================ MIN SET OF FEATURES =========================================== 
# ------- Results Processing and Minimum Set of features -----
cleaned_df = (
    df_res
    .sort_values(
        by=['alpha','n_key_nonzero','logloss'],
        ascending=[True, True, True]  
    )
    .drop_duplicates(
        subset=['alpha','n_key_nonzero'],
        keep='first'                     # within each (α, len(key_Features)) group, lowest log-loss comes first
    )
    .reset_index(drop=True)
)
# The following code chunk just organizes the columns and create a new column
# "var_addition" which specifies at each step which variable was added
col = cleaned_df.pop('n_key_nonzero')
idx = cleaned_df.columns.get_loc('key_names')
cleaned_df.insert(loc=idx, column='n_key_nonzero', value=col)
def _to_list(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x
cleaned_df['key_names'] = cleaned_df['key_names'].apply(_to_list)
# Each variable addition vector whould be performed independently
sto = []
for alpha in cleaned_df['alpha'].unique():
    subset = cleaned_df[cleaned_df['alpha'] == alpha].copy()
    prev = []                  
    adds = []
    for curr in subset['key_names']:
        new_vars = [v for v in curr if v not in prev]
        adds.append(new_vars)
        prev = curr
    subset['var_addition'] = adds
    col = subset.pop('var_addition')
    pos = subset.columns.get_loc('key_names') + 1
    subset.insert(pos, 'var_addition', col)
    sto.append(subset)
cleaned_df = pd.concat(sto, ignore_index=True)
# Maximum AUPRC index and lowest AUPRC Val
# I ran the script once and it looks like pure 
# lasso regression had better auprc performance
# Next stage of the analysis wll be made on alpha =1.
temp = cleaned_df[cleaned_df['alpha']==1].copy().reset_index(drop=True)

if AUPRC_BASED:
    best_idx   = temp['auprc'].idxmax()
    best_row   = temp.loc[best_idx]
    max_auprc  = best_row['auprc']
    sem        = best_row['auprc_sem']
    threshold  = max_auprc - sem
    # The following is the smallest set of features that reaches the AUPRC
    # lower boundary defined before. I will set this using the data series
    # from alpha = 1
    within_sem      = temp[temp['auprc'] >= threshold]
    min_idx         = within_sem['n_key_nonzero'].idxmin()
    min_vars        = within_sem.loc[min_idx, 'n_key_nonzero']
    min_var_info    = within_sem.loc[min_idx, 'key_coeff_metrics']
    # The following information is necessary to build the regressor from
    # scratch and get the performance on the external validation set.
    min_var_names   = within_sem.loc[min_idx, 'key_names']
    intercept       = within_sem.loc[min_idx, 'intercept']
    intercept_lo    = within_sem.loc[min_idx, 'intercept_lo']
    intercept_hi    = within_sem.loc[min_idx, 'intercept_hi']
    intercept_sd    = within_sem.loc[min_idx, 'intercept_sd']
    intercept_sem   = within_sem.loc[min_idx, 'intercept_sem']
    # Some processing stuff since I loaded the results from a csv file
    cleaned = re.sub(r"np\.float64\(\s*([^)]+?)\s*\)", r"\1", min_var_info)
    min_var_info = ast.literal_eval(cleaned)

    coef_means = np.array([
        min_var_info[f][f"{f}_mean"]
        for f in min_var_names
    ], dtype=float)

else:

    best_idx        = temp["logloss"].idxmin()
    best_row        = temp.loc[best_idx]
    min_logloss     = best_row['logloss']
    sem             = best_row['logloss_sem']
    threshold       = min_logloss + sem
    # The following is the smallest set of features that lead to a 
    # Logloss <= minlogloss + sem. I will set this using the data series
    # from alpha = 1
    within_sem      = temp[temp['logloss'] <= threshold]
    max_idx         = within_sem['n_key_nonzero'].idxmin()  # minimum features within the SEM range
    min_vars        = within_sem.loc[max_idx, 'n_key_nonzero']
    min_var_info    = within_sem.loc[max_idx, 'key_coeff_metrics']
    # The following information is necessary to build the regressor from
    # scratch and get the performance on the external validation set.
    
    #: AUPRC METRICS
    auprc     = within_sem.loc[max_idx, 'auprc']
    lo_auprc  = within_sem.loc[max_idx, 'auprc_lo']
    hi_auprc  = within_sem.loc[max_idx, 'auprc_hi']
    
    #: AUROC METRICS
    auroc     = within_sem.loc[max_idx, 'auroc']
    lo_auroc  = within_sem.loc[max_idx, 'auroc_lo']
    hi_auroc  = within_sem.loc[max_idx, 'auroc_hi']  
    
    #: LOGLOSS METRICS
    logloss         = within_sem.loc[max_idx, 'logloss']
    lo_logloss      = within_sem.loc[max_idx,'logloss_lo']
    hi_logloss      = within_sem.loc[max_idx,'logloss_hi']
    
    min_var_names   = within_sem.loc[max_idx, 'key_names']
    intercept       = within_sem.loc[max_idx, 'intercept']
    intercept_lo    = within_sem.loc[max_idx, 'intercept_lo']
    intercept_hi    = within_sem.loc[max_idx, 'intercept_hi']
    intercept_sd    = within_sem.loc[max_idx, 'intercept_sd']
    intercept_sem   = within_sem.loc[max_idx, 'intercept_sem']
    # Some processing stuff since I loaded the results from a csv file
    cleaned = re.sub(r"np\.float64\(\s*([^)]+?)\s*\)", r"\1", min_var_info)
    min_var_info = ast.literal_eval(cleaned)
    coef_means = np.array([
        min_var_info[f][f"{f}_mean"]
        for f in min_var_names
    ], dtype=float)
    

# %% #*============== TEXT SUMMARY FILE AND CLASSIFIER ON COEFFICIENTS AVERAGE =====================
# ------------------------- final model ---------------------------
class FixedCoefLogistic(BaseEstimator, ClassifierMixin):
    def __init__(self, coef, intercept=0.0):
        self.coef_      = np.atleast_2d(coef)
        self.intercept_ = np.array([intercept])
    def predict_proba(self, X):
        scores = expit(X @ self.coef_.T + self.intercept_)
        return np.hstack([1 - scores, scores])
    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)
final_model = FixedCoefLogistic(coef=coef_means, intercept=intercept)
joblib.dump(final_model, model_path)

#:TXT FILE
# I will save the highlights in a text file.
# df with the informaition about the coefficients 
records = []
for feature, metrics in min_var_info.items():
    records.append({
        "feature": feature,
        "mean":    metrics[f"{feature}_mean"],
        "low":     metrics[f"{feature}_low"],
        "hi":      metrics[f"{feature}_hi"],
        "sd":      metrics[f"{feature}_sd"],
        "sem":     metrics[f"{feature}_sem"],
    })
df = (
    pd.DataFrame.from_records(records)
      .assign(abs_mean=lambda d: d['mean'].abs()) # Here I added a new column 'abs_mean' with the absolute values
      .sort_values('abs_mean', ascending=False)
      .drop(columns='abs_mean')
      .reset_index(drop=True)
)
df.to_csv(path_coefficients, index = False)

def write_and_print(f, text):
    f.write(text + "\n")
    print(text)

with open(path_txt_file, "w") as f:
    write_and_print(f, "="*70)

    if AUPRC_BASED:
        write_and_print(f, f"Mode: AUPRC-based selection")
        write_and_print(f, f"Max AUPRC              = {max_auprc:.4f} ± SEM {sem:.4f}")
        write_and_print(f, f"AUPRC threshold (max_AUPRC - SEM) = {threshold:.4f}")
        write_and_print(f, f"Selected # features    = {min_vars}")
    else:
        write_and_print(f, f"Mode: Log-Loss-based selection")
        write_and_print(f, f"Min Log-Loss           = {min_logloss:.4f} ± SEM {sem:.4f}")
        write_and_print(f, f"Log-Loss threshold (min_logloss + SEM) = {threshold:.4f}")
        write_and_print(f, f"Selected # features    = {min_vars}")
        write_and_print(f, "")
        write_and_print(f, f"Average performance metrics of the chosen model with {min_vars} vars:")
        write_and_print(f, f"  LogLoss = {logloss:.4f} (95% CI [{lo_logloss:.4f}, {hi_logloss:.4f}])")
        write_and_print(f, f"  AUPRC = {auprc:.4f} (95% CI [{lo_auprc:.4f}, {hi_auprc:.4f}])")
        write_and_print(f, f"  AUROC = {auroc:.4f} (95% CI [{lo_auroc:.4f}, {hi_auroc:.4f}])")

    write_and_print(f, "")
    write_and_print(f, f"Intercept           = {intercept:.6f} ± {intercept_sem:.6f}")
    write_and_print(f, f"Intercept 95% CI    = [{intercept_lo:.6f}, {intercept_hi:.6f}]")
    write_and_print(f, f"Coefficients stored at: {path_coefficients}")
    write_and_print(f, "")
    write_and_print(f, "Non-zero coefficients:")
    write_and_print(f, "-"*11)

    # dump each variable’s stats
    for _, row in df.iterrows():
        feature0 = row["feature"]
        mean     = row["mean"]
        low_th   = row["low"]
        hi_th    = row["hi"]
        sd       = row["sd"]
        sem_var  = row["sem"]

        write_and_print(f, f"Name in Dataframe:    {feature0}")
        write_and_print(f, f"Average:              {mean:.4f}")
        write_and_print(f, f"95% CI:               [{low_th:.4f}, {hi_th:.4f}]")
        write_and_print(f, f"sd={sd:.4f}, sem={sem_var:.4f}")
        write_and_print(f, "-"*11)

    write_and_print(f, "="*70)

# %% #*=============================================================================================
# --------------------- PLOTS --------------------------
# This is the plot with the metrics names vs coefficients
"""y_pos = list(range(len(df)))
fig, ax = StandardFig(
    width   = "double",   # "single" (90 mm) | "double" (183 mm) | numeric (mm)
    height  = len(df)*0.05,
    base_pt = 8,
    scale   = 1.15
)

# 1) draw shading regions BEFORE the errorbars
#    grab current x‐limits (you may want to set them manually afterward)
ax.set_xlim(df['low'].min() * 1.1, df['hi'].max() * 1.1)
xmin, xmax = ax.get_xlim()

# negative region (black), positive region (green)
ax.axvspan(xmin, 0,    color='black', alpha=0.1)
ax.axvspan(0,    xmax, color='green', alpha=0.1)

# 2) actual errorbar plot
plt.errorbar(
    df['mean'], y_pos,
    xerr=[df['mean'] - df['low'], df['hi'] - df['mean']],
    fmt='*', color='black', markersize=4,
    ecolor='black', capsize=3
)
labels = df_names.loc[df["feature"], "new_name"]
plt.yticks(y_pos, labels)
plt.gca().invert_yaxis()

# 4) add a mini‐legend for the shaded regions
neg_patch = Patch(facecolor='black', alpha=0.2,
                  label='β < 0: Higher values of the signal are associated with lower IDH risk')
pos_patch = Patch(facecolor='green', alpha=0.2,
                  label='β > 0: Higher values of the signal are associated with higher IDH risk')
ax.legend(handles=[neg_patch, pos_patch],
          loc='lower left', frameon=True) # framon 

# 5) titles and finish
plt.xlabel('Coefficient estimate')
plt.title('Logistic Regression Coefficients')
plt.tight_layout()
plt.savefig(
    os.path.join(graphics_dir, "LogisticREG_Coefficients.pdf"),
    format='pdf',
    dpi=300
)
plt.show() if SHOW_PLOTS else None
plt.close()
"""

# %%


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
y_pos = list(range(len(df)))
fig, ax = StandardFig(width="double", height=len(df)*0.06, scale=1.2)

ax.set_xlim(df['low'].min()*1.1, df['hi'].max()*1.1)
xmin, xmax = ax.get_xlim()

ax.axvspan(xmin, 0,    color='black', alpha=0.15)   # β < 0  region
ax.axvspan(0,    xmax, color='red',   alpha=0.15)   # β > 0  region

ax.errorbar(
    df['mean'], y_pos,
    xerr=[df['mean'] - df['low'], df['hi'] - df['mean']],
    fmt='*', color='black', markersize=4,
    ecolor='black', capsize=3
)

labels = df_names.loc[df["feature"], "new_name"]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()

arrow_y        = -0.09                     
arrow_label_y  = arrow_y - 0.01            
header_y       = arrow_label_y + 0.016       
arrow_kw       = dict(arrowstyle='-|>', lw=1.2, clip_on=False)
ax.annotate(
    '', xy=(xmax, arrow_y), xytext=(0, arrow_y),
    xycoords=('data', 'axes fraction'),
    arrowprops={**arrow_kw, 'color': 'red'}
)
ax.text(xmax - 0.3, arrow_label_y,
        '  Higher IDH likelihood',
        ha='left', va='top',
        transform=ax.get_xaxis_transform())
ax.annotate(
    '', xy=(xmin, arrow_y), xytext=(0, arrow_y),
    xycoords=('data', 'axes fraction'),
    arrowprops={**arrow_kw, 'color': 'black'}
)
ax.text(xmin + 0.2, arrow_label_y,
        'Lower IDH likelihood  ',
        ha='right', va='top',
        transform=ax.get_xaxis_transform())

ax.text(-0.35, header_y,
        'Outcome associated with a higher signal value',
        ha='center', va='bottom',
        transform=ax.get_xaxis_transform(), weight='bold')

fig.subplots_adjust(bottom=-0.05)

ax.set_xlabel('Average Coefficients Across 10-Fold Models')
ax.set_title('Logistic Regression Coefficients from the Feature Selection Process', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(graphics_dir, "LogisticREG_Coefficients.pdf"),
            format='pdf', dpi=300)
if SHOW_PLOTS:
    fig.show()
plt.close()

# %%
# AUPRC vs NONZERO COeffs ---------------------------------
filtered_df = cleaned_df[cleaned_df["n_key_nonzero"] >= 1]
color_map = {0.5: 'black', 1.0: 'green'}
metrics = {
    'auprc': {
        'label': 'AUPRC',
        'marker': 'o',
        'linestyle': '-',
        'lo_col': 'auprc_lo',
        'hi_col': 'auprc_hi',
    },
    'recall_fpr20': {
        'label': 'recall_fpr20',
        'marker': 's',
        'linestyle': '--',
        'lo_col': 'recall_lo',
        'hi_col': 'recall_hi',
    },
    'auroc': {
        'label': 'AUROC',
        'marker': '^',
        'linestyle': '-.',
        'lo_col': 'auroc_lo',
        'hi_col': 'auroc_hi',
    },
    
    'logloss': {
        'label': 'Log-loss',
        'marker': 's',
        'linestyle': '--',
        'lo_col': 'logloss_lo',
        'hi_col': 'logloss_hi',
    },
    'llauprc': {
        'label': 'Logloss + 0.2*(1-AUPRC)',
        'marker': '^',
        'linestyle': '-.',
        'lo_col': 'llauprc_lo',
        'hi_col': 'llauprc_hi',
    },
    
}

LAM = 0.2
for key, props in metrics.items():
    fig, ax = StandardFig(
        width   = "single",   # "single" (90 mm) | "double" (183 mm) | numeric (mm)
        height  = 0.75,       
        base_pt = 8,          
        scale   = 1.00
    )
    
    for alpha, grp0 in filtered_df.groupby("alpha"):
        grp = (
            grp0
            .sort_values("n_key_nonzero")            
            .drop_duplicates("n_key_nonzero")         
        )
        if key == "llauprc":
            grp["llauprc_manual"] = (
                grp["logloss"] + LAM * (1.0 - grp["auprc"])
            )
            # crude bounds (same sign-flip logic; ignore covariance)
            grp["llauprc_lo_manual"] = (
                grp["logloss_lo"] + LAM * (1.0 - grp["auprc_hi"])
            )
            grp["llauprc_hi_manual"] = (
                grp["logloss_hi"] + LAM * (1.0 - grp["auprc_lo"])
            )
            y  = grp["llauprc_manual"].to_numpy()
            lo = grp["llauprc_lo_manual"].to_numpy()
            hi = grp["llauprc_hi_manual"].to_numpy()
        else:
            y  = grp[key].to_numpy()
            lo = grp[props["lo_col"]].to_numpy()
            hi = grp[props["hi_col"]].to_numpy()

        x   = grp["n_key_nonzero"].to_numpy()
        c   = color_map[alpha]
        ax.plot(
            x, y,
            marker      = props["marker"],
            linestyle   = props["linestyle"],
            color       = c,
            label       = fr"$\alpha={alpha:0.2f}$",
            zorder      = 3           
        )
        ax.fill_between(
            x, lo, hi,
            alpha       = 0.20 if alpha == 0.5 else 0.30,
            color       = c,
            linewidth   = 0,          
            zorder      = 2
        )

    ax.set(
        title   = f"CV • {props['label']} vs Total Variables",
        xlabel  = "Total Variables",
        ylabel  = props["label"],
    )
    ax.legend()
    filename = os.path.join(graphics_dir, f"ElasticNET_{props['label']}_change.pdf")
    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else None
    plt.close(fig)

# %%
