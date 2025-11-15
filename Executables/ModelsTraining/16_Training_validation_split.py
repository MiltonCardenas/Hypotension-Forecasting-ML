# ---------------------------------------------------------------------
# Training and Validation split
# This script will split the data into training and validation sets.
# to ensure an stratified split and keep the dialysis onset distribution
# evenly across divisions
# v.1.0 author: Milton Dario Cardenas 
# -----------------------------------------------------------------------

"""


 /$$$$$$$              /$$                      /$$$$$$            /$$ /$$   /$$    
| $$__  $$            | $$                     /$$__  $$          | $$|__/  | $$    
| $$  \ $$  /$$$$$$  /$$$$$$    /$$$$$$       | $$  \__/  /$$$$$$ | $$ /$$ /$$$$$$  
| $$  | $$ |____  $$|_  $$_/   |____  $$      |  $$$$$$  /$$__  $$| $$| $$|_  $$_/  
| $$  | $$  /$$$$$$$  | $$      /$$$$$$$       \____  $$| $$  \ $$| $$| $$  | $$    
| $$  | $$ /$$__  $$  | $$ /$$ /$$__  $$       /$$  \ $$| $$  | $$| $$| $$  | $$ /$$
| $$$$$$$/|  $$$$$$$  |  $$$$/|  $$$$$$$      |  $$$$$$/| $$$$$$$/| $$| $$  |  $$$$/
|_______/  \_______/   \___/   \_______/       \______/ | $$____/ |__/|__/   \___/  
                                                        | $$                        
                                                        | $$                        
                                                        |__/                        
"""


# %% ===================================================================
import os
import pandas as pd
import numpy as np
import sys 
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
#main_dir = r"d:/"
main_dir = "/ihome/rparker/mdc147/PhD_Project"
os.chdir(main_dir)

# Instances --------------------------------------------------------------
sys.path.append("Auxiliar_codes")
from classes_repository import PathManager, GlobalParameters, ProcessingAuxiliar
from characterization import StandardFig
pm = PathManager().paths
pa = ProcessingAuxiliar()
gp = GlobalParameters()

#: Input - Output paths ---------------------------------------------------
DIR_PROCESSED       = "Data/Processed_datasets/Summary/Merged_M24H15"
PATH_DYN            = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Signals.pkl")
PATH_SSUM           = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Summary.pkl") 

#:Output
DIR_OUTPUT          = "Models_storage/DynamicPredictionModel/StratifiedDivision"
PATH_DIVISION       = os.path.join(DIR_OUTPUT, "stratified_division_summary.pkl")
PATH_DIVISIONTXT    = os.path.join(DIR_OUTPUT, "stratified_division_info.txt")
DIR_GRAPHICS        = "Documents/Multimedia/DataSplit"


# Main Split Parameters -------------------------------------------------
test_validationRatio    = 0.20
random_seed             = 1957
plot_results            = True # If false, the results are stored but not displayed
folds_CV1               = 5
folds_CV2               = 10

# Data Loading ---------------------------------------------------------
df_dyn              = pd.read_pickle(PATH_DYN)
df_summary0          = pd.read_pickle(PATH_SSUM)

SESSIONS    =  df_dyn["session"].unique().tolist()
df_summary  =  df_summary0[df_summary0["session"].isin(SESSIONS)].copy()

#: DATA UP TO IDH ------------- (JUST TO CHARACTERIZE THE OBSERVATION-WISE ONSET)
IDH_sessions    = df_summary[df_summary['IDH_01'] == True].session.unique().tolist()
condition1      = ~(df_dyn['session'].isin(IDH_sessions))
condition2      = (df_dyn['session'].isin(IDH_sessions) & (df_dyn['Time_until_IDH_01'] < 0))
df_dyn          = df_dyn[condition1 | condition2].copy()


# %%  ====================================================================
# Customized Stratified Division function --------------------------------
def stratified_custom_division(session_labels, split_ratio, df_summary, random_seed):
    # ------------------------------------------------------------------------------------
    # This function aims to make each patient appear only once in each fold
    # as much as possible, but we are still prioritizing the stratification of the sessions
    # based on the time distribution of positive samples and the IDH incidence.
    # ------------------------------------------------------------------------------------
    # First, I will append information of which patient the sessions comes from
    # sessions_label contains information of the first split (test/validation)
    session_label_n = session_labels.merge(df_summary[['session', 'FIN_study_id']], on='session', how='left')
    # GroupShuffleSplit makes sure that the same patient is not in the same fold
    rng = np.random.RandomState(random_seed)
    total_iterations = 50
    gss = GroupShuffleSplit(n_splits=total_iterations, test_size=split_ratio, random_state=rng) 
    # Here we have information about about the different stratification categories
    # and we want to know the initial distribution of the classes in the training set
    # before forcing the patient to be in different folds
    global_counts = session_label_n['stratify_col'].value_counts()
    global_distribution = global_counts / global_counts.sum()
    categories = global_distribution.index
    # Now I will perform a loop here, the idea is to find the best split that minimizes the difference 
    # between the global distribution and the subset distribution
    best_split = None
    best_diff = float('inf')
    # Helper to measure distribution difference 
    def distribution_diff(df_subset):
        subset_counts = df_subset['stratify_col'].value_counts()
        subset_dist = subset_counts / subset_counts.sum()
        subset_dist = subset_dist.reindex(categories).fillna(0.0)
        # L1 difference
        diff = np.sum(np.abs(subset_dist - global_distribution))
        return diff
    # Iteration over possible splits
    for train_idx, test_idx in gss.split(
        session_label_n, 
        groups=session_label_n['FIN_study_id'] # This is the patient identifier
        ):
        # Subset the data
        train_df = session_label_n.iloc[train_idx]
        diff = distribution_diff(train_df)
        if diff < best_diff:
            best_diff = diff
            best_split = (train_idx, test_idx)
            
        train_idx_new, test_idx_new = best_split
        g1_sessions = session_label_n.iloc[train_idx_new]['session'].unique().tolist() # This is the training set
        g2_sessions = session_label_n.iloc[test_idx_new]['session'].unique().tolist() # This is the validation set

    return g1_sessions, g2_sessions

#  Here I will label the sessions based on whether or not IDH was triggered
session_label = df_dyn.groupby('session')['Ground_truth'].max().reset_index()
# I gotta know whats the closer measurement to the IDH event for each positive session 
# In order to stratify this time as well
pos_times = (
    df_dyn[df_dyn['Ground_truth'] == 1]
    .groupby('session')['elapsed_time'].max()
    .reset_index(name='time_pos')
)
#  I will merge both time and label information to stratify the sessions
session_label = session_label.merge(pos_times, on='session', how='left')
session_label['time_bin'] = pd.qcut(
    session_label['time_pos'], 
    q=5, 
    duplicates='drop' # This basically changes the number of bins just if the boundaries are repeated
)
# sessions without IDH will trigger a NaN value in the time_bin column
# so i will replace the NaN values with 'NEG' to avoid errors in the stratification
session_label['time_bin'] = session_label['time_bin'].astype(str).replace('nan', 'NEG')
# Here I will create a new column to stratify the sessions based on the time_bin and the Ground_truth
session_label['stratify_col'] = session_label['Ground_truth'].astype(str) + "_" + session_label['time_bin']
# 10.1.5 Here I will stratify the sessions based on the new column
training_sessions, validation_sessions = stratified_custom_division(session_label, test_validationRatio, df_summary, random_seed)

# Training Subset Information ----------------------------------
training_subset = df_dyn[df_dyn['session'].isin(training_sessions)].copy().reset_index(drop=True) 
poslist_tra     = training_subset[training_subset['Ground_truth'] == 1].session.unique().tolist()
poslis_pat_tra  = training_subset[training_subset['Ground_truth'] == 1]['FIN_study_id'].unique().tolist()
total_patients_tra          = training_subset['FIN_study_id'].nunique()
slist_tra                   = training_subset.session.unique().tolist()
IDH_tra                     = len(poslist_tra)/len(slist_tra) * 100
positive_amount_training    = training_subset['Ground_truth'].sum()
idh_tra_m                   = positive_amount_training / ( positive_amount_training + training_subset['Ground_truth'].count())


# Validation Subset Information ---------------------------------
validation_subset = df_dyn[df_dyn['session'].isin(validation_sessions)].copy().reset_index(drop=True)
poslist_val             = validation_subset[validation_subset['Ground_truth'] == 1].session.unique().tolist()
slist_val               = validation_subset.session.unique().tolist()
neglist_val             = validation_subset[validation_subset['Ground_truth'] == 0].session.unique().tolist()
IDH_val= len(poslist_val)/len(slist_val) * 100
poslist_patients_val    = validation_subset[validation_subset['Ground_truth'] == 1]['FIN_study_id'].nunique()
all_patients_val        = validation_subset['FIN_study_id'].nunique()
p_val                   = validation_subset['Ground_truth'].sum()
n_val                   = validation_subset['Ground_truth'].count() - p_val
idh_val_m               = p_val / (p_val + validation_subset['Ground_truth'].count())

# %% ==========================================================
# Here I will divide the sessions into 10 folds and create the column "fold_CV10"
def split_internal_folds(labels:pd.DataFrame, folds:int, seed:int, custom_name=None) -> pd.DataFrame:
    # StratifiedGroupKFold() -> This cross-validation object is a variation of StratifiedKFold attempts to
    # return stratified folds with non-overlapping groups (Patients). The folds are made by
    # preserving the percentage of samples for each class. Each group will appear exactly once in the test set across all folds (the
    # number of distinct groups has to be at least equal to the number of folds).
    sgkf = StratifiedGroupKFold(folds, 
                                shuffle=True,
                                random_state=seed)
    
    groups  = labels['FIN_study_id'].values     # each fold will avoid to share this group
    y       = labels['stratify_col'].values     # each fold will tend to keep this distribution similarly
    X       = labels['session'].values          # These are the sessions IDs to be split
    
    iterator = sgkf.split(X, y, groups)         # This is basically an iterator with the train/val indices at each cv
    for i, (train_index, test_index) in enumerate(iterator):
        # train_index basically points out the index of everything outside each fold so 
        # in this case I'm interested in "test_index"
        target_indices = labels.index[test_index]       # Here I look for test_index  values in labels indexes
        
        if custom_name:
            labels.loc[target_indices, 'fold'] = i
        else:
            labels.loc[target_indices, f'fold_CV_{folds}'] = i
    return labels

temp_labels = session_label[session_label['session'].isin(slist_tra)].reset_index(drop=True)
if 'FIN_study_id' not in temp_labels.columns:
    temp_labels = temp_labels.merge(
        df_summary[['session', 'FIN_study_id']], 
        on='session', 
        how='left'
        ) 

train_labels = split_internal_folds(temp_labels, folds_CV2, random_seed)
if f'fold_CV_{folds_CV2}' not in training_subset.columns:
    target = ['session', f'fold_CV_{folds_CV2}' ]
    training_subset = pd.merge(
        training_subset,
        train_labels[target], 
        on='session', 
        how='left'
        )


# %% ===================================================================================
# I'll check the stratification here
patients = []
sessions = []
for f in training_subset[f'fold_CV_{folds_CV2}'].unique().tolist():
    subset = training_subset[training_subset[f'fold_CV_{folds_CV2}']==f].copy()
    fold_patients = subset['FIN_study_id'].unique().tolist()
    fold_sessions = subset['session'].unique().tolist() 
    patients.extend(fold_patients)
    sessions.extend(fold_sessions)
    
    # Fold Composition:
    fold_pos_s      = subset[subset['Ground_truth']==1]['session'].nunique()    # session-wise
    fold_neg_s      = subset[subset['Ground_truth']==0]['session'].nunique()
    fold_pos_m      = len(subset[subset['Ground_truth']==1])                    # measurement-wise
    fold_neg_m      = len(subset[subset['Ground_truth']==0])
    idh_incidence_s = fold_pos_s / (fold_pos_s + fold_neg_s) * 100
    idh_incidence_m = fold_pos_m / (fold_pos_m + fold_neg_m) * 100
    
    # More Information for the text-file
    print(f"Fold {f}")
    print(f"Unique Patients = {len(fold_patients)}")
    print(f"unique Sessions = {len(fold_sessions)}")
    print(f"IDH Incidence Sessions-Wise = {idh_incidence_s}")
    print(f"IDH Incidence Mesurements-Wise = {idh_incidence_m}")
    print(f"\n")
    
p_train_initial = training_subset['FIN_study_id'].nunique()
p_train_total   = len(patients)
difference = p_train_initial - p_train_total
print(f"The Initial number of patients at the training set is {p_train_initial}")
print(f"The sum of patients from all the Folds is: {p_train_total}")


# %% ==========================================================================================
# (Training_set) This is very similar to the previous code chunk, however now I'll split
# the training set in 3 Folds
train_labels = split_internal_folds(temp_labels, folds_CV1, random_seed, custom_name='fold')
training_subset = pd.merge(
    training_subset,
    train_labels[['session', 'fold']],
    on='session',
    how='left'
)
training_subset['fold'] = training_subset['fold'] + 1 # 1, 2, 3 ...

# Summary ----------------------------------------------------------------------
p_samp = training_subset[training_subset['Ground_truth'] == 1]
n_samp = training_subset[training_subset['Ground_truth'] == 0]
ratio = len(p_samp) / len(training_subset)
print('\n=================================================================================')
print(f"Folds Information ⬇️⬇️⬇️")
# Classes distribution at the folds
for i in [int(item) for item in training_subset['fold'].unique().tolist()]:
    fold = i
    pf1 = training_subset[(training_subset['Ground_truth'] == 1) & (training_subset['fold'] == fold)]['session'].nunique()
    pf2 = training_subset[(training_subset['Ground_truth'] == 1) & (training_subset['fold'] == fold)]['FIN_study_id'].nunique()
    tot1 = training_subset[training_subset['fold'] == fold]['session'].nunique()
    dist1 = pf1 / tot1
    print(f"Unique IDH Patients at fold {fold}: {pf2} (this plus the patients at the validation set should be the same if patients are not repeated)")
    print(f"IDH sessions at fold {fold}: {pf1}, Total sessions at fold {fold}: {tot1}, IDH incidence: {dist1*100:.3f}%")
    print(f" Positive observations at fold {fold}: {training_subset[training_subset['fold'] == fold]['Ground_truth'].sum()}")

# %% ===========================================================
# ---------- Final File summary processing and storage ---------
target = session_label['session'].isin(slist_val)
validation_labels = session_label[target].copy()
target = ['session', 'FIN_study_id']
if 'FIN_study_id' not in validation_labels:
    validation_labels = pd.merge(
        validation_labels,
        validation_subset[target],
        on='session',
        how='left'
    )
validation_labels = validation_labels.reindex(columns=train_labels.columns, fill_value = np.nan)
validation_labels['fold'] = "validation"
validation_labels = validation_labels[validation_labels.fold == 'validation'].drop_duplicates(keep='first')
division_summary = pd.concat([train_labels, validation_labels])
division_summary.to_pickle(PATH_DIVISION)


# %% ===========================================================
# --------------------- PLOTS  --------------------------------
# (1) 5 Internal stratified divisions (training)
df_pos = training_subset[training_subset['Ground_truth'] == 1]
fold_col = f'fold'
folds   = sorted(df_pos[fold_col].unique())

data_per_fold = [
    df_pos[df_pos[fold_col] == f]['elapsed_time']
    for f in folds
]

fig, ax = StandardFig(
                width="double",    #  90 mm wide
                height=0.5,      
                base_pt=8,         
                scale=1        
            )
box = ax.boxplot(
    data_per_fold,
    tick_labels = [str(int(f)) for f in folds],  # shift to 1-based fold numbering
    patch_artist=True,
    boxprops    = dict(facecolor='lightgray', color='black', linewidth=1),
    whiskerprops= dict(color='black', linewidth=1),
    capprops    = dict(color='black', linewidth=1),
    medianprops = dict(color='green', linewidth=2),
    flierprops  = dict(marker='x', markeredgecolor='black', markerfacecolor='none'),
)
ax.set_xlabel('Fold')
ax.set_ylabel('Elapsed Time (minutes)')
ax.set_title('Elapsed Time Distribution Of Pre-Hypotensive Signals (XGBoost Model)')
fig.tight_layout()
fig.savefig(
    os.path.join(DIR_GRAPHICS,f'PreHypotensiveOnsetTimeBoxPlotCV{folds_CV1}.pdf'),
    format='pdf',
    dpi=300
)
if plot_results:
    plt.show()

# (2) 11 Internal stratified divisions (training)
df_pos = training_subset[training_subset['Ground_truth'] == 1]
fold_col = f'fold_CV_{folds_CV2}'
folds   = sorted(df_pos[fold_col].unique())
data_per_fold = [
    df_pos[df_pos[fold_col] == f]['elapsed_time']
    for f in folds
]
fig, ax = StandardFig(
                width="double",    #  90 mm wide
                height=0.5,           
                scale=1        
            )
box = ax.boxplot(
    data_per_fold,
    tick_labels = [str(int(f) + 1) for f in folds],  # shift to 1-based fold numbering
    patch_artist=True,
    boxprops    = dict(facecolor='lightgray', color='black', linewidth=1),
    whiskerprops= dict(color='black', linewidth=1),
    capprops    = dict(color='black', linewidth=1),
    medianprops = dict(color='green', linewidth=2),
    flierprops  = dict(marker='x', markeredgecolor='black', markerfacecolor='none'),
)
ax.set_xlabel('Fold')
ax.set_ylabel('Elapsed Time (minutes)')
ax.set_title('Elapsed Time Distribution Of Pre-Hypotensive Signals (Feature Selection)')
fig.tight_layout()
fig.savefig(
    os.path.join(DIR_GRAPHICS, f'PreHypotensiveOnsetTimeBoxPlotCV{folds_CV2}.pdf'),
    format='pdf',
    dpi=300
)
if plot_results:
    plt.show()

# (3) Output information to a text file 
with open(PATH_DIVISIONTXT, 'w') as f:
    f.write('ALL OBSERVATION-WISE INFORMATION IS AGGREGATED UP TO THE FIRST IDH ONSET (WITHOUT USING THE ONSET)')
    f.write('=================================================================================\n')
    f.write(f"Training set information ⬇️⬇️⬇️\n")
    f.write(f"sessions at the training set: {len(slist_tra)}\n")
    f.write(f"IDH sessions at the training set: {len(poslist_tra)}\n")
    f.write(f"Patiets at the training set: {total_patients_tra}\n")
    f.write(f"IDH patients at the training set: {len(poslis_pat_tra)}\n")
    f.write(f"IDH incidence at the training set (sessions): {IDH_tra:.2f} %\n")
    f.write(f"Positive samples at the training set: {positive_amount_training}\n")
    f.write(f"Negative samples at the training set: {len(training_subset) - positive_amount_training}\n")
    f.write(f"IDH Incidence (Observations-wise): {idh_tra_m*100:.3f} %\n")
    f.write('==================================================================================\n')
    
    f.write(f"Folds Information CV = {folds_CV2} ⬇️⬇️⬇️\n")
    for fold in sorted(training_subset[f'fold_CV_{folds_CV2}'].unique().tolist()):
        subset = training_subset[training_subset[f'fold_CV_{folds_CV2}']==fold].copy()
        fold_patients = subset['FIN_study_id'].unique().tolist()
        fold_sessions = subset['session'].unique().tolist() 
        patients.extend(fold_patients)
        sessions.extend(fold_sessions)
        
        # Fold Composition:
        fold_pos_s      = subset[subset['Ground_truth']==1]['session'].nunique()    # session-wise
        fold_neg_s      = subset[subset['Ground_truth']==0]['session'].nunique()
        fold_pos_m      = len(subset[subset['Ground_truth']==1])                    # measurement-wise
        fold_neg_m      = len(subset[subset['Ground_truth']==0])
        idh_incidence_s = fold_pos_s / (fold_pos_s + fold_neg_s) * 100
        idh_incidence_m = fold_pos_m / (fold_pos_m + fold_neg_m) * 100
        
        # More Information for the text-file
        f.write(f"Fold {fold + 1}\n")
        f.write(f"Unique Patients = {len(fold_patients)}\n")
        f.write(f"Unique Sessions = {len(fold_sessions)}\n")
        f.write(f"IDH Incidence Sessions-Wise = {idh_incidence_s} %\n")
        f.write(f"IDH Incidence Observations-Wise = {idh_incidence_m} %\n")
        f.write(f"---------------------------------------\n")

    f.write("============================================================================\n")
    f.write(f"Folds Information CV = {folds_CV1} ⬇️⬇️⬇️\n")
    for i in sorted([int(item) for item in training_subset['fold'].unique().tolist()]):
        fold = i
        subset = training_subset[training_subset['fold'] == fold].copy()
        fold_patients = subset['FIN_study_id'].unique().tolist()
        fold_sessions = subset['session'].unique().tolist() 
        patients.extend(fold_patients)
        sessions.extend(fold_sessions)
        
        # Fold Composition:
        fold_pos_s      = subset[subset['Ground_truth']==1]['session'].nunique()    # session-wise
        fold_neg_s      = subset[subset['Ground_truth']==0]['session'].nunique()
        fold_pos_m      = len(subset[subset['Ground_truth']==1])                    # measurement-wise
        fold_neg_m      = len(subset[subset['Ground_truth']==0])
        idh_incidence_s = fold_pos_s / (fold_pos_s + fold_neg_s) * 100
        idh_incidence_m = fold_pos_m / (fold_pos_m + fold_neg_m) * 100
        
        # More Information for the text-file
        f.write(f"Fold {fold}\n")
        f.write(f"Unique Patients = {len(fold_patients)}\n")
        f.write(f"Unique Sessions = {len(fold_sessions)}\n")
        f.write(f"IDH Incidence Sessions-Wise = {idh_incidence_s} %\n")
        f.write(f"IDH Incidence Observations-Wise = {idh_incidence_m} %\n")
        f.write(f"---------------------------------------\n")
        
    
    f.write('=================================================================================\n')
    f.write(f"Validation set information ⬇️⬇️⬇️\n")
    f.write(f"Total patients in the validation set: {all_patients_val}\n")
    f.write(f"Unique IDH Patients in the validation set: {poslist_patients_val}\n")
    f.write(f"Total sessions in the validation set: {len(slist_val)}\n")
    f.write(f"IDH sessions in the validation set: {len(poslist_val)}\n")
    f.write(f"IDH incidence in the validation set: {IDH_val:.2f} %\n")
    f.write(f"Positive samples in the validation set: {p_val}\n")
    f.write(f"Negative samples in the validation set: {n_val}\n")
    f.write(f"IDH Incidence (Observations-wise): {idh_val_m *100:.3f} %\n")
    f.write('===================================================================================\n')
    
print(f"Training and Validation summary saved in {PATH_DIVISIONTXT}")
