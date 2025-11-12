""" 
---------------------------------------------------------------------------------------------------
This script merges the data from both Databases and filter out sessions depending on the session's
cohort design. This is the last processing step before the feature selection. This script also 
characterizes the signals and labels positive pre-hypotensive observations and plots useful 
information
----------------------------------------------------------------------------------------------------

 /$$$$$$$$ /$$                     /$$       /$$$$$$$              /$$                      
| $$_____/|__/                    | $$      | $$__  $$            | $$                      
| $$       /$$ /$$$$$$$   /$$$$$$ | $$      | $$  \ $$  /$$$$$$  /$$$$$$    /$$$$$$         
| $$$$$   | $$| $$__  $$ |____  $$| $$      | $$  | $$ |____  $$|_  $$_/   |____  $$        
| $$__/   | $$| $$  \ $$  /$$$$$$$| $$      | $$  | $$  /$$$$$$$  | $$      /$$$$$$$        
| $$      | $$| $$  | $$ /$$__  $$| $$      | $$  | $$ /$$__  $$  | $$ /$$ /$$__  $$        
| $$      | $$| $$  | $$|  $$$$$$$| $$      | $$$$$$$/|  $$$$$$$  |  $$$$/|  $$$$$$$        
|__/      |__/|__/  |__/ \_______/|__/      |_______/  \_______/   \___/   \_______/        
                                                                                            
                                                                                           
                                                                                            
 /$$$$$$$                                                            /$$                    
| $$__  $$                                                          |__/                    
| $$  \ $$ /$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$   /$$$$$$$ /$$$$$$$ /$$ /$$$$$$$   /$$$$$$ 
| $$$$$$$//$$__  $$ /$$__  $$ /$$_____/ /$$__  $$ /$$_____//$$_____/| $$| $$__  $$ /$$__  $$
| $$____/| $$  \__/| $$  \ $$| $$      | $$$$$$$$|  $$$$$$|  $$$$$$ | $$| $$  \ $$| $$  \ $$
| $$     | $$      | $$  | $$| $$      | $$_____/ \____  $$\____  $$| $$| $$  | $$| $$  | $$
| $$     | $$      |  $$$$$$/|  $$$$$$$|  $$$$$$$ /$$$$$$$//$$$$$$$/| $$| $$  | $$|  $$$$$$$
|__/     |__/       \______/  \_______/ \_______/|_______/|_______/ |__/|__/  |__/ \____  $$
                                                                                   /$$  \ $$
                                                                                  |  $$$$$$/
                                                                                   \______/ 

"""
# %% #* ============================ DATA LOADING AND LIBRARIES ====================================
import numpy as np
import pandas as pd
import time, datetime
import os, gc
from IPython.display import display
main_dir = "/ihome/rparker/mdc147/PhD_Project"; os.chdir(main_dir)
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from Auxiliar_codes.classes_repository import PathManager as P
from Auxiliar_codes.classes_repository import GlobalParameters as GP
from Auxiliar_codes.classes_repository import StandardNames as SN
from Auxiliar_codes.classes_repository import ProcessingAuxiliar as PA
from Auxiliar_codes                    import dynamic_processing, static_processing, characterization
from Auxiliar_codes.machine_learning_auxiliar import FeaturesManagerV2

#: INSTANCES -----------------------------------------------------
pm      = P().paths
gp      = GP()    
sn      = SN()
pa      = PA()
fm      = FeaturesManagerV2()

#: PARAMETERS  --------------------------------------------------
IDH_WINDOW   = 35 #minutes
IDH_LAGTIME  = 5 #minutes

#: PATHS INPUT ---------------------------------------------------
PATH_M24   = "Data/Processed_datasets/Summary/Mladi24/df_dyn_events.pkl"
PATH_SSM24 = "Data/Processed_datasets/Summary/Mladi24/Sessions_Summary.pkl"
PATH_H15   = "Data/Processed_datasets/Summary/Hidenic15/df_dyn_events.pkl"
PATH_SSH15 = "Data/Processed_datasets/Summary/Hidenic15/Sessions_Summary.pkl"

#: PATHS OUTPUT -------------------------------------------------
DIR_PROCESSED       = "Data/Processed_datasets/Summary/Merged_M24H15"
PATH_DYN_OUT        = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Signals.pkl")
PATH_SSUM_OUT       = os.path.join(DIR_PROCESSED, "df_events_Merged_M24&H15_Summary.pkl") 
PATH_IDH_SUM        = os.path.join(DIR_PROCESSED, "df_events_IDH_summary.csv") 

DIR_CHARACTERIZATION    = "Documents/Multimedia/DataCharacterization/AllSignalsDescription"
PATH_DESCRIPTIVE_TXT    = os.path.join(DIR_CHARACTERIZATION, "simulation_time_characterization.txt")
PATH_SESS_PATIENTS_TXT  = os.path.join(DIR_CHARACTERIZATION, "patients_and_sessions_characterization.txt")
PATH_COHORT_SUMMARY     =  os.path.join(DIR_CHARACTERIZATION, "cohort_tests.txt")
PATH_TXTDensity         =  os.path.join(DIR_CHARACTERIZATION, "SamplingTimeInfo.txt")
PATH_VAR_IMPUTATION     =   os.path.join(DIR_CHARACTERIZATION, "VarImputation.csv")

# %% #* =============================== MERGING PROCESS ============================================
#: IDENTIFIER CHANGE ---
df_m24      = pd.read_pickle(PATH_M24)
df_ssm24    = pd.read_pickle(PATH_SSM24)
df_H15      = pd.read_pickle(PATH_H15)
df_ssh15    = pd.read_pickle(PATH_SSH15)

df_m24['session']           = df_m24['session'].astype(str) + '_M24'
df_m24['FIN_study_id']      = df_m24['FIN_study_id'].astype(str) + '_M24'
df_ssm24['session']         = df_ssm24['session'].astype(str) + '_M24'
df_ssm24['FIN_study_id']    = df_ssm24['FIN_study_id'].astype(str) + '_M24'
df_H15['session']           = df_H15['session'].astype(str) + '_H15'
df_H15['FIN_study_id']      = df_H15['FIN_study_id'].astype(str) + '_H15'
df_ssh15['session']         = df_ssh15['session'].astype(str) + '_H15'
df_ssh15['FIN_study_id']    = df_ssh15['FIN_study_id'].astype(str) + '_H15'

#: CONCATENATION -------
col_M24         = df_m24.columns.tolist()
col_H15         = df_H15.columns.tolist()
common_columns  = [item for item in col_M24 if item in col_H15]
df_dyn          = pd.concat([df_m24[common_columns], df_H15[common_columns]], axis=0)
del df_m24 ; del df_H15 
col_ssM24       = df_ssm24.columns.tolist()
col_ssH15       = df_ssh15.columns.tolist()
common_columns  = [item for item in col_ssM24 if item in col_ssH15]
df_summary      = pd.concat([df_ssm24[common_columns], df_ssh15[common_columns]], axis=0)
del df_ssm24 ; del df_ssh15 ; gc.collect()

#: Temporal-----------------------------------------
cut             = [0, 40, 50, 60,  70,  95] 
labels          = ["<= 40", "<= 50", "<= 60", "<= 70", ">= 71"]
ACCI_Mapping    = {
    "<= 40": 0,
    "<= 50": 1,
    "<= 60": 2,
    "<= 70": 3,
    ">= 71": 4,   
}
df_summary["age_adj_comorbidity_score"] = (
    df_summary["age_adj_comorbidity_score"].astype("Int64")
)
df_summary["AACI_Group"] = pd.cut(
    df_summary["age"], bins=cut, right=True, labels=labels
)
mask = df_summary["age_adj_comorbidity_score"].isna()
df_summary.loc[mask, "age_adj_comorbidity_score"] = (
    df_summary.loc[mask, "AACI_Group"]
              .map(ACCI_Mapping)
              .astype("Int64")
)
mapping = df_summary.set_index("session")["age_adj_comorbidity_score"]
df_dyn["age_adj_comorbidity_score"] = df_dyn["session"].map(mapping)


# %% #* =============================== COHORT DESIGN ============================================
#: (1) Exclution of hypotensive sessions with no data within the range 5-40 minutes before the first hypotensive onset. 
df_summary["IDH_01"] = df_summary["IDH_01"].astype(bool)
hypo_info            =  df_summary.set_index("session")["IDH_01"]
def label_positive_observations(session, windowSize, IDH_LAGTIME):
    # define the two negative bounds
    t_lower = -(IDH_LAGTIME + windowSize)   #  −(5+35) = −40
    t_upper   = - IDH_LAGTIME                 #  −5
    mask = (
        (session['Time_until_IDH_01']   >= t_lower)
        & (session['Time_until_IDH_01'] <= t_upper)
    )
    session['Ground_truth'] = mask.astype(int)  
    return session

df_dyn = (
    df_dyn.groupby("session")[df_dyn.columns]
    .apply(label_positive_observations, IDH_WINDOW, IDH_LAGTIME)
    .reset_index(drop=True)
)
pos_info = df_dyn.groupby("session")["Ground_truth"].any()
df_summary["Has_positive_obs"] = df_summary["session"].map(pos_info).fillna(False)
print(f"Length of total dialysis sessions labeled as hypotensive {hypo_info.sum()}")
print(f"Length of total dialysis sessions with a Ground truth observation {pos_info.sum()}")

# This information goes back to the summary file:
cohort_tests = (
    ("UF_quality ∈ {'dg1','dg2'}",
        lambda d: d["UF_confidence"].isin(["dg1", "dg2"])),
    
    ("BP_quality ∈ {'dg1','dg2'}",
        lambda d: d["SBP_confidence"].isin(["dg1", "dg2"])),
    
    ("Pulse_quality ∈ {'dg1','dg2'}",
        lambda d: d["Pulse_confidence"].isin(["dg1", "dg2"])),
    
    ("Initial_weight_quality ∈ {'dg1','dg2'}",
        lambda d: d["Initial_weight_confidence"].isin(["dg1", "dg2"])),
    
    ("RespiratoryRateQuality ∈ {'dg1','dg2'}",
        lambda d: d["RR_confidence"].isin(["dg1", "dg2"])),
    
    ("If hypotensive, has a valid positive obs",
        lambda d: (~d["IDH_01"]) | (d["Has_positive_obs"] == True)),
    
    ("If IDH_01 then First_IDH_01 > 5",
        lambda d: (~d["IDH_01"]) | (d["First_IDH_01"] > 5)),
    
    ("Hypo_01 == False", lambda d: d["Hypo_01"] == False),
    
    ("18 ≤ age ≤ 90", lambda d: d["age"].between(18, 90)),
    ("unknown_sex == 0", lambda d: d["unknown_sex"] == 0),
    ("asian_race == 0",  lambda d: d["asian_race"]  == 0),
    ("other_race == 0",  lambda d: d["other_race"]  == 0),
)
report, sess_del, sess_remaining = characterization.cohort_report(df_summary, cohort_tests)
display(report)

#: SESSIONS FILTERING ----------------------------------------------
mask = pd.Series(True, index=df_summary.index)
for _, rule in cohort_tests:
    mask &= rule(df_summary)
remaining_sessions = df_summary.loc[mask, 'session'].unique().tolist()
print(f"Remaining sessions ({len(remaining_sessions)}):")

#: FINAL SESSIONS -------------------------------------------------
df_dyn             = df_dyn[df_dyn["session"].isin(remaining_sessions)].copy()
df_ssum            = df_summary[df_summary["session"].isin(remaining_sessions)].copy()
s_dyn              = df_dyn["session"].nunique()
print(f"Remaining sessions at the processed training set ({s_dyn}):")


#: NEW PREDICTORs Initial_UFR , BFR0: 
mapping_ufr0 = df_dyn.groupby("session").apply(lambda s: s.loc[(s["simulation_time"]==60), "UF_rate"].values[0])
mapping_BFR0 = df_dyn.groupby("session").apply(lambda s: s.loc[(s["simulation_time"]==60), "BFR"].values[0])
df_dyn["Initial_UF_rate"]    = df_dyn["session"].map(mapping_ufr0)
df_dyn["Initial_BFR"]        = df_dyn["session"].map(mapping_BFR0)


#: TARGET SIGNALS PROCESSING --------------------------------------
target_signals  = fm.identify_predictors(df_dyn.columns.tolist())
identifiers     = [item for item in sn.identifier_signals() if item in df_dyn.columns]
outside         = [x for x in df_dyn.columns.tolist() if x not in target_signals]
print(f"The following signals are not considered predictors in this analysis {outside}")

target_base     = [x for x in fm.signals_base if x != "DBP"]
new_obs_flags   = [f"new_observation_{base}" for base in target_base]
median_flags    = [x for x in df_dyn.columns if x.endswith("median")]
column_new_obs  = ["session", "simulation_time"] + ["SBP"] + new_obs_flags + median_flags
median_storage  = {}
for base in target_base:
    obs_col = f"new_observation_{base}"
    df_dyn[obs_col] = df_dyn[obs_col].fillna(False)

    if "VAR" in dynamic_processing.phi_mapping_CW[base]:
        flag_col = f"CW_{base}_VAR_median"
        df_dyn[flag_col] = df_dyn[flag_col].fillna(False)

        valid = (df_dyn[obs_col]) & ~(df_dyn[flag_col])
        #CW_median_var = df_dyn.loc[valid, f"CW_{base}_VAR"].median()
        CW_median_var = 0 # In the end i filled the CW initial var to 0 until a new observation happens
        
        df_dyn.loc[df_dyn[flag_col], f"CW_{base}_VAR"] = CW_median_var
        median_storage[f"CW_{base}_VAR"] = CW_median_var

    if "VAR" in dynamic_processing.phi_mapping_W[base]:
        flag_col = f"W_{base}_VAR_median"
        df_dyn[flag_col] = df_dyn[flag_col].fillna(False)

        valid = (df_dyn[obs_col]) & ~(df_dyn[flag_col])
        W_median_var = df_dyn.loc[valid, f"W_{base}_VAR"].median()

        df_dyn.loc[df_dyn[flag_col], f"W_{base}_VAR"] = W_median_var
        median_storage[f"W_{base}_VAR"] = W_median_var


df_VAR_imputation = pd.DataFrame(
    list(median_storage.items()),
    columns=["Signal", "Imputed value when n<2"]
)
subset_new_obs  = df_dyn[column_new_obs].copy()
    
df_dyn          = df_dyn.loc[:, identifiers + new_obs_flags + median_flags + target_signals + ["Ground_truth"]].copy()
for col in target_signals:
    df_dyn[col] = pd.to_numeric(df_dyn[col], errors="coerce")
nan = df_dyn[target_signals].isna().sum() 
# (1) Some Signals remain nan (specially those related to RR and O2Sat.)
# We can either delete those sessions or keep them. I kept them.

# %% #* ========================================= Storage 1 =======================================
df_dyn = pa.columns_sorting(df_dyn)
df_dyn.to_pickle(PATH_DYN_OUT)          # FINAL PROCESSED DATASET
df_summary.to_pickle(PATH_SSUM_OUT)     # FINAL PROCESSED SESSION'S SUMMARY
df_VAR_imputation.to_csv(PATH_VAR_IMPUTATION)
print("Done") 

# %% #* ============================== CHARACTERIZATION ============================================
#: SIGNALS -------------------------------------
char_signals = [x for x in target_signals if x not in ["sex", "White_race", "black_race", "age"]]
summary      = characterization.summarize_target_signals(df_dyn, char_signals)
display(summary)

def summarize(series: pd.Series) -> pd.Series:
    return pd.Series({
        "N"      : series.count(),
        "Mean"   : series.mean(),
        "SD"     : series.std(),
        "Median" : series.median(),
        "Q1"     : series.quantile(0.25),
        "Q3"     : series.quantile(0.75),
    })

#: SESSION - LEVEL
by_session = (
    df_dyn
      .groupby("session")
      .agg(
          Duration          = ("elapsed_time", "max"),
          Total_UF          = ("UF", "max"),
          Average_UF_rate   = ("UF_rate", "mean"),         # drop if UF_rate not present
          IDH_any           = ("Time_until_IDH_01", "any")
      )
)
session_numeric = by_session.drop(columns="IDH_any").apply(summarize).T


# initial values:
target = ["SBP", "MAP", "UF_rate", "Pulse", 
          "DT","RR", "BFR", "DC", "Initial_weight", "IBW", "BMI", "Blood_volume"]

by_session2= (
    df_dyn.loc[df_dyn["simulation_time"]==60, :]
    .groupby("session")
    .agg(
        **{f"Initial_{f}": (f, "mean") for f in target}
    )
)
sessions_info2 = by_session2.apply(summarize).T
sessions_info2 = sessions_info2.rename(index={"Initial_Initial_weight": "Initial_weight",
                                              "Initial_IBW": "IBW",
                                              "Initial_BMI": "BMI"})

session_numeric = pd.concat([session_numeric, sessions_info2], axis=0)
session_numeric["Hypotensive sesssions"] = by_session["IDH_any"].sum()
session_numeric["IDH incidence (%)"] = by_session["IDH_any"].mean() * 100
session_numeric.index.name = "Session-level variables"


# %% #* =========================== IDH Characterization ===========================================
import matplotlib.pyplot as plt
data = df_ssum["First_IDH_01"].dropna()
fIDH = data.describe()

fig, ax = characterization.StandardFig()
counts, bins, patches = plt.hist(
    data,
    bins=30,
    color='green',
    edgecolor='black',
    alpha=0.75
)

total = len(data)
median = np.median(data)
ax.axvline(median, color='darkkhaki', linestyle='--', linewidth=2,
            label=f'Median = {median:.1f}')
ax.set_xlabel("Time Since Dialysis Start (min)")
ax.set_ylabel("Total IDH Sessions")
ax.set_title(f"Histogram of First IDH Onset Time (S={total})")
ax.grid(axis='y', linestyle=':', linewidth=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(DIR_CHARACTERIZATION, "HistogramFirstIDHOnet.pdf"))
fig.show()
print(len(data))

fig, ax = characterization.StandardFig()
counts, bins, patches = ax.hist(
    data,
    bins=30,
    density=True,      # scale so total area = 1
    cumulative=True,   # make it cumulative
    color='darkkhaki',
    edgecolor='black',
    alpha=0.75
)
ax.axvline(
    median,
    color='green',
    linestyle='--',
    linewidth=2,
    label=f'Median = {median:.1f} min'
)
ax.set_xlabel("Time Since Dialysis Start (min)")
ax.set_ylabel("Ratio of IDH Sessions")
ax.set_title(f"Cumulative Density of First IDH Onset Time (S={total})")
ax.grid(axis='y', linestyle=':', linewidth=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(DIR_CHARACTERIZATION, "CumulativeFirstIDHOnet.pdf"))
fig.show()



#: PATIENT LEVEL --------------------------------------------
by_patient0 = ( # From observation level to session level
    df_dyn
      .groupby(["FIN_study_id", "session"])
      .agg(
          sessions      = ("session", "nunique"),
          IDH_patient   = ("Time_until_IDH_01", "any"),
          age           = ("age", "median"),
          weight        = ("Initial_weight", "median"),
          IBW           = ("IBW", "median"),
          BMI           = ("BMI", "median"),
          AACHARLSON    = ("age_adj_comorbidity_score", "median"),
          male          = ("sex", lambda x: (x == 1).median()),  
          female        = ("sex", lambda x: (x == 0).median()), 
          black         = ("black_race", "median"),
          white         = ("White_race", "median")
      )
)

by_patient = ( # From session level to patient level
    by_patient0.groupby("FIN_study_id")
    .agg(
        sessions      = ("sessions", "sum"),
        IDH_patient   = ("IDH_patient", "any"),
        age           = ("age", "median"),
        weight        = ("weight", "median"),
        IBW           = ("IBW", "median"),
        BMI           = ("BMI", "median"),
        AACHARLSON    = ("AACHARLSON", "median"),
        male          = ("male", "any"),  
        female        = ("female","any"), 
        black         = ("black", "median"),
        white         = ("white", "median")
    )
)

patient_numeric = (
    by_patient[["sessions", "age", "weight", "IBW",
                "BMI", "AACHARLSON"]]
      .apply(summarize).T
)
patient_numeric["IDH incidence (%)"] = by_patient["IDH_patient"].mean() * 100
patient_numeric["% male"]   = by_patient["male"].mean()   * 100
patient_numeric["% female"]   = by_patient["female"].mean()   * 100
patient_numeric["% black"]  = by_patient["black"].mean()  * 100
patient_numeric["% white"]  = by_patient["white"].mean()  * 100
patient_numeric.index.name  = "Patient-level variables"

#patient_numeric.loc["age", ["Mean", "SD"]] = patient_numeric.loc["age", ["Mean", "SD"]].round()
#patient_numeric.loc["sessions", ["Mean", "SD"]] = patient_numeric.loc["sessions", ["Mean", "SD"]].round()
session_char_table = session_numeric.reset_index()
patient_char_table = patient_numeric.reset_index()
display(session_char_table)
display(patient_char_table)


#: TIME UNTIL HYPOTENSION OF LABELS BEFORE IDH ----------------------------------------------------

data = df_dyn[df_dyn["Ground_truth"] == 1]["Time_until_IDH_01"]
print(len(data))

fig, ax = characterization.StandardFig()
ax.hist(
    data,
    bins=35,
    color='darkkhaki',
    edgecolor='black',
    alpha=0.75
)

ax.set_xlabel("Time Until Hypotension (min)")
ax.set_ylabel("Frequency")
ax.set_title(f"Distribution of Time to Hypotension\nFor Positive Class Observations ($O={len(data)}$)")
fig.tight_layout()
fig.savefig(os.path.join(DIR_CHARACTERIZATION, "HistogramPositiveClassTimeUntilHypotension.pdf"))



# %% #* ============================ ACTIONS DIFFERENCE OVER TIME ===================================
group1 = df_dyn[df_dyn["Time_until_IDH_01"] == -5]
group2 = df_dyn[df_dyn["Time_until_IDH_01"] == -15]

uf_data  = [group1["UF_rate"], group2["UF_rate"]]
bfr_data = [group1["BFR"],     group2["BFR"]]
labels   = ["-5 min", "-15 min"]

fig, axes = characterization.StandardFig(
    width="double",
    height=0.5,
    subplots=(1, 2)
)

boxprops     = dict(linewidth=1, facecolor="lightgray")
medianprops  = dict(linewidth=1.5, color="black")
whiskerprops = dict(linewidth=1)
capprops     = dict(linewidth=1)
flierprops   = dict(marker='x', markersize=5, markerfacecolor='black', markeredgecolor='black')

# UF_rate boxplot
axes[0].boxplot(
    uf_data, labels=labels, patch_artist=True, showfliers=True,
    boxprops=boxprops, medianprops=medianprops,
    whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops
)
axes[0].set_ylabel("UFR (mL/kg/min)")
axes[0].set_title("Ultrafiltration Rate (UFR) Near Hypotensive Episode", pad=6)
axes[0].set_xlabel("Time until IDH onset")
axes[0].tick_params(axis='y', labelsize=10)

# BFR boxplot
axes[1].boxplot(
    bfr_data, labels=labels, patch_artist=True, showfliers=True,
    boxprops=boxprops, medianprops=medianprops,
    whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops
)
axes[1].set_ylabel("BFR (mL/kg/min)")
axes[1].set_title("Blood Flow Rate (BFR) Near Hypotensive Episode", pad=6)
axes[1].set_xlabel("Time until IDH onset")
axes[1].tick_params(axis='y', labelsize=10)
plt.tight_layout()
fig.savefig(os.path.join(DIR_CHARACTERIZATION, "UFR_BFR_5min_vs_15min.pdf"))
plt.show()


# %% #* =================================== SAMPLING TIME FILE =====================================
def safe_freq(x):
    T = x.max()
    return len(x) / T if T > 0 else np.nan

with open(PATH_TXTDensity, "w") as f:
    # This just opens the file path and resets the content
    pass  
for col in target_base:
    print(f"Analyzing {col}")
    density = (
        df_dyn[
            (df_dyn[f"new_observation_{col}"]) &
            (df_dyn["elapsed_time"] >= 0)
        ]
        .groupby("session")
        .agg(
            observations=("elapsed_time", "count"),
            sampling_frequency=("elapsed_time", safe_freq)
        )
    )

    density["sampling_period"] = 1 / density["sampling_frequency"]
    density_statistics = density.apply(summarize).T
    display(density_statistics)
    
    with open(PATH_TXTDensity, 'a') as f:
        f.write("="*20 + "\n")
        f.write(f"Sampling Information of {col}\n")
        f.write(density_statistics.to_string())
        f.write("\n\n")
        

# %% #** ================================== DATA LABELING PLOT =====================================

good_sessions = ["10000_H15", "10110_H15", "30119_H15", "7423_H15"]
sessions = df_dyn[df_dyn["Ground_truth"]==True]["session"].unique().tolist()
obj      = np.random.choice(sessions)
hypo_s   = df_dyn[(df_dyn["session"] == "7423_H15")]


# IDH ONSET ----
hypo_s       = hypo_s[hypo_s["simulation_time"]<240] 
conds        = (hypo_s["SBP"] < 90) & (hypo_s["MAP"] < 65)
idh_times    = hypo_s.loc[conds, "simulation_time"].sort_values().unique()
first_idh    = idh_times[0]
non_idh_ts   = hypo_s.loc[~conds, "simulation_time"].sort_values()
next_non_idh = non_idh_ts[non_idh_ts > first_idh].iloc[0]    

# PLOT ---------
fig, ax = characterization.StandardFig(width="double", height=0.6, scale=1.1)
ax.scatter(hypo_s["simulation_time"], hypo_s["SBP"], label="SBP", color="black")
ax.scatter(hypo_s["simulation_time"], hypo_s["MAP"], label="MAP", color="green")

# SHADED AREAS --
gray_start, gray_end = first_idh - 40, first_idh - 5
ax.axvspan(gray_start, gray_end,              color="gray", alpha=0.30)  # pre‑IDH
ax.axvspan(first_idh,  next_non_idh - 1,      color="red",  alpha=0.30)  # post‑IDH

# This helps to place the labels
y_min, y_max = (hypo_s[["SBP", "MAP"]].min().min(),
                hypo_s[["SBP", "MAP"]].max().max())
y_mid  = (y_max + y_min) / 2
y_off  = (y_max - y_min) * 0.4    

# ANNOTATIONS AND OTHER STUFF ---
ax.text(gray_start, y_mid - y_off, "$t_\mathrm{IDH}^{lower}$", ha="right", va="bottom")
ax.text(gray_end,   y_mid - y_off, "$t_\mathrm{IDH}^{upper}$", ha="center", va="bottom")
ax.axvline(first_idh, color="red")
ax.text(first_idh, y_mid, "IDH onset ($t_{\mathrm{IDH}}$)", color="red",
        rotation=90, ha="right", va="center")  
ax.axvline(60, color="gray", linestyle = '--')
ax.text(60, y_mid, "Dialysis Start ($t_{\mathrm{ds}}$)", color="gray",
        rotation=90, ha="right", va="center")  

# (t1, Xt1), (t2, Xt2), ...
inside = hypo_s[(hypo_s["simulation_time"] >= gray_start) &
                (hypo_s["simulation_time"] <= gray_end)]
for idx, (_, row) in enumerate(inside.iterrows(), start=1):
    t_i   = row["simulation_time"]
    sbp_i = row["SBP"]
    label = f"$\\vec{{X}}_{{t_{idx}}}$"
    # draw a thin gray vertical line at t_i
    ax.axvline(
        x=t_i,
        color="gray",
        linestyle="--",
        linewidth=0.8,
    )
    # place label at the SBP level
    ax.text(
        t_i,
        sbp_i-10,
        label,
        rotation=0,
        ha="left",
        va="bottom",
        color="gray",
        fontsize=11
    )
ax.set_xlabel("Simulation time (min)")
ax.set_ylabel("Pressure (mmHg)")
ax.set_title("Window Definition for Positive Class Labeling")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.savefig("Documents/Multimedia/Other/IDHNotation.pdf")
plt.tight_layout()
plt.show()



# %% #* ==================== PLOT: HYPOTENSIVE SESSIONS OF HYPOTENSIVE PATIENTS ====================
#: How many hypotensive sessions on average had those hypotensive patients?
temp0      = df_ssum[(df_ssum["IDH_01"] == True)].groupby("FIN_study_id")["IDH_01"].agg("sum")
des_hypo_pat     = temp0.describe()
totalIDHP    = temp0.index.nunique()

# bins of width 1
min_count = temp0.min()
max_count = temp0.max()
bins = range(min_count, max_count + 2)

fig, ax = characterization.StandardFig()
counts, bin_edges, patches = ax.hist(
    temp0.values,
    bins=bins,
    align='left',
    rwidth=0.8,
    color='green',
    edgecolor='black'
)
ax.set_xticks(bins)
ax.set_xlabel("IDH Sessions")
ax.set_ylabel("Total IDH Patients")
ax.set_title(f"Histogram of Total IDH Sessions per Patient (P={totalIDHP})")
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
fig.tight_layout()
fig.savefig(os.path.join(DIR_CHARACTERIZATION, "IDHSessionsPerPatient.pdf"))
fig.show()

# Cumulative histogram new figure
fig2, ax2 = characterization.StandardFig()
counts_cum, bin_edges_cum, patches_cum = ax2.hist(
    temp0.values,
    bins=bins,
    align='left',
    rwidth=0.8,
    cumulative=True,
    density=True,
    color='darkkhaki',
    edgecolor='black'
)
ax2.set_xticks(bins)
ax2.set_xlabel("IDH Sessions")
ax2.set_ylabel("Ratio of IDH Patients")
ax2.set_title(f"Cumulative Density Histogram of \nTotal IDH Sessions per Patient (P={totalIDHP})")
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax2.grid(axis='y', which='major', linestyle='--', alpha=0.7)
fig2.tight_layout()
fig2.savefig(os.path.join(DIR_CHARACTERIZATION, "IDHSessionsPerPatient_Cumulative.pdf"))
fig2.show()

# %% #* =============================== STORAGE2 ====================================================

session_labels = (
    df_dyn
    .groupby(["FIN_study_id", "session"])["Time_until_IDH_01"]
    .any()                          
    .reset_index(name="IDH")
)
temp = df_dyn.groupby("session")[["FIN_study_id"]].apply(lambda x: x.iloc[0]).reset_index()
out = pd.merge(
    temp,
    session_labels,
    on = ["FIN_study_id","session"],
    how="left"
)
with open(PATH_DESCRIPTIVE_TXT, 'w') as f:
    f.write(summary.to_string())

with open(PATH_SESS_PATIENTS_TXT, "w") as f:
    f.write(session_char_table.to_string())
    f.write("\n")
    f.write("First IDH Onset Time Info:")
    f.write(fIDH.to_string())
    f.write("\n\n")
    f.write(patient_char_table.to_string())
    f.write("\n\n")
    f.write("IDH Onsets per patient:\n")
    f.write(des_hypo_pat.to_string())

with open(PATH_COHORT_SUMMARY, "w") as f:
    f.write(report.to_string())

out.to_csv(PATH_IDH_SUM, index=False)   # IDH SUMMARY

# %% #* ============================== PLOT: SIGNALS TRANSFORMATIONS ===============================
# ---------------
SBP_dynamic = ["SBP"]
SBP_dynamic.extend(dynamic_processing.get_derived_column_names("SBP", "W",  dynamic_processing.phi_mapping_W.get("SBP")))
SBP_dynamic.extend(dynamic_processing.get_derived_column_names("SBP", "CW", dynamic_processing.phi_mapping_CW.get("SBP")))
# -------------
def hstep(ax, x, y, *, where="post", label=None, **plot_kw):
    """
    This Draws a step curve without vertical connectors.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if where not in {"post", "pre"}:
        raise ValueError("where must be 'post' or 'pre'")

    if where == "post":
        x0, x1, yy = x[:-1], x[1:], y[:-1]
    else:  # 'pre'
        x0, x1, yy = x[:-1], x[1:], y[1:]

    first = True
    for xi0, xi1, yi in zip(x0, x1, yy):
        ax.hlines(yi, xi0, xi1,
                  label=label if first else None,
                  **plot_kw)
        first = False
    return ax
# ------------------

def get_color(col: str) -> str:
    if col == "SBP":
        return "olive"
    if col.startswith("CW"):
        return "green"
    if col.startswith("W"):
        return "black"
    return "olive"

marker_base = "s"
marker_event = "x"

session = "6772_M24"
ses_df = df_dyn.loc[df_dyn["session"] == session]

op_cols = {}
for col in SBP_dynamic:
    if col == "SBP":
        continue
    op = col.split("_")[-1]
    op_cols.setdefault(op, []).append(col)

subplot_groups = [
    ("LWA", ["LWA"]),
    ("MAX", ["MAX"]),
    ("MIN", ["MIN"]),
    ("DELTA", ["DELTA"]),
    ("TWA", ["TWA"]),
    ("WSLOPE", ["WSLOPE"]),
    ("VAR", ["VAR"]),
]

n_panels = 1 + len(subplot_groups)
fig, axes = characterization.StandardFig(
    width="double",
    height=1.345,
    subplots=(n_panels, 1),
    constrained_layout=False,
    scale=0.9,
)

ms = plt.rcParams["lines.markersize"]

# --- SBP base signal ----------------------------------------------------------
base_color = get_color("SBP")
hstep(
    axes[0],
    ses_df["simulation_time"],
    ses_df["SBP"],
    where="post",
    color=base_color,
    linewidth=1.0,
    label="SBP",
)
axes[0].plot(
    ses_df["simulation_time"],
    ses_df["SBP"],
    linestyle="none",
    marker=marker_base,
    color=base_color,
    markersize=ms * 0.75,
)
axes[0].plot(
    ses_df.loc[ses_df["new_observation_SBP"], "simulation_time"],
    ses_df.loc[ses_df["new_observation_SBP"], "SBP"],
    linestyle="none",
    marker=marker_event,
    color=base_color,
    markersize=ms * 2,
)
axes[0].axvline(60, linestyle="--", color="gray", label="Dialysis Start")
axes[0].set_ylabel("SBP (mmHg)")
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(30))
axes[0].set_title(
    r"SBP transformations on the minute‑by‑minute grid ($t_m$) and the event‑grid time ($t_e$)"
)
axes[0].legend(loc="upper right")

# --- Remaining subplots -------------------------------------------------------
for i, (group_name, ops) in enumerate(subplot_groups, start=1):
    ax = axes[i]
    plotted_any = False
    for op in ops:
        for col in op_cols.get(op, []):
            c = get_color(col)
            hstep(
                ax,
                ses_df["simulation_time"],
                ses_df[col],
                where="post",
                color=c,
                linewidth=1.0,
                label=col,
            )
            ax.plot(
                ses_df["simulation_time"],
                ses_df[col],
                linestyle="none",
                marker=marker_base,
                color=c,
                markersize=ms * 0.75,
            )
            ax.plot(
                ses_df.loc[ses_df["new_observation_SBP"], "simulation_time"],
                ses_df.loc[ses_df["new_observation_SBP"], col],
                linestyle="none",
                marker=marker_event,
                color=c,
                markersize=ms * 2,
            )
            plotted_any = True

    if plotted_any:
        ax.axvline(60, linestyle="--", color="gray", label="Dialysis Start")
        ax.set_ylabel(group_name)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.legend(loc="upper right", fontsize="x-small")
    else:
        ax.set_visible(False)

for ax in axes:
    if ax.get_visible():
        ax.grid(alpha=0.3)
        ax.set_xlabel("Simulation time (min)")

fig.tight_layout()
fig.savefig(
    os.path.join(DIR_CHARACTERIZATION, "SBP_transformations.pdf"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%