
"""
---------------------------------------------------------------------------------------------------
This script is an updated version of the static signals processing file,
the underlying mathematical background is presented in the Thesis Chapter about the signals processing/ 
Feature Engineering. The goal is to transform base signals X_base into 
derived signals X_derived by applying operators (or transformations) over different sets 
of time-series data. The code here aims to be clearer than the first version for future implementation.
----------------------------------------------------------------------------------------------------

  /$$$$$$   /$$                 /$$     /$$                  /$$$$$$  /$$                               /$$          
 /$$__  $$ | $$                | $$    |__/                 /$$__  $$|__/                              | $$          
| $$  \__//$$$$$$    /$$$$$$  /$$$$$$   /$$  /$$$$$$$      | $$  \__/ /$$  /$$$$$$  /$$$$$$$   /$$$$$$ | $$  /$$$$$$$
|  $$$$$$|_  $$_/   |____  $$|_  $$_/  | $$ /$$_____/      |  $$$$$$ | $$ /$$__  $$| $$__  $$ |____  $$| $$ /$$_____/
 \____  $$ | $$      /$$$$$$$  | $$    | $$| $$             \____  $$| $$| $$  \ $$| $$  \ $$  /$$$$$$$| $$|  $$$$$$ 
 /$$  \ $$ | $$ /$$ /$$__  $$  | $$ /$$| $$| $$             /$$  \ $$| $$| $$  | $$| $$  | $$ /$$__  $$| $$ \____  $$
|  $$$$$$/ |  $$$$/|  $$$$$$$  |  $$$$/| $$|  $$$$$$$      |  $$$$$$/| $$|  $$$$$$$| $$  | $$|  $$$$$$$| $$ /$$$$$$$/
 \______/   \___/   \_______/   \___/  |__/ \_______/       \______/ |__/ \____  $$|__/  |__/ \_______/|__/|_______/ 
                                                                          /$$  \ $$                                  
                                                                         |  $$$$$$/                                  
                                                                          \______/                                   
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
import os
import time, datetime
main_dir = "/ihome/rparker/mdc147/PhD_Project"; os.chdir(main_dir)

from Auxiliar_codes.classes_repository import PathManager as P
from Auxiliar_codes.classes_repository import GlobalParameters as GP
from Auxiliar_codes.classes_repository import StandardNames as SN
from Auxiliar_codes.classes_repository import ProcessingAuxiliar as PA

from Auxiliar_codes import dynamic_processing
from Auxiliar_codes import static_processing

#: INSTANCES -----------------------------------------------
pm      = P().paths
gp      = GP()    
sn      = SN()
pa      = PA()
    
#: PARAMETERS  ---------------------------------------------
db          = "Hidenic15"      # gp.databases[0]
n_horizon   = 6

#: PATHS INPUT ---------------------------------------------
PROCESSED_DIR   = f"Data/Processed_datasets/Summary/{db}"
PATH_DYN_EVENTS = os.path.join(PROCESSED_DIR, "df_dyn_events.pkl")
PATH_DYN0       =  pm['Summary'][db]['Dynamic Signals']
PATH_SUM        = os.path.join(PROCESSED_DIR, "Sessions_Summary.pkl")

#: PATHS OUTPUT --------------------------------------------
# The only output is the processed dataframes saved back to PATH_DYN_EVENTS, PATH_SUM.

#: DATA ----------------------------------------------------
df_dyn0                 = pd.read_pickle(PATH_DYN0) 
df_dyn                  =  pd.read_pickle(PATH_DYN_EVENTS)  # (1)
df_ssum                 =  pd.read_pickle(PATH_SUM)         # (2)
dfs_extracted, dfs_raw  = dynamic_processing.load_data(db)  # (3)
identifiers             = sn.identifier_signals()           # (4)
# (1) This is the output from "Executables/DataProcessing/13_DynamicSignalsProcessing.py"
# with the dynamic signals processing. 
# (2) This is dialysis sessions summary file.
# (3) Extracted and raw dataframes
# (4) Identifiers like patient, session, elapsed_time, chartdate, Initial_chartdate ... list[str]

#%% #* ===================  SIMULATION START AND DIALYSIS START IMPUTATION =========================
base_signals =  ['SBP', 'DBP', 'MAP','DC','Pulse', 'DT', 'RR', 'O2Sat', 'UF', 'UF_rate','BFR']
identifiers_2  =  ['FIN_study_id', 'session', 'Initial_chartdate', "Final_chartdate", 'chartdate', 'simulation_time']
t0 = time.time()
for name, df in dfs_extracted.items():
    print(f"Imputing Initial simulation times and dialysis start for {name} when needed ...")
    if name in ["uf", "bfr"]:
        # There is no filling process for UF, 
        continue 

    target_cols     = [x for x in df.columns if  x in base_signals and x != "MAP"]  # (4)
    columns_keep    = [x for x in df.columns if  x in base_signals] 
    df              = df[identifiers_2 + columns_keep].copy()
    df["Imputed_value"] = False
    df_raw          = dfs_raw[name]
    out             = []
    
    #: ITERATION LOOP FOR EACH SESSION TO FIND THE VALUE AT SIMULATION START --------------
    for patientID, patient in df.groupby("FIN_study_id"):

        # These are all the observations int he original set that correspond to the patient
        dfr_subset = df_raw.loc[lambda df_raw: df_raw["FIN_study_id"]==patientID].copy()
        
        for session, subset in patient.groupby("session", as_index=False):    
            cond1 = (subset["simulation_time"] == 0).any()
            cond2 = (subset["simulation_time"] ==60).any()
            if cond1 and cond2:
                out.append(subset) # rows already present
                continue
        
            #: IDENTIFIERS OF THE CURRENT SESSION -----------------------------------------
            patient_id      = subset["FIN_study_id"].iloc[0]
            initial_date    = subset["Initial_chartdate"].iloc[0]
            sim_start_date  = initial_date - pd.Timedelta(minutes=60)
            new_row_ss      = {# structure of the new row at simulation start
                                "FIN_study_id":     patient_id,
                                "session":          subset["session"].iloc[0],
                                "chartdate":        sim_start_date,
                                "simulation_time":  0,
                                "Imputed_value":    True,
                                "Initial_chartdate": initial_date
                            }   
            new_row_ds      = { # structure of the new row at dialysis start
                            "FIN_study_id":     patient_id,
                            "session":          subset["session"].iloc[0],
                            "chartdate":        initial_date,
                            "simulation_time":  60,
                            "Imputed_value":    True,
                            "Initial_chartdate": initial_date
                            }
            
            #: SIMULATION START AND DIALYSIS START IMPUTATION -----------------------------
            temp_base2       = dfr_subset[dfr_subset["chartdate"] <= initial_date]
            temp_base        = temp_base2[temp_base2["chartdate"]<= sim_start_date]
        
            for target_col in target_cols: # (5)
                event_names     = dynamic_processing.get_event_name(target_col) # (6)
                
                #: Simulation start
                if not cond1:
                    temp_filtered   = temp_base[temp_base["event_name"].isin(event_names)]
                    if not  temp_filtered.empty: # If a previous observation exists, we use it
                        idx_max             = temp_filtered["chartdate"].idxmax()
                        locf_ss            = temp_filtered.loc[idx_max, "result_val"]
                        new_row_ss[target_col] = locf_ss
                    else:                  # No previous signal, NOCB needed
                        idx_min             = subset["chartdate"].idxmin()
                        nocb_ss            = subset.loc[idx_min, target_col] # (7)
                        new_row_ss[target_col] = nocb_ss
                        
                #: Dialysis start
                if not cond2:
                    pre_dialysis = subset[subset["chartdate"] <= initial_date]
                    if pre_dialysis.empty:
                        temp_filtered2   = temp_base2[temp_base2["event_name"].isin(event_names)]
                        if not  temp_filtered2.empty: # If a previous observation exists, we use it
                            idx_max                 = temp_filtered2["chartdate"].idxmax()
                            locf_ds                 = temp_filtered2.loc[idx_max, "result_val"]
                            new_row_ds[target_col]  = locf_ds
                        else:
                            idx_min             = subset[subset["chartdate"]>subset["Initial_chartdate"]]["chartdate"].idxmin()
                            nocb_ds             = subset.loc[idx_min, target_col] # (7)
                            new_row_ds[target_col] = nocb_ds
                    else:
                        idx_max = pre_dialysis["chartdate"].idxmax()
                        new_row_ds[target_col] = pre_dialysis.loc[idx_max, target_col]       
                            
            #: ONCE THE NEW NEW ROWS GATHERED, THEY ARE PLUGGED INTO THE SESSION ----------------------
            if not cond1:
                subset = pd.concat([subset, pd.DataFrame([new_row_ss])],
                                ignore_index=True)
            if not cond2:
                subset = pd.concat([subset, pd.DataFrame([new_row_ds])],
                                ignore_index=True)
            out.append(subset)
        
    # Once all sessions evaluated, we rewrite the values of the extracted dataframes:
    out_df              = pd.concat(out, ignore_index=True)
    for col in target_cols:
        out_df[col]     = pd.to_numeric(out_df[col], errors="coerce")
    dfs_extracted[name] = out_df
# (4) This steps select the base_signal column present in the current df. We explicitly exclude "MAP"
# since this value will be calculated from "SBP" and "MAP". fforwarded signals can trigger hypotensive
# episodes, however they were not labeled since they fell outside the simulation time. 
# (5) for the dataframe "bp" there are three columns, so each SBP and DBP must be processed independently.
# (6) These are all the events related to the current base signal in the raw queries from the database
# We dont make any distinction between them and just get the closest value to the initial time. 
# (7) We prioritize taking the NOCB from the extracted dataframes (even if we could get this value from
# the raw queries) since these values already underwent the extraction process. 
# (8) If a value is present before the dialysis onset and after the simulation start
# (9) If no value, we use the same as the simulation start (LOCF or NOCB depending on the case)
del dfs_raw
del df_raw            
import gc; gc.collect()
#: Here the same but for blowflowrate -------------------------------------------------------------
for name, df in dfs_extracted.items():
    if name in [ "bfr"]:
        print(f"Imputing Initial simulation times and dialysis start for {name} when needed ...")
        df = (
            df.sort_values(by=["session", "simulation_time"],ascending=[True, True])
            .groupby("session", group_keys=False)
            .apply(dynamic_processing.fill_bfr) # This function is defined in the auxiliary file
            .reset_index(drop=True)
        )
        
    dfs_extracted[name] = df
tf = time.time()
elapsed = tf - t0
print(f"The total loop time is {datetime.timedelta(seconds = elapsed)}")

#: There are three session that dont start at simulation_time = 60 so I will delete them
# since im not sure why
flags = (
    dfs_extracted["uf"].groupby("session")["simulation_time"]
    .agg(
        simulation_start = lambda t: (t == 0).any(),
        dialysis_start   = lambda t: (t == 60).any()
    )
    .reset_index()
)
s_keep =  flags.loc[lambda df: (df["dialysis_start"]), "session"].unique()
dfs_extracted["uf"] = dfs_extracted["uf"][dfs_extracted["uf"]["session"].isin(s_keep)]

# %% #* ================== UF, UFR, BFR NORMALIZATION BY PATIENT'S WEIGHT ==========================
weights         = df_ssum.loc[:, ["session", "Initial_weight"]].set_index("session")
weights_mapping = weights["Initial_weight"]
sessions_delete = weights_mapping.index[weights_mapping.isna()]
print(f"\n A total of {len(sessions_delete)} sessions will be eliminated since there is no weight associated")
for name, df in dfs_extracted.items():
    if name  in ["uf", "bfr"]:
        target = [x for x in df.columns.tolist() if x in ["UF", "UF_rate", "BFR"]]
        for col in target:
            df.loc[:, col] = df.loc[:, col] / df.loc[:, "session"].map(weights_mapping)
            df = df[~df["session"].isin(sessions_delete)].copy()
    else:
        print(f"Family {name} doesnt need any weight normalization ...")
    df = df[~df["session"].isin(sessions_delete)].copy()
    dfs_extracted[name] = df
df_dyn = df_dyn[~df_dyn["session"].isin(sessions_delete)].copy()

#%% #* ===========  ADDITIONAL PRE-PROCESSING STEPS (ADDITIONAL TO EXTRACTION PROCESS) =============
df_dyn = df_dyn[lambda df_dyn: (df_dyn["chartdate"]<= df_dyn["Final_chartdate"])] 
for name, df in dfs_extracted.items():
    if name == "bp":
        associated_cols = ["SBP", "DBP", "MAP"]
    else:
        associated_cols = dynamic_processing.get_associated_columns(name)
    for base in associated_cols:
        lo, hi   = dynamic_processing.signals_limits[base]
        df[base] = df[base].clip(lower=lo, upper=hi)
# MAP calculation: after filling the most recent SBP and DBP values, we need compute MAP from them.
# Although these filled values could trigger an IDH definition, they were only used for
# simulation start and dialysis start imputation. Therefore, they are not officially included in the
# IDH01 (intradialytic hypotension) or HYPO01 (any hypotensive episode occurring from one
# hour before up to dialysis start) lists.
df_bp                       = dfs_extracted["bp"].copy()
condition                   = df_bp["MAP"].isna()
df_bp.loc[condition, "MAP"] = ((df_bp.loc[condition, "SBP"] + 2*df_bp.loc[condition, "DBP"])/3)
dfs_extracted["bp"]         = df_bp #dataframe back in the dictionary
del df_bp #Free memory

# %% #* =================== SIMULATION START AND DIALYSIS START COMPLETENESS =======================
for name, df in dfs_extracted.items():
    print(f"Processing analysis windows for: {name} ...")
    s   = df["session"].nunique()
    #: CHECK THAT EVERY SINGLE DATAFRAME IS COMPLETE ------------------------------------------
    flags = (
        df.groupby("session")["simulation_time"]
        .agg(
            simulation_start = lambda t: (t == 0).any(),
            dialysis_start   = lambda t: (t == 60).any()
        )
        .reset_index()
    )
    if name in ["uf", "bfr"]:
        complete = flags.loc[lambda df: (df["dialysis_start"]), "session"].nunique()
    else:
        complete = flags.loc[lambda df: (df["simulation_start"]) & (df["dialysis_start"]), "session"].nunique()
    print(f"There are a total of {s} sessions with iformation registered for {name} "
          f"with a {complete/s*100} % having information at tss, and tds")
    

# %% #* ========================= DEMOGRAPHICS AND COMORBIDITIES: ==================================
total_static = []
f_list1  = ["age", "sex", "White_race", "black_race", "Initial_weight",
            "BMI", "IBW", "Blood_volume", "age_adj_comorbidity_score"] # (1)
target_identifiers  = [x for x in identifiers if x in df_ssum.columns.tolist()]
other_auxiliar_cols = ['Height','IDH_01', 'First_IDH_01', 'Duration_1st_IDH_01', 'Prior_IDH_01', 'Hypo_01',
                        'First_Hypo_01','Duration_1st_Hypo_01','Prior_Hypo_01']
df_static           =  df_ssum[target_identifiers + f_list1 + other_auxiliar_cols].copy()
for col in f_list1 + ["Height"]:
    df_static[col] = df_static[col].astype("Float64")
# (1) (These signals were appended to the summary file at each extraction file)
#: Missing values policy (demo and comor): -----
# Height --
height_medians = ( # Medians per age
    df_static.groupby(["sex"], group_keys=True)["Height"]
    .median()
    .reset_index()
    .set_index("sex")
    .loc[:, "Height"] # name of the series = Height
)
condition = df_static["Height"].isna()
df_static.loc[condition, "Height"] = df_static.loc[condition, "sex"].map(height_medians)
# BMI --
condition = df_static["BMI"].isna()
df_static.loc[condition, "BMI"] = df_static.loc[condition, "Initial_weight"]/((df_static.loc[condition, "Height"]/100)**2) # kg/m**2
# Blood Volume --
condition = df_static["Blood_volume"].isna()
sessions  = df_static.loc[condition, "session"].unique()
for s in sessions:
    mask = df_static["session"] == s
    subset = df_static.loc[mask].copy()
    filled = static_processing.blood_volume(subset)
    df_static.loc[mask, "Blood_volume"] = filled["Blood_volume"]
# IBW ----
condition = df_static["IBW"].isna()
sessions  = df_static.loc[condition, "session"].unique()
for s in sessions:
    mask = df_static["session"] == s
    subset = df_static.loc[mask].copy()
    filled = static_processing.ideal_body_weight(subset)
    df_static.loc[mask, "IBW"] = filled["IBW"]
# Age Charlson Score ----
# Missing comorbidity scores are set to increase +1 every decase above 40 yrs.
cut             = [0, 40, 50, 60,  70,  95] 
labels          = ["<= 40", "<= 50", "<= 60", "<= 70", ">= 71"]
ACCI_Mapping    = {
    "<= 40": 0,
    "<= 50": 1,
    "<= 60": 2,
    "<= 70": 3,
    ">= 71": 4,   
}
df_static["age_adj_comorbidity_score"] = (
    df_static["age_adj_comorbidity_score"].astype("Int64")
)
df_static["AACI_Group"] = pd.cut(
    df_static["age"], bins=cut, right=True, labels=labels
)
mask = df_static["age_adj_comorbidity_score"].isna()
df_static.loc[mask, "age_adj_comorbidity_score"] = (
    df_static.loc[mask, "AACI_Group"]
              .map(ACCI_Mapping)
              .astype("Int64")
)
total_static.extend(f_list1)

# %% #* =================================== OTHER SIGNALS ==========================================

#: (Time since last Session) ---
temp = (
    df_static.groupby("FIN_study_id")[df_static.columns]
    .apply(
        lambda session: (session.loc[:,"Initial_chartdate"] - session.loc[:,"Final_chartdate"].shift())
        .fillna(pd.Timedelta(0))
    )
    .reset_index(level=0, drop=True)
)
df_static["LS_TSD"] =  temp.dt.total_seconds()/3600

#: Total Dialysis Sessions ---
df_static["TOT_DIALYSIS"] = df_static["session"].map(df_ssum.set_index("session")["Patient session"])
total_static.extend(["LS_TSD", "TOT_DIALYSIS"])

#: Quality Mask --------------
min_requirements  = [
    'UF_confidence',
    'Initial_weight_confidence',
    'SBP_confidence',
]
df_static["IDH_01"]  = df_static["IDH_01"].fillna(False)
df_ssum["QM"]        = df_ssum[min_requirements].isin(["dg1", "dg2"]).all(axis=1)
df_static["QM"]      = df_static["session"].map(df_ssum.set_index("session")["QM"])
df_static["idh_qm"]  = df_static["IDH_01"].fillna(False).astype(int)  * df_static["QM"].fillna(False).astype(int)

#: L6S Incidence ------------
def calc_incidence(patient: pd.DataFrame, horizon: int) -> pd.Series:
    incidence = pd.Series(index=patient.index, dtype=float)
    for idx, row in patient.iterrows():
        current_session = row.at["TOT_DIALYSIS"]

        if current_session == 0: 
            incidence.at[idx] = 0.0
        else:
            K_QM = patient[
                (patient["TOT_DIALYSIS"] < current_session) &
                (patient["QM"])
            ]
            K_hist = K_QM.tail(horizon)
            total = len(K_hist)
            if total > 0:
                IDH = K_hist["IDH_01"].astype(int).sum()
                incidence.at[idx] = IDH / total
            else:
                incidence.at[idx] = 0
    return incidence
df_static["L6SIDH_INC"] = (
    df_static
      .groupby("FIN_study_id")[df_static.columns]
      .apply(lambda grp: calc_incidence(grp, n_horizon))
      .reset_index(level=0, drop=True)
)
#: TOT IDH ---------- (No constrained by the horizon)
def tot_IDH(patient: pd.DataFrame) -> pd.Series:
    """
    For each session in `patient`, return the total number of prior IDH events
    among *all* previous QM-approved sessions (excluding the current one).
    """
    total_idh = pd.Series(index=patient.index, dtype=int)

    for idx, row in patient.iterrows():
        current_session = row.at["TOT_DIALYSIS"]

        if current_session == 0:
            total_idh.at[idx] = 0
        else:
            K_QM = patient[
                (patient["TOT_DIALYSIS"] < current_session) &
                (patient["QM"])
            ]
            total_idh.at[idx] = int(K_QM["IDH_01"].astype(int).sum())
    return total_idh
df_static["TOT_IDH"] = (
    df_static
      .groupby("FIN_study_id")[df_static.columns]
      .apply(tot_IDH)
      .reset_index(level=0, drop=True)
)
total_static.extend(["TOT_IDH", "L6SIDH_INC"])
subset = df_static[["session", "FIN_study_id", "IDH_01", "QM", "idh_qm", "TOT_IDH"]]


# %% #* ============================ CURRENT SESSION TRANSFORMATIONS ===============================

#: Transformations list
phi_mapping_DT ={ # these are the transformations applied to each of the base signals
    "SBP":      [ "INITIAL", "DELTA", "TWA", "WSLOPE", "VAR", "MINSCORE", "MAXSCORE"],
    "MAP":      [ "INITIAL", "DELTA", "TWA", "WSLOPE", "VAR", "MINSCORE", "MAXSCORE"],
    "Pulse":    [ "INITIAL", "DELTA", "TWA", "WSLOPE", "VAR", "MINSCORE", "MAXSCORE"],
    "RR":       [ "INITIAL", "DELTA", "TWA", "WSLOPE", "VAR"],
    "O2Sat":    [ "INITIAL", "DELTA", "TWA", "WSLOPE", "VAR"],
    "DC":       [ "INITIAL", "DELTA", "TWA", "WSLOPE", "VAR"],
    "DT":       [ "INITIAL", "DELTA", "TWA", "WSLOPE", "VAR"],
    "UF":       [ "DELTA" ],
    "UF_rate":  [ "TWA", "MAXSCORE"],
    "BFR":      [ "TIMEt", "TWA"]            # time t can be both for Laste session's analysis and current session
}
 
#: Thresholds for MAX and MIN SCORES --------------------------------------------------
AUT_threshold_mapping = {
    "UF_rate": {
        "MAX": 0.167,
    },
    "SBP": {
        "MAX": 130,
        "MIN": 100,
    },
    "MAP": {
        "MAX": 97,
        "MIN": 70,
    },
    "Pulse": {
        "MAX": 90,
        "MIN": 60,
    },
}

#: Aggregated signals over dialysis time are calculated here -------------------------
dfs_transformed = {}               
for name, df in dfs_extracted.items():
    print(f"Computing Dialysis-Time transformations for {name}…")
    t0 = time.time()
    df = (df.sort_values(["session", "simulation_time"])
            .reset_index(drop=True))                        

    per_base_results = []                                   
    for base in dynamic_processing.get_associated_columns(name):
        res = (
            df.groupby("session", group_keys=True)[df.columns]
              .apply(dynamic_processing.dialysis_time_transformations,
                     base=base,
                     mapping=phi_mapping_DT,
                     aut_thresholds=AUT_threshold_mapping)
        )
        if "VAR" in phi_mapping_DT[base]:
            condition = (res[f"S_{base}_VAR"].isna())
            res.loc[condition, f"S_{base}_VAR"] = res[f"S_{base}_VAR"].median()
        per_base_results.append(res)

    # combine all bases side-by-side; add 'session' as an ordinary column
    new_dataframe = (
        pd.concat(per_base_results, axis=1)
          .reset_index()                                    
    )

    dfs_transformed[name] = new_dataframe                   #
    tf = time.time()
    print("Time elapsed:", datetime.timedelta(seconds=tf - t0))

#: All derived predictors into a single dataframe -------------------------------------
df_auxiliar = df_static[["FIN_study_id","session"]].copy()
list2 = []
for name, df in dfs_transformed.items():
    for base in dynamic_processing.get_associated_columns(name):
        S_signals       = dynamic_processing.get_derived_column_names(base, "S", phi_mapping_DT.get(base))
        df_static       = df_static.drop(columns=S_signals, errors="ignore") # this just resets the columns
        df_static     = pd.merge(
            df_static,
            df[["session"] + S_signals],
            left_on="session",
            right_on="session",
            how= "left" 
        )
        list2.extend(S_signals)
    
total_static.extend(["S_BFR_TIMEt"]) # The only predictor directly used from the current session
# this value is interpreted as the prescribed blood-flow rate (and must match the BFR at 5 minutes)

#: LS Signals -------------------------------------------------------------------------
def get_last_session_value(patient: pd.DataFrame, base: str) -> pd.DataFrame:
    col = f"L{base}"                      # reuse the helper’s column name
    for idx, row in patient.iterrows():
        current_session = row.at["TOT_DIALYSIS"]
        if current_session == 0:
            patient.at[idx, col] = pd.NA  # pd.NA works with dtype=object
            continue

        K_QM = patient[
            (patient["TOT_DIALYSIS"] < current_session) & (patient["QM"])
        ]
        if K_QM.empty:
            patient.at[idx, col] = pd.NA
        else:
            value = K_QM.loc[K_QM["TOT_DIALYSIS"].idxmax(), base]
            if isinstance(value, (pd.Series, np.ndarray)):
                value = value.item()
            patient.at[idx, col] = value
    return patient

df_static = df_static.reset_index(drop=True)
for name in list(dfs_transformed.keys()):
    for base in dynamic_processing.get_associated_columns(name):
        S_signals       = dynamic_processing.get_derived_column_names(base, "S", phi_mapping_DT.get(base))
        for signal in S_signals:
            df_static = (
                df_static
                .groupby("FIN_study_id", group_keys=False)[df_static.columns]
                .apply(get_last_session_value, signal)
                .reset_index(drop=True)
            )
list3 = [f"L{x}" for x in list2]
total_static.extend(list3)
subset = df_static[["FIN_study_id", "session", "QM", "S_Pulse_TWA", "LS_Pulse_TWA"]]

#: L6S Signals ------------------------------------------------------------------------
phi_mapping_L6S = {          # Transformation rules for L6S
    "DELTA":     ["AVG"],
    "TWA":       ["AVG"],
    "MAXSCORE":  ["MAX"],
    "MINSCORE":  ["MAX"],
}

def average(set_, base):
    return float(set_.loc[:, base].mean())

def max_(set_, base):
    return float(set_.loc[:, base].max())

operators_dispatch = {       # Map operator names to callables
    "AVG": average,
    "MAX": max_,
}

def get_last6_sessions_value(
        patient: pd.DataFrame,
        base: str,                # ← a *column name* to aggregate
        operators: list[str],     # ← list such as ["AVG"] or ["AVG","MAX"]
        horizon: int = 6,
    ) -> pd.DataFrame:

    for idx, row in patient.iterrows():
        current_session = row.at["TOT_DIALYSIS"]

        for op in operators:
            patient.at[idx, f"L6{base}_{op}"] = np.nan

        if current_session == 0:
            continue

        K_QM = patient[
            (patient["TOT_DIALYSIS"] < current_session) &
            (patient["QM"])
        ]
        if K_QM.empty:
            continue

        K_QM = (
            K_QM.sort_values("TOT_DIALYSIS", ascending=False)
                .head(horizon)
        )
        #: Here we apply all the operators
        for op in operators:
            func = operators_dispatch[op]
            patient.at[idx, f"L6{base}_{op}"] = func(K_QM, base)

    return patient

for name in list(dfs_extracted.keys()):
    for base in dynamic_processing.get_associated_columns(name):
        S_signals = dynamic_processing.get_derived_column_names(
            base, "S", phi_mapping_DT.get(base)
        )
        for signal in S_signals:
            # does this signal belong to ∪{DELTA, TWA, MAXSCORE, MINSCORE} ?
            family = next(
                (fam for fam in phi_mapping_L6S if fam in signal.upper()),
                None,
            )
            if family is None:
                continue # not an L-6 target
            ops_to_apply = phi_mapping_L6S[family]
            df_static = (
                df_static
                .groupby("FIN_study_id", group_keys=False)[df_static.columns]
                .apply(get_last6_sessions_value, signal, ops_to_apply)
                .reset_index(drop=True)
            )

targets = phi_mapping_L6S.keys()
list4 = [
    f"L6{signal}_{op}"
    for signal in list2            
    for fam, ops in phi_mapping_L6S.items()
    if fam in signal.upper()          
    for op in ops                   
]
total_static.extend(list4)
subset = df_static[["FIN_study_id","session", "QM", "S_UF_rate_TWA", "L6S_UF_rate_TWA_AVG","S_UF_rate_MAXSCORE", "L6S_UF_rate_MAXSCORE_MAX"] ]
#: LIDHO Signals ----------------------------------------------------------------------
phi_mapping_IDHO ={ # these are the transformations applied to each of the base signals
    "SBP":      [ "INITIAL", "DELTA", "TWA",  "VAR"],
    "MAP":      [ "INITIAL", "DELTA", "TWA",  "VAR"],
    "Pulse":    [ "INITIAL", "DELTA", "TWA",  "VAR"],
    "RR":       [ "INITIAL", "DELTA", "TWA",  "VAR"],
    "O2Sat":    [ "INITIAL", "DELTA", "TWA",  "VAR"],
    "DC":       [ "INITIAL", "DELTA", "TWA",  "VAR"],
    "DT":       [ "INITIAL", "DELTA", "TWA",  "VAR"],
    "UF":       [ "DELTA" ],
    "UF_rate":  [ "TWA", "MAXSCORE"],
    "BFR":      [ "TWA", "TIMEt",] 
}

IDH_onset_info = df_ssum.set_index("session")["First_IDH_01"] # This is in minutes since dialysis start
IDH_sessions   = df_ssum[df_ssum["IDH_01"]==True]["session"].unique().tolist()

#: The processing happens here
dfs_transformed = {}               
for name, df in dfs_extracted.items():
    print(f"Computing Dialysis-Time transformations for {name}…")
    t0 = time.time()
    df = (df.sort_values(["session", "simulation_time"])
            .reset_index(drop=True))         
    
    # I will append this information to all dataframes stored since this is needed for the analysis
    df["First_IDH_01"]    = df["session"].map(IDH_onset_info)    
    df["First_IDH_01"]       = df["First_IDH_01"].apply(pd.to_numeric) 
    df["simulation_time"] = df["simulation_time"].apply(pd.to_numeric)
    per_base_results = []    
                                   
    for base in dynamic_processing.get_associated_columns(name):
        df[base]              = df[base].apply(pd.to_numeric)
        res = (
            df.groupby("session", group_keys=True)[df.columns]
              .apply(dynamic_processing.cumulative_windows_up_to_IDH,
                     base=base,
                     mapping=phi_mapping_IDHO,
                     aut_thresholds=AUT_threshold_mapping)
        )
        
        if "VAR" in phi_mapping_IDHO[base]:
            condition = (res[f"IDHO_{base}_VAR"].isna() ) & (res[f"IDHO_{base}_VAR"].index.isin(IDH_sessions))
            res.loc[condition, f"IDHO_{base}_VAR"] = res[f"IDHO_{base}_VAR"].median()
        
        per_base_results.append(res)

    # combine all bases side-by-side; add 'session' as an ordinary column
    new_dataframe = (
        pd.concat(per_base_results, axis=1)
          .reset_index()                                    
    )
    dfs_transformed[name] = new_dataframe                
    tf = time.time()
    print("Time elapsed:", datetime.timedelta(seconds=tf - t0))

# Here I append the value to the current static frame:
list5  = []
for name, df in dfs_transformed.items():
    for base in dynamic_processing.get_associated_columns(name):
        IDHO_signals       = dynamic_processing.get_derived_column_names(base, "IDHO", phi_mapping_IDHO.get(base))
        df_static       = df_static.drop(columns=IDHO_signals, errors="ignore") # this just resets the columns
        df_static     = pd.merge(
            df_static,
            df[["session"] + IDHO_signals],
            left_on="session",
            right_on="session",
            how= "left" 
        )
        list5.extend(IDHO_signals)

def get_last_session_value(patient: pd.DataFrame, base: str) -> pd.DataFrame:
    col = f"L{base}"                      # reuse the helper’s column name
    for idx, row in patient.iterrows():
        current_session = row.at["TOT_DIALYSIS"]
        if current_session == 0:
            patient.at[idx, col] = pd.NA  # pd.NA works with dtype=object
            continue

        K_QM = patient[
            (patient["TOT_DIALYSIS"] < current_session) & (patient["idh_qm"]) #idh_qm is both idh positive and good quality session
        ]
        if K_QM.empty:
            patient.at[idx, col] = pd.NA
        else:
            value = K_QM.loc[K_QM["TOT_DIALYSIS"].idxmax(), base]
            if isinstance(value, (pd.Series, np.ndarray)):
                value = value.item()
            patient.at[idx, col] = value
    return patient

df_static = df_static.reset_index(drop=True)
for name in list(dfs_transformed.keys()):
    for base in dynamic_processing.get_associated_columns(name):
        IDHO_signals       = dynamic_processing.get_derived_column_names(base, "IDHO", phi_mapping_IDHO.get(base))
        for signal in IDHO_signals:
            df_static = (
                df_static
                .groupby("FIN_study_id", group_keys=False)[df_static.columns]
                .apply(get_last_session_value, signal)
                .reset_index(drop=True)
            )
list6 = [f"L{x}" for x in list5]    
total_static.extend(list6)
# %% #* =============================== STORAGE ====================================================

df_dyn = df_dyn.drop(columns=total_static, errors="ignore")
df_dyn = pd.merge(
    df_dyn, 
    df_static[["session"] + total_static],
    on = "session",
    how = "left"
)

df_ssum = df_ssum.drop(columns=total_static, errors="ignore")
df_ssum = pd.merge(
    df_ssum, 
    df_static[["session"] + total_static],
    on = "session",
    how = "left"
)
df_ssum.to_pickle(PATH_SUM)
df_dyn.to_pickle(PATH_DYN_EVENTS)
print("Done")
