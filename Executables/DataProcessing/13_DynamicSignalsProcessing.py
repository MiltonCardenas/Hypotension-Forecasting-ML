"""
---------------------------------------------------------------------------------------------------
This script is an updated version of the dynamic signals processing file,
the underlying mathematical background is presented in the Thesis Chapter about the signals processing/ 
Feature Engineering. The goal is to transform base signals X_base into 
derived signals X_derived by applying operators (or transformations) over different sets 
of time-series data. The code here aims to be clearer than the first version for future implementation.
----------------------------------------------------------------------------------------------------


 /$$$$$$$                                              /$$                                  
| $$__  $$                                            |__/                                  
| $$  \ $$ /$$   /$$ /$$$$$$$   /$$$$$$  /$$$$$$/$$$$  /$$  /$$$$$$$                        
| $$  | $$| $$  | $$| $$__  $$ |____  $$| $$_  $$_  $$| $$ /$$_____/                        
| $$  | $$| $$  | $$| $$  \ $$  /$$$$$$$| $$ \ $$ \ $$| $$| $$                              
| $$  | $$| $$  | $$| $$  | $$ /$$__  $$| $$ | $$ | $$| $$| $$                              
| $$$$$$$/|  $$$$$$$| $$  | $$|  $$$$$$$| $$ | $$ | $$| $$|  $$$$$$$                        
|_______/  \____  $$|__/  |__/ \_______/|__/ |__/ |__/|__/ \_______/                        
           /$$  | $$                                                                        
          |  $$$$$$/                                                                        
           \______/                                                                         
  /$$$$$$  /$$                               /$$                                            
 /$$__  $$|__/                              | $$                                            
| $$  \__/ /$$  /$$$$$$  /$$$$$$$   /$$$$$$ | $$  /$$$$$$$                                  
|  $$$$$$ | $$ /$$__  $$| $$__  $$ |____  $$| $$ /$$_____/                                  
 \____  $$| $$| $$  \ $$| $$  \ $$  /$$$$$$$| $$|  $$$$$$                                   
 /$$  \ $$| $$| $$  | $$| $$  | $$ /$$__  $$| $$ \____  $$                                  
|  $$$$$$/| $$|  $$$$$$$| $$  | $$|  $$$$$$$| $$ /$$$$$$$/                                  
 \______/ |__/ \____  $$|__/  |__/ \_______/|__/|_______/                                   
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
# %% #* =============================== DATA LOAD AND LIBRARIES ====================================
import numpy as np
import pandas as pd
import os
import warnings
from pandas.errors import DtypeWarning
import time, datetime
import sys
main_dir = "/ihome/rparker/mdc147/PhD_Project"; os.chdir(main_dir)
sys.path.append(os.path.join(main_dir, "Auxiliar_codes"))

from classes_repository import PathManager as P
from classes_repository import GlobalParameters as GP
from classes_repository import StandardNames as SN
from classes_repository import ProcessingAuxiliar as PA

import dynamic_processing
from  characterization import StandardFig

#: INSTANCES -----------------------------------------------
pm      = P().paths
gp      = GP()    
sn      = SN()
pa      = PA()
    
#: PARAMETERS  ---------------------------------------------
db     = "Hidenic15"      # gp.databases[0]
WSIZE   = 45          # Rolling window size in minutes

#: PATHS INPUT ---------------------------------------------------
PATH_DYN  =  pm['Summary'][db]['Dynamic Signals']
PATH_SUM  =  pm['Summary'][db]['Sessions Summary']

#: PATHS OUTPUT --------------------------------------------------
PROCESSED_DIR   = f"Data/Processed_datasets/Summary/{db}"
PATH_DYN_EVENTS = os.path.join(PROCESSED_DIR, "df_dyn_events.pkl")

#: DATA ----------------------------------------------------
df_dyn                  =  pd.read_pickle(PATH_DYN) # (1)
df_ssum                 =  pd.read_pickle(PATH_SUM) # (2)
dfs_extracted, dfs_raw  = dynamic_processing.load_data(db) # (3)
identifiers             = sn.identifier_signals() # This is a list of strings with all the identifiers (session, patient, ...)
# (1) This is the base dataframe with the timestamps from simulation start up to 15 min after dialysis end
# (2) This is the summary of characteristics of each session
# (3) Post- and Pre- Extraction data grouped in a dictionary.
#%% #* ===================  SIMULATION START AND DIALYSIS START IMPUTATION =========================
base_signals =  ['SBP', 'DBP', 'MAP','DC','Pulse', 'DT', 'RR', 'O2Sat', 'UF', 'UF_rate','BFR']
identifiers_2  =  ['FIN_study_id', 'session', 'Initial_chartdate', 'chartdate', 'simulation_time']
t0 = time.time()
for name, df in dfs_extracted.items():
    print(f"Processing {db}")
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
df_dyn = df_dyn[lambda df_dyn: (df_dyn["chartdate"]<= df_dyn["Final_chartdate"])] # (1)
for name, df in dfs_extracted.items():
    if name == "bp":
        associated_cols = ["SBP", "DBP", "MAP"]
    else:
        associated_cols = dynamic_processing.get_associated_columns(name)
    for base in associated_cols:
        lo, hi   = dynamic_processing.signals_limits[base]
        df[base] = df[base].clip(lower=lo, upper=hi)
# (1) Here I trim the latest 15 after dialysis END, this is a idea that is not longer useful
# (2) Values outside the boundaries were not analyzed in the extraction. This is a patch that
# revisits some of the limits of these signals.

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
    print(f"Processing Rolliwg windows for: {name} ...")
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
    
# %% #* =========================== ROLLING WINDOWS SIGNALS PROCESSING ==========================

#: Transformations assignation------------------------------------
phi_mapping_W ={ # these are the transformations applied to each of the base signals
    "SBP":      ["MAX", "MIN", "LWA", "TWA", "WSLOPE", "VAR"],
    "MAP":      ["MAX", "MIN", "LWA", "TWA", "WSLOPE", "VAR"],
    "Pulse":    ["MAX", "MIN", "LWA", "TWA", "WSLOPE", "VAR"],
    "RR":       ["MAX", "MIN", "LWA", "TWA", "WSLOPE", "VAR"],
    "O2Sat":    ["MAX", "MIN", "LWA", "TWA", "WSLOPE", "VAR"],
    "DC":       [],
    "DT":       [],
    "UF":       [], 
    "UF_rate":  ["TWA"],
    "BFR":      ["TWA"]
}
imputation_history = {}
for name, df in dfs_extracted.items():
    print(f"Computing rolling window transformations for {name}...")
    t0 = time.time()
    df = df.sort_values(by=["session", "simulation_time"],ascending=[True, True])
    
    for base in dynamic_processing.get_associated_columns(name): # One dataframe can have more than 1 base signal
        #: All the processing happens here
        df = (
            df.groupby("session", group_keys=False)[df.columns.tolist()]
            .apply(dynamic_processing.rolling_windows, base, WSIZE, phi_mapping_W)
        )

        #: Just if the operator was assigned to the base signal
        if "VAR" in phi_mapping_W[base]:
            #: Median-value imputation for W_X_VAR:
            df[f"W_{base}_VAR_median"]                  = False
            median                                      = df[f"W_{base}_VAR"].median()
            condition                                   = df[f"W_{base}_VAR"].isna()      
            df.loc[condition, f"W_{base}_VAR"]          = median
            df.loc[condition, f"W_{base}_VAR_median"]   = True
            imputation_history[f"W_{base}_VAR_median"]  = median
        
    #: The next just tracks how much time it takes 
    tf  = time.time()
    elapsed = tf - t0
    print(f"The total loop time is {datetime.timedelta(seconds = elapsed)}")
    
    #: Storage
    dfs_extracted[name] = df
    
# %% #* =========================== CUMULATIVE WINDOWS SIGNALS PROCESSING ==========================
#: Transformations assignation ---------------------------------------------------------------------
phi_mapping_CW ={ # these are the transformations applied to each of the base signals
    "SBP":      [ "DELTA", "TWA", "WSLOPE", "VAR"],
    "MAP":      [ "DELTA", "TWA", "WSLOPE", "VAR"],
    "Pulse":    [ "DELTA", "TWA", "WSLOPE", "VAR"],
    "RR":       [ "DELTA", "TWA", "WSLOPE", "VAR"],
    "O2Sat":    [ "DELTA", "TWA", "WSLOPE", "VAR"],
    "DC":       [ "DELTA", "TWA", "WSLOPE", "VAR"],
    "DT":       [ "DELTA", "TWA", "WSLOPE", "VAR"],
    "UF":       [], # The Delta UF in this case is the session's Total Uf which is the raw signal UF
    "UF_rate":  ["TWA"], #["TWA", "MAXSCORE"]
    "BFR":      ["TWA"]
}

for name, df in dfs_extracted.items():
    print(f"Computing cumulative window transformations for {name}...")
    t0 = time.time()
    df = df.sort_values(by=["session", "simulation_time"],ascending=[True, True]).reset_index(drop=True) #(1)
    for base in dynamic_processing.get_associated_columns(name): 
        #: All the processing happens here 
        df = (
            df.groupby("session", group_keys=False)[df.columns.tolist()]
            .apply(dynamic_processing.cumulative_windows, base, phi_mapping_CW)
        )
        
        if "VAR" in phi_mapping_CW[base]:   # Imputation is now applied at Final dyalysis signals merging to consider both databases
            #: Median-value imputation for W_X_VAR:
            df[f"CW_{base}_VAR_median"]                  = False
            condition                                    = df[f"CW_{base}_VAR"].isna()
            median                                       = df[f"CW_{base}_VAR"].median()
            df.loc[condition, f"CW_{base}_VAR"]          = median
            df.loc[condition, f"CW_{base}_VAR_median"]   = True
            imputation_history[f"CW_{base}_VAR_median"]  = median
            
    #: The next just tracks how much time it takes -------------------------
    tf  = time.time()
    elapsed = tf - t0
    print(f"The total loop time is {datetime.timedelta(seconds = elapsed)}")
    # (1) It is important to reset the index value, since the TWA here in the cumulative window uses
    # a intenger index assuming a monotonic increase of +1 between neighbor observations.
    #: Storage
    dfs_extracted[name] = df

# %% #* ============================== Event time-stamps and UF change =============================
# Here The idea is basically to merge all the different dataframes in dfs_extracted into  a single
# data frame "df_dyn" that stands for "dynamic signals". the idea is to forwardfill every single 
# missing value if necessary to have complete information at each time step where at least one of
# the signals was updated. 

# Columns to keep (These are columns from other analysis)
columns_to_keep = ['vasopressors', 'Ct[vasopressors]',
       'midodrine', 'Ct[midodrine]', 'albumin human', 'Ct[albumin human]',
       'mannitol', 'Ct[mannitol]']
temp            = df_dyn.copy()
total_dynamic   = []
new_signal_flags = []
for name, df in dfs_extracted.items():
    for base in dynamic_processing.get_associated_columns(name):
        W_predictors        = dynamic_processing.get_derived_column_names(base, "W",  phi_mapping_W.get(base))
        CW_predictors       = dynamic_processing.get_derived_column_names(base, "CW", phi_mapping_CW.get(base))
        all_predictors      = [base, *W_predictors, *CW_predictors]
        
        #: Here we gotta keep the imputation flags and new_observation flags along with the signals
        df[f"new_observation_{base}"]   = True
        imputation_flags                = [x for x in imputation_history.keys() if base in x]


        target_identifiers  = [x for x in identifiers if x in temp.columns.tolist()]
        target_columns      = target_identifiers + total_dynamic + columns_to_keep
        temp                = temp.drop(columns = all_predictors + [f"new_observation_{base}"] + imputation_flags, errors="ignore") # This resets the names
        temp                = pd.merge(
                                temp,
                                df[["session", "simulation_time", f"new_observation_{base}"] + all_predictors + imputation_flags],
                                on= ["session", "simulation_time"],
                                how="left"
                            )
        
        total_dynamic.extend(all_predictors) # These are all the derived dynamic signals for this study
        new_signal_flags.extend(f"new_observation_{base}")

# Here, I will keep all the rows with at least one new measurement in it. 
df_dyn_events = temp[temp[total_dynamic].notna().any(axis=1)]

#: Here I estimate the UF value at each simulation time to fill missing values ---------------------
def uf_imputation_ev(session: pd.DataFrame) -> pd.DataFrame:
    last_uf, last_ufr, last_t = None, None, None
    for idx, row in session.iterrows():
        t_now = row["simulation_time"]

        if pd.notna(row["UF"]):       # already has UF → update the state
            last_uf   = row["UF"]
            last_ufr  = row["UF_rate"]
            last_t    = t_now
            continue
        
        if last_uf is None:            # we haven't seen a valid UF yet
            session.at[idx, "UF"] = 0.0
        else:
            dt = t_now - last_t       
            session.at[idx, "UF"] = last_uf + last_ufr * dt
    return session
df_dyn_events=  (
    df_dyn_events.sort_values(["session", "simulation_time"])
        .groupby("session", group_keys=False)[df_dyn_events.columns]
        .apply(uf_imputation_ev)
)

#: Here I forward fill all other signals -----------------------------------------------------------
pre_dial0 = [] #All the transformations and missing signals before dialysis start will be set to 0 for UF_Rate and BFR
for base in ["UF_rate", "BFR"]:
    pre_dial0.extend(dynamic_processing.get_derived_column_names(base, "W",  phi_mapping_W.get(base)))
    pre_dial0.extend(dynamic_processing.get_derived_column_names(base, "CW", phi_mapping_CW.get(base)))
    pre_dial0.extend([base])

def session_fill(session: pd.DataFrame) -> pd.DataFrame:
    # Fills  all the missing values with LOCF
    # We will forwardfill everything, whenever a value was imputed, we will forward fill the 
    # flag so we can spot them in the processed dataframe #(just W, CW VAR signals!)
    target_ffill = total_dynamic + list(imputation_history.keys())
    session[target_ffill] = session[target_ffill].ffill()
    for col in pre_dial0: 
        session.loc[session["simulation_time"] <60, col] = 0
    return session

df_dyn_events = (
    df_dyn_events
      .sort_values(["session", "simulation_time"])
      .groupby("session", group_keys=False)[df_dyn_events.columns]
      .apply(session_fill)
)

#: Missing values check ----------------------------------------------------------------------------
for col in total_dynamic:
    missing = df_dyn_events[col].isna().sum()
    print(f"Signal {col} has {missing} missing values")


# %% #* ======================================= STORAGE ============================================
df_dyn_events = pa.columns_sorting(df_dyn_events)
df_dyn_events.to_pickle(PATH_DYN_EVENTS)

