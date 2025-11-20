"""
This module provides auxiliary functions and custom operators designed to transform raw 
time-series data into engineered predictive features. It implements the core logic for 
moving-window analysis (rolling statistics) and facilitates the extraction of historical 
context from previous dialysis sessions to capture long-term patient trends.
V.1.0 Author: Milton Dario Cardenas
"""

#: DEPENDENCIES -------------------------------------
import numpy as np
import pandas as pd
import sys, os
main_dir = "/ihome/rparker/mdc147/PhD_Project"; os.chdir(main_dir)
sys.path.append(os.path.join(main_dir, "Utils"))
from classes_repository import PathManager as P

# %% #* ====================================== GENERAL FUNCTIONS ===================================
def get_event_name(base:str):
    
    """
    This uses the name of the column "base" and retrieve all the event_names related
    """
    mapping = {
        "SBP":      ["Systolic BP", "Arterial Systolic Pressure", "Arterial Systolic Pr", "Systolic BP #2" ],
        "DBP":      ["Diastolic BP", "Arterial Diastolic Pressure", "Arterial Diastolic P"],
        "DC" :      ["Dialysis Conductivity (mS/cm)"],
        "Pulse":    ["Pulse"],
        "DT":       ["Dialysis Temperature"],
        "RR":       ["Respiratory Rate", "Total respiratory rate", "Respiratory rate #2", "Total Respiratory Rate #2"],
        "O2Sat":    ["O2 Saturation"]
    }
    
    return mapping.get(base)
    
    
def get_associated_columns(name):    
    mapping = {    
        'uf':       ["UF","UF_rate"],
        'bfr':      ["BFR"],
        'bp':       ["SBP", "MAP"],
        'pulse':    ["Pulse"],
        'dt':       ["DT"],
        'rr':       ["RR"],
        'o2sat':    ["O2Sat"],
        'dc':       ["DC"],
    }
    return mapping.get(name)


def load_data(db:str):
    """
    This loads the raw signals and extracted dataframes into two different dictionaries,
    the input is the dataframe ID: either "Mladi24" or "Hidenic15".
    """
    #: INSTANCES -----------------------------------------------
    pm = P().paths

    #: DATA DIRECTLY FROM THE DATABASES -------------------------
    raw_vars = {    # Name given in the structUre paths / version
        'bfr':    ('Dialysis Pulse',                  'v1'),
        'bp':       ('Dialysis Pressures (compiled)',   'v1'),
        'pulse':    ('Dialysis Pulse',                  'v1'),
        'dt':       ('Dialysis Temperature',            'v1'),
        'rr':       ('Respiratory Rate',                'v1'),
        'o2sat':    ('Oxygen Saturation',               'v1'),
        'dc':       ('Dialysis Conductivity',           'v1'),
    }

    #: DATA AFTER EXTRACTION ----------------------------------
    extracted_vars = {
        'uf':  ('Dialysis UF Total',             'v1'),
        'bfr': ('Dialysis Blood Flow',           'v1'),
        'bp':  ('Dialysis Pressures (compiled)', 'v2'),
        'pulse': ('Dialysis Pulse',              'v1'),
        'dt':  ('Dialysis Temperature',          'v1'),
        'rr':  ('Respiratory Rate',              'v1'),
        'o2sat': ('Oxygen Saturation',           'v1'),
        'dc':  ('Dialysis Conductivity',         'v1'),
    }

    #: FILE PATHS CREATION --------------------------------
    # This just converts previous informations into the format 
    # pm['Raw Queries'][db][Name][version] that pm understands and return the path. 
    raw_paths = {
        var: pm["Raw Queries"][db][name][version]
        for var, (name, version) in raw_vars.items()
    }

    ext_paths = {
        var: pm["Extracted Data"][db][name][version]
        for var, (name, version) in extracted_vars.items()
    }

    #: DICTIONARY WITH ALL THE RAW DATA FRAMES -----------
    dfs_raw  = {
        var: (
            pd.read_csv(raw_paths[var])
            .assign(
                chartdate   =lambda df: pd.to_datetime(df["chartdate"]),
                FIN_study_id=lambda df: df["FIN_study_id"].astype(str),
                result_val  =lambda df: pd.to_numeric(df["result_val"], errors="coerce")
            )
        )
        for var in raw_paths.keys()
    }

    #: DICTIONARY WITH ALL THE EXTRACTED DATA FRAMES -----
    dfs_extracted  = {
        var: (
            pd.read_pickle(ext_paths[var])
            .assign(
                chartdate=lambda df: pd.to_datetime(df["chartdate"]),
                FIN_study_id=lambda df: df["FIN_study_id"].astype(str)
            )
        )
        for var in ext_paths.keys()
    }
    return dfs_extracted, dfs_raw

def fill_bfr(session, base="BFR"):
    """
    Fill (or create) the BFR observation at dialysis start (simulation_time==60).
    """

    pre_obsv  = session.loc[session["simulation_time"] < 60, base]
    dial_obsv = session.loc[session["simulation_time"] > 60, base]
    
    #: CASE 1 – a row already exists at t = 60 --------------------------------
    has_row_60 = (session["simulation_time"] == 60).any()
    if has_row_60:
        mask_60 = session["simulation_time"] == 60
        if session.loc[mask_60, base].notna().any():
            session = session.loc[session["simulation_time"] >= 60].reset_index(drop=True)
            return session
        if not pre_obsv.dropna().empty:
            fill_val = pre_obsv.dropna().iloc[-1]            # forward fill
        elif not dial_obsv.dropna().empty:
            fill_val = dial_obsv.dropna().iloc[0]            # backward fill
        else:
            print("Warning! Session has no BFR measurements at all.")
            session = session.loc[session["simulation_time"] >= 60].reset_index(drop=True)
            return session

        session.loc[mask_60, base] = fill_val
        session = session.loc[session["simulation_time"] >= 60].reset_index(drop=True)
        return session

    #: CASE 2 – no row at t = 60 ➜ create one -------------------------------
    if not pre_obsv.dropna().empty:
        fill_val   = pre_obsv.dropna().iloc[-1]              # forward fill
        source_idx = pre_obsv.dropna().index[-1]
    elif not dial_obsv.dropna().empty:
        fill_val   = dial_obsv.dropna().iloc[0]              # backward fill
        source_idx = dial_obsv.dropna().index[0]
    else:
        print("Warning! Session has no BFR measurements at all.")
        session = session.loc[session["simulation_time"] >= 60].reset_index(drop=True)
        return session

    new_row = session.loc[source_idx].copy()
    new_row["simulation_time"] = 60
    new_row["chartdate"]       = session["Initial_chartdate"].iloc[0]
    new_row[base]              = fill_val
    new_row["Imputed_BFR"]     = True
    session = pd.concat([session, new_row.to_frame().T], ignore_index=True)
    session = session.sort_values("simulation_time").reset_index(drop=True)
    
    #: Finally, here I keep just the dialysis information --------------------------
    session = session.loc[session["simulation_time"] >= 60].reset_index(drop=True)
    return session

def get_derived_column_names(base, type_, operators_list):

    """" 
    This retunrs the names of tha base, type of window and all the operators listed in 
    operators_list
    """
    names = []
    for operator in operators_list:
        operator_name = f"{type_}_{base}_{operator}"
        names.append(operator_name)
    return names


# %% #*============================= Dynamic signals operators storage =============================

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


# %% #*========================================== Operators ========================================
def operator_MAX(set_, base):
    """ 
    This calcualtes the maximum value of the given set
    """
    max_ = set_[base].max()
    return float(max_)

def operator_MIN(set_, base):
    """ 
    This calcualtes the minimum value of the given set
    """
    min_ = set_[base].min()
    return float(min_)

def operator_LWA(set_, base, t_low):
    """ 
    This calculates the linearly weighted average
    ws = windowsize #!Check wj   = max(0, set_["simulation_time"] - ws) + 1
    """
    X    = set_[base] 
    t    = set_["simulation_time"] 
    wj   = t - t_low + 1
    lwa  = np.sum(X * wj) / np.sum(wj)
    return float(lwa)

def operator_TWA(set_, base, edge_case=None):
    """
    This calculates the time-weighted average.
    """
    if edge_case:
        return set_[base].iloc[0] # value at t_ss
    else:
        t0              = set_["simulation_time"].min()
        tn              = set_["simulation_time"].max()
        Xj              = set_[base]
        tj              = set_["simulation_time"]
        deltaTj         = (tj.shift(-1) - tj).fillna(0)         # (1)
        twa             =  np.sum(Xj * deltaTj) / (tn - t0)  
        if   tn - t0 ==0:
            print(f"ZEEROO DIVISON B")
        # (1) deltaTj will always be zero for the last observation and it wont be included
        #    in the time-weighted calculus tj.shift(-1) > tj and time will be positive
        return float(twa)
    
def operator_WSLOPE(set_, base, edge_case=None):
    """
    This calculates the time-weighted average.
    """
    if edge_case:
        return 0 # value at t_ss
    else:
        t0      = set_["simulation_time"].min()
        tj      = set_["simulation_time"]
        Xj      = set_[base]
        wj      = tj - t0 + 1
        tw      = (wj*tj).sum() / wj.sum()
        Xw      = (wj*Xj).sum() / wj.sum()
        wslope  = (wj*(tj - tw)*(Xj - Xw)).sum() / (wj*((tj - tw)**2)).sum()
        return float(wslope)
    
def operator_VAR(set_, base, edge_case=None):
    """ 
    This operator calculates the variability
    """
    n = np.shape(set_)[0]
    if edge_case or (n<2):
        return np.nan
    else:
        Xj      = set_[base]
        Xj_prev =  Xj.shift()
        var     = (Xj - Xj_prev).fillna(0).abs().sum() / (n-1)
        return float(var)
    
    
def operator_DELTA(set_, base, edge_case=None, custom_X0=None):
    """ 
    This function computes the total change of the base features between 
    the minimum and maximum time inside the window. 
    """
    if edge_case:
        return 0
    else:
        if custom_X0:
            X0  = custom_X0
        else:
            X0  = set_.loc[set_["simulation_time"].idxmin(), base]
            
        Xn      = set_.loc[set_["simulation_time"].idxmax(), base]    
        delta   = Xn - X0
        return float(delta)
    
def volume_estimation(Xprev_star, t_low):
    """ 
    This takes the nearest previous measurement and forecast the Ultrafiltration value at
    the window start depending on the elapsed time since the last observation and its UF
    """
    UF_prev      = Xprev_star["UF"].iloc[0]
    UFR_prev     = Xprev_star["UF_rate"].iloc[0]
    deltaT       = t_low -  Xprev_star["simulation_time"].iloc[0] # This is always positive
    UF0          = UF_prev + UFR_prev * deltaT
    return float(UF0)
    
def operator_initial(set_, base):
    """
    This operator calculates the value of the earliest time
    """
    idxmin = set_["simulation_time"].idxmin()
    X0 = set_.loc[idxmin, base]
    return float(X0)
    
def operator_AUT_max(set_, base, threshold):
    """
    Area ABOVE `threshold` (AUT_max) for signal `base`.
    """
    Xi   = set_[base].to_numpy(float)
    dt   = (set_["simulation_time"] - set_["simulation_time"].shift()).fillna(0).to_numpy(float)

    # element-wise positive excess × width
    return float(np.sum(np.maximum(0.0, Xi - threshold) * dt))

def operator_AUT_min(set_, base, threshold):
    """
    Area BELOW `threshold` (AUT_min) for signal `base`.
    """
    if len(set_) == 1:                       # single observation ⇒ assume 1-min width
        xi = set_[base].iloc[0]
        return float(max(0.0, threshold - xi) * 1.0)

    Xi   = set_[base].to_numpy(float)
    dt   = (set_["simulation_time"] - set_["simulation_time"].shift()).fillna(0).to_numpy(float)

    return float(np.sum(np.maximum(0.0, threshold - Xi) * dt))


def operator_TIMEt(set_, base, time=5):
    """
    Return the value of `base` at the closest observation *at or before*
    t = 60 + time.  If no such obs exists, returns nan.
    """
    cutoff = 60 + time
    sub    = set_[set_["simulation_time"] <= cutoff]
    if sub.empty:
        return float(np.nan)
    # find the row whose simulation_time is maximal but <= cutoff
    idxmax      = sub["simulation_time"].idxmax()
    target_base = set_.at[idxmax, base]
    return float(target_base)
    
    

# %% #*============================= Rolling Window Transformations ==============================
#: This is just a helper that makes the code able to run any transformation to any of the
#: Base signals depending on the list declared in the variable "phi_mapping_X"
operators_dispatch = {
    "MAX":    operator_MAX,
    "MIN":    operator_MIN,
    "LWA":    operator_LWA,
    "TWA":    operator_TWA,
    "WSLOPE": operator_WSLOPE,
    "VAR":    operator_VAR,
    "DELTA":  operator_DELTA,
    "INITIAL": operator_initial,
    "MAXSCORE": operator_AUT_max,
    "MINSCORE": operator_AUT_min,
    "TIMEt" : operator_TIMEt
}

def rolling_windows(session, base,  ws, mapping):
    """ 
    This iterates over all the observations and calculates the rolling windows transformations 
    at each time.
    """
    session         = session.sort_values(by="simulation_time")    
    for idx in session.index:

        #: Rolling window at current time ----------------------------------------------------------
        t_now           = session.loc[idx, "simulation_time"]
        
        #: Edge case flag --------------------------------------------------------------------------
        # This just tells each of the operators how to behave at the first observation of the simulation time  or dialysis time
        if base in ["UF", "UF_rate", "BFR"]:
            t_low       = max(60, t_now - ws)
            edge_case   = t_now == 60
        else:
            # Where the simulation_time <= 45, the t_low must always be st = 0
            t_low       = max(0, t_now - ws)
            edge_case   = t_now == 0
        
       
        window          = session[(session["simulation_time"] >= t_low) & (session["simulation_time"] <= t_now)]
        if (window["simulation_time"] == t_low).any():
            # In this case, no need to calculate the augmented case for some of the operators :)
            window_augmented    = window.copy()
            UF0 = None
        else:
            # Here I will attach that observation to the current window to create the augmented window
            # since simulation_time == 0 (t_low) will always exists, this condition will only trigger
            # after simulation time > 45
            jprev_star          = session[session["simulation_time"] < t_low]["simulation_time"].idxmax()
            Xprev_star          = session.loc[[jprev_star]].copy()   # (1)
            Xprev_star.loc[:, "simulation_time"] = t_low
            window_augmented    = pd.concat([Xprev_star, window], axis=0, ignore_index=True) 
            UF0                 = volume_estimation(Xprev_star, t_low) if base == "UF" else None    
        
        #: This applies all the transformations defined in the dictionary for each base signal -----
        for operator_name in mapping.get(base, []):
        
            operator = operators_dispatch.get(operator_name) #empty list if no operators at all
            
            if operator is None:
                raise ValueError(f"No operator {operator_name!r} registered for base {base!r}")
            
            if operator_name == "MAX":
                session.loc[idx, f"W_{base}_MAX"]       = operator(window, base)
                
            elif operator_name == "MIN":
                session.loc[idx, f"W_{base}_MIN"]       = operator(window, base)
                            
            elif operator_name == "VAR":
                session.loc[idx, f"W_{base}_VAR"]       = operator(window, base, edge_case)
            
            elif operator_name == "DELTA": #! This is only for UF here in rolling windows
                session.loc[idx, f"W_{base}_DELTA"]     = operator(window, base, edge_case, custom_X0=UF0)        
        
            elif operator_name == "LWA":
                session.loc[idx, f"W_{base}_LWA"]       = operator(window_augmented, base, t_low)

            elif operator_name =="TWA":
                session.loc[idx, f"W_{base}_TWA"]       = operator(window_augmented, base, edge_case)
                
            elif operator_name == "WSLOPE":
                session.loc[idx, f"W_{base}_WSLOPE"]    = operator(window_augmented, base, edge_case)

    # (1)   [[]] usage makes it return a dataframe instead of a series even if there is only one max index
    return session



# %% #*=========================== Cumulative Window Transformations ===============================

def cumulative_windows(session, base, mapping):
    """ 
    This iterates over all the observations and calculates the cumulative window transformations 
    at each time.
    """
    session         = session.sort_values(by="simulation_time")    
    for idx in session.index:

        #: Rolling window at current time ----------------------------------------------------------
        t_now           = session.loc[idx, "simulation_time"]
        
        if base in ["UF", "UF_rate", "BFR"]:
            t_edge          = 60
        else:
            t_edge          = 0
            
        #: Edge case flag --------------------------------------------------------------------------
        # This just tells each of the operators how to behave at the first observation of the simulation time  or dialysis time
        t_low               = 60 # This is equal to the horizon time
        edge_cases        = t_now <= t_low
        cumulative_window   = session[(session["simulation_time"] >= t_low) & (session["simulation_time"] <= t_now)]
        #: This applies all the transformations defined in the dictionary for each base signal -----
        
        if edge_cases:
        
            for operator_name in mapping.get(base, []):
                operator = operators_dispatch.get(operator_name) #empty list if no operators at all
                if operator is None:
                    raise ValueError(f"No operator {operator_name!r} registered for base {base!r}")
                
                elif operator_name == "VAR":
                    session.loc[idx, f"CW_{base}_VAR"]       = 0 if t_now != t_low else np.nan 
                    # This means that before the dialysis start, the signals are filled wit zero, 
                    # at the dialysis start (first value) the var is left as nan so the value is computed
                    # with the median value until the first entry different after start is registred
                    # and the signal's behavior backs to the definition for n>=2
                
                elif operator_name == "DELTA":
                    session.loc[idx, f"CW_{base}_DELTA"]     = 0     
            
                elif operator_name =="TWA":
                    session.loc[idx, f"CW_{base}_TWA"]       = session.loc[idx, base]
                    
                elif operator_name == "WSLOPE":
                    session.loc[idx, f"CW_{base}_WSLOPE"]    = 0
        
        else:
            #! if observation within dialysis, there must be always more than one observation! 
            for operator_name in mapping.get(base, []):
                operator = operators_dispatch.get(operator_name) #empty list if no operators at all
                if operator is None:
                    raise ValueError(f"No operator {operator_name!r} registered for base {base!r}")
                
                elif operator_name == "VAR":
                    session.loc[idx, f"CW_{base}_VAR"]       = operator(cumulative_window, base, edge_case=None) #all the edge cases here are handled through "if edge_cases:"
                
                elif operator_name == "DELTA":
                    session.loc[idx, f"CW_{base}_DELTA"]     = operator(cumulative_window, base, edge_case=None, custom_X0=None)   
            
                elif operator_name =="TWA":
                    session.loc[idx, f"CW_{base}_TWA"]       = operator(cumulative_window, base, edge_case=None)
                    
                elif operator_name == "WSLOPE":
                    session.loc[idx, f"CW_{base}_WSLOPE"]    = operator(cumulative_window, base, edge_case=None)
    return session

        
        
def dialysis_time_transformations(session, base, mapping, aut_thresholds):
    """
    Compute all 'dialysis-time' operators requested for this `base` signal on one
    session.  
    """
    out            = {}                                    
    session        = session.sort_values("simulation_time")
    dialysis_time  = session[session["simulation_time"] >= 60]

    total_obs      = len(dialysis_time)
    edge_case      = total_obs <= 1                        

    ths            = aut_thresholds.get(base, {})
    th_max         = ths.get("MAX", np.nan)
    th_min         = ths.get("MIN", np.nan)

    for op_name in mapping.get(base, []):                   
        operator = operators_dispatch.get(op_name)
        #: Only one observation (at the dialysis start, always true since these values were forward filled) -----
        if edge_case:
            X0_value = float(
                dialysis_time.loc[dialysis_time["simulation_time"] == 60, base].iloc[0]
            )
            pre_dialysis = session[session["simulation_time"] < 60]
            if not pre_dialysis.empty:
                twa = float(
                    pre_dialysis.loc[pre_dialysis["simulation_time"].idxmax(), base]
                )
            else:
                twa = X0_value
            
            if   op_name == "VAR":       out[f"S_{base}_VAR"]       = np.nan
            elif op_name == "DELTA":     out[f"S_{base}_DELTA"]     = 0
            elif op_name == "TWA":       out[f"S_{base}_TWA"]       = twa
            elif op_name == "WSLOPE":    out[f"S_{base}_WSLOPE"]    = 0
            elif op_name == "INITIAL":   out[f"S_{base}_INITIAL"]   = X0_value
            elif op_name == "MINSCORE":  out[f"S_{base}_MINSCORE"]  = max(0, th_min - X0_value)
            elif op_name == "MAXSCORE":  out[f"S_{base}_MAXSCORE"]  = max(0, X0_value - th_max)
            elif op_name == "TIMEt":     out[f"S_{base}_TIMEt"]     = X0_value
            continue                     # go to next operator
        
        #: More than 1 observation ---
        if operator is None:
            raise ValueError(f"No operator {op_name!r} registered for {base!r}")

        if   op_name == "MINSCORE": out[f"S_{base}_MINSCORE"] = operator(dialysis_time, base, threshold=th_min)
        elif op_name == "MAXSCORE": out[f"S_{base}_MAXSCORE"] = operator(dialysis_time, base, threshold=th_max)
        elif op_name == "TIMEt":    out[f"S_{base}_TIMEt"]    = operator(dialysis_time, base, time=5)
        else:                            # VAR, DELTA, TWA, WSLOPE, INITIAL
            out[f"S_{base}_{op_name}"] = operator(dialysis_time, base)
    return pd.Series(out)
            


def cumulative_windows_up_to_IDH(session, base, mapping, aut_thresholds):
    """ 
    This iterates over all the observations and calculates the cumulative window transformations 
    at each time.
    """
    session             = session.sort_values(by="simulation_time")    
    First_IDH           = session["First_IDH_01"].iloc[0] + 60  # First IDH is in minutes since dialysis start

    #:The analysis just trigger if IDH
    out            = {}   
    if not np.isnan(First_IDH):
        cw = session[(session["simulation_time"] >= 60) & (session["simulation_time"]<=First_IDH)].copy()
        if not (cw["simulation_time"] == First_IDH).any():
            new_row                    = cw.loc[cw["simulation_time"].idxmax(), :].copy()
            t_diff                     = First_IDH - new_row["simulation_time"]
            new_row["chartdate"]       = new_row["chartdate"] + pd.Timedelta(minutes=t_diff)
            new_row["simulation_time"] = First_IDH
            
            #: The value at t_IDH is different than the LOCF only for UF 
            if base == "UF":
                UF_prev                    = new_row["UF"]
                UFR_prev                   = new_row["UF_rate"]
                UF_IDH                     = UF_prev + UFR_prev*t_diff
                new_row.at[base]           = UF_IDH
                
            new_row_df                      = new_row.to_frame().T          
            cw_augmented                    = pd.concat([cw, new_row_df], ignore_index=True)
            cw_augmented[base]              = cw_augmented[base].apply(pd.to_numeric)
        else:
            cw_augmented               = cw

        edge_case   = len(cw) <=1
        ths         = aut_thresholds.get(base, {})
        th_max      = ths.get("MAX", np.nan)
        for op_name in mapping.get(base, []):                   
            operator = operators_dispatch.get(op_name)
            #: Only one observation (at the dialysis start, always true since these values were forward filled) -----
            if edge_case:
                
                X0_value       = cw.loc[cw["simulation_time"]==60, base].iloc[0]
                pre_dialysis   = session[session["simulation_time"] < 60]
                if not pre_dialysis.empty:
                    twa   = pre_dialysis.loc[pre_dialysis["simulation_time"].idxmax(), base]
                else:
                    twa   = X0_value
                
                if   op_name == "VAR":       out[f"IDHO_{base}_VAR"]       = np.nan
                elif op_name == "DELTA":     out[f"IDHO_{base}_DELTA"]     = 0
                elif op_name == "TWA":       out[f"IDHO_{base}_TWA"]       = twa
                elif op_name == "WSLOPE":    out[f"IDHO_{base}_WSLOPE"]    = 0
                elif op_name == "INITIAL":   out[f"IDHO_{base}_INITIAL"]   = X0_value
                elif op_name == "TIMEt":     out[f"IDHO_{base}_TIMEt"]     = X0_value
                elif op_name == "MAXSCORE":  out[f"IDHO_{base}_MAXSCORE"]  = max(0, X0_value - th_max)                  
            
            else:
            #: More than 1 observation ---
                if operator is None:
                    raise ValueError(f"No operator {op_name!r} registered for {base!r}")
                if   op_name == "TIMEt":    out[f"IDHO_{base}_TIMEt"]      = operator(cw_augmented, base, time=5)
                elif op_name == "MAXSCORE": out[f"IDHO_{base}_MAXSCORE"]   = operator(cw_augmented, base, threshold=th_max)
                elif op_name == "DELTA":    out[f"IDHO_{base}_DELTA"]      = operator(cw_augmented, base)
                else:                            # VAR, DELTA, TWA, WSLOPE, INITIAL
                    out[f"IDHO_{base}_{op_name}"] = operator(cw_augmented, base)

    else: # this is when no IDH happened within the dialysis time
        for op_name in mapping.get(base, []):                   
            operator = operators_dispatch.get(op_name)
            if   op_name == "VAR":       out[f"IDHO_{base}_VAR"]       =  np.nan
            elif op_name == "DELTA":     out[f"IDHO_{base}_DELTA"]     =  np.nan
            elif op_name == "TWA":       out[f"IDHO_{base}_TWA"]       =  np.nan
            elif op_name == "WSLOPE":    out[f"IDHO_{base}_WSLOPE"]    =  np.nan
            elif op_name == "INITIAL":   out[f"IDHO_{base}_INITIAL"]   =  np.nan
            elif op_name == "TIMEt":     out[f"IDHO_{base}_TIMEt"]     =  np.nan
            elif op_name == "MAXSCORE":  out[f"IDHO_{base}_MAXSCORE"]  =  np.nan
            continue                     # go to next operator

    return pd.Series(out)
            
#: Base Signal Limits -----------------------
signals_limits = {
    "SBP":      (60, 200),       
    "MAP":      (40, 160),    
    "DBP":      (30, 120),      
    "DC":       (10, 16),        
    "Pulse":    (40, 160),        
    "BFR":      (0, 15),         
    "DT":       (35, 41),        
    "RR":       (5, 60),         
    "O2Sat":    (70, 100),        
    "UF":       (0, 100),         
    "UF_rate":  (0, 0.6),       
}
        
        
