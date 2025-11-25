""""
This script stores different classes for the project. 
v.1.0 Author: Dario Cardenas
"""
import numpy as np
import pandas as pd 
import mysql.connector as sql

# %% #: ============================================================================================
class DataQuery():
    """
    Serves as the interface between Python scripts and the SQL server hosting medical data.
    
    The primary objective of this class is to make easier the retrieval of information 
    directly from the server to the current workspace for downstream analysis, 
    storage, or processing. It includes optimized methods for handling large-scale 
    datasets through batch processing.

    Methods
    -------
    __init__()
        Initializes the class instance and loads the necessary credentials to 
        authenticate and access the server.

    query(sql_string)
        Executes a custom raw SQL query string and returns the results as a 
        pandas DataFrame.

    distinct_value(dataset, table, column)
        Retrieves unique row values from a specified column within a given 
        dataset and table. Returns the result as a pandas DataFrame.

    query_largeDataSource(event_names)
        Executes a query on large datasets in batches of 1 million entries to 
        manage memory usage. Filters results to include only records matching 
        the target 'event_names'. Returns a pandas DataFrame.

    query_largeDataSource_patientsList_similar(patient_ids, event_names)
        Executes a batch query (1M entries) filtering exclusively for a specific 
        list of patient IDs (e.g., dialysis patients). It retrieves entries where 
        the 'event_name' is distinct but *similar* (fuzzy match/like) to the 
        provided target list. Returns a pandas DataFrame.

    query_largeDataSource_patientsList(patient_ids, event_names)
        Similar to the method above, but enforces an *exact* match on the 
        'event_name'. Queries in batches of 1M entries, filtering for specific 
        patient IDs and exact target events. Returns a pandas DataFrame.

    query_largeDataSource_patientsList_NoDiv(patient_ids)
        Executes a batch query (1M entries) filtering exclusively for a specific 
        list of patient IDs. Unlike the other methods, this applies *no filtering* on 'event_name', returning all available records for the specified patients. 
        Returns a pandas DataFrame.

    show_tables(database_name)
        Retrieves and returns a list of all table names contained within the 
        specified database.

    get_methods()
        Returns a list of all available methods within this class for reference.
    """
    
    def __init__(self, user, password, port=3306):

        self.user = user
        self.password = password
        self.port = port

    def query(self, database, query, custom_columns=False):

        connection = sql.connect(host= '', 
                                 user=self.user, 
                                 password=self.password, 
                                 database=database,
                                 port=self.port)
        
        cursor = connection.cursor()
        cursor.execute(query)

        # Dataframe building
        table_rows = cursor.fetchall()

        if custom_columns:
            column_names = custom_columns
        else:
            column_names = [column[0] for column in cursor.description]

        cursor.close()
        connection.close()

        df = pd.DataFrame(table_rows, columns=column_names)

        return df

    def distinct_value(self, database_name, table_name, column_name, limit=None):

        query = f"""
        SELECT DISTINCT {column_name}
        FROM (
            SELECT {column_name}
            FROM {database_name}.{table_name}
            {f'LIMIT {limit}' if limit is not None else ''}
        ) AS sample;
        """
        df = self.query(database_name, query)

        return df
        
    def query_largeDataSource(self, database_name, table_name, patient_id_col, event_column, event_name, limit=1000000, num_batches=None):
        
        last_id = 0
        frames = []  
        batch_count = 0  

        while True:
            query = f"""
            SELECT *
            FROM {database_name}.{table_name}
            WHERE {patient_id_col} > {last_id} AND {event_column} = '{event_name}'
            ORDER BY {patient_id_col}
            LIMIT {limit};
            """
            df_batch = self.query(database_name, query)
 
            if df_batch.empty or (num_batches is not None and batch_count >= num_batches):
                break  # Exit loop if no data is returned or batch limit is reached

            frames.append(df_batch) 
            last_id = df_batch[patient_id_col].max()  # Update last_id to the max ID of the batch
            batch_count += 1 

        df_hidenic15 = pd.concat(frames, ignore_index=True)

        return df_hidenic15
    
    def query_largeDataSource_patientsList_similar(
        self, 
        database_name, 
        table_name, 
        patient_id_col, 
        event_column, 
        event_names,        # a list of event name substrings
        patients_list, 
        limitPerBatch=1000000, 
        num_batches=None
        ):

        
        last_id = 0
        frames = []
        batch_count = 0
        patients_list_str = ', '.join(str(pid) for pid in patients_list)
        
        # Build the OR condition for partial (LIKE) matches, e.g.:
        #   (EventCode LIKE '%K%' OR EventCode LIKE '%Na%' OR EventCode LIKE '%BUN%' ...)
        like_clauses = [f"{event_column} LIKE '%{ename}%'" for ename in event_names]
        or_condition = " OR ".join(like_clauses)

        while True:
            query = f"""
                SELECT *
                FROM {database_name}.{table_name}
                WHERE {patient_id_col} > {last_id}
                AND {patient_id_col} IN ({patients_list_str})
                AND ({or_condition})
                ORDER BY {patient_id_col}
                LIMIT {limitPerBatch};
            """
            
            df_batch = self.query(database_name, query)
            if df_batch.empty or (num_batches is not None and batch_count >= num_batches):
                break

            frames.append(df_batch)
            last_id = df_batch[patient_id_col].max()
            batch_count += 1

        df = pd.concat(frames, ignore_index=True)
        return df

    def query_largeDataSource_patientsList_NoDiv(self, database_name, table_name, patient_id_col, patients_list, limitPerBatch=1000000, num_batches=None):
        last_id = 0
        frames = []  
        batch_count = 0  

        patients_list_str = ', '.join(str(id) for id in patients_list)

        while True:
            query = f"""
            SELECT *
            FROM {database_name}.{table_name}
            WHERE {patient_id_col} > {last_id} AND {patient_id_col} IN ({patients_list_str}) 
            ORDER BY {patient_id_col}
            LIMIT {limitPerBatch};
            """
            df_batch = self.query(database_name, query)
            
            if df_batch.empty or (num_batches is not None and batch_count >= num_batches):
                break  # Exit loop if no data is returned or batch limit is reached

            frames.append(df_batch) 
            last_id = df_batch[patient_id_col].max()  # Update last_id to the max ID of the batch
            batch_count += 1 

        df = pd.concat(frames, ignore_index=True)

        return df

    def query_largeDataSource_patientsList(self, database_name, table_name, patient_id_col, event_column, event_names, patients_list, limitPerBatch=1000000, num_batches=None):
        last_id = 0
        frames = []  
        batch_count = 0  

        patients_list_str = ', '.join(str(id) for id in patients_list)

        while True:
            query = f"""
            SELECT *
            FROM {database_name}.{table_name}
            WHERE {patient_id_col} > {last_id} AND {patient_id_col} IN ({patients_list_str}) AND {event_column} IN ({event_names})
            ORDER BY {patient_id_col}
            LIMIT {limitPerBatch};
            """
            df_batch = self.query(database_name, query)
            
            if df_batch.empty or (num_batches is not None and batch_count >= num_batches):
                break  # Exit loop if no data is returned or batch limit is reached

            frames.append(df_batch) 
            last_id = df_batch[patient_id_col].max()  # Update last_id to the max ID of the batch
            batch_count += 1 

        df = pd.concat(frames, ignore_index=True)

        return df

    def show_tables(self, database):
        query = f"SHOW TABLES IN {database}"
        tables = self.query(database, query)

        return tables
    
    def get_methods(cls):
        return [func for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__")]


# %% #: ============================================================================================
# Data Policies
class DataExtraction:
    
    """
    This class manages the data extraction and filtering process for a dataset (Pandas DataFrame) 
    provided during initialization.

    The final output consists of:
    1. A cleaned DataFrame containing only the rows that passed the specified set of tests.
    2. The original DataFrame with an added "Extraction policies" column. Rows excluded from the 
       final cohort are labeled with the specific test(s) they failed.
    3. An executive summary detailing total rows deleted, sessions affected, unique sessions 
       removed, and other metrics per specified test.

    Additional Parameters:
    - initial_counter (int): A number that modifies the final report object.
    - report (bool): If False, the executive summary is not generated.

    Methodology:
    Most methods in this class test the input DataFrame without modifying it directly. Actual 
    filtering occurs only when the finalizer method, `perform_extraction()`, is called. 
    Intermediate methods function as quality checks that:

    1. Create a copy of the dataframe to ensure no in-place modification occurs during testing.
    2. Apply logic to create a boolean mask (Pandas Series) indicating pass/fail status per row.
    3. Store the results in `self.storage` (a dictionary). This includes booleans for keeping rows, 
       IDs of sessions triggering exclusion criteria, and IDs of excluded indices.
    """

    def __init__(self, df0, initial_counter=0, report=True):

        if 'session' not in df0.columns:
            if 'chartdate' not in df0.columns:
                self.df0 = df0.sort_values(by=['FIN_study_id', 'study_id']).reset_index(drop=True)
            else:
                self.df0 = df0.sort_values(by=['FIN_study_id','chartdate']).reset_index(drop=True)
        else:
            self.df0 = df0.sort_values(by=['session', 'chartdate']).reset_index(drop=True)

        self.storage = {}
        self.i = initial_counter
        self.report = report

    def get_methods(cls):
        "This method returns the names of the methods in the class."
        return [func for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__")]

    def constrain_maximum_change(self, max_change, min_value, max_value):
        """
        Identify and handle consecutive measurements with differences exceeding a threshold.

        Rules:
            - If one of the two measurements falls within the valid range, keep it.
            - If both are inside or outside the range, delete both.
        """
        
        df0 = self.df0.copy()
        df0['diff'] = df0.groupby('session')['result_val'].diff().abs().fillna(0)
        df0['valid_diff'] = df0['diff'] <= max_change
        df0['in_range'] = df0['result_val'].between(min_value, max_value)

        df0['keep'] = True
        out_of_range_diff = df0.loc[~df0['valid_diff']]
        for session, group in out_of_range_diff.groupby('session'):
            for idx in group.index:
                # Get indices of current row and previous row
                current_row = idx
                prev_row = idx - 1 if idx - 1 in df0.index and df0.loc[idx, 'session'] == df0.loc[idx - 1, 'session'] else None

                if prev_row is not None:
                    current_in_range = df0.loc[current_row, 'in_range']
                    prev_in_range = df0.loc[prev_row, 'in_range']

                    if current_in_range and not prev_in_range:
                        # Keep the current row, delete the previous row
                        df0.loc[prev_row, 'keep'] = False
                    elif prev_in_range and not current_in_range:
                        # Keep the previous row, delete the current row
                        df0.loc[current_row, 'keep'] = False
                    elif not prev_in_range and not current_in_range:
                        # Delete both rows
                        df0.loc[[prev_row, current_row], 'keep'] = False
                    else:
                        # Both are in range, delete both rows
                        df0.loc[[prev_row, current_row], 'keep'] = False

        keep_mask = df0['keep']
        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': out_of_range_diff['session'].unique().tolist(),
            'indices_with_problems': df0.loc[~keep_mask].index.tolist(),
            'problematic_rows_preview': df0.loc[~keep_mask]
        }
        self.i += 1

    def delete_repeated_entries(self):

        df0 = self.df0.copy()
        # keep Mask 
        mask = ~df0.duplicated(subset=['FIN_study_id','chartdate', 'event_name', 'result_val'], keep='first')
        problematic_indices = df0.index[~mask].tolist()  

        if 'session'   in df0.columns:
            problematic_sessions = df0.loc[~mask, 'session'].unique().tolist() 
        else:
            problematic_sessions = np.nan

        self.storage[self.i] = {
            'keep_mask': mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def error_entries_deleting(self):

        df0 = self.df0.copy()
        # Mask
        mask = ((df0['event_tag'] == 'DateTime Correction') | 
                (df0['result_stat'] == 'In Error'))
        
        problematic_indices = df0.index[mask].tolist()
        if 'session' in df0.columns:
             problematic_sessions = df0.loc[mask, 'session'].unique().tolist()
        else:
            problematic_sessions = np.nan

        self.storage[self.i] = {
            'keep_mask': ~mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def delete_preliminary(self):

        df0 = self.df0.copy()
        # Mask
        mask = (df0['result_stat'] == 'Preliminary') 
        problematic_indices = df0.index[mask].tolist()

        if 'session' in df0.columns:
             problematic_sessions = df0.loc[mask, 'session'].unique().tolist()
        else:
            problematic_sessions = np.nan

        self.storage[self.i] = {
            'keep_mask': ~mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def one_value_per_session(self):

        df0 = self.df0.copy()

        # Group by session and check if all result_val values are the same
        grouped = df0.groupby('session')['result_val']
        consistent_mask = grouped.transform(lambda x: x.nunique() == 1)

        problematic_indices = df0.index[~consistent_mask].tolist()
        problematic_sessions = df0.loc[~consistent_mask, 'session'].unique().tolist()

        self.storage[self.i] = {
            'keep_mask': consistent_mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def drop_missing_values(self):
            
            df0 = self.df0.copy()
            mask = df0['result_val'].notnull()
            problematic_indices = df0.index[~mask].tolist()
            problematic_sessions = df0.loc[~mask, 'session'].unique().tolist()
    
            self.storage[self.i] = {
                'keep_mask': mask,
                'sessions_with_problems': problematic_sessions,
                'indices_with_problems': problematic_indices
            }
            self.i += 1
    
    def minimum_entries_per_session(self, min_entries):

        df0 = self.df0.copy()
        # Mask
        mask = df0.groupby('session').transform('count')['chartdate'] >= min_entries
        
        problematic_sessions = df0.loc[~mask, 'session'].unique().tolist()
        problematic_indices = df0.index[~mask].tolist()

        self.storage[self.i] = {
            'keep_mask': mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def minimum_entries_per_session2(self, min_entries):
        """
        Counts measurements only within the session range (Initial_chartdate to Final_chartdate)
        and identifies sessions with fewer than the specified minimum number of entries.
        """
        df0 = self.df0.copy()
        
        def process(group):
            initial_chartdate = group['Initial_chartdate'].iloc[0]
            final_chartdate = group['Final_chartdate'].iloc[0]
            within_session = group[(group['chartdate'] >= initial_chartdate) & (group['chartdate'] <= final_chartdate)]
            return within_session['chartdate'].count() >= min_entries

        # Apply process function to each session
        mask = df0.groupby('session').apply(process)
        problematic_sessions = mask[~mask].index.tolist()  # Sessions that did not meet min_entries
        problematic_indices = df0[df0['session'].isin(problematic_sessions)].index.tolist()

        # Store results
        self.storage[self.i] = {
            'keep_mask': df0['session'].isin(mask[mask].index),  #
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def delete_NAN_reg_dates(self):

        df0 = self.df0.copy()
        mask = df0['reg_date'].notnull()
        problematic_indices = df0.index[~mask].tolist()

        self.storage[self.i] = {
            'keep_mask': mask,
            'sessions_with_problems': None,
            'indices_with_problems': problematic_indices
        }
        self.i += 1
    
    def delete_NAN_specify_columns(self, columns):

        df0 = self.df0.copy()
        mask = df0[columns].notnull().all(axis=1)
        problematic_indices = df0.index[~mask].tolist()

        self.storage[self.i] = {
            'keep_mask': mask,
            'sessions_with_problems': None,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def minimum_time_per_session(self, min_time):
            
            df0 = self.df0.copy()
            # Mask
            mask = df0.groupby('session')['elapsed_time'].transform(lambda x: x.max() - x.min()) >= min_time
            
            problematic_sessions = df0.loc[~mask, 'session'].unique().tolist()
            problematic_indices = df0.index[~mask].tolist()
    
            self.storage[self.i] = {
                'keep_mask': mask,
                'sessions_with_problems': problematic_sessions,
                'indices_with_problems': problematic_indices
            }
            self.i += 1

    def delete_TST_close_to_end(self, last_entry_offset=10):
        
        df0 = self.df0.copy()
        
        mask_to_keep = pd.Series(True, index=df0.index)
        problematic_sessions = []
        problematic_indices = []

        # Iterattion over sessions
        for session, group in df0.groupby('session'):

            final_chartdate = group['Final_chartdate'].iloc[0]
            tst_indices = group[group['event_name'] == 'Treatment Start Time'].index

            for idx in tst_indices:
                time_diff = (final_chartdate - group.loc[idx, 'chartdate']).total_seconds() / 60
                if 0 <= time_diff <= last_entry_offset:
                    mask_to_keep[idx] = False
                    problematic_indices.append(idx)
                else:
                    continue

        problematic_sessions= df0.loc[problematic_indices, 'session'].unique().tolist()
        problematic_indices = df0[~mask_to_keep]

        self.storage[self.i] = {
            'keep_mask': mask_to_keep,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def delete_second_TST_after_a_valid_TST(self, uf_difference_threshold=50):
        """
        This method only deletes the second TST EVENT after a valid TST event.
        Modifies logic to account for UF differences.
        """
        df0 = self.df0.copy()

        mask_to_keep = pd.Series(True, index=df0.index)
        problematic_sessions = []
        problematic_indices = []

        # Iterate over each session
        for session, group in df0.groupby('session'):

            tst_events = group[group['event_name'] == 'Treatment Start Time'].sort_values(by='chartdate')
            uf_events = group[group['event_name'] == 'Dialysis UF total']

            if len(tst_events) > 1:
                for i in range(len(tst_events) - 1):
                    first_tst_idx = tst_events.index[i]
                    second_tst_idx = tst_events.index[i + 1]

                    # Closest "Dialysis UF total" values for both TST events
                    first_tst_time = tst_events.loc[first_tst_idx, 'chartdate']
                    second_tst_time = tst_events.loc[second_tst_idx, 'chartdate']

                    # Closest UF values to each TST event
                    uf_to_first_tst = uf_events.iloc[(uf_events['chartdate'] - first_tst_time).abs().argsort()[:1]]
                    uf_to_second_tst = uf_events.iloc[(uf_events['chartdate'] - second_tst_time).abs().argsort()[:1]]

                    if not uf_to_first_tst.empty and not uf_to_second_tst.empty:
                        first_uf_val = uf_to_first_tst['result_val'].values[0]
                        second_uf_val = uf_to_second_tst['result_val'].values[0]

                        if abs(second_uf_val) < uf_difference_threshold:
                            # Keep the second TST and delete all UF entries before the second TST
                            mask_to_keep[first_tst_idx] = False
                            uf_entries_to_delete = uf_events[uf_events['chartdate'] < second_tst_time].index
                            mask_to_keep[uf_entries_to_delete] = False
                            problematic_indices.extend(uf_entries_to_delete.tolist())
                        else:
                            # Delete the second TST if the UF value of the first is closer and lower
                            if first_uf_val < second_uf_val:
                                mask_to_keep[second_tst_idx] = False
                                problematic_indices.append(second_tst_idx)

            if problematic_indices:
                problematic_sessions.append(session)

        problematic_indices = list(set(problematic_indices))
        problematic_sessions = (
            df0.loc[problematic_indices, 'session'].unique().tolist()
            if problematic_indices else []
        )

        self.storage[self.i] = {
            'keep_mask': mask_to_keep,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': df0.loc[problematic_indices] if problematic_indices else pd.DataFrame()
        }
        self.i += 1

    def delete_sessions_with_three_or_more_TST(self, x=0.3):
        """
        Deletes all data entries in sessions that have three or more "Treatment Start Time" (TST) events,
        only if the time span of the TST values is greater than 30% of the total session time.
        Sessions with zero time span are ignored.
        """
        df0 = self.df0.copy()

        mask_to_keep = pd.Series(True, index=df0.index)
        problematic_sessions = []
        problematic_indices = []

        tst_events = df0[df0['event_name'] == 'Treatment Start Time']
        session_tst_counts = tst_events.groupby('session').size()
        sessions_to_check = session_tst_counts[session_tst_counts >= 3].index.tolist()

        # Process each session
        for session in sessions_to_check:
            session_rows = df0[df0['session'] == session]
            tst_session_events = tst_events[tst_events['session'] == session]

            tst_time_span = (tst_session_events['chartdate'].max() - tst_session_events['chartdate'].min()).total_seconds()
            session_time_span = (session_rows['chartdate'].max() - session_rows['chartdate'].min()).total_seconds()

            if session_time_span == 0:
                continue

            # Time span exceeds x% of the total session time
            if tst_time_span / session_time_span > x/100:
                session_indices = session_rows.index
                mask_to_keep[session_indices] = False
                problematic_sessions.append(session)
                problematic_indices.extend(session_indices.tolist())

        self.storage[self.i] = {
            'keep_mask': mask_to_keep,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def delete_Uf_entries_at_tstart_event(self):

        def filter_entries(group):
            keep_mask = pd.Series(True, index=group.index)
            if len(group) > 1:
                first_three = group.head(3)
                if 'Dialysis UF total' in first_three['event_name'].values and 'Treatment Start Time' in first_three['event_name'].values:
                    keep_mask.loc[first_three[first_three['event_name'] == 'Dialysis UF total'].index] = False
            return keep_mask

        df0 = self.df0.copy()
        mask_to_keep = df0.groupby(['session', 'chartdate']).apply(filter_entries)
        mask_to_keep = mask_to_keep.reset_index(level=[0,1], drop=True)
        problematic_indices = df0.index[~mask_to_keep].tolist() 
        problematic_sessions = df0.loc[problematic_indices, 'session'].unique().tolist()

        self.storage[self.i] = {
            'keep_mask': mask_to_keep,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1
    
    def static_result_val(self):

        """
        Identifies sessions where 'result_val' values don't change over the session time span.

        """
        df0 = self.df0.copy()
        constant_sessions = df0.groupby('session')['result_val'].nunique().loc[lambda x: x == 1].index.tolist()
        keep_mask = ~df0['session'].isin(constant_sessions)
        indices_with_problems = df0.index[~keep_mask].tolist()

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': constant_sessions,
            'indices_with_problems': indices_with_problems
        }
        self.i += 1

    def upper_lower_boundaries_DeleteSession_policy(self, column, upper_bound, lower_bound):
        """
        This method identifies sessions where at least one entry has a value above the upper_bound or
        below the lower_bound, and marks all entries in those sessions for deletion.
        """
        df0 = self.df0.copy()
        sessions_with_problems = []
        indices_with_problems = []

        mask_outside_bounds = (df0[column] > upper_bound) | (df0[column] < lower_bound)

        if 'session' in df0.columns:
            problematic_sessions = df0[mask_outside_bounds]['session'].unique()
            sessions_with_problems.extend(problematic_sessions)
            keep_mask = ~df0['session'].isin(problematic_sessions)
        else:
            # No session column, default behavior based on individual entries
            keep_mask = ~mask_outside_bounds

        indices_with_problems.extend(df0[~keep_mask].index.tolist())

        result = {
            'keep_mask': keep_mask,
            'indices_with_problems': indices_with_problems
        }

        if sessions_with_problems:
            result['sessions_with_problems'] = sessions_with_problems

        return result

    def upper_lower_boundaries_DeletePatients_policy(self, column, upper_bound, lower_bound):
        """
        This method identifies patients where at least one entry has a value above the upper_bound or
        below the lower_bound, and marks all entries of these patients for deletion.
        """
 
        df0 = self.df0.copy()
        patients_with_problems = []
        indices_with_problems = []

        mask_outside_bounds = (df0[column] > upper_bound) | (df0[column] < lower_bound)

        # Identify patients with any problematic values
        problematic_patients = df0[mask_outside_bounds]['FIN_study_id'].unique()
        patients_with_problems.extend(problematic_patients)

        keep_mask = ~df0['FIN_study_id'].isin(problematic_patients)
        indices_with_problems.extend(df0[~keep_mask].index.tolist())

        result = {
            'keep_mask': keep_mask,
            'indices_with_problems': indices_with_problems
        }

        if patients_with_problems:
            result['patients_with_problems'] = patients_with_problems

        return result

    def perform_extraction(self):
        """
        This method performs data extraction in a parallel order based on stored keep_mask values.
        Each policy is applied independently, and only entries that satisfy all policies are kept.
        Returns:
            - A polished DataFrame with only the entries that pass all extraction policies.
            - The original DataFrame with a column indicating which policies triggered deletion.
            - A detailed report of the extraction process.
        """
        self.df0["Extraction Policies Triggered"] = ""
        
        if self.report:
            processing_steps = sorted(self.storage.keys())
            report_data = {
                "Entries Deleted": [],
                "Sessions Affected": [],
                "Total Sessions Deleted": []
            }
            triggered_policies = {idx: set() for idx in self.df0.index}  # Use a set to avoid duplicates

            # Individual and combined tracking
            combined_mask = pd.Series(True, index=self.df0.index)
            for step in processing_steps:
                mask = self.storage[step]['keep_mask']
                combined_mask &= mask
                failed_indices = mask.index[~mask]
                
                # Trackinkg of the policies triggered for each entry
                for idx in failed_indices:
                    triggered_policies[idx].add(step)

                step_df = self.df0[mask].copy()
                entries_deleted = len(self.df0) - mask.sum()  
                sessions_affected = len(self.storage[step]['sessions_with_problems']) if 'session' in self.df0.columns else np.nan
                
                if 'session' in self.df0.columns:
                    # Sessions fully deleted in the current step
                    sessions_deleted = set(self.df0['session']) - set(step_df['session'])
                    total_sessions_deleted = len(sessions_deleted)
                else:
                    total_sessions_deleted = np.nan

                report_data["Entries Deleted"].append(entries_deleted)
                report_data["Sessions Affected"].append(sessions_affected)
                report_data["Total Sessions Deleted"].append(total_sessions_deleted)

            # Extraction
            df = self.df0[combined_mask].reset_index(drop=True)

            # Report Dataframe creation
            final_entries_deleted = len(self.df0) - combined_mask.sum()
            if 'session' in self.df0.columns:
                final_sessions_deleted = set(self.df0['session']) - set(df['session'])
                final_total_sessions_deleted = len(final_sessions_deleted)
            else:
                final_sessions_deleted = np.nan
                final_total_sessions_deleted = np.nan

            report_data["Entries Deleted"].append(final_entries_deleted)
            report_data["Sessions Affected"].append(final_total_sessions_deleted)  # No specific sessions affected count for the final row
            report_data["Total Sessions Deleted"].append(final_total_sessions_deleted)

            self.df0["Extraction Policies Triggered"] = self.df0.index.map(
                lambda idx: ";".join(map(str, sorted(triggered_policies[idx]))) if triggered_policies[idx] else ""
            )
            report = pd.DataFrame(report_data, index=[f"Step {i}" for i in processing_steps] + ["Final"])

        else:
            processing_steps = sorted(self.storage.keys())
            combined_mask = pd.Series(True, index=self.df0.index)
            for step in processing_steps:
                mask = self.storage[step]['keep_mask']
                combined_mask &= mask

            df = self.df0[combined_mask].reset_index(drop=True)
            report = None

        return df, self.df0, report

    def only_increasing_or_constantValues(self, uf_drop):

        df = self.df0.copy()

        def is_non_decreasing(x):
            return  (x.diff().fillna(0) >= -uf_drop).all()
        
        is_non_decreasing_mask = df.groupby('session')['result_val'].apply(is_non_decreasing)
        final_mask = df['session'].map(is_non_decreasing_mask)

        self.storage[self.i] = {
            'keep_mask': final_mask,
            'sessions_with_problems': df.loc[~final_mask, 'session'].unique().tolist(),
            'indices_with_problems': df.index[~final_mask].tolist()
        }
        self.i += 1

    def allow_close_duplicated_chartdates(self, threshold):
        
        df = self.df0.copy()
        
        # Duplicated entries and UF difference
        duplicated_condition = df.duplicated(subset=['session', 'chartdate', 'event_name'], keep=False)
        temp_diff = df.loc[duplicated_condition].groupby(['session', 'chartdate', 'event_name'], observed=True)['result_val'].diff().abs()
        
        # keep_mask
        difference_condition = temp_diff <= threshold
        keep_mask = (~duplicated_condition) | difference_condition

        problematic_sessions = df.loc[duplicated_condition & ~difference_condition, 'session'].unique().tolist()
        problematic_indices = df.loc[duplicated_condition & ~difference_condition].index.tolist()

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def duplicated_entries_keep_modified(self):
        """
        Deletes duplicated entries but prioritizes keeping those with 'Modified' status.
        If no 'Modified' status exists for duplicates, keeps the first occurrence.
        """
        df = self.df0.copy()


        duplicated_condition = df.duplicated(subset=['session', 'chartdate', 'event_name'], keep=False)
        duplicated_df = df[duplicated_condition]
        
        # Process duplicates: keep 'Modified' if exists, otherwise first
        to_keep_indices = []
        for _, group in duplicated_df.groupby(['session', 'chartdate', 'event_name']):
            if (group['result_stat'] == 'Modified').any():
                modified_index = group[group['result_stat'] == 'Modified'].index[0]
                to_keep_indices.append(modified_index)
            else:
                # Keep the first occurrence
                first_index = group.index[0]
                to_keep_indices.append(first_index)

        # rows
        final_keep_mask = df.index.isin(to_keep_indices) | ~duplicated_condition

        # sessions
        sessions_with_problems = df.loc[~final_keep_mask, 'session'].unique().tolist()
        indices_with_problems = df.index[~final_keep_mask].tolist()

        self.storage[self.i] = {
            'keep_mask': final_keep_mask,
            'sessions_with_problems': sessions_with_problems,
            'indices_with_problems': indices_with_problems
        }
        self.i += 1

    def delete_sessions_with_excessive_duplicates(self, max_duplicates):
        df = self.df0.copy()
        
        # Identify duplicated pairs based on session, chartdate, and event_name
        duplicated_entries = df.duplicated(subset=['session', 'chartdate', 'event_name'], keep=False)
        
        # Filter DataFrame to only duplicated entries and count them per session
        duplicate_counts_per_session = df[duplicated_entries].groupby('session').size()
        
        # Identify sessions where the duplicate count meets or exceeds the max_duplicates threshold
        problematic_sessions = duplicate_counts_per_session[duplicate_counts_per_session > max_duplicates].index.tolist()
        
        # Create a mask to keep entries not in problematic sessions
        keep_mask = ~df['session'].isin(problematic_sessions)
        
        # Track indices of problematic entries
        problematic_indices = df.index[~keep_mask].tolist()
        
        # Store results
        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def only_one_outlier_per_session(self, max_val, min_val):

        df = self.df0.copy()
        mask = df.groupby('session')['result_val'].transform(lambda x: (x < min_val).sum() <= 1)
        mask = df.groupby('session')['result_val'].transform(lambda x: (x > max_val).sum() <= 1)

        final_mask = mask & mask
        problematic_sessions = df.loc[~final_mask, 'session'].unique().tolist()
        problematic_indices = df.index[~final_mask].tolist()

        self.storage[self.i] = {
            'keep_mask': mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1
               
    def exclude_sessions_starting_nonzero(self, threshold):

        df = self.df0.copy()
        first_result_val = df.groupby('session')['result_val'].transform('first')
        keep_mask = abs(first_result_val) <= threshold

        problematic_sessions = df.loc[~keep_mask, 'session'].unique().tolist()
        problematic_indices = df.index[~keep_mask].tolist()

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def exclude_sessions_with_no_TST(self):

        df = self.df0.copy()
        has_tst = df.groupby('session')['event_name'].transform(lambda x: 'Treatment Start Time' in x.values)
        keep_mask = has_tst

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': df.loc[~keep_mask, 'session'].unique().tolist(),
            'indices_with_problems': df.index[~keep_mask].tolist()
        }
        self.i += 1

    def exclude_mismatches(self):

        df = self.df0.copy()
        session_has_mismatch = df.groupby('session')['TSTmismatch'].transform(lambda x: 'mismatch' in x.values)
        keep_mask = ~session_has_mismatch

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': df.loc[session_has_mismatch, 'session'].unique().tolist(),
            'indices_with_problems': df.index[session_has_mismatch].tolist()
        }
        self.i += 1

    def exclude_sessions_with_high_result_val_rate(self, uf_rate_threshold):

        df = self.df0.copy()
        df['result_rate'] = pd.to_numeric(df['result_rate'], errors='coerce')
        sessions_with_high_uf_rate = df[df['result_rate'].abs() > uf_rate_threshold]['session'].unique()
        keep_mask = ~df['session'].isin(sessions_with_high_uf_rate)

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': sessions_with_high_uf_rate.tolist(),
            'indices_with_problems': df.index[~keep_mask].tolist()
        }
        self.i += 1

    def exclude_sessions_with_large_time_diff(self, max_time_diff_minutes):

        df = self.df0.copy()
        sessions_with_large_diff = df[df['time_diff'] > max_time_diff_minutes]['session'].unique()
        keep_mask = ~df['session'].isin(sessions_with_large_diff)

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': sessions_with_large_diff.tolist(),
            'indices_with_problems': df.index[~keep_mask].tolist()
        }

        self.i += 1

    def exclude_different_utc_sessions(self):

        df = self.df0.copy()
        sorted_df = df.sort_values(by=['session', 'chartdate'])
        start_utc = df.groupby('session')['UTCoffset'].transform('first')
        end_utc = df.groupby('session')['UTCoffset'].transform('last')
        keep_mask = start_utc == end_utc

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': df.loc[~keep_mask, 'session'].unique().tolist(),
            'indices_with_problems': df.index[~keep_mask].tolist()
        }
        self.i += 1

    def upper_lower_boundaries_DeleteEntry(self, max_, min_):
        df = self.df0.copy()
        
        # mask
        entries_to_keep = (df['result_val'] <= max_) & (df['result_val'] >= min_)
        if 'session' in df.columns:
            sessions_with_problems = df.loc[~entries_to_keep, 'session'].unique().tolist()
        else: 
            sessions_with_problems = np.nan

        indices_with_problems = df.loc[~entries_to_keep].index.tolist()

        self.storage[self.i] = {
            'keep_mask': entries_to_keep,
            'sessions_with_problems': sessions_with_problems,
            'indices_with_problems': indices_with_problems
        }
        self.i += 1

    def delete_first_value_if_0(self):
        df = self.df0.copy()
        df = df.sort_values(by=["session", "simulation_time"])

        first_entries = df.groupby('session').first().reset_index()
        sessions_with_problems = first_entries[first_entries['result_val'] == 0]['session'].tolist()

        indices_with_problems = []
        for session in sessions_with_problems:
            session_df = df[df['session'] == session]
            first_index = session_df.nsmallest(1, 'simulation_time').index[0]
            indices_with_problems.append(first_index)
            
        entries_to_keep = pd.Series(True, index=df.index)
        entries_to_keep.loc[indices_with_problems] = False

        self.storage[self.i] = {
            'keep_mask': entries_to_keep,
            'sessions_with_problems': sessions_with_problems,
            'indices_with_problems': indices_with_problems
        }
        self.i += 1
                               
    def delete_zeros_before_the_first_positive_value(self):
        df = self.df0.copy()
        df = df.sort_values(by=["session", "simulation_time"])

        indices_with_problems = []
        sessions_with_problems = []

        for session_id, group in df.groupby("session"):
            group_sorted = group.sort_values("simulation_time")

            result_vals = group_sorted["result_val"]
            positive_mask = result_vals > 0

            if positive_mask.any():
                # First positive value index
                first_positive_idx = positive_mask.idxmax()
                before_positive = group_sorted.loc[:first_positive_idx - 1]
                zero_before = before_positive[before_positive["result_val"] == 0.0]

                if not zero_before.empty:
                    sessions_with_problems.append(session_id)
                    indices_with_problems.extend(zero_before.index.tolist())
            else:
                # No positive value: delete *all* zero entries
                zero_all = group_sorted[group_sorted["result_val"] == 0.0]
                if not zero_all.empty:
                    sessions_with_problems.append(session_id)
                    indices_with_problems.extend(zero_all.index.tolist())

        entries_to_keep = pd.Series(True, index=df.index)
        entries_to_keep.loc[indices_with_problems] = False

        self.storage[self.i] = {
            'keep_mask': entries_to_keep,
            'sessions_with_problems': sessions_with_problems,
            'indices_with_problems': indices_with_problems
        }
        self.i += 1
 
    def exclude_sessions_below_uf_threshold(self, UF_threshold):
        """
        Exclude sessions where the maximum UF value is below the specified UF_threshold.
        
        """

        df = self.df0.copy()

        max_uf_per_session = df.groupby('session')['result_val'].transform('max')
        keep_mask = max_uf_per_session >= UF_threshold
        
        problematic_sessions = df.loc[~keep_mask, 'session'].unique().tolist()
        problematic_indices = df.index[~keep_mask].tolist()
        
        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': problematic_sessions,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def include_sessions_starting_Nonzero(self, R2_threshold, Uf_min_regression):
        df = self.df0.copy()
        df.sort_values(by=['session', 'chartdate'], inplace=True)
        
        initial_result_val = df.groupby('session')['result_val'].transform('first')
        initial_R2 = df.groupby('session')['R2'].transform('first')
        
        keep_mask = ~((initial_result_val > Uf_min_regression) & (initial_R2 < R2_threshold))
        
        deleted_sessions = df.loc[~keep_mask, 'session'].unique().tolist()
        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': deleted_sessions,
            'indices_with_problems': df.index[~keep_mask].tolist()
        }
        self.i += 1
        
        return deleted_sessions

    def exclude_nonincreasing_or_constant_values2(self, max_diminution):

        df = self.df0.copy()
        def is_valid_session(x):
            if len(x) < 2:
                return [True] * len(x)
            validity = [True] * len(x)
            for i in range(1, len(x)):
                if x.iloc[i] < x.iloc[i-1] - max_diminution:
                    return [False] * len(x)
            return validity

        valid_sessions = df.groupby('session')['result_val'].apply(is_valid_session)
        keep_mask = pd.Series([item for sublist in valid_sessions for item in sublist], index=df.index)

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'sessions_with_problems': df.loc[~keep_mask, 'session'].unique().tolist(),
            'indices_with_problems': df.index[~keep_mask].tolist()
        }
        self.i += 1

    def NCFE_fixed_Threshold(self, time_threshold=30):
        """
        Analyze a single event within each session to verify if the total information gap between
        Initial_chartdate and Final_chartdate exceeds the specified threshold.

        Parameters:
        - time_threshold: Integer representing the maximum allowable gap in minutes.

        """
        df_query = self.df0.copy()

        mask = np.ones(len(df_query), dtype=bool)
        sessions_with_problems = []
        indices_with_problems = []

        def session_gap_check(session_group):
            session_id = session_group['session'].iloc[0]

            # total time gap
            start_time = session_group['Initial_chartdate'].iloc[0]
            end_time = session_group['Final_chartdate'].iloc[0]
            total_gap = (end_time - start_time).total_seconds() / 60  # Gap in minutes

            # Check if the gap exceeds the threshold
            if total_gap > time_threshold:
                mask[session_group.index] = False
                sessions_with_problems.append(session_id)
                indices_with_problems.extend(session_group.index.tolist())

        df_query.groupby('session').apply(session_gap_check)

        return {
            'keep_mask': pd.Series(mask, index=df_query.index),
            'sessions_with_problems': sessions_with_problems,
            'indices_with_problems': indices_with_problems
        }
    
    def duplicated_FIN_study_id_policy(self):

        df_query = self.df0.copy()

        """
        Finds duplicated FIN_study_id and keeps valid entries that dont contain NaN values.
        """
        keep_mask = [1] * len(df_query)
        duplicated_df = df_query[df_query.duplicated(subset =['FIN_study_id', 'reg_date'], keep=False)]
        duplicated_ids = duplicated_df['FIN_study_id'].unique()

        problem_ids = []
        problem_indices = []

        for fin_id in duplicated_ids:
            fin_rows = df_query[df_query['FIN_study_id'] == fin_id]
            rows_with_nan = fin_rows[fin_rows.isna().any(axis=1)]
            
            if not rows_with_nan.empty:
                for index in rows_with_nan.index:
                    keep_mask[index] = 0
                    problem_indices.append(index)
                problem_ids.append(fin_id)
            else:
                for index in fin_rows.index[1:]:
                    keep_mask[index] = 0
                    problem_indices.append(index)
                problem_ids.append(fin_id)

        keep_mask = keep_mask = pd.Series(keep_mask, dtype=bool)

        self.storage[self.i] = {
            'keep_mask': keep_mask,
            'problem_ids': problem_ids,
            'problem_indices': problem_indices
        }
        self.i += 1

    def drop_nan_columns_specify_columns(self, columns):
        df = self.df0.copy()

        def is_valid(value):
            return pd.notnull(value) and value not in {"", "NaT", "None", "N/A"}

        mask = df[columns].applymap(is_valid).all(axis=1)
        problematic_indices = df.index[~mask].tolist()

        self.storage[self.i] = {
            'keep_mask': mask,
            'indices_with_problems': problematic_indices
        }
        self.i += 1

    def drop_duplicates_specify_columns(self, column):
        df = self.df0.copy()
        mask = df.duplicated(subset=column, keep='first')
        problematic_indices = df.index[mask].tolist()

        self.storage[self.i] = {
            'keep_mask': ~mask,
            'indices_with_problems': problematic_indices
        }
        self.i += 1
