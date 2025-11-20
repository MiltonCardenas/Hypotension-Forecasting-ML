
""""
This script stores different classes for the project. 
v.1.0 Author: Dario Cardenas
"""
import pandas as pd 
import mysql.connector as sql

# %% ===============================================================================================
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
