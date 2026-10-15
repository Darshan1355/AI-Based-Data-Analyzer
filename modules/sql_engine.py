import time
import re
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import pandasql as ps

# Forbidden SQL statements/commands to enforce safe, read-only querying
FORBIDDEN_KEYWORDS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bINSERT\b", r"\bUPDATE\b", r"\bALTER\b",
    r"\bTRUNCATE\b", r"\bCREATE\b", r"\bREPLACE\b", r"\bATTACH\b", r"\bDETACH\b",
    r"\bPRAGMA\b", r"\bEXEC\b", r"\bEXECUTE\b", r"\bGRANT\b", r"\bREVOKE\b"
]

def validate_sql_query(query: str) -> Tuple[bool, str]:
    """
    Validates that a SQL query is read-only, safe, and targets the 'data' table.
    Returns (is_valid, validation_message).
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."

    cleaned_query = query.strip()
    
    # Check for forbidden mutating statements
    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, cleaned_query, flags=re.IGNORECASE):
            match = re.search(pattern, cleaned_query, flags=re.IGNORECASE).group()
            return False, f"Security Violation: Mutating SQL command '{match}' is not permitted. Only read-only SELECT queries are allowed."

    # Verify query starts with SELECT or WITH (for CTEs)
    normalized = re.sub(r"/\*.*?\*/", "", cleaned_query, flags=re.DOTALL) # remove comments
    normalized = re.sub(r"--.*?\n", "", normalized)
    normalized = normalized.strip()
    
    if not (re.match(r"^SELECT\b", normalized, flags=re.IGNORECASE) or 
            re.match(r"^WITH\b", normalized, flags=re.IGNORECASE)):
        return False, "Query must start with a SELECT or WITH statement."

    return True, "Query validation passed."

def execute_sql_query(
    query: str, 
    df: pd.DataFrame, 
    default_limit: Optional[int] = None
) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Safely executes an SQL query against the provided DataFrame using pandasql.
    Returns (success, result_df, metadata_dict).
    """
    if df is None or df.empty:
        return False, None, {"error": "DataFrame is empty or not loaded.", "execution_time": 0}

    is_valid, msg = validate_sql_query(query)
    if not is_valid:
        return False, None, {"error": msg, "execution_time": 0}

    exec_query = query.strip()
    if exec_query.endswith(";"):
        exec_query = exec_query[:-1].strip()

    # Optional default limit if query does not specify LIMIT
    if default_limit and not re.search(r"\bLIMIT\b", exec_query, flags=re.IGNORECASE):
        exec_query = f"{exec_query} LIMIT {default_limit}"

    start_time = time.perf_counter()
    try:
        # Use pandasql with explicit env mapping 'data': df
        result_df = ps.sqldf(exec_query, {"data": df})
        execution_time = time.perf_counter() - start_time
        
        row_count = len(result_df) if result_df is not None else 0
        col_count = len(result_df.columns) if result_df is not None else 0
        
        metadata = {
            "execution_time": round(execution_time, 4),
            "rows": row_count,
            "columns": col_count,
            "executed_sql": exec_query,
            "is_empty": row_count == 0
        }
        return True, result_df, metadata
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        error_str = str(e)
        # Clean up pandasql SQLite internal trace
        if "OperationalError:" in error_str:
            error_str = error_str.split("OperationalError:")[-1].strip()
        return False, None, {
            "error": f"SQL Execution Error: {error_str}",
            "execution_time": round(execution_time, 4),
            "executed_sql": exec_query
        }
