import re
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

def detect_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes DataFrame and detects data quality issues:
    - Missing values
    - Duplicate rows
    - Type inconsistencies / numeric stored as object
    - Date columns stored as text
    - Inconsistent text formatting (leading/trailing whitespace)
    - Constant or near-constant columns
    - Outliers in numeric columns
    - High missing rate columns
    - Potentially irrelevant columns (e.g. constant, pure IDs)
    Computes a transparent 0-100 data quality score.
    """
    if df is None or df.empty:
        return {
            "score": 0,
            "score_breakdown": {},
            "issues": [],
            "summary": {"total_rows": 0, "total_columns": 0, "duplicate_rows": 0, "missing_cells": 0}
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    issues = []

    # 1. Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / total_rows) * 100, 2)
    if duplicate_count > 0:
        issues.append({
            "id": "duplicates",
            "type": "Duplicate Rows",
            "column": "(Dataset Level)",
            "affected_count": duplicate_count,
            "percentage": duplicate_pct,
            "explanation": f"Dataset contains {duplicate_count} exact duplicate rows ({duplicate_pct}%).",
            "recommendation": "Remove duplicate rows to avoid skewed statistical metrics and biased models.",
            "action_type": "remove_duplicates",
            "default_enabled": True
        })

    # 2. Missing values & High missing rates
    total_missing_cells = 0
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        total_missing_cells += null_count
        if null_count > 0:
            null_pct = round((null_count / total_rows) * 100, 2)
            is_high_missing = null_pct >= 50.0
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            if is_high_missing:
                rec = "Consider dropping this column or imputing if domain knowledge permits."
                action = "handle_high_missing"
            elif is_numeric:
                rec = "Impute with median/mean or drop rows with null values."
                action = "impute_numeric"
            else:
                rec = "Impute with mode (most frequent value), constant 'Unknown', or drop rows."
                action = "impute_categorical"

            issues.append({
                "id": f"missing_{col}",
                "type": "High Missing Values" if is_high_missing else "Missing Values",
                "column": col,
                "affected_count": null_count,
                "percentage": null_pct,
                "explanation": f"Column '{col}' has {null_count} missing values ({null_pct}%).",
                "recommendation": rec,
                "action_type": action,
                "default_enabled": True,
                "is_numeric": is_numeric
            })

    # 3. Constant or Near-Constant columns
    for col in df.columns:
        val_counts = df[col].value_counts(dropna=False)
        if len(val_counts) == 1:
            issues.append({
                "id": f"constant_{col}",
                "type": "Constant Column",
                "column": col,
                "affected_count": total_rows,
                "percentage": 100.0,
                "explanation": f"Column '{col}' has only one unique value ('{val_counts.index[0]}') across all rows.",
                "recommendation": "Consider removing this column as it provides zero variance or analytical signal.",
                "action_type": "remove_column",
                "default_enabled": False
            })
        elif len(val_counts) > 1 and total_rows > 20:
            top_pct = round((val_counts.iloc[0] / total_rows) * 100, 2)
            if top_pct >= 95.0:
                issues.append({
                    "id": f"near_constant_{col}",
                    "type": "Near-Constant Column",
                    "column": col,
                    "affected_count": int(val_counts.iloc[0]),
                    "percentage": top_pct,
                    "explanation": f"Column '{col}' is dominated by a single value ('{val_counts.index[0]}') in {top_pct}% of rows.",
                    "recommendation": "Review whether this column is relevant for your analysis or model.",
                    "action_type": "review_column",
                    "default_enabled": False
                })

    # 4. Potentially Irrelevant / Pure ID columns
    for col in df.columns:
        if df[col].nunique(dropna=True) == total_rows and total_rows > 20:
            if re.search(r"(id|uuid|guid|index|key|hash)", col, re.IGNORECASE) or df[col].dtype == object:
                issues.append({
                    "id": f"id_col_{col}",
                    "type": "Potential Identifier Column",
                    "column": col,
                    "affected_count": total_rows,
                    "percentage": 100.0,
                    "explanation": f"Column '{col}' has 100% unique values and appears to be an identifier.",
                    "recommendation": "Identifiers are usually irrelevant for statistical aggregations or machine learning. Keep only if needed for indexing.",
                    "action_type": "review_column",
                    "default_enabled": False
                })

    # 5. Date columns stored as text
    date_patterns = [
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{2}/\d{2}/\d{4}$",
        r"^\d{4}/\d{2}/\d{2}$",
        r"^\d{2}-\d{2}-\d{4}$",
        r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    ]
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "string":
            non_null_samples = df[col].dropna().astype(str).str.strip().head(50)
            if len(non_null_samples) >= 5:
                # Check regex matching
                matched_count = 0
                for val in non_null_samples:
                    if any(re.match(pat, val) for pat in date_patterns):
                        matched_count += 1
                if matched_count / len(non_null_samples) >= 0.8:
                    issues.append({
                        "id": f"date_as_text_{col}",
                        "type": "Date Stored as Text",
                        "column": col,
                        "affected_count": total_rows - int(df[col].isna().sum()),
                        "percentage": round(((total_rows - int(df[col].isna().sum())) / total_rows) * 100, 2),
                        "explanation": f"Column '{col}' contains date-formatted strings but is stored as text (object).",
                        "recommendation": "Convert to DateTime data type to enable temporal queries and time-series charts.",
                        "action_type": "convert_date",
                        "default_enabled": True
                    })

    # 6. Numeric columns stored as object (e.g. '$1,200', '15.5%', or dirty numeric strings)
    for col in df.columns:
        if df[col].dtype == object:
            # Check if non-null values can be coerced to numeric after stripping currency/commas/percent
            non_null_series = df[col].dropna().astype(str).str.strip()
            if len(non_null_series) >= 5:
                cleaned_series = non_null_series.str.replace(r"[$,€£%]", "", regex=True).str.replace(",", "")
                coerced = pd.to_numeric(cleaned_series, errors="coerce")
                valid_num_pct = round((coerced.notna().sum() / len(non_null_series)) * 100, 2)
                # If majority are numeric but stored as object
                if valid_num_pct >= 85.0 and coerced.notna().sum() > 3:
                    issues.append({
                        "id": f"numeric_as_text_{col}",
                        "type": "Numeric Stored as Text",
                        "column": col,
                        "affected_count": int(coerced.notna().sum()),
                        "percentage": valid_num_pct,
                        "explanation": f"Column '{col}' contains {valid_num_pct}% numeric values but is stored as text/object.",
                        "recommendation": "Clean non-numeric symbols and convert to float/integer type.",
                        "action_type": "convert_numeric",
                        "default_enabled": True
                    })

    # 7. Inconsistent text formatting (leading/trailing whitespace or mixed casing)
    for col in df.columns:
        if df[col].dtype == object:
            str_series = df[col].dropna().astype(str)
            whitespace_issue_count = int((str_series != str_series.str.strip()).sum())
            if whitespace_issue_count > 0:
                ws_pct = round((whitespace_issue_count / total_rows) * 100, 2)
                issues.append({
                    "id": f"whitespace_{col}",
                    "type": "Leading/Trailing Whitespace",
                    "column": col,
                    "affected_count": whitespace_issue_count,
                    "percentage": ws_pct,
                    "explanation": f"Column '{col}' has {whitespace_issue_count} records with leading or trailing whitespace.",
                    "recommendation": "Trim whitespace to prevent query matching failures and category fragmentation.",
                    "action_type": "trim_whitespace",
                    "default_enabled": True
                })
            
            # Categorical label casing inconsistency
            # e.g., 'Yes', 'yes', 'YES '
            unique_raw = set(str_series.str.strip())
            unique_lower = set(str_series.str.strip().str.lower())
            if len(unique_raw) > len(unique_lower) and len(unique_lower) <= 30:
                diff_count = len(unique_raw) - len(unique_lower)
                issues.append({
                    "id": f"casing_{col}",
                    "type": "Inconsistent Categorical Casing",
                    "column": col,
                    "affected_count": diff_count,
                    "percentage": round((diff_count / max(1, len(unique_raw))) * 100, 2),
                    "explanation": f"Column '{col}' has {diff_count} redundant category variants due to mixed letter casing.",
                    "recommendation": "Normalize text casing (Title Case or Lowercase) to harmonize categorical values.",
                    "action_type": "standardize_casing",
                    "default_enabled": True
                })

    # 8. Outliers in numeric columns (IQR Method)
    total_outliers = 0
    total_numeric_cells = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        total_numeric_cells += len(series)
        if len(series) >= 10:
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_count = len(outliers)
                total_outliers += outlier_count
                if outlier_count > 0:
                    outlier_pct = round((outlier_count / len(series)) * 100, 2)
                    issues.append({
                        "id": f"outlier_{col}",
                        "type": "Potential Outliers",
                        "column": col,
                        "affected_count": outlier_count,
                        "percentage": outlier_pct,
                        "lower_bound": round(float(lower_bound), 3),
                        "upper_bound": round(float(upper_bound), 3),
                        "explanation": f"Column '{col}' has {outlier_count} values ({outlier_pct}%) outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}].",
                        "recommendation": "Cap outliers at IQR bounds (Winsorize), replace with median, or review domain legitimacy.",
                        "action_type": "handle_outliers",
                        "default_enabled": False
                    })

    # Calculate Data Quality Score (0 - 100)
    # Deductions:
    # Missing cells: up to 30 pts
    # Duplicates: up to 25 pts
    # Outliers: up to 20 pts
    # Type / format: up to 15 pts
    # Constant columns: up to 10 pts
    score = 100.0
    deduction_missing = min(30.0, (total_missing_cells / max(1, total_cells)) * 100)
    deduction_duplicates = min(25.0, (duplicate_count / max(1, total_rows)) * 100)
    deduction_outliers = min(20.0, (total_outliers / max(1, total_numeric_cells)) * 50) if total_numeric_cells > 0 else 0
    format_issue_count = sum(1 for iss in issues if iss["action_type"] in ["trim_whitespace", "convert_date", "convert_numeric", "standardize_casing"])
    deduction_format = min(15.0, format_issue_count * 3.0)
    constant_col_count = sum(1 for iss in issues if iss["action_type"] == "remove_column")
    deduction_constant = min(10.0, constant_col_count * 5.0)

    total_deductions = deduction_missing + deduction_duplicates + deduction_outliers + deduction_format + deduction_constant
    final_score = max(0, min(100, int(round(score - total_deductions))))

    breakdown = {
        "initial_score": 100,
        "missing_deduction": round(deduction_missing, 1),
        "duplicate_deduction": round(deduction_duplicates, 1),
        "outlier_deduction": round(deduction_outliers, 1),
        "format_deduction": round(deduction_format, 1),
        "constant_deduction": round(deduction_constant, 1),
        "final_score": final_score
    }

    summary = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "duplicate_rows": duplicate_count,
        "missing_cells": total_missing_cells,
        "missing_percentage": round((total_missing_cells / max(1, total_cells)) * 100, 2),
        "issue_count": len(issues)
    }

    return {
        "score": final_score,
        "score_breakdown": breakdown,
        "issues": issues,
        "summary": summary
    }

def apply_preprocessing(
    df: pd.DataFrame, 
    operations_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies user-selected preprocessing operations non-destructively to a copy of df.
    Returns (cleaned_df, report_stats).
    """
    if df is None:
        return df, {}

    cleaned = df.copy()
    stats = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "duplicates_removed": 0,
        "missing_imputed": 0,
        "rows_dropped": 0,
        "columns_converted": 0,
        "columns_removed": 0,
        "outliers_treated": 0,
        "whitespace_trimmed": 0,
        "casing_standardized": 0,
        "modifications": []
    }

    # 1. Remove duplicates
    if operations_config.get("remove_duplicates", False):
        before_count = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        dups_removed = before_count - len(cleaned)
        stats["duplicates_removed"] = dups_removed
        if dups_removed > 0:
            stats["modifications"].append(f"Removed {dups_removed} duplicate rows.")

    # 2. Trim whitespace
    trim_cols = operations_config.get("trim_whitespace_cols", [])
    for col in trim_cols:
        if col in cleaned.columns and (cleaned[col].dtype == object or str(cleaned[col].dtype) == "string"):
            cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            stats["whitespace_trimmed"] += 1
    if stats["whitespace_trimmed"] > 0:
        stats["modifications"].append(f"Trimmed leading/trailing whitespace on {stats['whitespace_trimmed']} text column(s).")

    # 3. Standardize text casing
    casing_config = operations_config.get("standardize_casing_cols", {})
    # casing_config: {col: 'title' | 'lower' | 'upper'}
    for col, case_type in casing_config.items():
        if col in cleaned.columns and (cleaned[col].dtype == object or str(cleaned[col].dtype) == "string"):
            if case_type == "title":
                cleaned[col] = cleaned[col].apply(lambda x: x.title() if isinstance(x, str) else x)
            elif case_type == "lower":
                cleaned[col] = cleaned[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
            elif case_type == "upper":
                cleaned[col] = cleaned[col].apply(lambda x: x.upper() if isinstance(x, str) else x)
            stats["casing_standardized"] += 1
    if stats["casing_standardized"] > 0:
        stats["modifications"].append(f"Standardized text casing on {stats['casing_standardized']} column(s).")

    # 4. Convert numeric stored as text
    numeric_convert_cols = operations_config.get("convert_numeric_cols", [])
    for col in numeric_convert_cols:
        if col in cleaned.columns:
            try:
                # Remove common currency symbols and commas
                cleaned[col] = cleaned[col].astype(str).str.replace(r"[$,€£%]", "", regex=True).str.replace(",", "")
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                stats["columns_converted"] += 1
                stats["modifications"].append(f"Converted column '{col}' to numeric (float).")
            except Exception as e:
                stats["modifications"].append(f"Failed to convert column '{col}' to numeric: {str(e)}")

    # 5. Parse dates
    date_config = operations_config.get("date_parse_cols", {})
    # date_config: {col: format_str or 'auto'}
    for col, date_fmt in date_config.items():
        if col in cleaned.columns:
            try:
                if date_fmt and date_fmt != "auto":
                    cleaned[col] = pd.to_datetime(cleaned[col], format=date_fmt, errors="coerce")
                else:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                stats["columns_converted"] += 1
                stats["modifications"].append(f"Converted column '{col}' to DateTime.")
            except Exception as e:
                stats["modifications"].append(f"Failed to parse dates for column '{col}': {str(e)}")

    # 6. Handle missing values
    missing_config = operations_config.get("missing_strategies", {})
    # missing_config: {col: {"strategy": "mean"|"median"|"mode"|"drop_rows"|"constant", "value": ...}}
    for col, strat_dict in missing_config.items():
        if col in cleaned.columns and cleaned[col].isna().sum() > 0:
            strategy = strat_dict.get("strategy")
            null_count_before = int(cleaned[col].isna().sum())
            
            if strategy == "drop_rows":
                rows_before = len(cleaned)
                cleaned = cleaned.dropna(subset=[col])
                dropped = rows_before - len(cleaned)
                stats["rows_dropped"] += dropped
                stats["modifications"].append(f"Dropped {dropped} row(s) containing nulls in '{col}'.")
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(cleaned[col]):
                mean_val = cleaned[col].mean()
                cleaned[col] = cleaned[col].fillna(mean_val)
                stats["missing_imputed"] += null_count_before
                stats["modifications"].append(f"Imputed {null_count_before} missing value(s) in '{col}' with mean ({mean_val:.2f}).")
            elif strategy == "median" and pd.api.types.is_numeric_dtype(cleaned[col]):
                median_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median_val)
                stats["missing_imputed"] += null_count_before
                stats["modifications"].append(f"Imputed {null_count_before} missing value(s) in '{col}' with median ({median_val:.2f}).")
            elif strategy == "mode":
                mode_series = cleaned[col].mode()
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]
                    cleaned[col] = cleaned[col].fillna(mode_val)
                    stats["missing_imputed"] += null_count_before
                    stats["modifications"].append(f"Imputed {null_count_before} missing value(s) in '{col}' with mode ('{mode_val}').")
            elif strategy == "constant":
                const_val = strat_dict.get("value", "Unknown")
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    try:
                        const_val = float(const_val)
                    except ValueError:
                        const_val = 0
                cleaned[col] = cleaned[col].fillna(const_val)
                stats["missing_imputed"] += null_count_before
                stats["modifications"].append(f"Imputed {null_count_before} missing value(s) in '{col}' with constant '{const_val}'.")

    # 7. Outlier treatment
    outlier_config = operations_config.get("outlier_treatments", {})
    # outlier_config: {col: "cap"|"median"|"drop"}
    for col, method in outlier_config.items():
        if col in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[col]):
            series = cleaned[col].dropna()
            if len(series) >= 10:
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower = q25 - (1.5 * iqr)
                    upper = q75 + (1.5 * iqr)
                    outliers_mask = (cleaned[col] < lower) | (cleaned[col] > upper)
                    outlier_count = int(outliers_mask.sum())
                    
                    if outlier_count > 0:
                        if method == "cap":
                            cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
                            stats["outliers_treated"] += outlier_count
                            stats["modifications"].append(f"Capped (Winsorized) {outlier_count} outlier(s) in '{col}' to [{lower:.2f}, {upper:.2f}].")
                        elif method == "median":
                            med_val = series.median()
                            cleaned.loc[outliers_mask, col] = med_val
                            stats["outliers_treated"] += outlier_count
                            stats["modifications"].append(f"Replaced {outlier_count} outlier(s) in '{col}' with median ({med_val:.2f}).")
                        elif method == "drop":
                            rows_before = len(cleaned)
                            cleaned = cleaned[~outliers_mask]
                            dropped = rows_before - len(cleaned)
                            stats["rows_dropped"] += dropped
                            stats["modifications"].append(f"Dropped {dropped} row(s) containing outliers in '{col}'.")

    # 8. Remove selected columns (ONLY with explicit user confirmation)
    remove_cols = operations_config.get("remove_columns", [])
    valid_remove = [c for c in remove_cols if c in cleaned.columns]
    if valid_remove:
        cleaned = cleaned.drop(columns=valid_remove)
        stats["columns_removed"] = len(valid_remove)
        stats["modifications"].append(f"Removed {len(valid_remove)} column(s): {', '.join(valid_remove)}.")

    stats["processed_rows"] = len(cleaned)
    stats["processed_cols"] = len(cleaned.columns)
    stats["final_missing"] = int(cleaned.isna().sum().sum())
    stats["final_duplicates"] = int(cleaned.duplicated().sum())

    return cleaned, stats
