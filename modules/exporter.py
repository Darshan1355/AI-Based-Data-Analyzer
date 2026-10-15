import io
import json
from typing import Dict, Any, Optional
import pandas as pd

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encodes a DataFrame into UTF-8 CSV bytes."""
    if df is None:
        return b""
    return df.to_csv(index=False).encode("utf-8")

def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Encodes a DataFrame into Excel (.xlsx) bytes using openpyxl."""
    if df is None:
        return b""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()

def multi_sheet_excel_bytes(sheet_dict: Dict[str, pd.DataFrame]) -> bytes:
    """Exports multiple DataFrames into a single multi-tab Excel workbook."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheet_dict.items():
            if df is not None and not df.empty:
                safe_name = sheet_name[:31]
                df.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()

def generate_quality_report_markdown(quality_data: Dict[str, Any], dataset_name: str = "Dataset") -> str:
    """Generates a comprehensive Markdown quality audit report."""
    score = quality_data.get("score", 0)
    summary = quality_data.get("summary", {})
    breakdown = quality_data.get("score_breakdown", {})
    issues = quality_data.get("issues", [])

    lines = [
        f"# Data Quality Audit Report: {dataset_name}",
        f"**Overall Data Quality Score: {score}/100**",
        "",
        "## 1. Executive Summary",
        f"- **Total Records (Rows):** {summary.get('total_rows', 0):,}",
        f"- **Total Features (Columns):** {summary.get('total_columns', 0):,}",
        f"- **Total Data Cells:** {summary.get('total_cells', 0):,}",
        f"- **Exact Duplicate Rows:** {summary.get('duplicate_rows', 0):,} ({summary.get('duplicate_rows', 0)/max(1, summary.get('total_rows', 1))*100:.2f}%)",
        f"- **Total Missing Cells:** {summary.get('missing_cells', 0):,} ({summary.get('missing_percentage', 0):.2f}%)",
        f"- **Total Issues Identified:** {len(issues)}",
        "",
        "## 2. Score Calculation Breakdown",
        "The Data Quality Score starts at 100 points, applying evidence-based deductions:",
        f"- **Baseline:** 100 points",
        f"- **Missing Value Deduction:** -{breakdown.get('missing_deduction', 0)} pts (proportional to null ratio)",
        f"- **Duplicate Row Deduction:** -{breakdown.get('duplicate_deduction', 0)} pts (proportional to duplicate ratio)",
        f"- **Outlier Deduction:** -{breakdown.get('outlier_deduction', 0)} pts (proportional to numeric outliers outside 1.5x IQR)",
        f"- **Formatting / Type Inconsistency:** -{breakdown.get('format_deduction', 0)} pts (dates stored as strings, whitespace, casing)",
        f"- **Constant / Low-Variance Columns:** -{breakdown.get('constant_deduction', 0)} pts",
        f"- **Calculated Final Score:** **{score}/100**",
        "",
        "## 3. Detected Issues & Recommended Actions"
    ]

    if not issues:
        lines.append("No critical data quality issues were detected. Dataset is clean and ready for analysis.")
    else:
        for idx, iss in enumerate(issues, start=1):
            lines.append(f"### {idx}. {iss.get('type')} on Column: `{iss.get('column')}`")
            lines.append(f"- **Affected Records:** {iss.get('affected_count', 0):,} ({iss.get('percentage', 0)}%)")
            lines.append(f"- **Diagnosis:** {iss.get('explanation')}")
            lines.append(f"- **Recommended Action:** {iss.get('recommendation')}")
            lines.append("")

    return "\n".join(lines)

def generate_analysis_summary_markdown(
    df_orig: pd.DataFrame,
    df_proc: Optional[pd.DataFrame],
    quality_data: Dict[str, Any],
    dataset_name: str = "Dataset"
) -> str:
    """Generates an executive analysis and dataset summary report in Markdown."""
    lines = [
        f"# AI Data Analyzer: Dataset Summary & Preprocessing Report",
        f"**Target Dataset:** {dataset_name}",
        "",
        "## 1. Overview & Dimensions",
        f"- **Original Shape:** {df_orig.shape[0]:,} rows x {df_orig.shape[1]:,} columns",
    ]
    if df_proc is not None:
        lines.append(f"- **Processed Shape:** {df_proc.shape[0]:,} rows x {df_proc.shape[1]:,} columns")
        lines.append(f"- **Net Row Retention:** {(len(df_proc)/max(1, len(df_orig)))*100:.2f}%")

    lines.append("")
    lines.append("## 2. Inferred Column Types & Null Profile")
    lines.append("| Column Name | Inferred Type | Null Count | Null % | Unique Values |")
    lines.append("|---|---|---|---|---|")

    active_df = df_proc if df_proc is not None else df_orig
    for col in active_df.columns:
        null_c = int(active_df[col].isna().sum())
        null_p = round((null_c / len(active_df)) * 100, 2)
        uniq_c = int(active_df[col].nunique(dropna=True))
        lines.append(f"| `{col}` | {str(active_df[col].dtype)} | {null_c:,} | {null_p}% | {uniq_c:,} |")

    # Date ranges if available
    date_cols = [c for c in active_df.columns if pd.api.types.is_datetime64_any_dtype(active_df[c])]
    if date_cols:
        lines.append("")
        lines.append("## 3. Detected Date Ranges")
        for dc in date_cols:
            valid_dates = active_df[dc].dropna()
            if not valid_dates.empty:
                lines.append(f"- **`{dc}`:** From `{valid_dates.min()}` to `{valid_dates.max()}`")

    # Numeric summary statistics
    num_cols = active_df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        lines.append("")
        lines.append("## 4. Key Numeric Statistics")
        try:
            desc = active_df[num_cols].describe().round(2).to_markdown()
        except Exception:
            desc = f"```\n{active_df[num_cols].describe().round(2).to_string()}\n```"
        lines.append(desc)

    return "\n".join(lines)
