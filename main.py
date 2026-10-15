import os
import io
import time
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

# Import our custom modular components
from modules.llm_manager import (
    initialize_llm,
    generate_sql_query,
    explain_query_results,
    GROQ_MODELS,
    get_env_groq_key,
    fetch_lm_studio_models,
    normalize_lm_studio_url,
    is_lm_studio_reachable
)
from modules.sql_engine import execute_sql_query, validate_sql_query
from modules.preprocessor import detect_quality_issues, apply_preprocessing
from modules.visualizer import (
    generate_automated_dashboard,
    recommend_chart_for_dataframe,
    create_custom_chart
)
from modules.exporter import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    generate_quality_report_markdown,
    generate_analysis_summary_markdown
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 12px;
    }
    .score-badge {
        font-size: 24px;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
    }
    .score-high { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .score-mid { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .score-low { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def init_session_state():
    defaults = {
        "dataset_name": None,
        "df_original": None,
        "df_processed": None,
        "quality_report": None,
        "preprocessing_stats": None,
        "query_history": [],
        "current_query": None,
        "current_result": None,
        "current_meta": None,
        "current_explanation": None,
        "last_uploaded_file_id": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def reset_dataset_state():
    """Resets dataset-specific state when a new dataset is uploaded or reset requested."""
    st.session_state.df_original = None
    st.session_state.df_processed = None
    st.session_state.quality_report = None
    st.session_state.preprocessing_stats = None
    st.session_state.query_history = []
    st.session_state.current_query = None
    st.session_state.current_result = None
    st.session_state.current_meta = None
    st.session_state.current_explanation = None
    st.session_state.dataset_name = None


# ---------------------------------------------------------
# Sidebar: Configuration & Upload
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # LLM Provider Configuration
    st.subheader("🤖 AI Model Provider")
    
    # Auto-detect if LM Studio is reachable on localhost
    lm_is_up = is_lm_studio_reachable("http://127.0.0.1:1234")
    # In cloud environments or when LM Studio is down, default to Groq Cloud (index 1)
    default_provider_idx = 0 if lm_is_up else 1

    provider = st.radio(
        "Select Provider",
        ["LM Studio (Local)", "Groq Cloud"],
        index=default_provider_idx,
        help="LM Studio runs locally on your machine with zero cloud data transmission. Groq connects to fast cloud inference."
    )
    
    if provider == "LM Studio (Local)":
        lm_base_url = st.text_input(
            "LM Studio Base URL",
            value="http://127.0.0.1:1234",
            help="LM Studio server endpoint (e.g., http://127.0.0.1:1234 or a public tunnel URL like https://xyz.ngrok-free.app)."
        )
        norm_url = normalize_lm_studio_url(lm_base_url)
        is_local_host = any(h in norm_url for h in ["localhost", "127.0.0.1", "0.0.0.0"])

        # Fetch models dynamically from LM Studio
        found_ok, discovered_models, disc_msg = fetch_lm_studio_models(lm_base_url)
        default_model = "meta-llama-3.1-8b-instruct-abliterated"
        model_idx = discovered_models.index(default_model) if default_model in discovered_models else 0
        
        lm_model = st.selectbox(
            "LM Studio Model",
            discovered_models,
            index=model_idx,
            help="Select the model loaded in LM Studio."
        )
        llm_ready, llm, llm_status_msg = initialize_llm(
            provider="LM Studio (Local)",
            model=lm_model,
            base_url=lm_base_url
        )
        
        if not found_ok and is_local_host:
            st.warning(
                "⚠️ **Local LM Studio Not Reachable**\n\n"
                "- **Running on Localhost?** Ensure LM Studio is open, server is started, and CORS is enabled.\n"
                "- **Running on Streamlit Cloud?** The public cloud server cannot reach `localhost` on your home computer. To use LM Studio with cloud deployment, expose your local port with a secure tunnel (e.g., ngrok or Cloudflare Tunnel) and paste the HTTPS URL above, or switch to **Groq Cloud** above."
            )
        elif found_ok:
            st.success("🛡️ **Local-Only Mode Active**: 100% offline. No prompts or dataset values leave your machine.")
    else:
        env_key = get_env_groq_key()
        has_secret = bool(env_key)
        
        groq_api_key = st.text_input(
            "Groq API Key",
            value=env_key,
            type="password",
            placeholder="gsk_..." if not has_secret else "Loaded from Streamlit Secrets / .env",
            help="Loaded from Streamlit Cloud Secrets or .env by default. You can override it here."
        )
        selected_model = st.selectbox(
            "Groq Model",
            GROQ_MODELS,
            index=0
        )
        llm_ready, llm, llm_status_msg = initialize_llm(
            provider="Groq",
            model=selected_model,
            api_key=groq_api_key
        )
        if not groq_api_key:
            st.warning("⚠️ **Groq API Key Missing**: Set `GROQ_API_KEY = \"your-key\"` in Streamlit Cloud Secrets (App Settings > Secrets), or paste it above.")
        else:
            st.caption("🔒 Cloud Inference: Connected securely to Groq API.")

    if llm_ready:
        st.success(llm_status_msg)
    else:
        st.warning(llm_status_msg)

    st.markdown("---")
    
    # CSV Upload
    st.subheader("📁 Dataset Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=["csv"],
        help="Upload standard comma or semicolon delimited CSV files."
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_uploaded_file_id != file_id:
            # New dataset detected: reset previous state to prevent contamination
            reset_dataset_state()
            try:
                # Robust CSV reading with encoding fallback
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding="latin-1")
                
                st.session_state.df_original = df.copy()
                st.session_state.df_processed = df.copy()
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.last_uploaded_file_id = file_id
                
                # Automatically run initial quality detection
                with st.spinner("Analyzing dataset quality and structure..."):
                    st.session_state.quality_report = detect_quality_issues(df)
                    
                st.success(f"Loaded: `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

    # Sidebar Dataset Status & Reset
    if st.session_state.df_processed is not None:
        st.markdown("---")
        st.subheader("📊 Dataset Status")
        st.write(f"**Name:** `{st.session_state.dataset_name}`")
        st.write(f"**Rows:** {len(st.session_state.df_processed):,} | **Cols:** {len(st.session_state.df_processed.columns)}")
        
        is_modified = st.session_state.preprocessing_stats is not None
        if is_modified:
            st.info("Status: **Preprocessed Data Active**")
        else:
            st.caption("Status: **Original Data Active**")
            
        if st.button("🔄 Reset to Original Data", use_container_width=True):
            st.session_state.df_processed = st.session_state.df_original.copy()
            st.session_state.preprocessing_stats = None
            st.session_state.quality_report = detect_quality_issues(st.session_state.df_original)
            st.success("Reset data to original state.")
            st.rerun()


# ---------------------------------------------------------
# Main App Header & Navigation
# ---------------------------------------------------------
st.title("📈 AI-Based Data Analyzer")
st.markdown("Automate data profiling, natural language SQL queries, interactive visualization, and cleaning with AI.")

# If no file uploaded, show informative empty state
if st.session_state.df_processed is None:
    st.info("👋 Welcome! Please upload a CSV dataset from the left sidebar to begin analysis.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 💬 Natural Language to SQL")
        st.write("Ask questions in plain English. The AI generates safe SQLite queries and executes them instantly with `pandasql`.")
    with col2:
        st.markdown("### 🛠️ Preprocessing Assistant")
        st.write("Automatically detect missing values, duplicates, outliers, date anomalies, and format inconsistencies with user-controlled fixes.")
    with col3:
        st.markdown("### 📊 Automated Visualizations")
        st.write("Instant visualization dashboard showing distributions, correlations, time trends, and categorical breakdowns.")
    st.stop()

# Active DataFrame references
df_curr = st.session_state.df_processed
df_orig = st.session_state.df_original
dataset_name = st.session_state.dataset_name or "Dataset"

# Main Navigation Tabs
tab_overview, tab_sql, tab_preprocess, tab_visualize, tab_export = st.tabs([
    "📋 Dataset Overview",
    "💬 AI Query Generator",
    "🛠️ Data Preprocessing",
    "📊 Data Visualization",
    "📥 Export Center"
])


# =========================================================
# TAB 1: DATASET OVERVIEW
# =========================================================
with tab_overview:
    st.header(f"Dataset Overview: {dataset_name}")
    
    # Key High-Level Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Rows", f"{len(df_curr):,}", delta=f"{len(df_curr) - len(df_orig):,} vs orig" if len(df_curr) != len(df_orig) else None)
    m2.metric("Total Columns", f"{len(df_curr.columns):,}")
    m3.metric("Duplicate Rows", f"{df_curr.duplicated().sum():,}")
    m4.metric("Missing Cells", f"{df_curr.isna().sum().sum():,}")
    
    q_score = st.session_state.quality_report.get("score", 100) if st.session_state.quality_report else 100
    score_cls = "score-high" if q_score >= 80 else ("score-mid" if q_score >= 50 else "score-low")
    m5.markdown(f"**Quality Score**<br><span class='score-badge {score_cls}'>{q_score} / 100</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Preview & Schema
    col_prev, col_schema = st.columns([3, 2])
    with col_prev:
        st.subheader("Data Preview")
        st.dataframe(df_curr.head(100), use_container_width=True)
        st.caption(f"Showing first {min(100, len(df_curr))} rows.")
    
    with col_schema:
        st.subheader("Inferred Column Schema & Profile")
        schema_df = pd.DataFrame({
            "Column": df_curr.columns,
            "Type": [str(t) for t in df_curr.dtypes],
            "Nulls": df_curr.isna().sum().values,
            "Null %": [round((n / len(df_curr)) * 100, 1) for n in df_curr.isna().sum().values],
            "Unique": [df_curr[c].nunique(dropna=True) for c in df_curr.columns]
        })
        st.dataframe(schema_df, use_container_width=True)

    # Date ranges if any
    date_cols = [c for c in df_curr.columns if pd.api.types.is_datetime64_any_dtype(df_curr[c])]
    if date_cols:
        st.subheader("Detected Temporal Ranges")
        date_summary = []
        for dc in date_cols:
            valid_d = df_curr[dc].dropna()
            if not valid_d.empty:
                date_summary.append({
                    "Date Column": dc,
                    "Start Date": str(valid_d.min()),
                    "End Date": str(valid_d.max()),
                    "Total Days": (valid_d.max() - valid_d.min()).days
                })
        if date_summary:
            st.dataframe(pd.DataFrame(date_summary), use_container_width=True)

    # Statistical Summaries
    st.subheader("Summary Statistics")
    tab_num_stat, tab_cat_stat = st.tabs(["Numeric Variables", "Categorical Variables"])
    with tab_num_stat:
        num_cols = df_curr.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            st.dataframe(df_curr[num_cols].describe().round(3).T, use_container_width=True)
        else:
            st.info("No numerical columns found in dataset.")
    with tab_cat_stat:
        cat_cols = df_curr.select_dtypes(include=["object", "string", "category"]).columns
        if len(cat_cols) > 0:
            st.dataframe(df_curr[cat_cols].describe().T, use_container_width=True)
        else:
            st.info("No categorical columns found in dataset.")

    # Data Quality Calculation Breakdown
    if st.session_state.quality_report:
        with st.expander("ℹ️ Data Quality Score Formula & Deduction Details"):
            b = st.session_state.quality_report.get("score_breakdown", {})
            st.markdown(f"""
            - **Base Score:** 100 points
            - **Missing Values Penalty:** -{b.get('missing_deduction', 0)} pts (proportional to total missing cells)
            - **Duplicate Rows Penalty:** -{b.get('duplicate_deduction', 0)} pts (proportional to duplicate percentage)
            - **Outlier Penalty:** -{b.get('outlier_deduction', 0)} pts (based on IQR 1.5x deviations)
            - **Formatting & Type Inconsistency Penalty:** -{b.get('format_deduction', 0)} pts (whitespace, unparsed dates, casing)
            - **Constant Columns Penalty:** -{b.get('constant_deduction', 0)} pts (zero variance features)
            - **Final Certified Score:** **{q_score} / 100**
            """)


# =========================================================
# TAB 2: AI QUERY GENERATOR (FEATURE 1)
# =========================================================
with tab_sql:
    st.header("💬 AI Natural-Language Query Generator")
    st.caption("Ask questions about your data in plain English. The AI generates safe SQLite SQL and executes it with `pandasql`.")

    # Target data selector (Processed vs Original)
    col_target, col_limit = st.columns([3, 1])
    with col_target:
        query_target = st.radio(
            "Query Target Data",
            ["Cleaned / Processed Data (Recommended)", "Original Raw Data"],
            index=0,
            horizontal=True
        )
    with col_limit:
        default_limit = st.number_input("Max Result Rows", min_value=10, max_value=5000, value=500, step=50)

    target_df = df_curr if "Cleaned" in query_target else df_orig

    # Natural Language Input
    user_query_input = st.text_input(
        "Ask anything about the data:",
        placeholder="e.g., Which category has the highest average value? Or: Show the top 5 records by age.",
        key="nl_query_input"
    )

    col_btn_gen, col_btn_manual = st.columns([1, 1])
    with col_btn_gen:
        run_ai_btn = st.button("🚀 Generate & Execute SQL Query", type="primary", use_container_width=True)
    with col_btn_manual:
        show_manual_editor = st.checkbox("Toggle Manual SQL Editor", value=False)

    manual_sql_query = None
    if show_manual_editor:
        manual_sql_query = st.text_area(
            "Custom SQL Query (Table name is 'data')",
            value=st.session_state.current_query or "SELECT * FROM data LIMIT 10;",
            height=100
        )
        if st.button("▶️ Execute Manual SQL", use_container_width=True):
            st.session_state.current_query = manual_sql_query
            with st.spinner("Executing custom SQL query..."):
                ok, res_df, meta = execute_sql_query(manual_sql_query, target_df, default_limit=default_limit)
                if ok:
                    st.session_state.current_result = res_df
                    st.session_state.current_meta = meta
                    st.session_state.current_explanation = "Manual query executed."
                else:
                    st.error(meta.get("error"))

    # Handle AI Generation
    if run_ai_btn:
        if not user_query_input or not user_query_input.strip():
            st.warning("Please enter a question about your data first.")
        elif not llm_ready:
            st.error(f"Cannot generate SQL: {llm_status_msg}")
        else:
            with st.spinner("Generating SQL query from question..."):
                schema_str = "\n".join([f"- {col}: {str(target_df[col].dtype)}" for col in target_df.columns])
                success, generated_sql = generate_sql_query(llm, schema_str, user_query_input)
                
                if not success:
                    st.error(f"SQL Generation Error: {generated_sql}")
                else:
                    st.session_state.current_query = generated_sql
                    
                    # Safe Execution
                    with st.spinner("Executing generated SQL..."):
                        exec_ok, res_df, meta = execute_sql_query(generated_sql, target_df, default_limit=default_limit)
                        if exec_ok:
                            st.session_state.current_result = res_df
                            st.session_state.current_meta = meta
                            
                            # Natural Language Explanation
                            preview_str = res_df.head(5).to_string() if res_df is not None and not res_df.empty else "Empty Result"
                            exp_ok, explanation = explain_query_results(llm, schema_str, generated_sql, preview_str)
                            st.session_state.current_explanation = explanation if exp_ok else "No explanation available."
                            
                            # Append to query history
                            st.session_state.query_history.append({
                                "question": user_query_input,
                                "sql": generated_sql,
                                "rows": len(res_df),
                                "time": meta.get("execution_time", 0)
                            })
                        else:
                            st.error(meta.get("error"))
                            st.session_state.current_result = None
                            st.session_state.current_meta = meta
                            st.session_state.current_explanation = None

    # Display Query Results & Views
    if st.session_state.current_query is not None:
        st.markdown("---")
        
        # SQL View (Always available in expandable container)
        with st.expander("🔍 Generated SQL Query & Security Audit", expanded=True):
            st.code(st.session_state.current_query, language="sql")
            is_valid, val_msg = validate_sql_query(st.session_state.current_query)
            if is_valid:
                st.caption(f"🛡️ Security Status: {val_msg} (Read-Only Verified)")
            else:
                st.error(f"⚠️ Validation Warning: {val_msg}")

        # Execution Status & Metrics
        if st.session_state.current_meta:
            m = st.session_state.current_meta
            c1, c2, c3 = st.columns(3)
            c1.metric("Execution Time", f"{m.get('execution_time', 0):.4f}s")
            c2.metric("Result Rows", f"{m.get('rows', 0):,}")
            c3.metric("Result Columns", f"{m.get('columns', 0):,}")

        # Natural Language Explanation
        if st.session_state.current_explanation:
            st.info(f"💡 **AI Interpretation:** {st.session_state.current_explanation}")

        # Output Views: Table View, Chart View, SQL View
        result_df = st.session_state.current_result
        if result_df is not None and not result_df.empty:
            view_tabs = st.tabs(["📊 Table View", "📈 Chart View", "💾 Export Result"])
            
            # Sub-Tab 1: Table View
            with view_tabs[0]:
                st.dataframe(result_df, use_container_width=True)
                st.caption(f"Showing {len(result_df)} row(s). Click column headers to sort.")

            # Sub-Tab 2: Chart View with automatic recommendations
            with view_tabs[1]:
                rec = recommend_chart_for_dataframe(result_df)
                st.caption(f"💡 **AI Chart Recommendation:** {rec.get('chart_type')} chart — {rec.get('reason')}")
                
                chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)
                chart_types = ["Bar", "Line", "Scatter", "Pie", "Histogram", "Box"]
                rec_idx = chart_types.index(rec.get("chart_type")) if rec.get("chart_type") in chart_types else 0
                
                with chart_col1:
                    sel_chart_type = st.selectbox("Chart Type", chart_types, index=rec_idx, key="query_chart_type")
                with chart_col2:
                    all_cols = list(result_df.columns)
                    default_x_idx = all_cols.index(rec.get("x")) if rec.get("x") in all_cols else 0
                    sel_x = st.selectbox("X-Axis", all_cols, index=default_x_idx, key="query_chart_x")
                with chart_col3:
                    num_or_all = ["None"] + all_cols
                    default_y = rec.get("y")
                    default_y_idx = num_or_all.index(default_y) if (default_y and default_y in num_or_all) else 0
                    sel_y = st.selectbox("Y-Axis", num_or_all, index=default_y_idx, key="query_chart_y")
                with chart_col4:
                    cat_options = ["None"] + [c for c in all_cols if c != sel_x and c != sel_y]
                    sel_color = st.selectbox("Color / Category", cat_options, index=0, key="query_chart_color")

                # Optional Aggregation
                agg_options = ["None", "Mean", "Sum", "Count", "Min", "Max"]
                sel_agg = st.selectbox("Aggregation", agg_options, index=0, key="query_chart_agg")

                fig, summary_plot_df, plot_desc = create_custom_chart(
                    result_df,
                    chart_type=sel_chart_type,
                    x_col=sel_x,
                    y_col=sel_y if sel_y != "None" else None,
                    color_col=sel_color if sel_color != "None" else None,
                    aggregation=sel_agg
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(plot_desc)
                else:
                    st.warning(plot_desc)

            # Sub-Tab 3: Export Result
            with view_tabs[2]:
                st.subheader("Download Query Output")
                d_c1, d_c2, d_c3 = st.columns(3)
                with d_c1:
                    csv_data = dataframe_to_csv_bytes(result_df)
                    st.download_button(
                        label="📥 Download Result CSV",
                        data=csv_data,
                        file_name=f"query_result_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with d_c2:
                    excel_data = dataframe_to_excel_bytes(result_df, sheet_name="Query_Result")
                    st.download_button(
                        label="📥 Download Result Excel",
                        data=excel_data,
                        file_name=f"query_result_{int(time.time())}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                with d_c3:
                    sql_text = st.session_state.current_query.encode("utf-8")
                    st.download_button(
                        label="📥 Download SQL (.sql)",
                        data=sql_text,
                        file_name=f"query_{int(time.time())}.sql",
                        mime="text/plain",
                        use_container_width=True
                    )
        elif result_df is not None and result_df.empty:
            st.warning("Query executed successfully, but returned 0 records.")


# =========================================================
# TAB 3: DATA PREPROCESSING (FEATURE 2)
# =========================================================
with tab_preprocess:
    st.header("🛠️ AI Data Preprocessing Assistant")
    st.caption("Inspect data quality, review automated recommendations, configure transformations, and preview changes non-destructively.")

    # Quality Report Summary
    quality_rep = st.session_state.quality_report
    if not quality_rep:
        with st.spinner("Analyzing data quality..."):
            quality_rep = detect_quality_issues(df_curr)
            st.session_state.quality_report = quality_rep

    score = quality_rep.get("score", 100)
    summary_q = quality_rep.get("summary", {})
    issues = quality_rep.get("issues", [])

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Data Quality Score", f"{score}/100")
    col_s2.metric("Issues Detected", f"{len(issues)}")
    col_s3.metric("Missing Value Cells", f"{summary_q.get('missing_cells', 0):,} ({summary_q.get('missing_percentage', 0)}%)")
    col_s4.metric("Exact Duplicate Rows", f"{summary_q.get('duplicate_rows', 0):,}")

    st.markdown("---")
    
    st.subheader("📋 Preprocessing Configuration & Recommendations")
    if not issues:
        st.success("🎉 No significant data quality issues detected! Dataset is clean.")
    else:
        st.write("Configure which cleaning operations to apply. Expand each card to customize strategies.")

    # Building dynamic configuration dictionary from user controls
    ops_config: Dict[str, Any] = {
        "remove_duplicates": False,
        "trim_whitespace_cols": [],
        "standardize_casing_cols": {},
        "convert_numeric_cols": [],
        "date_parse_cols": {},
        "missing_strategies": {},
        "outlier_treatments": {},
        "remove_columns": []
    }

    # Group issues by categories for structured UI
    for idx, iss in enumerate(issues):
        with st.expander(f"⚠️ {iss['type']} — `{iss['column']}` ({iss['affected_count']} affected, {iss['percentage']}%)", expanded=iss.get("default_enabled", False)):
            st.markdown(f"**Diagnosis:** {iss['explanation']}")
            st.markdown(f"**Recommendation:** {iss['recommendation']}")
            
            enable_op = st.checkbox("Enable this operation", value=iss.get("default_enabled", False), key=f"enable_{iss['id']}")
            
            if enable_op:
                act = iss["action_type"]
                col_name = iss["column"]
                
                # 1. Duplicates
                if act == "remove_duplicates":
                    ops_config["remove_duplicates"] = True
                    st.info("Will deduplicate entire rows while keeping first occurrence.")

                # 2. Missing Values
                elif act in ["impute_numeric", "impute_categorical", "handle_high_missing"]:
                    is_num = iss.get("is_numeric", False)
                    if is_num:
                        strat = st.selectbox(
                            f"Missing strategy for '{col_name}'",
                            ["median", "mean", "mode", "constant", "drop_rows"],
                            index=0,
                            key=f"missing_strat_{col_name}"
                        )
                    else:
                        strat = st.selectbox(
                            f"Missing strategy for '{col_name}'",
                            ["mode", "constant", "drop_rows"],
                            index=0,
                            key=f"missing_strat_{col_name}"
                        )
                    
                    const_val = None
                    if strat == "constant":
                        const_val = st.text_input(f"Replacement constant for '{col_name}'", value="0" if is_num else "Unknown", key=f"const_{col_name}")

                    ops_config["missing_strategies"][col_name] = {
                        "strategy": strat,
                        "value": const_val
                    }

                # 3. Whitespace
                elif act == "trim_whitespace":
                    ops_config["trim_whitespace_cols"].append(col_name)
                    st.info(f"Will strip leading and trailing whitespace from string values in '{col_name}'.")

                # 4. Casing
                elif act == "standardize_casing":
                    case_choice = st.selectbox(
                        f"Target casing for '{col_name}'",
                        ["title", "lower", "upper"],
                        index=0,
                        key=f"casing_{col_name}"
                    )
                    ops_config["standardize_casing_cols"][col_name] = case_choice

                # 5. Date Conversion
                elif act == "convert_date":
                    date_format = st.selectbox(
                        f"Date format for '{col_name}'",
                        ["auto", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"],
                        index=0,
                        key=f"date_fmt_{col_name}"
                    )
                    ops_config["date_parse_cols"][col_name] = date_format

                # 6. Numeric Conversion
                elif act == "convert_numeric":
                    ops_config["convert_numeric_cols"].append(col_name)
                    st.info(f"Will remove currency symbols/commas and convert '{col_name}' to numeric.")

                # 7. Outliers
                elif act == "handle_outliers":
                    out_method = st.selectbox(
                        f"Outlier treatment for '{col_name}'",
                        ["cap", "median", "drop"],
                        index=0,
                        key=f"outlier_method_{col_name}",
                        help="Cap (Winsorize to IQR lower/upper bound), replace with median, or drop outlier rows."
                    )
                    ops_config["outlier_treatments"][col_name] = out_method

                # 8. Column Removal (Strictly user-confirmed)
                elif act in ["remove_column", "review_column"]:
                    confirm_remove = st.checkbox(f"⚠️ Confirm permanent removal of column '{col_name}'", value=False, key=f"confirm_rm_{col_name}")
                    if confirm_remove:
                        ops_config["remove_columns"].append(col_name)

    st.markdown("---")
    
    # Action Workflow: Preview -> Apply -> Reset
    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        preview_btn = st.button("👁️ Preview Proposed Changes", use_container_width=True)
    with col_act2:
        apply_btn = st.button("✅ Apply Preprocessing", type="primary", use_container_width=True)
    with col_act3:
        reset_btn = st.button("🔄 Reset to Original Data", use_container_width=True)

    if reset_btn:
        st.session_state.df_processed = st.session_state.df_original.copy()
        st.session_state.preprocessing_stats = None
        st.session_state.quality_report = detect_quality_issues(st.session_state.df_original)
        st.success("Reverted all modifications. Original dataset restored.")
        st.rerun()

    # Preview logic
    if preview_btn:
        with st.spinner("Generating preprocessing preview..."):
            preview_df, preview_stats = apply_preprocessing(df_curr, ops_config)
            st.subheader("Preview of Proposed Changes")
            st.write(f"**Projected Dimensions:** {len(preview_df):,} rows x {len(preview_df.columns)} columns (Net Change: {len(preview_df) - len(df_curr):,} rows)")
            st.dataframe(preview_df.head(25), use_container_width=True)
            if preview_stats.get("modifications"):
                st.write("**Planned Modifications:**")
                for mod in preview_stats["modifications"]:
                    st.write(f"- {mod}")
            else:
                st.info("No modifications selected.")

    # Apply logic
    if apply_btn:
        with st.spinner("Applying preprocessing transformations..."):
            cleaned_df, stats = apply_preprocessing(df_orig, ops_config)
            st.session_state.df_processed = cleaned_df
            st.session_state.preprocessing_stats = stats
            # Recompute quality score on cleaned dataset
            st.session_state.quality_report = detect_quality_issues(cleaned_df)
            st.success("🎉 Preprocessing successfully applied!")
            st.rerun()

    # Before vs After Comparison (if preprocessing has been applied)
    if st.session_state.preprocessing_stats is not None:
        st.markdown("---")
        st.subheader("📊 Before vs After Preprocessing Comparison")
        st_info = st.session_state.preprocessing_stats
        
        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
        comp_col1.metric("Rows Retained", f"{len(df_curr):,}", delta=f"{len(df_curr) - len(df_orig):,}")
        comp_col2.metric("Duplicates Removed", f"{st_info.get('duplicates_removed', 0):,}")
        comp_col3.metric("Missing Imputed", f"{st_info.get('missing_imputed', 0):,}")
        comp_col4.metric("Columns Transformed", f"{st_info.get('columns_converted', 0) + st_info.get('whitespace_trimmed', 0):,}")

        if st_info.get("modifications"):
            with st.expander("📝 Applied Changes Audit Log", expanded=True):
                for m in st_info["modifications"]:
                    st.write(f"✅ {m}")

        # Quick downloads for cleaned dataset
        st.markdown("#### Download Cleaned Data")
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            st.download_button(
                "📥 Download Cleaned CSV",
                data=dataframe_to_csv_bytes(df_curr),
                file_name=f"cleaned_{dataset_name}",
                mime="text/csv",
                use_container_width=True
            )
        with dl_c2:
            st.download_button(
                "📥 Download Cleaned Excel (.xlsx)",
                data=dataframe_to_excel_bytes(df_curr, sheet_name="Cleaned_Data"),
                file_name=f"cleaned_{dataset_name.replace('.csv', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# =========================================================
# TAB 4: AUTOMATIC DATA VISUALIZATION (FEATURE 3)
# =========================================================
with tab_visualize:
    st.header("📊 Automatic Data Visualization Dashboard")
    st.caption("Instantly generated visual analytics based on data distributions, correlation matrices, and time trends.")

    viz_tabs = st.tabs(["🚀 Automated Visual Dashboard", "🎨 Custom Visual Explorer"])
    
    # Sub-tab 1: Automated Dashboard
    with viz_tabs[0]:
        with st.spinner("Generating visual insights..."):
            dashboard_cards = generate_automated_dashboard(df_curr)
            
        if not dashboard_cards:
            st.info("No charts could be generated for this dataset.")
        else:
            for item in dashboard_cards:
                st.subheader(item["title"])
                st.plotly_chart(item["fig"], use_container_width=True)
                st.info(f"📌 **Analytical Finding:** {item['description']}")
                
                # Download chart data
                if item.get("data") is not None:
                    chart_csv = dataframe_to_csv_bytes(item["data"])
                    st.download_button(
                        label=f"📥 Download Data for '{item['title']}' (CSV)",
                        data=chart_csv,
                        file_name=f"{item['id']}_data.csv",
                        mime="text/csv",
                        key=f"dl_chart_{item['id']}"
                    )
                st.markdown("---")

    # Sub-tab 2: Custom Visual Explorer
    with viz_tabs[1]:
        st.subheader("Interactive Custom Chart Builder")
        st.caption("Select columns, chart types, and aggregations to build custom interactive visualizations.")
        
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            cust_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Histogram", "Box"], key="cust_chart_type")
        with c_col2:
            cust_x = st.selectbox("X-Axis", list(df_curr.columns), key="cust_x")
        with c_col3:
            cust_y_options = ["None"] + list(df_curr.columns)
            cust_y = st.selectbox("Y-Axis", cust_y_options, index=0, key="cust_y")
        with c_col4:
            cust_color_options = ["None"] + [c for c in df_curr.columns if c != cust_x]
            cust_color = st.selectbox("Category / Color", cust_color_options, index=0, key="cust_color")

        cust_agg = st.selectbox("Aggregation Function", ["None", "Mean", "Sum", "Count", "Min", "Max", "Median"], index=0, key="cust_agg")
        
        fig_cust, cust_df, cust_desc = create_custom_chart(
            df_curr,
            chart_type=cust_type,
            x_col=cust_x,
            y_col=cust_y if cust_y != "None" else None,
            color_col=cust_color if cust_color != "None" else None,
            aggregation=cust_agg
        )
        if fig_cust:
            st.plotly_chart(fig_cust, use_container_width=True)
            st.caption(cust_desc)
            if cust_df is not None:
                st.download_button(
                    label="📥 Download Chart Aggregation Data (CSV)",
                    data=dataframe_to_csv_bytes(cust_df),
                    file_name=f"custom_chart_data_{int(time.time())}.csv",
                    mime="text/csv"
                )
        else:
            st.warning(cust_desc)


# =========================================================
# TAB 5: EXPORT CENTER (FEATURE 4 & 5)
# =========================================================
with tab_export:
    st.header("📥 Export Center")
    st.caption("Download original or processed datasets, Excel workbooks, query results, SQL scripts, and audit reports.")

    sec1, sec2 = st.columns(2)
    
    with sec1:
        st.subheader("📄 Dataset Exports")
        
        # Original Data
        st.markdown("**Original Dataset:**")
        orig_c1, orig_c2 = st.columns(2)
        with orig_c1:
            st.download_button(
                "📥 Original CSV",
                data=dataframe_to_csv_bytes(df_orig),
                file_name=f"original_{dataset_name}",
                mime="text/csv",
                use_container_width=True
            )
        with orig_c2:
            st.download_button(
                "📥 Original Excel (.xlsx)",
                data=dataframe_to_excel_bytes(df_orig, sheet_name="Original_Data"),
                file_name=f"original_{dataset_name.replace('.csv', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        st.markdown("**Active Processed Dataset:**")
        proc_c1, proc_c2 = st.columns(2)
        with proc_c1:
            st.download_button(
                "📥 Processed CSV",
                data=dataframe_to_csv_bytes(df_curr),
                file_name=f"processed_{dataset_name}",
                mime="text/csv",
                use_container_width=True
            )
        with proc_c2:
            st.download_button(
                "📥 Processed Excel (.xlsx)",
                data=dataframe_to_excel_bytes(df_curr, sheet_name="Processed_Data"),
                file_name=f"processed_{dataset_name.replace('.csv', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with sec2:
        st.subheader("💬 Query & SQL Exports")
        if st.session_state.current_result is not None:
            st.markdown(f"**Latest Query Output ({len(st.session_state.current_result):,} rows):**")
            q_c1, q_c2 = st.columns(2)
            with q_c1:
                st.download_button(
                    "📥 Query Result CSV",
                    data=dataframe_to_csv_bytes(st.session_state.current_result),
                    file_name="latest_query_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with q_c2:
                st.download_button(
                    "📥 Query Result Excel (.xlsx)",
                    data=dataframe_to_excel_bytes(st.session_state.current_result, sheet_name="Query_Result"),
                    file_name="latest_query_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            if st.session_state.current_query:
                st.download_button(
                    "📥 Generated SQL (.sql)",
                    data=st.session_state.current_query.encode("utf-8"),
                    file_name="generated_query.sql",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("No query results currently generated to export.")

    st.markdown("---")
    
    st.subheader("📋 Analytical Reports & Audit Documentation")
    r_c1, r_c2 = st.columns(2)
    
    # 1. Quality Audit Report
    with r_c1:
        st.markdown("#### Data Quality Audit Report")
        quality_rep = st.session_state.quality_report or detect_quality_issues(df_curr)
        report_md = generate_quality_report_markdown(quality_rep, dataset_name)
        
        st.download_button(
            "📥 Download Quality Audit Report (.md)",
            data=report_md.encode("utf-8"),
            file_name=f"quality_report_{dataset_name.replace('.csv', '')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        with st.expander("Preview Quality Report"):
            st.markdown(report_md)

    # 2. Executive Summary Report
    with r_c2:
        st.markdown("#### Executive Dataset Analysis Summary")
        summary_md = generate_analysis_summary_markdown(df_orig, df_curr, quality_rep, dataset_name)
        
        st.download_button(
            "📥 Download Analysis Summary (.md)",
            data=summary_md.encode("utf-8"),
            file_name=f"analysis_summary_{dataset_name.replace('.csv', '')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        with st.expander("Preview Analysis Summary"):
            st.markdown(summary_md)
