from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Modern theme template for Plotly charts
PLOT_THEME = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Plotly

def recommend_chart_for_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes DataFrame columns and data types to recommend an optimal chart type and axes.
    """
    if df is None or df.empty:
        return {"chart_type": "None", "reason": "DataFrame is empty.", "x": None, "y": None}

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    
    # Check if any object columns are parseable dates
    if not datetime_cols:
        for c in cols:
            if c not in numeric_cols and df[c].dtype == object:
                sample = df[c].dropna().head(10)
                try:
                    converted = pd.to_datetime(sample, format="mixed", errors="coerce")
                    if converted.notna().sum() >= len(sample) * 0.7:
                        datetime_cols.append(c)
                except Exception:
                    pass

    categorical_cols = [c for c in cols if c not in numeric_cols and c not in datetime_cols]

    # Rule 1: Date column + numeric column -> Time series line chart
    if datetime_cols and numeric_cols:
        return {
            "chart_type": "Line",
            "x": datetime_cols[0],
            "y": numeric_cols[0],
            "reason": f"Detected temporal trend between date '{datetime_cols[0]}' and numeric value '{numeric_cols[0]}'."
        }

    # Rule 2: Categorical + Numeric -> Bar chart
    if categorical_cols and numeric_cols:
        return {
            "chart_type": "Bar",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "reason": f"Comparing numeric '{numeric_cols[0]}' across categories of '{categorical_cols[0]}'."
        }

    # Rule 3: Two or more Numeric columns -> Scatter plot or Correlation
    if len(numeric_cols) >= 2:
        return {
            "chart_type": "Scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "reason": f"Evaluating relationship between numeric features '{numeric_cols[0]}' and '{numeric_cols[1]}'."
        }

    # Rule 4: Single Numeric column -> Histogram
    if len(numeric_cols) == 1:
        return {
            "chart_type": "Histogram",
            "x": numeric_cols[0],
            "y": None,
            "reason": f"Analyzing frequency distribution of numeric column '{numeric_cols[0]}'."
        }

    # Rule 5: Only Categorical -> Count Bar Chart or Pie Chart
    if categorical_cols:
        return {
            "chart_type": "Bar",
            "x": categorical_cols[0],
            "y": None,
            "reason": f"Displaying value frequencies for categorical column '{categorical_cols[0]}'."
        }

    return {"chart_type": "Bar", "x": cols[0], "y": None, "reason": "Default overview visualization."}


def create_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    aggregation: Optional[str] = "None",
    title: Optional[str] = None
) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame], str]:
    """
    Builds an interactive Plotly figure based on user selections.
    Returns (figure, summarized_dataframe, description).
    """
    if df is None or df.empty:
        return None, None, "Dataset is empty."

    plot_df = df.copy()

    # Handle aggregation if selected
    if aggregation and aggregation != "None" and x_col and y_col:
        try:
            agg_func = aggregation.lower()
            if agg_func in ["mean", "sum", "count", "min", "max", "median"]:
                group_cols = [x_col]
                if color_col and color_col != "None" and color_col in plot_df.columns:
                    group_cols.append(color_col)
                plot_df = plot_df.groupby(group_cols, as_index=False).agg({y_col: agg_func})
                plot_df = plot_df.sort_values(by=y_col, ascending=False)
        except Exception as e:
            return None, None, f"Aggregation error: {str(e)}"

    effective_title = title or f"{chart_type} Chart: {y_col or ''} by {x_col or ''}".strip()
    color_arg = color_col if (color_col and color_col != "None" and color_col in plot_df.columns) else None

    try:
        if chart_type == "Bar":
            if y_col and y_col in plot_df.columns:
                fig = px.bar(
                    plot_df, x=x_col, y=y_col, color=color_arg,
                    title=effective_title, template=PLOT_THEME,
                    color_discrete_sequence=COLOR_SEQUENCE
                )
            else:
                # Count frequencies of x_col
                counts = plot_df[x_col].value_counts().reset_index()
                counts.columns = [x_col, "Count"]
                fig = px.bar(
                    counts.head(25), x=x_col, y="Count",
                    title=f"Frequency Count: {x_col}", template=PLOT_THEME,
                    color_discrete_sequence=COLOR_SEQUENCE
                )
                plot_df = counts

        elif chart_type == "Line":
            fig = px.line(
                plot_df, x=x_col, y=y_col, color=color_arg,
                title=effective_title, template=PLOT_THEME,
                markers=True
            )

        elif chart_type == "Scatter":
            fig = px.scatter(
                plot_df, x=x_col, y=y_col, color=color_arg,
                title=effective_title, template=PLOT_THEME
            )

        elif chart_type == "Pie":
            names_col = x_col
            values_col = y_col if (y_col and y_col in plot_df.columns) else None
            if values_col:
                fig = px.pie(
                    plot_df, names=names_col, values=values_col,
                    title=effective_title, template=PLOT_THEME,
                    hole=0.3
                )
            else:
                counts = plot_df[names_col].value_counts().reset_index()
                counts.columns = [names_col, "Count"]
                fig = px.pie(
                    counts.head(15), names=names_col, values="Count",
                    title=f"Distribution: {names_col}", template=PLOT_THEME,
                    hole=0.3
                )
                plot_df = counts

        elif chart_type == "Histogram":
            fig = px.histogram(
                plot_df, x=x_col, color=color_arg,
                title=effective_title, template=PLOT_THEME,
                marginal="box", nbins=30
            )

        elif chart_type == "Box":
            fig = px.box(
                plot_df, x=x_col, y=y_col, color=color_arg,
                title=effective_title, template=PLOT_THEME,
                points="outliers"
            )

        else:
            return None, None, f"Unsupported chart type: {chart_type}"

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        desc = f"Displaying {chart_type} chart with {len(plot_df)} plotted records."
        return fig, plot_df, desc

    except Exception as e:
        return None, None, f"Chart generation failed: {str(e)}"


def generate_automated_dashboard(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Automatically generates a balanced suite of analytical charts based on dataset properties:
    - Missing values summary (if nulls exist)
    - Numeric distributions (Histograms / Box plots)
    - Categorical frequency counts (Top categories)
    - Time-series trend (if date column present)
    - Correlation heatmap (if multiple numeric columns)
    - Category vs Numeric aggregated comparison
    Returns list of dicts: {"title": str, "fig": Figure, "description": str, "data": DataFrame}
    """
    if df is None or df.empty:
        return []

    dashboard_items = []
    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols and c not in datetime_cols]

    # 1. Missing Values Overview Chart (if missing values exist)
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if not cols_with_missing.empty:
        missing_df = pd.DataFrame({
            "Column": cols_with_missing.index,
            "Missing Count": cols_with_missing.values,
            "Percentage": [round((v / len(df)) * 100, 2) for v in cols_with_missing.values]
        }).sort_values(by="Missing Count", ascending=False)

        fig_missing = px.bar(
            missing_df, x="Column", y="Missing Count", text="Percentage",
            title="Missing Values Summary by Column",
            template=PLOT_THEME, color="Missing Count",
            color_continuous_scale="Reds"
        )
        fig_missing.update_traces(texttemplate="%{text}%", textposition="outside")
        dashboard_items.append({
            "id": "missing_summary",
            "title": "Missing Values Overview",
            "fig": fig_missing,
            "description": f"Identified {len(missing_df)} column(s) with null entries. Peak missingness: {missing_df.iloc[0]['Column']} ({missing_df.iloc[0]['Percentage']}%).",
            "data": missing_df
        })

    # 2. Time-series trend (if date column exists and numeric column exists)
    if datetime_cols and numeric_cols:
        dcol = datetime_cols[0]
        ncol = numeric_cols[0]
        # Aggregate by date to keep points manageable
        try:
            temp_df = df[[dcol, ncol]].dropna().copy()
            temp_df["DateOnly"] = pd.to_datetime(temp_df[dcol]).dt.date
            ts_agg = temp_df.groupby("DateOnly", as_index=False)[ncol].mean().sort_values(by="DateOnly")
            fig_ts = px.line(
                ts_agg, x="DateOnly", y=ncol,
                title=f"Temporal Trend: Average {ncol} Over Time",
                template=PLOT_THEME, markers=True
            )
            dashboard_items.append({
                "id": "time_trend",
                "title": f"Time Trend: {ncol}",
                "fig": fig_ts,
                "description": f"Displays the trajectory of '{ncol}' across dates. Earliest date: {ts_agg['DateOnly'].min()}, Latest: {ts_agg['DateOnly'].max()}.",
                "data": ts_agg
            })
        except Exception:
            pass

    # 3. Correlation Heatmap (if at least 2 numeric columns)
    if len(numeric_cols) >= 2:
        valid_num_cols = [c for c in numeric_cols if df[c].nunique() > 1][:8] # limit to 8 columns max
        if len(valid_num_cols) >= 2:
            corr_matrix = df[valid_num_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap (Numeric Columns)",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                template=PLOT_THEME
            )
            dashboard_items.append({
                "id": "correlation_heatmap",
                "title": "Numeric Correlation Matrix",
                "fig": fig_corr,
                "description": "Examines linear correlation between numerical features. Values near +1 or -1 indicate strong positive or negative associations.",
                "data": corr_matrix.reset_index()
            })

    # 4. Primary Numeric Distributions (Histograms with Box Plot marginal)
    for col in numeric_cols[:2]:
        series = df[col].dropna()
        if len(series) > 0 and series.nunique() > 1:
            mean_val = series.mean()
            median_val = series.median()
            fig_dist = px.histogram(
                df, x=col, marginal="box", nbins=30,
                title=f"Distribution of {col}",
                template=PLOT_THEME,
                color_discrete_sequence=["#1f77b4"]
            )
            fig_dist.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text="Mean")
            fig_dist.add_vline(x=median_val, line_dash="dot", line_color="green", annotation_text="Median")
            dashboard_items.append({
                "id": f"dist_{col}",
                "title": f"Distribution: {col}",
                "fig": fig_dist,
                "description": f"Statistical distribution for '{col}': Mean = {mean_val:.2f}, Median = {median_val:.2f}, Std Dev = {series.std():.2f}.",
                "data": df[[col]].describe().reset_index()
            })

    # 5. Top Categorical Column Frequency
    for col in categorical_cols[:2]:
        val_counts = df[col].value_counts().head(12).reset_index()
        val_counts.columns = [col, "Count"]
        if not val_counts.empty:
            fig_cat = px.bar(
                val_counts, x=col, y="Count", text="Count",
                title=f"Frequency Count: Top Categories in {col}",
                template=PLOT_THEME,
                color_discrete_sequence=["#2ca02c"]
            )
            fig_cat.update_traces(textposition="outside")
            dashboard_items.append({
                "id": f"cat_{col}",
                "title": f"Category Breakdown: {col}",
                "fig": fig_cat,
                "description": f"Shows top categories in '{col}'. Highest occurrence: '{val_counts.iloc[0][col]}' with {val_counts.iloc[0]['Count']} occurrences.",
                "data": val_counts
            })

    # 6. Category vs Numeric Aggregation (if both available)
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        # Keep top 10 categories
        top_cats = df[cat_col].value_counts().head(10).index
        filtered = df[df[cat_col].isin(top_cats)]
        if not filtered.empty:
            cat_agg = filtered.groupby(cat_col)[num_col].agg(["mean", "count"]).reset_index()
            cat_agg.columns = [cat_col, f"Avg_{num_col}", "Count"]
            cat_agg = cat_agg.sort_values(by=f"Avg_{num_col}", ascending=False)
            
            fig_comp = px.bar(
                cat_agg, x=cat_col, y=f"Avg_{num_col}", text=f"Avg_{num_col}",
                title=f"Average {num_col} by {cat_col}",
                template=PLOT_THEME,
                color=f"Avg_{num_col}",
                color_continuous_scale="Viridis"
            )
            fig_comp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            dashboard_items.append({
                "id": f"cat_num_{cat_col}_{num_col}",
                "title": f"Average {num_col} by {cat_col}",
                "fig": fig_comp,
                "description": f"Compares average '{num_col}' across dominant categories of '{cat_col}'. Top: '{cat_agg.iloc[0][cat_col]}' ({cat_agg.iloc[0][f'Avg_{num_col}']:.2f}).",
                "data": cat_agg
            })

    return dashboard_items
