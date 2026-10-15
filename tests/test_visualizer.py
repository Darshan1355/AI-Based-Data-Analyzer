import pytest
import pandas as pd
from modules.visualizer import (
    recommend_chart_for_dataframe,
    create_custom_chart,
    generate_automated_dashboard
)

@pytest.fixture
def mixed_dataset():
    return pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "product": ["Widget A", "Widget B"] * 5,
        "quantity": [10, 15, 8, 22, 14, 19, 12, 25, 9, 16],
        "unit_price": [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0]
    })

def test_chart_recommendations(mixed_dataset):
    # Date + Numeric
    rec = recommend_chart_for_dataframe(mixed_dataset[["order_date", "quantity"]])
    assert rec["chart_type"] == "Line"
    assert rec["x"] == "order_date"

    # Categorical + Numeric
    rec2 = recommend_chart_for_dataframe(mixed_dataset[["product", "quantity"]])
    assert rec2["chart_type"] == "Bar"
    assert rec2["x"] == "product"

    # Two Numeric
    rec3 = recommend_chart_for_dataframe(mixed_dataset[["quantity", "unit_price"]])
    assert rec3["chart_type"] == "Scatter"

    # Single Numeric
    rec4 = recommend_chart_for_dataframe(mixed_dataset[["quantity"]])
    assert rec4["chart_type"] == "Histogram"

def test_create_custom_charts(mixed_dataset):
    # Bar chart
    fig, df_out, desc = create_custom_chart(mixed_dataset, "Bar", x_col="product", y_col="quantity")
    assert fig is not None
    assert df_out is not None

    # Line chart
    fig, df_out, desc = create_custom_chart(mixed_dataset, "Line", x_col="order_date", y_col="quantity")
    assert fig is not None

    # Scatter chart
    fig, df_out, desc = create_custom_chart(mixed_dataset, "Scatter", x_col="quantity", y_col="unit_price")
    assert fig is not None

    # Bar with Aggregation
    fig, df_out, desc = create_custom_chart(
        mixed_dataset, "Bar", x_col="product", y_col="quantity", aggregation="Sum"
    )
    assert fig is not None
    assert len(df_out) == 2  # Two distinct products

def test_automated_dashboard_generation(mixed_dataset):
    cards = generate_automated_dashboard(mixed_dataset)
    assert len(cards) >= 3
    card_ids = [c["id"] for c in cards]
    assert any("time_trend" in cid for cid in card_ids)
    assert any("dist_" in cid for cid in card_ids)
    assert any("cat_" in cid for cid in card_ids)
