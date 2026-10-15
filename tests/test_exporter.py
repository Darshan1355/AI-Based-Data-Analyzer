import io
import pytest
import pandas as pd
from modules.exporter import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    multi_sheet_excel_bytes,
    generate_quality_report_markdown,
    generate_analysis_summary_markdown
)
from modules.preprocessor import detect_quality_issues

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer": ["Alice", "Bob", "Charlie"],
        "spend": [120.5, 340.0, 50.25],
        "status": ["active", "active", "churned"]
    })

def test_csv_export(sample_df):
    csv_bytes = dataframe_to_csv_bytes(sample_df)
    assert len(csv_bytes) > 0
    read_df = pd.read_csv(io.BytesIO(csv_bytes))
    pd.testing.assert_frame_equal(sample_df, read_df)

def test_excel_export(sample_df):
    excel_bytes = dataframe_to_excel_bytes(sample_df, sheet_name="TestSheet")
    assert len(excel_bytes) > 0
    read_df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="TestSheet")
    pd.testing.assert_frame_equal(sample_df, read_df)

def test_multi_sheet_excel(sample_df):
    sheets = {
        "SheetOne": sample_df,
        "SheetTwo": sample_df.head(1)
    }
    multi_bytes = multi_sheet_excel_bytes(sheets)
    assert len(multi_bytes) > 0
    xls = pd.ExcelFile(io.BytesIO(multi_bytes))
    assert "SheetOne" in xls.sheet_names
    assert "SheetTwo" in xls.sheet_names

def test_markdown_reports(sample_df):
    quality = detect_quality_issues(sample_df)
    q_rep = generate_quality_report_markdown(quality, "Customers.csv")
    assert "Customers.csv" in q_rep
    assert "Data Quality Score" in q_rep

    sum_rep = generate_analysis_summary_markdown(sample_df, sample_df, quality, "Customers.csv")
    assert "Executive Dataset Analysis" in sum_rep or "Dataset Summary" in sum_rep
    assert "customer" in sum_rep
