import pytest
import pandas as pd
import numpy as np
from modules.preprocessor import detect_quality_issues, apply_preprocessing

@pytest.fixture
def sample_dirty_data():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 5],
        "name": [" Alice ", "Bob", "charlie", "David", "Eve", "Eve"],
        "age": [25.0, 30.0, None, 40.0, 150.0, 150.0],  # Outlier 150, null
        "salary_str": ["$50,000", "$60,000", "$55,000", "$70,000", "$80,000", "$80,000"],
        "join_date": ["2023-01-10", "2023-02-15", "2023-03-20", "2023-04-25", "2023-05-30", "2023-05-30"],
        "department": ["sales", "Sales", "SALES", "HR", "Sales", "Sales"],
        "constant_feature": [99, 99, 99, 99, 99, 99]
    })

def test_quality_issues_detection(sample_dirty_data):
    issues_info = detect_quality_issues(sample_dirty_data)
    
    assert "score" in issues_info
    assert 0 <= issues_info["score"] <= 100
    
    detected_types = [iss["type"] for iss in issues_info["issues"]]
    assert "Duplicate Rows" in detected_types
    assert "Missing Values" in detected_types
    assert "Constant Column" in detected_types
    assert "Date Stored as Text" in detected_types
    assert "Leading/Trailing Whitespace" in detected_types
    assert "Inconsistent Categorical Casing" in detected_types
    assert "Numeric Stored as Text" in detected_types

def test_duplicate_removal(sample_dirty_data):
    orig_copy = sample_dirty_data.copy()
    config = {"remove_duplicates": True}
    cleaned, stats = apply_preprocessing(sample_dirty_data, config)
    
    assert len(cleaned) == 5  # 1 duplicate removed
    assert stats["duplicates_removed"] == 1
    # Check original dataset preservation
    pd.testing.assert_frame_equal(sample_dirty_data, orig_copy)

def test_missing_value_handling_median(sample_dirty_data):
    config = {
        "missing_strategies": {
            "age": {"strategy": "median"}
        }
    }
    cleaned, stats = apply_preprocessing(sample_dirty_data, config)
    assert cleaned["age"].isna().sum() == 0
    assert stats["missing_imputed"] == 1

def test_missing_value_handling_drop_rows(sample_dirty_data):
    config = {
        "missing_strategies": {
            "age": {"strategy": "drop_rows"}
        }
    }
    cleaned, stats = apply_preprocessing(sample_dirty_data, config)
    assert len(cleaned) == 5
    assert cleaned["age"].isna().sum() == 0

def test_text_and_casing_normalization(sample_dirty_data):
    config = {
        "trim_whitespace_cols": ["name"],
        "standardize_casing_cols": {"department": "title"}
    }
    cleaned, stats = apply_preprocessing(sample_dirty_data, config)
    assert cleaned["name"].iloc[0] == "Alice"
    assert cleaned["department"].iloc[0] == "Sales"
    assert cleaned["department"].iloc[2] == "Sales"
    assert stats["whitespace_trimmed"] >= 1
    assert stats["casing_standardized"] >= 1

def test_date_and_numeric_conversion(sample_dirty_data):
    config = {
        "convert_numeric_cols": ["salary_str"],
        "date_parse_cols": {"join_date": "%Y-%m-%d"}
    }
    cleaned, stats = apply_preprocessing(sample_dirty_data, config)
    assert pd.api.types.is_numeric_dtype(cleaned["salary_str"])
    assert cleaned["salary_str"].iloc[0] == 50000.0
    assert pd.api.types.is_datetime64_any_dtype(cleaned["join_date"])

def test_outlier_treatment_capping():
    # Construct series with clear outliers
    normal_vals = list(range(20, 40))  # 20 points
    outliers = [200, 250]
    df = pd.DataFrame({"metric": normal_vals + outliers})
    
    config = {
        "outlier_treatments": {"metric": "cap"}
    }
    cleaned, stats = apply_preprocessing(df, config)
    assert stats["outliers_treated"] == 2
    assert cleaned["metric"].max() < 200

def test_column_removal(sample_dirty_data):
    config = {
        "remove_columns": ["constant_feature"]
    }
    cleaned, stats = apply_preprocessing(sample_dirty_data, config)
    assert "constant_feature" not in cleaned.columns
    assert "constant_feature" in sample_dirty_data.columns  # Original preserved
