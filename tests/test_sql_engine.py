import pytest
import pandas as pd
from modules.sql_engine import validate_sql_query, execute_sql_query

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 40],
        "score": [88.5, 92.0, 79.5, 95.0]
    })

def test_validate_safe_queries():
    assert validate_sql_query("SELECT * FROM data;")[0] is True
    assert validate_sql_query("SELECT name, age FROM data WHERE age > 25 ORDER BY score DESC;")[0] is True
    assert validate_sql_query("WITH high_scores AS (SELECT * FROM data WHERE score > 80) SELECT * FROM high_scores;")[0] is True

def test_reject_mutating_queries():
    assert validate_sql_query("DROP TABLE data;")[0] is False
    assert validate_sql_query("DELETE FROM data WHERE age < 30;")[0] is False
    assert validate_sql_query("INSERT INTO data VALUES ('Eve', 22, 90);")[0] is False
    assert validate_sql_query("UPDATE data SET score = 100;")[0] is False
    assert validate_sql_query("ALTER TABLE data ADD COLUMN extra TEXT;")[0] is False
    assert validate_sql_query("TRUNCATE data;")[0] is False

def test_execute_valid_query(sample_df):
    query = "SELECT name, score FROM data WHERE score > 80"
    ok, res, meta = execute_sql_query(query, sample_df)
    assert ok is True
    assert res is not None
    assert len(res) == 3
    assert list(res.columns) == ["name", "score"]
    assert meta["rows"] == 3
    assert meta["execution_time"] >= 0

def test_execute_empty_result(sample_df):
    query = "SELECT * FROM data WHERE age > 100"
    ok, res, meta = execute_sql_query(query, sample_df)
    assert ok is True
    assert res is not None
    assert len(res) == 0
    assert meta["is_empty"] is True

def test_execute_syntax_error(sample_df):
    query = "SELECT FORM data"  # Typo FORM
    ok, res, meta = execute_sql_query(query, sample_df)
    assert ok is False
    assert res is None
    assert "error" in meta

def test_execute_default_limit(sample_df):
    query = "SELECT * FROM data"
    ok, res, meta = execute_sql_query(query, sample_df, default_limit=2)
    assert ok is True
    assert len(res) == 2
