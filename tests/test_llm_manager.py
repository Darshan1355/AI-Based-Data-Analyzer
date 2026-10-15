import pytest
from modules.llm_manager import (
    clean_sql_output,
    initialize_llm,
    generate_sql_query,
    normalize_lm_studio_url,
    is_lm_studio_reachable,
    fetch_lm_studio_models,
    get_env_groq_key
)

def test_clean_sql_output():
    raw_markdown = "```sql\nSELECT * FROM data WHERE age > 30;\n```"
    cleaned = clean_sql_output(raw_markdown)
    assert cleaned == "SELECT * FROM data WHERE age > 30;"

    raw_generic = "```\nSELECT count(*) FROM data\n```"
    assert clean_sql_output(raw_generic) == "SELECT count(*) FROM data"

def test_normalize_lm_studio_url():
    assert normalize_lm_studio_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234/v1"
    assert normalize_lm_studio_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"
    assert normalize_lm_studio_url("127.0.0.1:1234") == "http://127.0.0.1:1234/v1"
    assert normalize_lm_studio_url("https://my-tunnel.ngrok-free.app/") == "https://my-tunnel.ngrok-free.app/v1"

def test_lm_studio_initialization():
    ok, llm, msg = initialize_llm(
        provider="LM Studio (Local)",
        model="meta-llama-3.1-8b-instruct-abliterated",
        base_url="http://127.0.0.1:1234"
    )
    assert ok is True
    assert llm is not None
    assert "LM Studio" in msg

def test_groq_missing_key():
    ok, llm, msg = initialize_llm(
        provider="Groq",
        api_key="   "
    )
    assert isinstance(ok, bool)
    assert isinstance(msg, str)

def test_generate_sql_with_no_llm():
    ok, err = generate_sql_query(None, "name: object", "Show names")
    assert ok is False
    assert "not initialized" in err.lower()

def test_unreachable_lm_studio():
    # Connecting to an invalid port should safely return False without crashing
    reachable = is_lm_studio_reachable("http://127.0.0.1:59999")
    assert reachable is False
    
    ok, models, msg = fetch_lm_studio_models("http://127.0.0.1:59999")
    assert ok is False
    assert "meta-llama-3.1-8b-instruct-abliterated" in models

def test_get_env_groq_key_type():
    key = get_env_groq_key()
    assert isinstance(key, str)
