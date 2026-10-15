import os
import re
from typing import Tuple, Optional, Any
from dotenv import load_dotenv

# Load .env if present, otherwise fall back to .env.example
if os.path.exists(".env"):
    try:
        load_dotenv(".env", encoding="utf-8-sig")
    except Exception:
        load_dotenv(".env")
elif os.path.exists(".env.example"):
    try:
        load_dotenv(".env.example", encoding="utf-8-sig")
    except Exception:
        load_dotenv(".env.example")

try:
    from langchain.chat_models import init_chat_model
except ImportError:
    from langchain_community.chat_models import init_chat_model  # type: ignore

# Default Groq Models (fallback list; fetch_groq_models() queries the account dynamically)
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "allam-2-7b"
]

def fetch_groq_models(api_key: Optional[str] = None) -> Tuple[bool, list, str]:
    """Queries Groq endpoint to retrieve available chat models for the current API key."""
    import requests
    effective_key = api_key.strip() if api_key and api_key.strip() else get_env_groq_key()
    fallback_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "allam-2-7b"
    ]
    if not effective_key:
        return False, fallback_models, "Groq API key not provided."
        
    try:
        headers = {"Authorization": f"Bearer {effective_key}"}
        resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            chat_models = [
                m["id"] for m in data 
                if not any(x in m["id"].lower() for x in ["whisper", "guard", "safeguard", "orpheus"])
            ]
            if chat_models:
                # Prioritize fast gpt-oss-20b if available
                if "openai/gpt-oss-20b" in chat_models:
                    chat_models.remove("openai/gpt-oss-20b")
                    chat_models.insert(0, "openai/gpt-oss-20b")
                return True, chat_models, f"Discovered {len(chat_models)} active Groq model(s)."
        return False, fallback_models, f"Groq responded with status {resp.status_code}."
    except Exception as e:
        return False, fallback_models, f"Could not fetch Groq models: {e}"

def get_env_groq_key() -> str:
    """
    Retrieve Groq API key with priority:
    1. Streamlit secrets (st.secrets["GROQ_API_KEY"]) - for Streamlit Community Cloud
    2. Environment variable (os.environ["GROQ_API_KEY"]) - from .env or system env
    Never exposes or prints the key.
    """
    # 1. Check Streamlit secrets (used in Streamlit Community Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip()
            if key and not key.startswith("your-") and not key.startswith("gsk_placeholder"):
                return key
    except Exception:
        pass

    # 2. Check environment variable (used in local development)
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key and not env_key.startswith("your-") and not env_key.startswith("gsk_placeholder"):
        return env_key

    return ""

def is_lm_studio_reachable(base_url: Optional[str] = "http://127.0.0.1:1234") -> bool:
    """Quickly tests if the LM Studio endpoint is actively reachable."""
    import requests
    norm_url = normalize_lm_studio_url(base_url)
    try:
        resp = requests.get(f"{norm_url}/models", timeout=0.8)
        return resp.status_code == 200
    except Exception:
        return False

def normalize_lm_studio_url(base_url: Optional[str] = None) -> str:
    """Normalizes LM Studio URL to ensure proper protocol and /v1 suffix."""
    url = (base_url or "http://127.0.0.1:1234").strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url

def fetch_lm_studio_models(base_url: Optional[str] = "http://127.0.0.1:1234") -> Tuple[bool, list, str]:
    """Queries LM Studio endpoint to retrieve loaded models."""
    import requests
    norm_url = normalize_lm_studio_url(base_url)
    default_model = "meta-llama-3.1-8b-instruct-abliterated"
    try:
        resp = requests.get(f"{norm_url}/models", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            models = [m.get("id") for m in data if m.get("id")]
            if models:
                if default_model not in models:
                    models.insert(0, default_model)
                else:
                    # Move default_model to front
                    models.remove(default_model)
                    models.insert(0, default_model)
                return True, models, f"Discovered {len(models)} model(s) from LM Studio."
            return True, [default_model], "Connected to LM Studio."
        return False, [default_model], f"LM Studio responded with HTTP status {resp.status_code}."
    except Exception as e:
        return False, [default_model], f"Could not connect to LM Studio at {norm_url} ({type(e).__name__})."

def initialize_llm(
    provider: str = "LM Studio (Local)",
    model: str = "meta-llama-3.1-8b-instruct-abliterated",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Tuple[bool, Any, str]:
    """
    Initializes the LLM instance based on provider selection.
    Supports Groq (cloud) and LM Studio (local OpenAI-compatible endpoint).
    Returns (success, llm_instance_or_none, message).
    """
    try:
        if provider == "Groq":
            effective_key = api_key.strip() if api_key and api_key.strip() else get_env_groq_key()
            if not effective_key:
                return False, None, "Groq API Key is missing. Please configure GROQ_API_KEY in Streamlit Secrets, .env, or enter it in the sidebar."
            
            effective_base_url = base_url.strip() if base_url and base_url.strip() else "https://api.groq.com/openai/v1"
            effective_model = model.strip() if model and model.strip() else "openai/gpt-oss-20b"
            llm = init_chat_model(
                model=effective_model,
                model_provider="openai",
                base_url=effective_base_url,
                api_key=effective_key,
                temperature=0.0
            )
            return True, llm, f"Connected to Groq Cloud ({effective_model})"
            
        elif provider == "LM Studio (Local)":
            effective_base_url = normalize_lm_studio_url(base_url or "http://127.0.0.1:1234")
            effective_model = model.strip() if model and model.strip() else "meta-llama-3.1-8b-instruct-abliterated"
            effective_key = api_key.strip() if api_key and api_key.strip() else "lm-studio"
            
            llm = init_chat_model(
                model=effective_model,
                model_provider="openai",
                base_url=effective_base_url,
                api_key=effective_key,
                temperature=0.0
            )
            return True, llm, f"Connected to LM Studio ({effective_model} at {effective_base_url})"
        else:
            return False, None, f"Unsupported provider: {provider}"
    except Exception as e:
        return False, None, f"Failed to initialize {provider} LLM: {str(e)}"

def clean_sql_output(raw_sql: str) -> str:
    """Strips markdown code fences and cleans SQL string."""
    cleaned = raw_sql.strip()
    # Remove markdown code blocks if any
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Strip leading/trailing whitespaces and semicolons for consistency
    cleaned = cleaned.strip()
    return cleaned

def generate_sql_query(llm: Any, df_schema_str: str, question: str) -> Tuple[bool, str]:
    """
    Generates a safe SQLite query for the given question and schema.
    Returns (success, sql_query_or_error_message).
    """
    if not llm:
        return False, "LLM is not initialized. Please configure Groq or LM Studio in the sidebar."

    llm_input = f"""
You are an expert SQLite developer.
Table Name: data
Table Schema (Column Types):
{df_schema_str}

User Question: {question}

Instructions:
1. Write a standard, valid SQLite SELECT query that answers the user question.
2. The table name in the query must be 'data'.
3. Only write read-only SELECT queries. Do not include INSERT, UPDATE, DELETE, DROP, or any mutating statements.
4. Output the raw SQL query in plain text ONLY. Do NOT use markdown code blocks (no ```sql).
5. Do NOT include any explanations, greetings, or notes.
6. If the question cannot be answered by the schema, output only 'Error: Cannot answer question with provided schema'.
"""
    try:
        response = llm.invoke(llm_input)
        content = response.content if hasattr(response, "content") else str(response)
        cleaned = clean_sql_output(content)
        
        if "error:" in cleaned.lower() or cleaned.strip() == "Error":
            return False, cleaned
            
        return True, cleaned
    except Exception as e:
        err_msg = str(e)
        if "Connection" in err_msg or "10054" in err_msg or "ConnectError" in err_msg:
            return False, f"Connection failed: Unable to reach LLM endpoint. If running on Streamlit Cloud with LM Studio, note that localhost cannot reach your local PC directly. For cloud deployment, switch to Groq Cloud in the sidebar or configure a public tunnel. ({err_msg})"
        return False, f"LLM Query Generation failed: {err_msg}"

def explain_query_results(llm: Any, df_schema_str: str, query: str, result_summary_str: str) -> Tuple[bool, str]:
    """
    Generates a natural language explanation of the query result.
    Returns (success, explanation_or_error).
    """
    if not llm:
        return False, "LLM is not initialized."

    llm_input_explain = f"""
Table Name: data 
Table Schema:
{df_schema_str}

Executed SQL Query:
{query}

Query Result Preview:
{result_summary_str}

Instruction:
Provide a clear, concise, professional English explanation of what this query computed and what the result means based on the data.
"""
    try:
        explanation = llm.invoke(llm_input_explain)
        content = explanation.content if hasattr(explanation, "content") else str(explanation)
        return True, content.strip()
    except Exception as e:
        return False, f"Explanation failed: {str(e)}"
