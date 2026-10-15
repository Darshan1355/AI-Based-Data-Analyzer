# 📊 AI-Based Data Analyzer

An intelligent, interactive data analytics and profiling web platform powered by Streamlit, Pandas, Plotly, Groq API, and LM Studio.

The application allows users to upload raw tabular datasets (CSV), automatically audit data quality, perform natural language SQL queries, preprocess and clean data non-destructively, generate interactive visual dashboards, and export clean datasets and reports in multiple formats.

---

## 🌟 Key Features

* **💬 AI Natural-Language Query Generator**:
  * Ask questions about your data in plain English.
  * Generates clean, read-only SQLite queries.
  * Validates SQL queries to strictly prohibit mutating statements (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, etc.).
  * Executes queries locally in-memory using `pandasql`.
  * Multi-view results: Sortable **Table View**, **Chart View** with automated chart type recommendations, and syntax-highlighted **SQL View**.
  * Download query results as CSV, Excel (`.xlsx`), or SQL (`.sql`).

* **🛠️ Automated Data Quality Audit & Preprocessing Assistant**:
  * Scans datasets for missing values, exact duplicates, text stored as dates, numbers stored as text, whitespace, inconsistent categorical casing, constant columns, and statistical outliers (1.5x IQR).
  * Computes an evidence-based **Data Quality Score (0–100)** with transparent deductions.
  * Preprocessing operations: deduplication, median/mean/mode/constant imputation, whitespace trimming, casing normalization, date parsing, outlier capping (Winsorization), and column dropping.
  * **Non-destructive**: Preview proposed modifications before applying; original data is never overwritten and can be restored at any time via **Reset to Original**.

* **📈 Automated Visual Analytics Dashboard**:
  * Instant visualization suite generated upon CSV upload: missing value summaries, numeric distribution histograms with mean/median markers, correlation heatmaps, categorical frequency counts, and temporal trendlines.
  * **Custom Visual Explorer**: Build custom interactive Plotly charts (Bar, Line, Scatter, Pie, Histogram, Box) with custom axes, colors, and aggregations.
  * Export summarized chart data to CSV.

* **📥 Multi-Format Export Center**:
  * One-click downloads for original and cleaned datasets in both CSV and formatted Excel (`.xlsx`) workbooks.
  * Download generated SQL queries.
  * Export Markdown Data Quality Audit Reports and Executive Dataset Analysis Summaries.

* **🤖 Dual LLM Backend**:
  * **LM Studio (Local-Only Mode)**: Connect to local offline models (e.g. `meta-llama-3.1-8b-instruct-abliterated` at `http://127.0.0.1:1234`). Guarantees zero cloud data transmission.
  * **Groq Cloud**: Lightning-fast cloud inference with Llama 3 models using the Groq API.

---

## 🏗️ Architecture & Project Structure

```
├── main.py                     # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── sample_data.csv             # Synthetic benchmark dataset for testing
├── .gitignore                  # Git exclusions (secrets, caches, virtualenvs)
├── .env.example                # Example environment file template
├── .streamlit/
│   └── secrets.toml.example    # Streamlit Cloud secrets template
├── modules/
│   ├── __init__.py             # Package init
│   ├── llm_manager.py          # Groq & LM Studio provider integration
│   ├── sql_engine.py           # Safe SQL validation and pandasql execution
│   ├── preprocessor.py         # Quality issue detection and non-destructive cleaning
│   ├── visualizer.py           # Automated Plotly dashboard and chart recommendation
│   └── exporter.py             # CSV, Excel, and Markdown report generators
└── tests/
    ├── test_sql_engine.py      # Unit tests for SQL safety and execution
    ├── test_preprocessor.py    # Unit tests for data quality and cleaning
    ├── test_visualizer.py      # Unit tests for charting and dashboard
    ├── test_exporter.py        # Unit tests for CSV/Excel/report exports
    └── test_llm_manager.py     # Unit tests for LLM providers and secrets handling
```

---

## 💻 Local Setup & Execution

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12 (Python 3.13 also supported)
* Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone <your-repository-url>
cd Data_analyzer

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Unit Tests
```bash
python -m pytest -v
```

### 4. Run the Streamlit Application
```bash
streamlit run main.py
```
Open your browser to `http://localhost:8501`.

### 5. Using Local LM Studio
1. Start LM Studio on your computer.
2. Load your model (e.g., `meta-llama-3.1-8b-instruct-abliterated`).
3. Start the local server on `http://127.0.0.1:1234` with CORS enabled.
4. In the application sidebar, select **LM Studio (Local)**. The app will auto-detect your running server and model.

---

## ☁️ Streamlit Community Cloud Deployment Guide

### Step 1: Push Code to GitHub
Ensure your local branch is committed and pushed to GitHub:
```bash
git add .
git commit -m "Prepare AI Data Analyzer for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Create App on Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
2. Click **New app**.
3. Select your repository: `Data_analyzer`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `main.py`.
6. Select **Python version**: `3.11` (or `3.12`).

### Step 3: Configure Cloud Secrets
In the Streamlit Cloud deployment wizard, open **Advanced settings > Secrets** (or navigate to **App Settings > Secrets** after creating the app).

Paste your Groq API key:
```toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
```

Save and click **Deploy!**

> ℹ️ **Note about LM Studio in the Cloud:** When deployed on Streamlit Community Cloud, `localhost` refers to the cloud container, not your home PC. The application will automatically detect this and default to **Groq Cloud**. If you wish to connect your local LM Studio to the cloud app, expose your local port via a secure tunnel (e.g. ngrok or Cloudflare Tunnel) and provide the HTTPS URL in the sidebar.

---

## 🔒 Security & Privacy

* **Strict Read-Only SQL**: The application parses and validates every generated query before execution. Mutating commands such as `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, and `TRUNCATE` are blocked immediately.
* **Secret Protection**: API keys are retrieved from `st.secrets` or environment variables and are masked in the UI. Keys are never printed in console logs or embedded in generated files.
* **Local-Only Guarantee**: When **LM Studio (Local)** is selected, all prompts, schema information, and query results remain strictly within your local environment.

---

## 📄 License
MIT License.
