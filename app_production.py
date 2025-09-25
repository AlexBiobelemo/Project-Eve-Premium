import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
import io
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import time
import base64
import warnings
import pickle
from scipy import stats
import threading
import queue
import gc
import psutil
import os
from pathlib import Path

# Pandas import with availability check
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    PANDAS_VERSION = pd.__version__
except ImportError:
    PANDAS_AVAILABLE = False
    PANDAS_VERSION = "Not installed"

# SQL and Database libraries (optional imports)
try:
    import sqlite3
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import StaticPool
    SQL_AVAILABLE = True
    SQL_VERSION = sqlalchemy.__version__
except ImportError as e:
    SQL_AVAILABLE = False
    SQL_VERSION = "Not installed"
    # Don't show warning here, show it in the SQL tab instead

# Only import ML libraries if available
try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
    from sklearn.metrics import (
        mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score,
        silhouette_score, davies_bouldin_score, calinski_harabasz_score,
        mean_absolute_error, explained_variance_score, max_error, median_absolute_error,
        balanced_accuracy_score, matthews_corrcoef, roc_auc_score
    )
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.feature_selection import RFE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning(" Machine Learning features disabled: scikit-learn not available")

warnings.filterwarnings('ignore')

# --- Production Configuration ---
st.set_page_config(
    page_title="ProjectEve Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "ProjectEve Enterprise Analytics Platform - Advanced Data Intelligence Suite"
    }
)

# Custom CSS for professional styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Streamlit's theme variables for better integration */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
    
    /* Adaptive theme colors that work with both light and dark modes */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --info-color: #17a2b8;
        --light-gray: rgba(248, 249, 250, 0.8);
        --dark-gray: rgba(52, 58, 64, 0.9);
        --card-bg: rgba(255, 255, 255, 0.95);
        --border-color: rgba(0, 0, 0, 0.1);
        --shadow-light: rgba(0, 0, 0, 0.08);
        --text-primary: inherit;
        --text-secondary: rgba(0, 0, 0, 0.6);
    }
    
    /* Dark mode adaptations */
    @media (prefers-color-scheme: dark) {
        :root {
            --light-gray: rgba(32, 33, 35, 0.8);
            --dark-gray: rgba(255, 255, 255, 0.9);
            --card-bg: rgba(32, 33, 35, 0.95);
            --border-color: rgba(255, 255, 255, 0.1);
            --shadow-light: rgba(0, 0, 0, 0.3);
            --text-secondary: rgba(255, 255, 255, 0.6);
        }
    }
    
    /* Streamlit integration - blend with native components */
    .stApp {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Keep sidebar functionality natural - don't override Streamlit's built-in behavior */
    
    /* Adaptive header that blends with Streamlit */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--info-color) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 16px var(--shadow-light);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        opacity: 0.95;
    }
    
    /* Adaptive KPI Cards that blend with Streamlit metrics */
    .kpi-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px var(--shadow-light);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-light);
    }
    
    .kpi-value {
        font-size: clamp(1.5rem, 3vw, 2rem);
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
        line-height: 1.2;
    }
    
    .kpi-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin: 0.25rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Adaptive section headers */
    .section-header {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 2px 8px var(--shadow-light);
    }
    
    .section-header h3 {
        margin: 0;
        color: var(--text-primary);
        font-size: clamp(1.1rem, 2.5vw, 1.3rem);
        font-weight: 600;
    }
    
    .section-header p {
        margin: 0.5rem 0 0 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Executive summary boxes that blend naturally */
    .exec-summary {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px var(--shadow-light);
        transition: transform 0.2s ease;
    }
    
    .exec-summary:hover {
        transform: translateY(-1px);
    }
    
    .exec-summary h4 {
        margin: 0 0 1rem 0;
        color: var(--primary-color);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .exec-summary p {
        margin: 0;
        line-height: 1.6;
        color: var(--text-primary);
    }
    
    /* Status badges with better integration */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: 1px solid transparent;
    }
    
    .status-success {
        background: rgba(40, 167, 69, 0.1);
        color: var(--success-color);
        border-color: rgba(40, 167, 69, 0.2);
    }
    
    .status-warning {
        background: rgba(255, 193, 7, 0.1);
        color: #856404;
        border-color: rgba(255, 193, 7, 0.2);
    }
    
    .status-error {
        background: rgba(220, 53, 69, 0.1);
        color: var(--danger-color);
        border-color: rgba(220, 53, 69, 0.2);
    }
    
    /* Responsive design for mobile and tablets */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        
        .kpi-card, .section-header, .exec-summary {
            padding: 1rem;
        }
        
        .kpi-value {
            font-size: 1.5rem;
        }
        
        /* Let Streamlit handle sidebar responsiveness naturally */
    }
    
    /* Subtle animations for better UX */
    .kpi-card, .section-header, .exec-summary {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Gentle Streamlit branding removal - keep functionality intact */
    #MainMenu {visibility: visible;}
    #footer {visibility: visible;}
    
    /* Improve spacing with Streamlit elements */
    .stMetric {
        margin: 0.5rem 0;
    }
    
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px var(--shadow-light);
    }
    </style>
    """, unsafe_allow_html=True)

# Professional header component
def render_professional_header():
    st.markdown("""
    <div class="main-header">
        <h1>Project Eve</h1>
        <p>Enterprise Data Intelligence Suite</p>
    </div>
    """, unsafe_allow_html=True)

# Professional data display utilities
def create_kpi_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create a professional KPI card"""
    delta_html = ""
    if delta:
        color_class = "success" if delta_color == "normal" else delta_color
        delta_html = f'<p style="color: var(--{color_class}-color); margin: 0.5rem 0 0 0; font-size: 0.9rem;">{delta}</p>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_section_header(title: str, description: str = None):
    """Create a professional section header"""
    desc_html = f'<p style="margin: 0.5rem 0 0 0; color: #6c757d;">{description}</p>' if description else ""
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
        {desc_html}
    </div>
    """, unsafe_allow_html=True)

def create_status_badge(status: str, badge_type: str = "success"):
    """Create a status badge"""
    return f'<span class="status-badge status-{badge_type}">{status}</span>'

def create_professional_table(df: pd.DataFrame, title: str = None, max_rows: int = None):
    """Create a professional table display with native Streamlit features"""
    if title:
        st.subheader(title)
    
    # Show table with native Streamlit dataframe (includes search, download, expand options)
    if max_rows and len(df) > max_rows:
        st.info(f"Showing first {max_rows:,} of {len(df):,} rows. Full table available below.")
        display_df = df.head(max_rows)
    else:
        display_df = df
    
    # Native Streamlit dataframe with all built-in features
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

def create_executive_summary(title: str, content: str):
    """Create an executive summary box"""
    st.markdown(f"""
    <div class="exec-summary">
        <h4 style="margin: 0 0 1rem 0; color: var(--primary-color);">{title}</h4>
        <p style="margin: 0; line-height: 1.6;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def format_large_number(num):
    """Format large numbers with appropriate suffixes"""
    if pd.isna(num):
        return "N/A"
    if isinstance(num, str):
        return num
    
    abs_num = abs(num)
    if abs_num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}" if isinstance(num, float) else str(num)

def create_metrics_grid(metrics_dict: dict, columns: int = 4):
    """Create a grid of metrics with professional styling"""
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics_dict.items()):
        with cols[i % columns]:
            formatted_value = format_large_number(value) if isinstance(value, (int, float)) else str(value)
            create_kpi_card(label, formatted_value)

# --- Enhanced Constants ---
FILE_TYPES: List[str] = ["CSV", "Excel", "JSON"]
CHART_OPTIONS: List[str] = [
    "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot",
    "Correlation Heatmap", "Pie Chart", "Violin Plot", "Map View", "Anomaly Plot"
]
AGG_OPTIONS: List[str] = ['None', 'mean', 'sum', 'median', 'count', 'min', 'max', 'std', 'var']
THEME_OPTIONS: List[str] = ["plotly", "plotly_dark", "seaborn", "ggplot2", "simple_white", "presentation"]
ACCESSIBILITY_THEMES: List[str] = ["Light", "Dark", "Colorblind Friendly"]
COLOR_PALETTES: List[str] = ["Viridis", "Plasma", "Inferno", "Magma", "Turbo", "Cividis", "Blues", "Greens"]
ML_MODELS: List[str] = ["RandomForest", "MLP", "IsolationForest"] if SKLEARN_AVAILABLE else []
CLEANING_METHODS: List[str] = ["mean", "median", "mode", "drop", "forward_fill", "backward_fill", "interpolate"]
ANOMALY_METHODS: List[str] = ["IsolationForest", "Z-Score", "Modified Z-Score", "IQR"] if SKLEARN_AVAILABLE else ["Z-Score", "Modified Z-Score", "IQR"]

# Import NLP Testing module
try:
    from nlp_test_integration import render_nlp_test_ui
    NLP_TESTING_AVAILABLE = True
except ImportError:
    NLP_TESTING_AVAILABLE = False

# Memory and performance limits
MAX_MEMORY_MB = 1024  # 1GB memory limit
MAX_ROWS_DISPLAY = 50000
MAX_ROWS_PROCESSING = 100000
SAMPLE_SIZE_LARGE = 15000
SAMPLE_SIZE_MEDIUM = 5000

# --- Production Logging Setup ---
def setup_production_logging():
    """Setup comprehensive logging for production environment."""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_dir / f"enterprise_app_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler for errors only
        error_handler = logging.FileHandler(log_dir / f"enterprise_errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        
        return True
    except Exception as e:
        st.error(f"Logging setup failed: {e}")
        return False

# Initialize logging
LOGGING_ENABLED = setup_production_logging()

# --- Production Session State Management ---
def init_session_state():
    """Initialize session state with production defaults and validation."""
    try:
        session_keys = {
            'chart_configs': [],
            'data_loaded': False,
            'filter_state': {},
            'last_uploaded_files': [],
            'dfs': {},
            'selected_df': None,
            'trained_models': {},
            'eda_cache': {},
            'performance_metrics': {},
            'chat_history': [],
            'cleaning_suggestions': {},
            'anomaly_results': {},
            'theme_preference': "Light",
            'accessibility_mode': False,
            'preview_data': {},
            'memory_usage': 0,
            'processing_errors': [],
            'app_start_time': time.time(),
            'user_preferences': {
                'auto_sample': True,
                'show_warnings': True,
                'enable_caching': True,
                'performance_mode': 'balanced'
            },
            'sql_connections': {},
            'sql_query_history': [],
            'sql_results': {},
            'file_workflows': {},  # Store workflow state for each file
            'active_file_tabs': [],  # Track open files
            'current_file_tab': None,  # Currently selected file tab
            'merge_configuration': {}  # CSV merge settings
        }
        
        for key, default_value in session_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
        # Memory monitoring
        if 'memory_monitor' not in st.session_state:
            st.session_state.memory_monitor = {'last_check': time.time(), 'peak_usage': 0}
            
    except Exception as e:
        if LOGGING_ENABLED:
            logging.error(f"Session state initialization error: {e}")
        st.error(f"Session initialization failed: {e}")

# --- Memory and Performance Monitoring ---
def monitor_memory_usage() -> Dict[str, Any]:
    """Monitor application memory usage."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        cpu_percent = process.cpu_percent()
        
        # Update peak usage
        if memory_mb > st.session_state.memory_monitor['peak_usage']:
            st.session_state.memory_monitor['peak_usage'] = memory_mb
        
        st.session_state.memory_usage = memory_mb
        
        return {
            'current_mb': memory_mb,
            'peak_mb': st.session_state.memory_monitor['peak_usage'],
            'cpu_percent': cpu_percent,
            'warning': memory_mb > MAX_MEMORY_MB * 0.8,
            'critical': memory_mb > MAX_MEMORY_MB
        }
    except Exception as e:
        if LOGGING_ENABLED:
            logging.error(f"Memory monitoring error: {e}")
        return {'error': str(e)}

def cleanup_memory():
    """Cleanup memory and trigger garbage collection."""
    try:
        # Clear old cached data
        current_time = time.time()
        
        # Clear old EDA cache (older than 30 minutes)
        keys_to_remove = []
        for key, value in st.session_state.eda_cache.items():
            if isinstance(value, dict) and 'timestamp' in value:
                if current_time - value['timestamp'] > 1800:  # 30 minutes
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state.eda_cache[key]
        
        # Clear old trained models (keep only last 5)
        if len(st.session_state.trained_models) > 5:
            sorted_models = sorted(
                st.session_state.trained_models.items(),
                key=lambda x: x[1].get('timestamp', 0),
                reverse=True
            )
            st.session_state.trained_models = dict(sorted_models[:5])
        
        # Force garbage collection
        gc.collect()
        
        if LOGGING_ENABLED:
            logging.info("Memory cleanup completed")
            
    except Exception as e:
        if LOGGING_ENABLED:
            logging.error(f"Memory cleanup error: {e}")

# --- Enhanced Data Validation ---
def validate_dataframe(df: pd.DataFrame, max_size_mb: float = MAX_MEMORY_MB) -> Dict[str, Any]:
    """Comprehensive dataframe validation."""
    try:
        if df is None or df.empty:
            return {'valid': False, 'error': 'DataFrame is empty or None'}
        
        # Size validation
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_usage > max_size_mb:
            return {'valid': False, 'error': f'DataFrame too large: {memory_usage:.1f}MB > {max_size_mb}MB'}
        
        # Shape validation
        if len(df) > MAX_ROWS_PROCESSING:
            return {'valid': False, 'warning': f'Large dataset: {len(df):,} rows. Consider sampling.'}
        
        # Column validation
        if len(df.columns) > 1000:
            return {'valid': False, 'error': f'Too many columns: {len(df.columns)} > 1000'}
        
        # Data type validation
        problematic_dtypes = []
        for col, dtype in df.dtypes.items():
            if str(dtype) == 'object':
                # Check for mixed types
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    types = set(type(x).__name__ for x in sample_values)
                    if len(types) > 2:  # Allow some mixed types
                        problematic_dtypes.append(col)
        
        validation_result = {
            'valid': True,
            'memory_mb': memory_usage,
            'shape': df.shape,
            'dtypes_count': df.dtypes.value_counts().to_dict(),
            'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'problematic_columns': problematic_dtypes
        }
        
        if problematic_dtypes:
            validation_result['warnings'] = f"Mixed data types in columns: {', '.join(problematic_dtypes[:5])}"
        
        return validation_result
        
    except Exception as e:
        return {'valid': False, 'error': f'Validation failed: {str(e)}'}

# --- Enhanced Data Loading with Production Features ---
@st.cache_data(show_spinner="Loading data with enterprise optimization...", max_entries=3, ttl=3600)
def load_data_production(file_content: bytes, file_name: str, file_type: str, 
                        encoding_fallback: bool = True) -> Optional[pd.DataFrame]:
    """Production-grade data loading with comprehensive error handling."""
    try:
        start_time = time.time()
        df = None
        
        # File size check
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > MAX_MEMORY_MB:
            st.warning(f"Large file detected: {file_size_mb:.1f}MB. Loading may be slow.")
        
        # Load based on file type
        if file_type == "CSV":
            # Try multiple encodings for robustness
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16'] if encoding_fallback else ['utf-8']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_content),
                        encoding=encoding,
                        low_memory=False,
                        on_bad_lines='warn'  # Don't fail on bad lines
                    )
                    break
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    if encoding == encodings[-1]:  # Last encoding attempt
                        raise e
                    continue
                    
        elif file_type == "Excel":
            try:
                # Try openpyxl first
                df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            except Exception:
                try:
                    # Fallback to xlrd for older Excel files
                    df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
                except Exception as e:
                    raise ValueError(f"Excel reading failed with both engines: {e}")
                    
        elif file_type == "JSON":
            try:
                df = pd.read_json(io.BytesIO(file_content))
            except ValueError:
                # Try reading as lines of JSON
                df = pd.read_json(io.BytesIO(file_content), lines=True)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if df is None or df.empty:
            st.warning(f"File {file_name} appears to be empty or could not be parsed")
            return None
        
        # Data validation
        validation_result = validate_dataframe(df)
        if not validation_result['valid']:
            if 'error' in validation_result:
                raise ValueError(validation_result['error'])
            elif 'warning' in validation_result:
                st.warning(validation_result['warning'])
        
        # Enhanced data cleaning
        original_shape = df.shape
        
        # Clean column names
        df.columns = [
            re.sub(r'[^\w\s]', '_', str(col)).strip('_').replace(' ', '_')
            for col in df.columns
        ]
        
        # Handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Sample large datasets
        if len(df) > MAX_ROWS_PROCESSING:
            sample_size = min(MAX_ROWS_PROCESSING, len(df))
            df = df.sample(n=sample_size, random_state=42)
            st.info(f"Large dataset sampled: {sample_size:,} of {original_shape[0]:,} rows")
        
        # Calculate quality score
        quality_score = calculate_data_quality_score_enhanced(df)
        
        # Add comprehensive metadata
        load_time = time.time() - start_time
        df.attrs = {
            'source_file': file_name,
            'load_time': datetime.now().isoformat(),
            'processing_time': load_time,
            'quality_score': quality_score,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'validation_result': validation_result,
            'encoding_used': 'utf-8',  # Default assumption
            'file_size_mb': file_size_mb
        }
        
        if LOGGING_ENABLED:
            logging.info(f"Successfully loaded {file_name}: {df.shape} in {load_time:.2f}s")
        
        return df
        
    except Exception as e:
        error_msg = f"Failed to load {file_name}: {str(e)}"
        if LOGGING_ENABLED:
            logging.error(f"Data loading error: {error_msg}")
        st.error(error_msg)
        return None

def calculate_data_quality_score_enhanced(df: pd.DataFrame) -> float:
    """Enhanced data quality scoring with more comprehensive metrics."""
    try:
        if df is None or df.empty:
            return 0.0
            
        score = 100.0
        
        # Missing data penalty (0-40 points)
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 40
        
        # Duplicate rows penalty (0-20 points)
        duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
        score -= duplicate_ratio * 20
        
        # Data type consistency reward (0-15 points)
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        total_cols = len(df.columns)
        if total_cols > 0:
            score += (numeric_cols / total_cols) * 15
        
        # Column name quality (0-10 points)
        clean_names = sum(1 for col in df.columns if col.isidentifier())
        if total_cols > 0:
            score += (clean_names / total_cols) * 10
        
        # Data distribution quality (0-15 points)
        if numeric_cols > 0:
            numeric_data = df.select_dtypes(include=[np.number])
            # Check for constant columns
            constant_cols = sum(1 for col in numeric_data.columns if numeric_data[col].nunique() <= 1)
            if numeric_cols > 0:
                score += ((numeric_cols - constant_cols) / numeric_cols) * 15
        
        return max(0.0, min(100.0, score))
        
    except Exception as e:
        if LOGGING_ENABLED:
            logging.error(f"Quality score calculation error: {e}")
        return 50.0  # Default score on error

# --- Enhanced EDA Functions ---
@st.cache_data(show_spinner="Computing advanced analytics...", max_entries=3, ttl=1800)
def compute_eda_summary_enhanced(df_hash: str, shape: Tuple[int, int], 
                               sample_size: int = None) -> Dict[str, Any]:
    """Enhanced EDA with production-grade performance and insights."""
    try:
        df = st.session_state.selected_df
        if df is None or df.empty:
            return {'error': 'No data available'}
        
        # Use sampling for large datasets
        if sample_size is None:
            sample_size = min(SAMPLE_SIZE_LARGE, len(df))
        
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
        
        insights = []
        recommendations = []
        
        # Basic info
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df_sample.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Dataset size insights
        if len(df) > 50000:
            insights.append("Large dataset detected - using optimized processing")
            recommendations.append("Consider enabling performance mode for faster processing")
        
        # Missing data analysis
        missing_data = df_sample.isnull().sum().sum()
        missing_pct = (missing_data / (len(df_sample) * len(df_sample.columns))) * 100
        
        if missing_pct > 20:
            insights.append(f"High missing data rate: {missing_pct:.1f}%")
            recommendations.append("Consider data imputation or removal of sparse columns")
        elif missing_pct > 5:
            insights.append(f"Moderate missing data: {missing_pct:.1f}%")
        
        # Data types analysis
        if len(numeric_cols) > len(categorical_cols):
            insights.append("Numeric-heavy dataset - suitable for statistical analysis and ML")
            recommendations.append("Consider correlation analysis and regression modeling")
        elif len(categorical_cols) > len(numeric_cols):
            insights.append("Category-heavy dataset - ideal for classification and segmentation")
            recommendations.append("Consider clustering and classification approaches")
        
        # Data quality insights
        quality_score = df.attrs.get('quality_score', 0)
        if quality_score > 85:
            insights.append("High data quality - ready for advanced analytics")
        elif quality_score > 70:
            insights.append("Good data quality with minor issues")
            recommendations.append("Review data cleaning suggestions")
        else:
            insights.append("Data quality issues detected")
            recommendations.append("Prioritize data cleaning before analysis")
        
        # Memory and performance insights
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_usage > 100:
            insights.append(f"Large memory footprint: {memory_usage:.1f}MB")
            recommendations.append("Consider data sampling for faster processing")
        
        # Statistical insights for numeric data
        if len(numeric_cols) > 0:
            try:
                correlation_matrix = df_sample[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr_val = abs(correlation_matrix.iloc[i, j])
                        if corr_val > 0.8 and not np.isnan(corr_val):
                            high_corr_pairs.append((
                                correlation_matrix.columns[i],
                                correlation_matrix.columns[j],
                                corr_val
                            ))
                
                if high_corr_pairs:
                    insights.append(f"High correlations found between {len(high_corr_pairs)} feature pairs")
                    recommendations.append("Consider feature selection to reduce redundancy")
                
                # Outlier detection summary
                outlier_cols = []
                for col in numeric_cols[:5]:  # Check first 5 numeric columns
                    col_data = df_sample[col].dropna()
                    if len(col_data) > 10:
                        z_scores = np.abs(stats.zscore(col_data))
                        if (z_scores > 3).sum() / len(col_data) > 0.05:  # More than 5% outliers
                            outlier_cols.append(col)
                
                if outlier_cols:
                    insights.append(f"Outliers detected in {len(outlier_cols)} columns")
                    recommendations.append("Consider outlier treatment for robust analysis")
                    
            except Exception as e:
                if LOGGING_ENABLED:
                    logging.warning(f"Statistical analysis error: {e}")
        
        return {
            'shape': shape,
            'sample_size': len(df_sample),
            'insights': insights,
            'recommendations': recommendations,
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'missing_values': int(missing_data),
            'missing_percentage': missing_pct,
            'memory_usage_mb': memory_usage,
            'quality_score': quality_score,
            'timestamp': time.time()
        }
        
    except Exception as e:
        error_msg = f"EDA computation error: {str(e)}"
        if LOGGING_ENABLED:
            logging.error(error_msg)
        return {'error': error_msg}

# --- Enhanced Anomaly Detection ---
@st.cache_data(show_spinner="Detecting anomalies with advanced methods...", max_entries=2, ttl=1800)
def detect_anomalies_production(df_hash: str, method: str, params: Dict[str, Any],
                              features: List[str] = None) -> Dict[str, Any]:
    """Production-grade anomaly detection with multiple methods and validation."""
    try:
        df = st.session_state.selected_df
        if df is None or df.empty:
            return {"error": "No data available"}
        
        # Feature selection
        if features:
            numeric_cols = [col for col in features if col in df.columns and df[col].dtype in ['int64', 'float64']]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {"error": "No numeric columns available for anomaly detection"}
        
        # Sample for performance
        sample_size = min(SAMPLE_SIZE_LARGE, len(df))
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            sample_note = f" (analyzed {len(df_sample):,} sample rows)"
        else:
            df_sample = df
            sample_note = ""
        
        # Prepare data
        data = df_sample[numeric_cols].dropna()
        if len(data) == 0:
            return {"error": "No valid data points after removing missing values"}
        
        original_index = data.index
        
        # Apply anomaly detection method
        start_time = time.time()
        
        if method == "IsolationForest" and SKLEARN_AVAILABLE:
            contamination = params.get('contamination', 0.1)
            contamination = max(0.001, min(0.5, contamination))  # Validate range
            
            iso = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=min(100, max(10, len(data) // 100))  # Adaptive n_estimators
            )
            outliers = iso.fit_predict(data)
            anomaly_scores = -iso.decision_function(data)
            
        elif method == "Z-Score":
            threshold = params.get('threshold', 3.0)
            threshold = max(1.0, min(5.0, threshold))  # Validate range
            
            z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
            outliers = (z_scores > threshold).any(axis=1)
            outliers = np.where(outliers, -1, 1)  # Convert to standard format
            anomaly_scores = z_scores.max(axis=1)
            
        elif method == "Modified Z-Score":
            threshold = params.get('threshold', 3.5)
            threshold = max(1.0, min(10.0, threshold))  # Validate range
            
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            # Avoid division by zero
            mad = np.where(mad == 0, 1e-8, mad)
            modified_z_scores = np.abs(0.6745 * (data - median) / mad)
            outliers = (modified_z_scores > threshold).any(axis=1)
            outliers = np.where(outliers, -1, 1)
            anomaly_scores = modified_z_scores.max(axis=1)
            
        elif method == "IQR":
            multiplier = params.get('multiplier', 1.5)
            multiplier = max(0.1, min(5.0, multiplier))  # Validate range
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
            outliers = np.where(outliers, -1, 1)
            
            # Calculate distance-based anomaly scores
            lower_distances = (lower_bound - data).clip(lower=0).max(axis=1)
            upper_distances = (data - upper_bound).clip(lower=0).max(axis=1)
            anomaly_scores = np.maximum(lower_distances, upper_distances)
            
        else:
            return {"error": f"Unsupported anomaly detection method: {method}"}
        
        processing_time = time.time() - start_time
        outlier_count = int(np.sum(outliers == -1))
        
        # Validation and quality checks
        anomaly_rate = outlier_count / len(outliers) * 100
        quality_warnings = []
        
        if anomaly_rate > 50:
            quality_warnings.append(f"Very high anomaly rate ({anomaly_rate:.1f}%) - consider adjusting parameters")
        elif anomaly_rate == 0:
            quality_warnings.append("No anomalies detected - parameters might be too strict")
        
        # Prepare results
        result = {
            "outliers": outliers,
            "anomaly_scores": anomaly_scores,
            "method": method,
            "params": params,
            "index": original_index,
            "columns": numeric_cols,
            "outlier_count": outlier_count,
            "anomaly_rate": anomaly_rate,
            "processing_time": processing_time,
            "sample_note": sample_note,
            "quality_warnings": quality_warnings,
            "data_shape": data.shape,
            "timestamp": time.time()
        }
        
        if LOGGING_ENABLED:
            logging.info(f"Anomaly detection completed: {method}, {outlier_count} anomalies found in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        error_msg = f"Anomaly detection failed: {str(e)}"
        if LOGGING_ENABLED:
            logging.error(f"Anomaly detection error: {error_msg}")
        return {"error": error_msg}

# --- Enhanced Machine Learning Functions ---
if SKLEARN_AVAILABLE:
    def train_ml_model_production(df: pd.DataFrame, x_cols: List[str], y_col: str, 
                                model_type: str, algorithm: str, **kwargs) -> Dict[str, Any]:
        """Production-grade ML model training with comprehensive validation and monitoring."""
        try:
            # Input validation
            if df is None or df.empty:
                return {"error": "Empty dataset provided"}
            
            available_x_cols = [col for col in x_cols if col in df.columns]
            if not available_x_cols:
                return {"error": "No valid feature columns found"}
            
            if y_col not in df.columns:
                return {"error": f"Target column '{y_col}' not found"}
            
            # Data preparation with validation
            start_time = time.time()
            
            # Sample large datasets for performance
            if len(df) > MAX_ROWS_PROCESSING:
                df_sample = df.sample(n=SAMPLE_SIZE_LARGE, random_state=42)
                st.info(f"Using sample of {len(df_sample):,} rows for training")
            else:
                df_sample = df
            
            X = df_sample[available_x_cols].copy()
            y = df_sample[y_col].copy()
            
            # Handle missing values in target
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) == 0:
                return {"error": "No valid samples after removing missing target values"}
            
            # Feature preprocessing
            preprocessing_time = time.time()
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            label_encoders = {}
            
            for col in categorical_cols:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
                except Exception as e:
                    if LOGGING_ENABLED:
                        logging.warning(f"Label encoding failed for column {col}: {e}")
                    X[col] = 0  # Fill with default value
            
            # Handle missing values in features
            X = X.fillna(X.mean(numeric_only=True)).fillna(0)
            
            # Validate final data
            if X.isnull().any().any():
                return {"error": "Data still contains missing values after preprocessing"}
            
            preprocessing_time = time.time() - preprocessing_time
            
            # Train-test split with stratification for classification
            try:
                if model_type == "classification" and len(np.unique(y)) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
            except Exception:
                # Fallback without stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Model initialization and training
            model_start_time = time.time()
            
            if algorithm == "RandomForest":
                if model_type == "regression":
                    model = RandomForestRegressor(
                        n_estimators=kwargs.get('n_estimators', 100),
                        max_depth=kwargs.get('max_depth', None),
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=kwargs.get('n_estimators', 100),
                        max_depth=kwargs.get('max_depth', None),
                        random_state=42,
                        n_jobs=-1
                    )
                    
            elif algorithm == "MLP":
                # Feature scaling for MLP
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if model_type == "regression":
                    model = MLPRegressor(
                        hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (50, 50)),
                        max_iter=kwargs.get('max_iter', 200),
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1
                    )
                else:
                    model = MLPClassifier(
                        hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (50, 50)),
                        max_iter=kwargs.get('max_iter', 200),
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1
                    )
                
                # Train with scaled data
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
            else:
                return {"error": f"Unsupported algorithm: {algorithm}"}
            
            # Train model (for non-MLP algorithms)
            if algorithm != "MLP":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            training_time = time.time() - model_start_time
            
            # Model evaluation
            try:
                if model_type == "regression":
                    metrics = {
                        'mae': float(mean_absolute_error(y_test, y_pred)),
                        'mse': float(mean_squared_error(y_test, y_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                        'r2': float(r2_score(y_test, y_pred)),
                        'explained_variance': float(explained_variance_score(y_test, y_pred))
                    }
                else:
                    metrics = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    }
                    
                    # Add probability-based metrics if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            if algorithm == "MLP":
                                y_proba = model.predict_proba(X_test_scaled)
                            else:
                                y_proba = model.predict_proba(X_test)
                                
                            if len(np.unique(y_test)) == 2:  # Binary classification
                                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
                        except Exception as e:
                            if LOGGING_ENABLED:
                                logging.warning(f"Probability metrics calculation failed: {e}")
                
            except Exception as e:
                if LOGGING_ENABLED:
                    logging.error(f"Metrics calculation error: {e}")
                metrics = {"error": f"Metrics calculation failed: {str(e)}"}
            
            total_time = time.time() - start_time
            
            # Prepare comprehensive result
            result = {
                "algorithm": algorithm,
                "model_type": model_type,
                "metrics": metrics,
                "training_time": training_time,
                "total_time": total_time,
                "preprocessing_time": preprocessing_time,
                "n_samples": len(X),
                "n_features": len(available_x_cols),
                "feature_names": available_x_cols,
                "test_size": len(X_test),
                "model": model,
                "label_encoders": label_encoders,
                "timestamp": time.time()
            }
            
            # Add algorithm-specific information
            if algorithm == "RandomForest" and hasattr(model, 'feature_importances_'):
                # Convert feature importance to safe format
                try:
                    feature_importance = {}
                    for idx, col_name in enumerate(available_x_cols):
                        feature_importance[str(col_name)] = float(model.feature_importances_[idx])
                    result["feature_importance"] = feature_importance
                except Exception as imp_e:
                    if LOGGING_ENABLED:
                        logging.warning(f"Feature importance calculation failed: {imp_e}")
                    result["feature_importance"] = {}
            elif algorithm == "MLP":
                result["scaler"] = scaler
                result["convergence"] = model.n_iter_ < kwargs.get('max_iter', 200)
            
            if LOGGING_ENABLED:
                logging.info(f"Model training completed: {algorithm} {model_type} in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            if LOGGING_ENABLED:
                logging.error(f"ML training error: {error_msg}")
            return {"error": error_msg}
else:
    def train_ml_model_production(*args, **kwargs):
        return {"error": "Machine learning features not available - scikit-learn not installed"}

# --- Enhanced Data Cleaning ---
@st.cache_data(show_spinner="Generating cleaning suggestions...", max_entries=2, ttl=1800)
def suggest_cleaning_production(df_hash: str, max_suggestions: int = 15) -> List[Dict[str, Any]]:
    """Production-grade data cleaning suggestions with comprehensive analysis."""
    try:
        df = st.session_state.selected_df
        if df is None or df.empty:
            return []
        
        # Use sampling for performance
        sample_size = min(SAMPLE_SIZE_MEDIUM, len(df))
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        suggestions = []
        
        for col in df_sample.columns:
            col_data = df_sample[col]
            col_full_data = df[col]  # Use full dataset for statistics
            
            # Missing data analysis
            missing_count = col_full_data.isnull().sum()
            missing_pct = missing_count / len(df)
            
            if missing_pct > 0.8:
                suggestions.append({
                    'type': 'drop_column',
                    'column': col,
                    'description': f"Drop '{col}' (missing: {missing_pct:.1%})",
                    'severity': 'high',
                    'impact': 'high',
                    'confidence': 'high'
                })
            elif missing_pct > 0.3:
                suggestions.append({
                    'type': 'drop_column_conditional',
                    'column': col,
                    'description': f"Consider dropping '{col}' (missing: {missing_pct:.1%})",
                    'severity': 'medium',
                    'impact': 'medium',
                    'confidence': 'medium'
                })
            elif missing_pct > 0.05:
                # Suggest appropriate imputation method
                if col_data.dtype in ['int64', 'float64']:
                    method = 'median' if col_data.skew() > 1 else 'mean'
                    suggestions.append({
                        'type': 'impute_numeric',
                        'column': col,
                        'description': f"Impute '{col}' with {method} (missing: {missing_pct:.1%})",
                        'severity': 'medium',
                        'impact': 'low',
                        'confidence': 'high'
                    })
                else:
                    suggestions.append({
                        'type': 'impute_categorical',
                        'column': col,
                        'description': f"Impute '{col}' with mode (missing: {missing_pct:.1%})",
                        'severity': 'medium',
                        'impact': 'low',
                        'confidence': 'high'
                    })
            
            # Data type optimization
            if col_data.dtype == 'object':
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    # Check for numeric conversion
                    try:
                        pd.to_numeric(non_null_data.head(100), errors='raise')
                        suggestions.append({
                            'type': 'convert_numeric',
                            'column': col,
                            'description': f"Convert '{col}' to numeric type",
                            'severity': 'low',
                            'impact': 'medium',
                            'confidence': 'high'
                        })
                    except:
                        pass
                    
                    # Check for datetime conversion
                    try:
                        pd.to_datetime(non_null_data.head(100), errors='raise')
                        suggestions.append({
                            'type': 'convert_datetime',
                            'column': col,
                            'description': f"Convert '{col}' to datetime type",
                            'severity': 'low',
                            'impact': 'medium',
                            'confidence': 'medium'
                        })
                    except:
                        pass
                    
                    # Check for high cardinality categorical data
                    cardinality_ratio = col_data.nunique() / len(col_data)
                    if cardinality_ratio > 0.8 and col_data.nunique() > 100:
                        suggestions.append({
                            'type': 'high_cardinality_warning',
                            'column': col,
                            'description': f"'{col}' has very high cardinality ({col_data.nunique()} unique values)",
                            'severity': 'medium',
                            'impact': 'medium',
                            'confidence': 'high'
                        })
            
            # Outlier detection for numeric columns
            if col_data.dtype in ['int64', 'float64'] and len(col_data.dropna()) > 10:
                try:
                    clean_data = col_data.dropna()
                    z_scores = np.abs(stats.zscore(clean_data))
                    outlier_pct = (z_scores > 3).sum() / len(clean_data)
                    
                    if outlier_pct > 0.1:  # More than 10% outliers
                        suggestions.append({
                            'type': 'handle_outliers',
                            'column': col,
                            'description': f"Handle outliers in '{col}' ({outlier_pct:.1%} extreme values)",
                            'severity': 'medium',
                            'impact': 'medium',
                            'confidence': 'medium'
                        })
                except:
                    pass
            
            # Duplicate value analysis
            if col_data.dtype == 'object':
                duplicate_ratio = 1 - (col_data.nunique() / len(col_data.dropna()))
                if duplicate_ratio > 0.95 and col_data.nunique() < 5:  # Almost all same values
                    suggestions.append({
                        'type': 'low_variance_warning',
                        'column': col,
                        'description': f"'{col}' has very low variance (mostly same values)",
                        'severity': 'low',
                        'impact': 'medium',
                        'confidence': 'high'
                    })
        
        # Sort suggestions by priority (severity, then impact, then confidence)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: (
            priority_order[x['severity']],
            priority_order[x['impact']],
            priority_order[x['confidence']]
        ))
        
        return suggestions[:max_suggestions]
        
    except Exception as e:
        if LOGGING_ENABLED:
            logging.error(f"Cleaning suggestions error: {e}")
        return []

def apply_cleaning_suggestion_production(suggestion: Dict[str, Any]) -> Tuple[bool, str]:
    """Apply cleaning suggestion with comprehensive error handling and rollback capability."""
    try:
        if st.session_state.selected_df is None:
            return False, "No data available"
        
        # Create backup
        df_backup = st.session_state.selected_df.copy()
        df = st.session_state.selected_df.copy()
        col = suggestion['column']
        
        if col not in df.columns:
            return False, f"Column '{col}' not found"
        
        success_msg = ""
        
        if suggestion['type'] == 'drop_column':
            df = df.drop(columns=[col])
            success_msg = f"Dropped column '{col}'"
            
        elif suggestion['type'] == 'impute_numeric':
            if df[col].dtype in ['int64', 'float64']:
                # Use median for skewed data, mean for normal
                if abs(df[col].skew()) > 1:
                    fill_value = df[col].median()
                    method = "median"
                else:
                    fill_value = df[col].mean()
                    method = "mean"
                df[col] = df[col].fillna(fill_value)
                success_msg = f"Imputed '{col}' with {method}: {fill_value:.2f}"
            else:
                return False, f"Column '{col}' is not numeric"
                
        elif suggestion['type'] == 'impute_categorical':
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                fill_value = mode_values.iloc[0]
            else:
                fill_value = 'Unknown'
            df[col] = df[col].fillna(fill_value)
            success_msg = f"Imputed '{col}' with mode: {fill_value}"
            
        elif suggestion['type'] == 'convert_numeric':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                success_msg = f"Converted '{col}' to numeric type"
            except Exception as e:
                return False, f"Numeric conversion failed: {str(e)}"
                
        elif suggestion['type'] == 'convert_datetime':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                success_msg = f"Converted '{col}' to datetime type"
            except Exception as e:
                return False, f"Datetime conversion failed: {str(e)}"
                
        elif suggestion['type'] == 'handle_outliers':
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clip outliers instead of removing them
                original_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                success_msg = f"Clipped {original_outliers} outliers in '{col}'"
            else:
                return False, f"Column '{col}' is not numeric"
        elif suggestion['type'] in ['low_variance_warning', 'high_cardinality_warning', 'drop_column_conditional']:
            # These are warning-only suggestions that don't perform actual operations
            success_msg = f"Warning acknowledged for '{col}': {suggestion['description']}"
        else:
            return False, f"Unsupported cleaning operation: {suggestion['type']}"
        
        # Validation check
        if df is None or df.empty:
            return False, "Cleaning operation resulted in empty dataset"
        
        # Apply changes to both selected_df and the main dfs dictionary
        st.session_state.selected_df = df
        
        # Also update the main dataframes dictionary
        current_file = st.session_state.get('current_file_tab')
        if current_file and current_file in st.session_state.dfs:
            st.session_state.dfs[current_file] = df.copy()
            if LOGGING_ENABLED:
                logging.info(f"Updated both selected_df and dfs['{current_file}'] with cleaning changes")
        
        if LOGGING_ENABLED:
            logging.info(f"Applied cleaning suggestion: {success_msg}")
        
        return True, success_msg
        
    except Exception as e:
        error_msg = f"Cleaning operation failed: {str(e)}"
        if LOGGING_ENABLED:
            logging.error(f"Cleaning application error: {error_msg}")
        
        # Restore backup if available
        if 'df_backup' in locals():
            st.session_state.selected_df = df_backup
        
        return False, error_msg

# --- SQL Query Functions ---
if SQL_AVAILABLE:
    @st.cache_resource
    def create_sql_connection(connection_string: str, connection_name: str) -> Dict[str, Any]:
        """Create and cache SQL database connection."""
        try:
            engine = create_engine(connection_string)
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return {
                'engine': engine,
                'connection_string': connection_string,
                'status': 'connected',
                'created_at': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'engine': None,
                'connection_string': connection_string,
                'status': 'error',
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
    
    def execute_sql_query(engine, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        try:
            start_time = time.time()
            
            # For SQLite in-memory databases, we need to ensure we use the same connection
            # that was used to create the tables. We'll use pandas.read_sql_query instead
            # which handles connections more reliably for in-memory databases
            
            if LOGGING_ENABLED:
                logging.info(f"SQL Execute - Query: {query[:100]}..." if len(query) > 100 else f"SQL Execute - Query: {query}")
                logging.info(f"SQL Execute - Engine URL: {engine.url}")
            
            try:
                # Check if it's a SELECT query by trying to read with pandas first
                query_lower = query.lower().strip()
                if query_lower.startswith('select'):
                    # Use pandas read_sql_query for SELECT statements - this handles in-memory DBs better
                    df = pd.read_sql_query(query, engine)
                    execution_time = time.time() - start_time
                    
                    if LOGGING_ENABLED:
                        logging.info(f"SQL Execute - Successfully returned {len(df)} rows in {execution_time:.3f}s")
                    
                    return {
                        'success': True,
                        'data': df,
                        'rows_affected': len(df),
                        'execution_time': execution_time,
                        'query': query,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    # For non-SELECT queries, use the traditional approach
                    with engine.connect() as conn:
                        if params:
                            result = conn.execute(text(query), params)
                        else:
                            result = conn.execute(text(query))
                        
                        rows_affected = result.rowcount
                        execution_time = time.time() - start_time
                        
                        if LOGGING_ENABLED:
                            logging.info(f"SQL Execute - Non-SELECT query completed, {rows_affected} rows affected in {execution_time:.3f}s")
                        
                        return {
                            'success': True,
                            'data': None,
                            'rows_affected': rows_affected,
                            'execution_time': execution_time,
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        }
            
            except Exception as inner_e:
                # If pandas approach fails, fall back to the original approach
                if LOGGING_ENABLED:
                    logging.warning(f"SQL Execute - Pandas approach failed: {inner_e}, falling back to SQLAlchemy")
                
                with engine.connect() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    
                    # Check if it's a SELECT query (returns data)
                    if result.returns_rows:
                        # Fetch the raw data first
                        raw_data = result.fetchall()
                        columns = list(result.keys())
                        
                        # Try to create a DataFrame
                        try:
                            df = pd.DataFrame(raw_data, columns=columns)
                        except (NameError, ImportError):
                            # Fallback: return raw data if pandas not available or not imported
                            df = {
                                'columns': columns,
                                'data': [list(row) for row in raw_data],
                                'error': 'pandas not available - returning raw data'
                            }
                        execution_time = time.time() - start_time
                        
                        return {
                            'success': True,
                            'data': df,
                            'rows_affected': len(df) if isinstance(df, pd.DataFrame) else len(df.get('data', [])),
                            'execution_time': execution_time,
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        # For INSERT, UPDATE, DELETE, etc.
                        rows_affected = result.rowcount
                        execution_time = time.time() - start_time
                        
                        return {
                            'success': True,
                            'data': None,
                            'rows_affected': rows_affected,
                            'execution_time': execution_time,
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        }
                    
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            if LOGGING_ENABLED:
                logging.error(f"SQL Execute - Failed: {error_msg}")
                logging.error(f"SQL Execute - Query was: {query}")
            
            return {
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_table_info(engine, schema: str = None) -> Dict[str, Any]:
        """Get information about tables in the database."""
        try:
            inspector = sqlalchemy.inspect(engine)
            
            tables_info = []
            
            # Try to get table names
            try:
                table_names = inspector.get_table_names(schema=schema)
                if LOGGING_ENABLED:
                    logging.info(f"Found {len(table_names)} tables: {table_names}")
            except Exception as e:
                if LOGGING_ENABLED:
                    logging.error(f"Failed to get table names: {e}")
                return {
                    'success': False,
                    'error': f'Failed to get table names: {str(e)}'
                }
            
            # If no tables found, try alternative methods for SQLite
            if not table_names and str(engine.url).startswith('sqlite'):
                try:
                    # Direct SQLite query
                    with engine.connect() as conn:
                        result = conn.execute(sqlalchemy.text("SELECT name FROM sqlite_master WHERE type='table'"))
                        table_names = [row[0] for row in result.fetchall()]
                        if LOGGING_ENABLED:
                            logging.info(f"SQLite direct query found {len(table_names)} tables: {table_names}")
                except Exception as e:
                    if LOGGING_ENABLED:
                        logging.error(f"SQLite direct query failed: {e}")
            
            for table_name in table_names:
                try:
                    columns = inspector.get_columns(table_name, schema=schema)
                    column_info = [{
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable'],
                        'default': col.get('default')
                    } for col in columns]
                    
                    tables_info.append({
                        'table_name': table_name,
                        'columns': column_info,
                        'column_count': len(column_info)
                    })
                except Exception as e:
                    if LOGGING_ENABLED:
                        logging.warning(f"Failed to get columns for table {table_name}: {e}")
                    # Add table without column info
                    tables_info.append({
                        'table_name': table_name,
                        'columns': [],
                        'column_count': 0,
                        'error': str(e)
                    })
            
            return {
                'success': True,
                'tables': tables_info,
                'table_count': len(tables_info),
                'schema': schema,
                'debug_info': f"Engine URL: {engine.url}"
            }
            
        except Exception as e:
            error_msg = f'Schema inspection failed: {str(e)}'
            if LOGGING_ENABLED:
                logging.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def validate_sql_query(query: str) -> Dict[str, Any]:
        """Basic SQL query validation."""
        query_lower = query.lower().strip()
        
        # Remove comments and extra whitespace
        import re
        query_clean = re.sub(r'--.*$', '', query_lower, flags=re.MULTILINE)
        query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)
        query_clean = ' '.join(query_clean.split())
        
        # Check for dangerous operations
        dangerous_keywords = [
            'drop table', 'drop database', 'truncate', 'delete from',
            'alter table', 'create table', 'insert into', 'update '
        ]
        
        is_dangerous = any(keyword in query_clean for keyword in dangerous_keywords)
        
        # Determine query type
        if query_clean.startswith('select'):
            query_type = 'SELECT'
        elif any(query_clean.startswith(kw) for kw in ['insert', 'update', 'delete']):
            query_type = 'MODIFY'
        elif any(query_clean.startswith(kw) for kw in ['create', 'alter', 'drop']):
            query_type = 'DDL'
        else:
            query_type = 'OTHER'
        
        return {
            'is_valid': len(query_clean) > 0,
            'is_dangerous': is_dangerous,
            'query_type': query_type,
            'cleaned_query': query_clean
        }

    def create_database_from_dataframes(dfs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create an in-memory SQLite database from loaded DataFrames."""
        # Check if pandas is available
        if not PANDAS_AVAILABLE:
            return {
                'success': False,
                'error': 'Pandas is not installed. Cannot create database from DataFrames.'
            }
        
        try:
            # Enhanced debug logging
            if LOGGING_ENABLED:
                logging.info(f"SQL Database Creation - Starting with {len(dfs_dict)} DataFrames")
                for name, df in dfs_dict.items():
                    logging.info(f"  - {name}: {df.shape if df is not None else 'None'} shape, empty: {df.empty if df is not None else 'N/A'}")
            
            # Create in-memory SQLite database with better connection pooling
            # Use poolclass=StaticPool to ensure all connections use the same database instance
            engine = create_engine(
                'sqlite:///:memory:', 
                echo=False,
                poolclass=StaticPool,
                connect_args={
                    'check_same_thread': False,  # Allow multi-threading
                },
                pool_pre_ping=True  # Validate connections before use
            )
            if LOGGING_ENABLED:
                logging.info(f"SQL Database - Created SQLite engine with StaticPool: {engine}")
            
            tables_created = []
            total_rows = 0
            
            for df_name, df in dfs_dict.items():
                if LOGGING_ENABLED:
                    logging.info(f"SQL Database - Processing DataFrame: {df_name}")
                
                if df is None:
                    if LOGGING_ENABLED:
                        logging.warning(f"SQL Database - Skipping {df_name}: DataFrame is None")
                    continue
                    
                if df.empty:
                    if LOGGING_ENABLED:
                        logging.warning(f"SQL Database - Skipping {df_name}: DataFrame is empty")
                    continue
                
                # Clean table name (remove file extensions, special characters)
                table_name = re.sub(r'[^\w]', '_', str(df_name).split('.')[0]).lower()
                if LOGGING_ENABLED:
                    logging.info(f"SQL Database - Table mapping: '{df_name}' -> '{table_name}'")
                
                # Write DataFrame to SQLite
                if LOGGING_ENABLED:
                    logging.info(f"SQL Database - Writing {len(df)} rows to table '{table_name}'")
                
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                
                if LOGGING_ENABLED:
                    logging.info(f"SQL Database - Successfully created table '{table_name}' with {len(df)} rows, {len(df.columns)} columns")
                
                tables_created.append({
                    'original_name': df_name,
                    'table_name': table_name,
                    'rows': len(df),
                    'columns': len(df.columns)
                })
                total_rows += len(df)
            
            if LOGGING_ENABLED:
                logging.info(f"SQL Database - Final result: Created {len(tables_created)} tables, {total_rows:,} total rows")
                logging.info(f"SQL Database - Table names: {[t['table_name'] for t in tables_created]}")
            
            return {
                'success': True,
                'engine': engine,
                'tables_created': tables_created,
                'total_tables': len(tables_created),
                'total_rows': total_rows,
                'connection_string': 'sqlite:///:memory: (from loaded data)'
            }
            
        except Exception as e:
            error_msg = str(e)
            if LOGGING_ENABLED:
                logging.error(f"SQL Database Creation Failed: {error_msg}")
                logging.error(f"SQL Database - Exception details: {type(e).__name__}: {error_msg}")
            
            return {
                'success': False,
                'error': f"Database creation failed: {error_msg}",
                'exception_type': type(e).__name__
            }

else:
    def create_sql_connection(*args, **kwargs):
        return {'status': 'error', 'error': 'SQL libraries not available'}
    
    def execute_sql_query(*args, **kwargs):
        return {'success': False, 'error': 'SQL libraries not available'}
    
    def get_table_info(*args, **kwargs):
        return {'success': False, 'error': 'SQL libraries not available'}
    
    def validate_sql_query(*args, **kwargs):
        return {'is_valid': False, 'error': 'SQL libraries not available'}
    
    def create_database_from_dataframes(*args, **kwargs):
        return {'success': False, 'error': 'SQL libraries not available'}

# --- Advanced Statistical Operations ---
def perform_advanced_statistics(df: pd.DataFrame, operation: str, **kwargs) -> Dict[str, Any]:
    """Perform advanced statistical operations on DataFrame."""
    try:
        result = {'success': False, 'data': None, 'message': ''}
        
        if operation == 'SUM':
            column = kwargs.get('column')
            if column and column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    value = df[column].sum()
                    result = {'success': True, 'data': value, 'message': f'Sum of {column}: {value:,.2f}'}
                else:
                    result['message'] = f'Column {column} is not numeric'
            else:
                # Sum all numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                sums = df[numeric_cols].sum()
                result = {'success': True, 'data': sums.to_dict(), 'message': f'Sum of all numeric columns calculated'}
        
        elif operation == 'COUNT':
            column = kwargs.get('column')
            condition = kwargs.get('condition')
            if column and column in df.columns:
                if condition:
                    # COUNTIF functionality
                    count = len(df[df[column] == condition])
                    result = {'success': True, 'data': count, 'message': f'Count of {column} where value = {condition}: {count:,}'}
                else:
                    count = df[column].count()
                    result = {'success': True, 'data': count, 'message': f'Count of non-null values in {column}: {count:,}'}
            else:
                count = len(df)
                result = {'success': True, 'data': count, 'message': f'Total row count: {count:,}'}
        
        elif operation == 'DISTINCT_COUNT':
            column = kwargs.get('column')
            if column and column in df.columns:
                count = df[column].nunique()
                result = {'success': True, 'data': count, 'message': f'Distinct count in {column}: {count:,}'}
            else:
                result['message'] = 'Column required for distinct count'
        
        elif operation == 'MIN_MAX':
            column = kwargs.get('column')
            if column and column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    min_val = df[column].min()
                    max_val = df[column].max()
                    result = {'success': True, 'data': {'min': min_val, 'max': max_val}, 
                             'message': f'{column} - Min: {min_val}, Max: {max_val}'}
                else:
                    result['message'] = f'Column {column} is not numeric'
            else:
                result['message'] = 'Column required for min/max'
        
        elif operation == 'GROUP_BY':
            group_column = kwargs.get('group_column')
            agg_column = kwargs.get('agg_column')
            agg_function = kwargs.get('agg_function', 'count')
            
            if group_column and group_column in df.columns:
                if agg_column and agg_column in df.columns:
                    grouped = df.groupby(group_column)[agg_column].agg(agg_function).reset_index()
                else:
                    grouped = df.groupby(group_column).size().reset_index(name='count')
                result = {'success': True, 'data': grouped, 'message': f'Grouped by {group_column}'}
            else:
                result['message'] = 'Group column required'
        
        elif operation == 'PIVOT':
            index_col = kwargs.get('index_column')
            columns_col = kwargs.get('columns_column') 
            values_col = kwargs.get('values_column')
            agg_func = kwargs.get('agg_function', 'sum')
            
            if all([index_col, columns_col, values_col]):
                if all(col in df.columns for col in [index_col, columns_col, values_col]):
                    pivot_table = pd.pivot_table(df, index=index_col, columns=columns_col, 
                                                values=values_col, aggfunc=agg_func, fill_value=0)
                    result = {'success': True, 'data': pivot_table.reset_index(), 
                             'message': f'Pivot table created with {index_col} as index, {columns_col} as columns'}
                else:
                    result['message'] = 'One or more specified columns not found'
            else:
                result['message'] = 'Index, columns, and values columns required for pivot'
        
        elif operation == 'UNIQUE':
            column = kwargs.get('column')
            if column and column in df.columns:
                unique_values = df[column].unique()
                result = {'success': True, 'data': unique_values.tolist(), 
                         'message': f'Found {len(unique_values)} unique values in {column}'}
            else:
                result['message'] = 'Column required for unique values'
        
        elif operation == 'CONDITIONAL':
            column = kwargs.get('column')
            condition = kwargs.get('condition')
            true_value = kwargs.get('true_value', 1)
            false_value = kwargs.get('false_value', 0)
            
            if column and column in df.columns and condition:
                df_copy = df.copy()
                try:
                    # Parse condition (e.g., "> 100", "== 'value'", "< 50.0")
                    import re
                    match = re.match(r'([><=]=?|!=)\s*(.+)', condition.strip())
                    if not match:
                        raise ValueError("Invalid condition format. Use: >, <, ==, !=, >=, <= followed by value")
                    
                    operator, value = match.groups()
                    try:
                        # Convert value to appropriate type
                        if df[column].dtype in ['int64', 'float64']:
                            value = float(value)
                        elif df[column].dtype == 'bool':
                            value = value.lower() in ('true', '1', 'yes')
                        else:
                            value = str(value)  # Handle categorical/string columns
                    except ValueError:
                        raise ValueError(f"Value '{value}' cannot be converted to column type {df[column].dtype}")
                    
                    # Apply condition
                    if operator == '==':
                        df_copy['conditional_result'] = np.where(df_copy[column] == value, true_value, false_value)
                    elif operator == '!=':
                        df_copy['conditional_result'] = np.where(df_copy[column] != value, true_value, false_value)
                    elif operator == '>':
                        df_copy['conditional_result'] = np.where(df_copy[column] > value, true_value, false_value)
                    elif operator == '<':
                        df_copy['conditional_result'] = np.where(df_copy[column] < value, true_value, false_value)
                    elif operator == '>=':
                        df_copy['conditional_result'] = np.where(df_copy[column] >= value, true_value, false_value)
                    elif operator == '<=':
                        df_copy['conditional_result'] = np.where(df_copy[column] <= value, true_value, false_value)
                    else:
                        raise ValueError(f"Unsupported operator: {operator}")
                    
                    result = {
                        'success': True,
                        'data': df_copy,
                        'message': f'Applied condition {column} {condition}'
                    }
                except Exception as e:
                    result = {'success': False, 'message': f'Conditional operation failed: {str(e)}'}
            else:
                result = {'success': False, 'message': 'Column and condition required for conditional operation'}
        
        return result
    
    except Exception as e:
        return {'success': False, 'data': None, 'message': f'Error: {str(e)}'}

def merge_csv_files(files_dict: Dict[str, pd.DataFrame], merge_config: Dict) -> Dict[str, Any]:
    """Merge multiple CSV files based on configuration."""
    try:
        if len(files_dict) < 2:
            return {'success': False, 'message': 'At least 2 files required for merge'}
        
        file_names = list(files_dict.keys())
        left_df = files_dict[file_names[0]]
        
        merge_type = merge_config.get('how', 'inner')
        left_on = merge_config.get('left_on')
        right_on = merge_config.get('right_on')
        
        merged_df = left_df
        
        for i in range(1, len(file_names)):
            right_df = files_dict[file_names[i]]
            
            if left_on and right_on:
                merged_df = pd.merge(merged_df, right_df, left_on=left_on, right_on=right_on, how=merge_type)
            else:
                # Try to find common columns
                common_cols = list(set(merged_df.columns) & set(right_df.columns))
                if common_cols:
                    merged_df = pd.merge(merged_df, right_df, on=common_cols[0], how=merge_type)
                else:
                    return {'success': False, 'message': f'No common columns found between files'}
        
        return {
            'success': True, 
            'data': merged_df,
            'message': f'Successfully merged {len(file_names)} files ({len(merged_df)} rows, {len(merged_df.columns)} columns)'
        }
    
    except Exception as e:
        return {'success': False, 'message': f'Merge failed: {str(e)}'}

# --- Enhanced Natural Language Processing ---
def process_natural_query_production(query: str) -> Dict[str, Any]:
    """Natural language query processing with pattern matching."""
    try:
        query_lower = query.lower().strip()
        response = {"action": None, "message": "", "success": False, "data": None}
        
        if not query_lower:
            response["message"] = "Please enter a query."
            return response
        
        df = st.session_state.selected_df
        if df is None or df.empty:
            response["message"] = "No data available. Please upload a dataset first."
            return response
        
        # Enhanced pattern matching with more robust regex
        
        # Data cleaning patterns
        clean_patterns = [
            (r"clean\s+(missing|nan|null)\s+(?:data\s+)?(?:in\s+)?(\w+)", "clean_missing"),
            (r"remove\s+(?:missing|nan|null)\s+(?:from\s+)?(\w+)", "clean_missing"),
            (r"fill\s+(?:missing|nan|null)\s+(?:in\s+)?(\w+)", "clean_missing"),
            (r"drop\s+column\s+(\w+)", "drop_column"),
            (r"remove\s+column\s+(\w+)", "drop_column")
        ]
        
        for pattern, action in clean_patterns:
            match = re.search(pattern, query_lower)
            if match:
                col_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                matching_cols = [col for col in df.columns if col_name.lower() in col.lower()]
                
                if matching_cols:
                    col = matching_cols[0]
                    
                    if action == "clean_missing":
                        missing_count = df[col].isnull().sum()
                        if missing_count == 0:
                            response["message"] = f"Column '{col}' has no missing values."
                            response["success"] = True
                            return response
                        
                        # Apply appropriate cleaning method
                        if df[col].dtype in ['int64', 'float64']:
                            if abs(df[col].skew()) > 1:
                                fill_value = df[col].median()
                                method = "median"
                            else:
                                fill_value = df[col].mean()
                                method = "mean"
                            df[col] = df[col].fillna(fill_value)
                            response["message"] = f"Filled {missing_count} missing values in '{col}' with {method}: {fill_value:.2f}"
                        else:
                            mode_values = df[col].mode()
                            fill_value = mode_values.iloc[0] if len(mode_values) > 0 else 'Unknown'
                            df[col] = df[col].fillna(fill_value)
                            response["message"] = f"Filled {missing_count} missing values in '{col}' with mode: '{fill_value}'"
                        
                        st.session_state.selected_df = df
                        response["success"] = True
                        response["action"] = "clean_data"
                        
                    elif action == "drop_column":
                        df_new = df.drop(columns=[col])
                        st.session_state.selected_df = df_new
                        response["message"] = f"Dropped column '{col}'. Dataset now has {len(df_new.columns)} columns."
                        response["success"] = True
                        response["action"] = "drop_column"
                    
                    return response
                else:
                    response["message"] = f"Column containing '{col_name}' not found. Available columns: {', '.join(list(df.columns)[:5])}"
                    return response
        
        # Statistical analysis patterns
        stats_patterns = [
            (r"(?:show|get|display)\s+(?:stats|statistics|summary)\s+(?:for\s+|of\s+)?(\w+)", "show_stats"),
            (r"describe\s+(\w+)", "show_stats"),
            (r"summarize\s+(\w+)", "show_stats")
        ]
        
        for pattern, action in stats_patterns:
            match = re.search(pattern, query_lower)
            if match:
                col_name = match.group(1)
                matching_cols = [col for col in df.columns if col_name.lower() in col.lower()]
                
                if matching_cols:
                    col = matching_cols[0]
                    
                    if df[col].dtype in ['int64', 'float64']:
                        stats_data = df[col].describe()
                        response["message"] = f"Statistics for '{col}':\n"
                        response["message"] += "\n".join([f"• {k.title()}: {v:.2f}" if isinstance(v, (int, float)) else f"• {k.title()}: {v}" for k, v in stats_data.items()])
                        
                        # Add additional insights
                        missing_count = df[col].isnull().sum()
                        if missing_count > 0:
                            response["message"] += f"\n• Missing Values: {missing_count:,} ({missing_count/len(df)*100:.1f}%)"
                        
                        # Outlier detection
                        z_scores = np.abs(stats.zscore(df[col].dropna()))
                        outliers = (z_scores > 3).sum()
                        if outliers > 0:
                            response["message"] += f"\n• Potential Outliers: {outliers:,}"
                            
                    else:
                        value_counts = df[col].value_counts().head(10)
                        response["message"] = f"Top values in '{col}':\n"
                        response["message"] += "\n".join([f"• {k}: {v:,}" for k, v in value_counts.items()])
                        
                        # Add additional insights
                        unique_count = df[col].nunique()
                        response["message"] += f"\n• Total Unique Values: {unique_count:,}"
                        response["message"] += f"\n• Most Common: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}"
                    
                    response["success"] = True
                    response["action"] = "show_stats"
                    return response
        
        # Filtering patterns
        filter_patterns = [
            (r"filter\s+(\w+)\s+(equals?|=|==)\s+([^\s,]+)", "filter_equal"),
            (r"filter\s+(\w+)\s+(>|greater\s+than)\s+([^\s,]+)", "filter_greater"),
            (r"filter\s+(\w+)\s+(<|less\s+than)\s+([^\s,]+)", "filter_less"),
            (r"show\s+(?:only\s+)?(?:rows\s+)?where\s+(\w+)\s+(equals?|=|==)\s+([^\s,]+)", "filter_equal")
        ]
        
        for pattern, action in filter_patterns:
            match = re.search(pattern, query_lower)
            if match:
                col_name, operator, value = match.groups()
                matching_cols = [col for col in df.columns if col_name.lower() in col.lower()]
                
                if matching_cols:
                    col = matching_cols[0]
                    original_rows = len(df)
                    
                    try:
                        # Type conversion
                        if df[col].dtype in ['int64', 'float64']:
                            value = float(value)
                        
                        # Apply filter
                        if action == "filter_equal" or operator in ['equals', 'equal', '=', '==']:
                            filtered_df = df[df[col] == value]
                        elif action == "filter_greater" or operator in ['>', 'greater']:
                            filtered_df = df[df[col] > value]
                        elif action == "filter_less" or operator in ['<', 'less']:
                            filtered_df = df[df[col] < value]
                        
                        st.session_state.selected_df = filtered_df
                        filtered_rows = len(filtered_df)
                        
                        response["message"] = f"Applied filter: {col} {operator} {value}"
                        response["message"] += f"\nResult: {filtered_rows:,} rows (from {original_rows:,} original rows)"
                        
                        if filtered_rows == 0:
                            response["message"] += "\n No rows match the filter criteria."
                        elif filtered_rows < original_rows * 0.1:
                            response["message"] += "\n Filter removed most data. Consider adjusting criteria."
                        
                        response["success"] = True
                        response["action"] = "filter_data"
                        return response
                        
                    except Exception as e:
                        response["message"] = f"Filter failed: {str(e)}"
                        return response
        
        # Chart creation patterns
        chart_patterns = [
            (r"(?:create|make|show|plot|draw)\s+(?:a\s+)?(\w+)\s+(?:chart|plot|graph)", "create_chart"),
            (r"(?:visualize|plot)\s+(\w+)\s+(?:vs\s+|against\s+)?(\w+)?", "create_chart_xy")
        ]
        
        for pattern, action in chart_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if action == "create_chart":
                    chart_type = match.group(1)
                    chart_type_map = {
                        'scatter': 'Scatter Plot',
                        'scatterplot': 'Scatter Plot',
                        'bar': 'Bar Chart',
                        'line': 'Line Chart',
                        'histogram': 'Histogram',
                        'hist': 'Histogram',
                        'box': 'Box Plot',
                        'violin': 'Violin Plot',
                        'pie': 'Pie Chart',
                        'heatmap': 'Correlation Heatmap'
                    }
                    
                    chart_name = chart_type_map.get(chart_type.lower())
                    if chart_name:
                        new_config = {
                            "chart_type": chart_name,
                            "id": len(st.session_state.chart_configs)
                        }
                        st.session_state.chart_configs.append(new_config)
                        response["message"] = f"Created {chart_name} configuration. Go to Visualizations tab to customize."
                        response["success"] = True
                        response["action"] = "create_chart"
                        return response
        
        # Default help response with suggestions based on data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        help_message = "I can help you with:\n\n"
        help_message += "**Data Cleaning:**\n"
        help_message += "• 'Clean missing data in [column]'\n"
        help_message += "• 'Drop column [column]'\n\n"
        
        help_message += "**Statistics:**\n"
        help_message += "• 'Show stats for [column]'\n"
        help_message += "• 'Describe [column]'\n\n"
        
        help_message += "**Filtering:**\n"
        help_message += "• 'Filter [column] equals [value]'\n"
        help_message += "• 'Show rows where [column] > [value]'\n\n"
        
        help_message += "**Visualization:**\n"
        help_message += "• 'Create scatter plot'\n"
        help_message += "• 'Make histogram'\n\n"
        
        if numeric_cols:
            help_message += f"**Available numeric columns:** {', '.join(numeric_cols[:3])}\n"
        if categorical_cols:
            help_message += f"**Available categorical columns:** {', '.join(categorical_cols[:3])}"
        
        response["message"] = help_message
        return response
        
    except Exception as e:
        error_msg = f"Query processing failed: {str(e)}"
        if LOGGING_ENABLED:
            logging.error(f"Natural language query error: {error_msg}")
        
        response["message"] = error_msg
        return response

# --- Enhanced UI Components ---
def create_production_layout():
    """Create responsive layout with enhanced themes."""
    try:
        theme = st.session_state.theme_preference
        
            
        if theme == "Colorblind Friendly":
            st.markdown("""
                <style>
                .stApp { 
                    filter: contrast(1.2) brightness(1.1);
                }
                .stMetric { 
                    background: linear-gradient(90deg, #0173B2 0%, #029E73 100%);
                    color: white;
                    padding: 1rem; 
                    border-radius: 10px;
                }
                .stPlotlyChart {
                    filter: saturate(1.3);
                }
                </style>
            """, unsafe_allow_html=True)
            
        elif theme == "Dark":
            st.markdown("""
                <style>
                .stApp { 
                    background-color: #1e1e1e; 
                    color: #ffffff; 
                }
                .stMetric { 
                    background: linear-gradient(90deg, #2d2d2d 0%, #404040 100%);
                    color: white;
                    padding: 1rem; 
                    border-radius: 10px;
                }
                .stSidebar .stSelectbox > div > div {
                    background-color: #2d2d2d;
                }
                </style>
            """, unsafe_allow_html=True)
        
        # Add performance indicator
        memory_info = monitor_memory_usage()
        if 'current_mb' in memory_info:
            if memory_info['warning']:
                st.sidebar.warning(f" High memory usage: {memory_info['current_mb']:.0f}MB")
            elif memory_info['current_mb'] > 100:
                st.sidebar.info(f"Memory: {memory_info['current_mb']:.0f}MB")
                
    except Exception as e:
        if LOGGING_ENABLED:
            logging.error(f"Layout creation error: {e}")

def show_performance_metrics():
    """Display performance metrics in sidebar."""
    try:
        with st.sidebar.expander(" Performance Monitor", expanded=False):
            memory_info = monitor_memory_usage()
            
            if 'error' not in memory_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Memory", f"{memory_info['current_mb']:.0f}MB", 
                             delta=f"Peak: {memory_info['peak_mb']:.0f}MB")
                
                with col2:
                    st.metric("CPU", f"{memory_info.get('cpu_percent', 0):.1f}%")
                
                # Performance indicators
                if memory_info['critical']:
                    st.error(" Critical memory usage!")
                elif memory_info['warning']:
                    st.warning(" High memory usage")
                else:
                    st.success("Normal performance")
                
                # Cached models info
                if st.session_state.trained_models:
                    st.info(f"{len(st.session_state.trained_models)} cached models")
                
                # Cleanup button
                if st.button(" Clean Memory", key="cleanup_memory_btn"):
                    cleanup_memory()
                    st.success("Memory cleaned!")
                    st.rerun()
                    
    except Exception as e:
        if LOGGING_ENABLED:
            logging.error(f"Performance metrics display error: {e}")

# --- Production Main Application ---
def render_basic_nlp_features(df: pd.DataFrame):
    """Render basic NLP features without external dependencies."""
    try:
        # Find text columns
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains text (more than 90% non-null and average length > 10)
                sample_data = df[col].dropna().head(100)
                if len(sample_data) > 0:
                    avg_length = sample_data.astype(str).str.len().mean()
                    if avg_length > 10:  # Likely text column
                        text_cols.append(col)
        
        if not text_cols:
            st.info(" No text columns detected in your dataset. Text columns should contain longer strings (>10 characters on average).")
            return
        
        st.success(f" Found {len(text_cols)} potential text columns: {', '.join(text_cols)}")
        
        # Column selection
        selected_text_col = st.selectbox(
            "Select text column for analysis:",
            text_cols,
            key="nlp_text_column_selector"
        )
        
        if selected_text_col:
            text_data = df[selected_text_col].dropna().astype(str)
            
            if len(text_data) == 0:
                st.warning("Selected column contains no valid text data.")
                return
            
            # Basic text statistics
            st.subheader(" Text Statistics")
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Total Texts", f"{len(text_data):,}")
            
            with stats_col2:
                avg_length = text_data.str.len().mean()
                st.metric("Avg Length", f"{avg_length:.1f} chars")
            
            with stats_col3:
                unique_texts = text_data.nunique()
                st.metric("Unique Texts", f"{unique_texts:,}")
            
            with stats_col4:
                word_counts = text_data.str.split().str.len().mean()
                st.metric("Avg Words", f"{word_counts:.1f}")
            
            # Text length distribution
            st.subheader("📏 Text Length Distribution")
            lengths = text_data.str.len()
            fig = px.histogram(
                x=lengths,
                nbins=50,
                title="Distribution of Text Lengths",
                labels={'x': 'Text Length (characters)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Most common words (simple implementation)
            st.subheader(" Word Frequency Analysis")
            
            # Simple word extraction and counting
            all_words = []
            sample_size = min(1000, len(text_data))  # Limit for performance
            sample_texts = text_data.head(sample_size)
            
            for text in sample_texts:
                # Basic word extraction
                words = str(text).lower().split()
                # Simple filtering
                filtered_words = [word.strip('.,!?;:()[]{}"') for word in words 
                                if len(word.strip('.,!?;:()[]{}"')) > 2]
                all_words.extend(filtered_words)
            
            if all_words:
                from collections import Counter
                word_counts = Counter(all_words)
                top_words = word_counts.most_common(20)
                
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                    
                    fig_words = px.bar(
                        words_df,
                        x='Frequency',
                        y='Word',
                        orientation='h',
                        title="Top 20 Most Frequent Words"
                    )
                    fig_words.update_layout(height=600)
                    st.plotly_chart(fig_words, use_container_width=True)
                    
                    # Show the data table
                    with st.expander(" Word Frequency Table", expanded=False):
                        st.dataframe(words_df, use_container_width=True)
            
            # Simple text search and filtering
            st.subheader(" Text Search & Filter")
            
            search_term = st.text_input(
                "Search for texts containing:",
                placeholder="Enter search term...",
                key="nlp_search_term"
            )
            
            if search_term:
                # Case-insensitive search
                matching_indices = text_data.str.contains(search_term, case=False, na=False)
                matching_texts = df[matching_indices]
                
                st.info(f"Found {len(matching_texts)} texts containing '{search_term}'")
                
                if len(matching_texts) > 0:
                    # Show sample results
                    display_count = min(10, len(matching_texts))
                    st.write(f"**Sample Results ({display_count} of {len(matching_texts)}):**")
                    
                    for i, (idx, row) in enumerate(matching_texts.head(display_count).iterrows()):
                        with st.expander(f"Result {i+1}: {str(row[selected_text_col])[:100]}...", expanded=False):
                            st.write(f"**Full Text:** {row[selected_text_col]}")
                            # Show other columns too
                            for col in df.columns:
                                if col != selected_text_col:
                                    st.write(f"**{col}:** {row[col]}")
                    
                    # Option to save filtered results
                    if st.button(" Save Filtered Results", key="save_nlp_filtered"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filtered_name = f"text_search_{search_term}_{timestamp}.csv"
                        st.session_state.dfs[filtered_name] = matching_texts
                        if filtered_name not in st.session_state.active_file_tabs:
                            st.session_state.active_file_tabs.append(filtered_name)
                        st.success(f"Filtered results saved as {filtered_name}")
                        st.rerun()
            
            # Basic text preprocessing preview
            st.subheader(" Text Preprocessing Preview")
            
            if len(text_data) > 0:
                sample_text = str(text_data.iloc[0])
                st.write("**Original Text:**")
                st.text_area("Original", sample_text, height=100, disabled=True)
                
                # Show preprocessing options
                preprocessing_options = st.multiselect(
                    "Select preprocessing steps:",
                    ["Lowercase", "Remove punctuation", "Remove extra spaces"],
                    default=["Lowercase"],
                    key="preprocessing_options"
                )
                
                processed_text = sample_text
                if "Lowercase" in preprocessing_options:
                    processed_text = processed_text.lower()
                if "Remove punctuation" in preprocessing_options:
                    import string
                    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
                if "Remove extra spaces" in preprocessing_options:
                    processed_text = ' '.join(processed_text.split())
                
                st.write("**Processed Text:**")
                st.text_area("Processed", processed_text, height=100, disabled=True)
    
    except Exception as e:
        st.error(f"Error in basic NLP features: {str(e)}")
        if LOGGING_ENABLED:
            logging.error(f"Basic NLP features error: {e}")

def main_production():
    """Production-grade main function with comprehensive error handling and monitoring."""
    try:
        # Load professional styling first
        load_custom_css()
        
        # Render professional header
        render_professional_header()
        
        # Initialize session state
        init_session_state()
        
        # Create production layout
        create_production_layout()
        
        # Professional control panel
        create_section_header("System Controls", "Configure theme, performance mode, and application settings")
        
        header_col1, header_col2, header_col3, header_col4 = st.columns([3, 1, 1, 1])
        
        with header_col2:
            current_theme = st.session_state.theme_preference
            # Handle legacy high contrast theme
            if current_theme == "High Contrast":
                current_theme = "Dark"
                st.session_state.theme_preference = "Dark"
            
            theme_index = 0  # Default to Light
            if current_theme in ACCESSIBILITY_THEMES:
                theme_index = ACCESSIBILITY_THEMES.index(current_theme)
            
            theme_choice = st.selectbox(
                "Theme",
                ACCESSIBILITY_THEMES,
                index=theme_index,
                key="theme_selector_prod"
            )
            if theme_choice != st.session_state.theme_preference:
                st.session_state.theme_preference = theme_choice
                st.rerun()
        
        with header_col3:
            perf_mode = st.selectbox(
                "Mode",
                ["Balanced", "Performance", "Quality"],
                key="perf_mode_selector"
            )
            st.session_state.user_preferences['performance_mode'] = perf_mode.lower()
        
        with header_col4:
            if st.button("Refresh", help="Refresh application", key="refresh_prod_btn"):
                st.rerun()
        
        # Performance monitoring in sidebar
        show_performance_metrics()
        
        # File upload section with enhanced validation
        st.subheader("Data Upload & Management")
        
        upload_col1, upload_col2 = st.columns([3, 1])
        
        with upload_col1:
            file_format = st.selectbox("File Type", FILE_TYPES, help="Select your data format")
            uploaded_files = st.file_uploader(
                "Upload Data Files (Multiple files supported)",
                type=['csv', 'xlsx', 'json', 'xls'],
                accept_multiple_files=True,
                help="Upload CSV, Excel, or JSON files. Large files will be automatically sampled for performance."
            )
        
        with upload_col2:
            if st.button("Load Demo Data", key="demo_data_prod_btn"):
                try:
                    # Create more realistic demo data
                    np.random.seed(42)
                    n_samples = 5000 if st.session_state.user_preferences['performance_mode'] == 'performance' else 8000
                    
                    dates = pd.date_range('2023-01-01', periods=n_samples, freq='4H')
                    
                    # Create correlated features for more realistic data
                    base_trend = np.sin(np.arange(n_samples) / 100) * 1000
                    noise = np.random.normal(0, 100, n_samples)
                    
                    demo_data = pd.DataFrame({
                        'timestamp': dates,
                        'sales_amount': np.maximum(0, base_trend + np.random.lognormal(6, 0.8, n_samples) + noise),
                        'profit_margin': np.random.beta(2, 5, n_samples),
                        'customer_segment': np.random.choice(
                            ['Enterprise', 'SMB', 'Consumer', 'Government'], 
                            n_samples, p=[0.3, 0.4, 0.25, 0.05]
                        ),
                        'product_category': np.random.choice(
                            ['Software', 'Hardware', 'Services', 'Support'], n_samples
                        ),
                        'region': np.random.choice(
                            ['North America', 'Europe', 'Asia Pacific', 'Latin America'], 
                            n_samples, p=[0.4, 0.3, 0.2, 0.1]
                        ),
                        'sales_rep_performance': np.random.normal(75, 15, n_samples).clip(0, 100),
                        'customer_satisfaction': np.random.normal(4.1, 0.9, n_samples).clip(1, 5),
                        'deal_complexity': np.random.choice(
                            ['Simple', 'Moderate', 'Complex', 'Enterprise'], 
                            n_samples, p=[0.4, 0.3, 0.2, 0.1]
                        ),
                        'days_to_close': np.random.gamma(2, 15, n_samples).astype(int).clip(1, 365)
                    })
                    
                    # Add realistic missing values and outliers
                    missing_indices = np.random.choice(demo_data.index, size=int(0.03 * len(demo_data)), replace=False)
                    demo_data.loc[missing_indices[:len(missing_indices)//2], 'profit_margin'] = np.nan
                    demo_data.loc[missing_indices[len(missing_indices)//2:], 'customer_satisfaction'] = np.nan
                    
                    # Add some outliers
                    outlier_indices = np.random.choice(demo_data.index, size=int(0.02 * len(demo_data)), replace=False)
                    demo_data.loc[outlier_indices, 'sales_amount'] *= np.random.uniform(3, 8, len(outlier_indices))
                    
                    # Calculate quality score
                    demo_data.attrs = {
                        'source_file': 'enterprise_demo_data.csv',
                        'load_time': datetime.now().isoformat(),
                        'quality_score': calculate_data_quality_score_enhanced(demo_data)
                    }
                    
                    st.session_state.dfs['enterprise_demo_data.csv'] = demo_data
                    st.session_state.selected_df = demo_data
                    st.session_state.data_loaded = True
                    
                    st.success(f"Demo dataset loaded! ({len(demo_data):,} rows × {len(demo_data.columns)} columns)")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Demo data creation failed: {e}")
                    if LOGGING_ENABLED:
                        logging.error(f"Demo data creation error: {e}")
        
        # Process uploaded files
        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            
            # Reset state if new files uploaded
            if st.session_state.last_uploaded_files != current_file_names:
                st.session_state.chart_configs = []
                st.session_state.filter_state = {}
                st.session_state.last_uploaded_files = current_file_names
                st.session_state.dfs = {}
                st.session_state.trained_models = {}
                cleanup_memory()
            
            # Process files with progress tracking
            with st.spinner("Processing files with enterprise-grade validation..."):
                progress_bar = st.progress(0)
                load_metrics = {'files_loaded': 0, 'total_rows': 0, 'errors': 0}
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        start_time = time.time()
                        
                        # File size validation
                        file_content = uploaded_file.read()
                        file_size_mb = len(file_content) / (1024 * 1024)
                        
                        uploaded_file.seek(0)
                        
                        if file_size_mb > 500:  # 500MB limit
                            st.error(f"File {uploaded_file.name} is too large ({file_size_mb:.1f}MB > 500MB)")
                            load_metrics['errors'] += 1
                            continue
                        
                        # Load data
                        df = load_data_production(file_content, uploaded_file.name, file_format)
                        load_time = time.time() - start_time
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        if df is not None:
                            st.session_state.dfs[uploaded_file.name] = df
                            load_metrics['files_loaded'] += 1
                            load_metrics['total_rows'] += len(df)
                            
                            # Enhanced success message
                            quality_score = df.attrs.get('quality_score', 0)
                            quality_icon = "🟢" if quality_score > 80 else "🟡" if quality_score > 60 else "🔴"
                            
                            speed_mb_s = file_size_mb / load_time if load_time > 0 else 0
                            
                            st.success(
                                f"**{uploaded_file.name}**: "
                                f"{df.shape[0]:,} × {df.shape[1]} | "
                                f"{speed_mb_s:.1f} MB/s | "
                                f"Quality: {quality_score:.0f}/100 {quality_icon}"
                            )
                        else:
                            st.error(f"Failed to load {uploaded_file.name}")
                            load_metrics['errors'] += 1
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        load_metrics['errors'] += 1
                        if LOGGING_ENABLED:
                            logging.error(f"File processing error for {uploaded_file.name}: {e}")
                
                # Display summary metrics
                if load_metrics['files_loaded'] > 0:
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Files Loaded", load_metrics['files_loaded'])
                    with metric_col2:
                        st.metric("Total Rows", f"{load_metrics['total_rows']:,}")
                    with metric_col3:
                        st.metric("Errors", load_metrics['errors'])
                    with metric_col4:
                        success_rate = (load_metrics['files_loaded'] / len(uploaded_files)) * 100
                        st.metric("Success Rate", f"{success_rate:.0f}%")
        
        # Check if any data is loaded
        if not st.session_state.dfs and not st.session_state.data_loaded:
            # Enhanced welcome screen
            st.info("Welcome to the Enterprise Data Analytics Platform")
            
            # Feature showcase
            feature_col1, feature_col2, feature_col3 = st.columns(3)
            
            with feature_col1:
                st.markdown("""
                ### AI-Powered Analytics
                - Natural language queries
                - Intelligent data cleaning
                - Automated anomaly detection
                - Smart visualizations
                - Advanced ML algorithms
                """)
            
            with feature_col2:
                st.markdown("""
                ### Production Features
                - Memory optimization
                - Error recovery systems
                - Performance monitoring
                - Comprehensive logging
                - Data validation
                """)
            
            with feature_col3:
                st.markdown("""
                ### Enterprise Capabilities
                - Large dataset handling
                - Multi-file processing
                - Advanced visualizations
                - Export capabilities
                - Accessibility features
                """)
            
            return
        
        if not st.session_state.dfs:
            st.error("No files were successfully loaded. Please check your file formats and try again.")
            return
        
        # Multi-File Tab System
        st.subheader("Multi-File Workspace")
        
        # File management controls
        file_control_col1, file_control_col2, file_control_col3 = st.columns([2, 1, 1])
        
        with file_control_col1:
            # Create file tabs
            available_files = list(st.session_state.dfs.keys())
            if available_files:
                # Initialize active tabs if empty
                if not st.session_state.active_file_tabs:
                    st.session_state.active_file_tabs = [available_files[0]]
                    st.session_state.current_file_tab = available_files[0]
                
                # File tabs
                selected_tab = st.radio(
                    "Active Files:",
                    st.session_state.active_file_tabs,
                    horizontal=True,
                    key="file_tab_selector_main"
                )
                
                st.session_state.current_file_tab = selected_tab
        
        with file_control_col2:
            # Add file to workspace
            unopened_files = [f for f in available_files if f not in st.session_state.active_file_tabs]
            if unopened_files:
                file_to_add = st.selectbox(
                    "Add File:",
                    ["Select file..."] + unopened_files,
                    key="add_file_selector"
                )
                
                if file_to_add != "Select file..." and st.button("➕ Add", key="add_file_btn"):
                    if file_to_add not in st.session_state.active_file_tabs:
                        st.session_state.active_file_tabs.append(file_to_add)
                        st.session_state.current_file_tab = file_to_add
                        st.rerun()
        
        with file_control_col3:
            # Close current file tab
            if len(st.session_state.active_file_tabs) > 1:
                if st.button(" Close Tab", key="close_tab_btn"):
                    current_tab = st.session_state.current_file_tab
                    st.session_state.active_file_tabs.remove(current_tab)
                    st.session_state.current_file_tab = st.session_state.active_file_tabs[0]
                    st.rerun()
        
        # CSV Merge functionality
        if len(available_files) > 1:
            with st.expander("🔗 Merge CSV Files", expanded=False):
                merge_col1, merge_col2 = st.columns(2)
                
                with merge_col1:
                    files_to_merge = st.multiselect(
                        "Select files to merge:",
                        available_files,
                        key="merge_file_selector"
                    )
                    
                    merge_type = st.selectbox(
                        "Merge Type:",
                        ["inner", "outer", "left", "right"],
                        key="merge_type_selector"
                    )
                
                with merge_col2:
                    if len(files_to_merge) >= 2:
                        # Find common columns for merge
                        first_df = st.session_state.dfs[files_to_merge[0]]
                        common_cols = first_df.columns.tolist()
                        
                        for file in files_to_merge[1:]:
                            df_cols = st.session_state.dfs[file].columns.tolist()
                            common_cols = [col for col in common_cols if col in df_cols]
                        
                        if common_cols:
                            merge_on = st.selectbox(
                                "Merge on column:",
                                common_cols,
                                key="merge_on_selector"
                            )
                            
                            if st.button("🔗 Merge Files", key="merge_files_btn"):
                                try:
                                    merge_dict = {f: st.session_state.dfs[f] for f in files_to_merge}
                                    merge_config = {'how': merge_type, 'left_on': merge_on, 'right_on': merge_on}
                                    
                                    result = merge_csv_files(merge_dict, merge_config)
                                    
                                    if result['success']:
                                        merged_name = f"merged_{len(st.session_state.dfs)}.csv"
                                        st.session_state.dfs[merged_name] = result['data']
                                        st.session_state.active_file_tabs.append(merged_name)
                                        st.session_state.current_file_tab = merged_name
                                        st.success(result['message'])
                                        st.rerun()
                                    else:
                                        st.error(result['message'])
                                except Exception as e:
                                    st.error(f"Merge failed: {str(e)}")
                        else:
                            st.warning("No common columns found between selected files")
        
        # Get current file
        if not st.session_state.current_file_tab or st.session_state.current_file_tab not in st.session_state.dfs:
            st.error("No active file tab")
            return
        
        selected_file = st.session_state.current_file_tab
        df = st.session_state.dfs[selected_file]
        st.session_state.selected_df = df
        
        if df is None or df.empty:
            st.error("The selected dataset is empty.")
            return
        
        st.session_state.data_loaded = True
        df_hash = str(hash(str(df.shape) + str(df.columns.tolist())))
        
        # Professional dataset overview
        create_section_header(
            f"Dataset Overview: {selected_file}", 
            f"Analyzing {len(df):,} records across {len(df.columns)} dimensions"
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Calculate basic metrics for display
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Professional KPI grid
        overview_metrics = {
            "Total Records": f"{len(df):,}",
            "Columns": f"{len(df.columns)}",
            "Data Quality": f"{100-missing_pct:.1f}%",
            "Memory Usage": f"{memory_mb:.1f}MB",
            "Numeric Fields": f"{len(numeric_cols)}",
            "Categorical Fields": f"{len(categorical_cols)}"
        }
        
        create_metrics_grid(overview_metrics, 6)
        
        # Main application tabs with enhanced features
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
            "AI Assistant",
            "Analytics", 
            "Advanced Stats",
            "Data Explorer",
            "Data Cleaning",
            "Anomaly Detection",
            "PowerBI Viz", 
            "Visualizations",
            "ML Studio",
            "SQL Query",
            "NLP Testing",
            "⚙Settings"
        ])
        
        with tab1:
            st.header("NLP AI Assistant")
            st.markdown("*Ask questions about your data in natural language*")
            
            # Enhanced chat interface
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            if st.session_state.chat_history:
                with st.container():
                    for i, chat in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                        with st.chat_message("user"):
                            st.write(chat["query"])
                        with st.chat_message("assistant"):
                            st.write(chat["response"])
                            if chat.get("success"):
                                st.success("Action completed successfully")
            
            # Chat input with suggestions
            example_queries = [
                "Clean missing data in sales_amount",
                "Show stats for customer_satisfaction",
                "Filter region equals Europe",
                "Create scatter plot",
                "Drop column with high missing values"
            ]
            
            with st.expander("Example Queries", expanded=False):
                for query in example_queries:
                    if st.button(f"{query}", key=f"example_{query}"):
                        response = process_natural_query_production(query)
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": response["message"],
                            "success": response["success"]
                        })
                        st.rerun()
            
            # Main chat input
            if query := st.chat_input("Ask about your data...", key="main_chat_input"):
                with st.spinner("Processing your question..."):
                    response = process_natural_query_production(query)
                    
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": response["message"],
                        "success": response["success"]
                    })
                    
                    if response["success"]:
                        st.success("" + response["message"])
                    else:
                        st.info("" + response["message"])
                    
                    if response.get("action"):
                        st.rerun()
        
        with tab2:
            create_section_header("Enterprise Data Analytics", "Automated insights, statistical analysis, and data profiling")
            
            try:
                # Compute enhanced EDA
                eda_data = compute_eda_summary_enhanced(df_hash, df.shape)
                
                if 'error' in eda_data:
                    st.error(f"Analytics computation failed: {eda_data['error']}")
                else:
                    # Professional insights layout
                    insight_col1, insight_col2 = st.columns([2, 1])
                    
                    with insight_col1:
                        # Key insights with professional formatting
                        insights = eda_data.get('insights', ['No insights available'])
                        if insights:
                            create_executive_summary(
                                "Key Data Insights",
                                "\n\n".join([f"• {insight}" for insight in insights[:4]])
                            )
                        
                        # AI Recommendations
                        recommendations = eda_data.get('recommendations', [])
                        if recommendations:
                            rec_text = "\n\n".join([f"• {rec}" for rec in recommendations[:3]])
                            create_executive_summary("AI Recommendations", rec_text)
                    
                    with insight_col2:
                        # Quick stats visualization
                        if len(numeric_cols) > 0:
                            st.write("**Numeric Trends (Sample)**")
                            for col in numeric_cols[:3]:
                                try:
                                    sample_data = df[col].dropna().sample(min(1000, len(df[col].dropna())), random_state=42)
                                    fig_mini = px.histogram(
                                        x=sample_data, 
                                        nbins=15, 
                                        height=120, 
                                        title=f"{col} Distribution"
                                    )
                                    fig_mini.update_layout(
                                        showlegend=False, 
                                        margin=dict(l=0, r=0, t=25, b=0),
                                        title={"font": {"size": 10}},
                                        xaxis_title="",
                                        yaxis_title=""
                                    )
                                    fig_mini.update_xaxes(showticklabels=False)
                                    fig_mini.update_yaxes(showticklabels=False)
                                    st.plotly_chart(fig_mini, use_container_width=True, key=f"mini_hist_{col}")
                                except Exception as e:
                                    if LOGGING_ENABLED:
                                        logging.warning(f"Mini histogram error for {col}: {e}")
                    
                    # Detailed analytics in expandable section
                    with st.expander("Detailed Dataset Analysis", expanded=False):
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown("**Dataset Overview**")
                            overview_metrics = {
                                "Metric": ["Total Rows", "Columns", "Memory Usage", "Missing Data %", "Quality Score"],
                                "Value": [
                                    f"{len(df):,}",
                                    f"{len(df.columns)}",
                                    f"{eda_data.get('memory_usage_mb', 0):.1f} MB",
                                    f"{eda_data.get('missing_percentage', 0):.1f}%",
                                    f"{eda_data.get('quality_score', 0):.0f}/100"
                                ]
                            }
                            st.dataframe(
                                pd.DataFrame(overview_metrics),
                                hide_index=True, 
                                use_container_width=True
                            )
                        
                        with detail_col2:
                            st.markdown("**Column Type Analysis**")
                            type_metrics = {
                                "Type": ["Numeric", "Categorical", "DateTime", "Missing Data"],
                                "Count": [
                                    len(numeric_cols),
                                    len(categorical_cols),
                                    len(datetime_cols),
                                    int(eda_data.get('missing_values', 0))
                                ]
                            }
                            st.dataframe(
                                pd.DataFrame(type_metrics),
                                hide_index=True, 
                                use_container_width=True
                            )
                        
                        # Performance metrics
                        st.markdown("**Processing Performance**")
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        with perf_col1:
                            st.metric("Sample Size", f"{eda_data.get('sample_size', 0):,}")
                        with perf_col2:
                            processing_time = time.time() - st.session_state.get('app_start_time', time.time())
                            st.metric("Session Time", f"{processing_time:.0f}s")
                        with perf_col3:
                            st.metric("Cached Models", len(st.session_state.trained_models))
            
            except Exception as e:
                st.error(f"Analytics computation failed: {str(e)}")
                if LOGGING_ENABLED:
                    logging.error(f"Anomaly detection error: {e}")
        
        with tab3:
            st.header("Advanced Statistical Operations")
            st.markdown("*Statistical functions: SUM, COUNT, MIN/MAX, PIVOT, GROUP BY and more*")
            
            if df is None or df.empty:
                st.error("No data available for statistical operations")
            else:
                # Statistical operations selector
                stats_col1, stats_col2 = st.columns([1, 2])
                
                with stats_col1:
                    st.subheader("Operations")
                    
                    operation = st.selectbox(
                        "Select Operation:",
                        ["SUM", "COUNT", "DISTINCT_COUNT", "MIN_MAX", "GROUP_BY", "PIVOT", "UNIQUE", "CONDITIONAL"],
                        key="advanced_stats_operation"
                    )
                    
                    # Dynamic inputs based on operation
                    if operation in ["SUM", "DISTINCT_COUNT", "MIN_MAX", "UNIQUE"]:
                        column = st.selectbox(
                            "Select Column:",
                            df.columns.tolist(),
                            key="stats_column_selector"
                        )
                    
                    elif operation == "COUNT":
                        column = st.selectbox(
                            "Select Column (optional):",
                            ["All Rows"] + df.columns.tolist(),
                            key="count_column_selector"
                        )
                        
                        if column != "All Rows":
                            condition = st.text_input(
                                "Condition (COUNTIF):",
                                placeholder="e.g., 'Premium' or 100",
                                key="count_condition"
                            )
                    
                    elif operation == "GROUP_BY":
                        group_column = st.selectbox(
                            "Group By Column:",
                            df.columns.tolist(),
                            key="group_by_column"
                        )
                        agg_column = st.selectbox(
                            "Aggregate Column (optional):",
                            ["Count Only"] + df.columns.tolist(),
                            key="agg_column_selector"
                        )
                        if agg_column != "Count Only":
                            agg_function = st.selectbox(
                                "Aggregation Function:",
                                ["sum", "mean", "median", "min", "max", "std"],
                                key="agg_function_selector"
                            )
                    
                    elif operation == "PIVOT":
                        index_column = st.selectbox(
                            "Index Column:",
                            df.columns.tolist(),
                            key="pivot_index_column"
                        )
                        columns_column = st.selectbox(
                            "Columns:",
                            df.columns.tolist(),
                            key="pivot_columns_column"
                        )
                        values_column = st.selectbox(
                            "Values:",
                            df.select_dtypes(include=[np.number]).columns.tolist(),
                            key="pivot_values_column"
                        )
                        agg_function = st.selectbox(
                            "Aggregation:",
                            ["sum", "mean", "count", "min", "max"],
                            key="pivot_agg_function"
                        )
                    
                    elif operation == "CONDITIONAL":
                        column = st.selectbox(
                            "Column:",
                            df.columns.tolist(),
                            key="conditional_column"
                        )
                        condition = st.text_input(
                            "Condition (e.g., > 100):",
                            key="conditional_condition"
                        )
                        true_value = st.text_input(
                            "Value if True:",
                            value="1",
                            key="conditional_true_value"
                        )
                        false_value = st.text_input(
                            "Value if False:",
                            value="0",
                            key="conditional_false_value"
                        )
                    
                    # Execute button
                    if st.button("Execute Operation", key="execute_stats_operation"):
                        with st.spinner("Performing statistical operation..."):
                            try:
                                # Prepare parameters
                                kwargs = {}
                                
                                if operation in ["SUM", "DISTINCT_COUNT", "MIN_MAX", "UNIQUE"]:
                                    kwargs['column'] = column
                                elif operation == "COUNT":
                                    if column != "All Rows":
                                        kwargs['column'] = column
                                        if 'condition' in locals() and condition:
                                            try:
                                                # Try to convert condition to appropriate type
                                                if df[column].dtype in ['int64', 'float64']:
                                                    kwargs['condition'] = float(condition)
                                                else:
                                                    kwargs['condition'] = condition
                                            except:
                                                kwargs['condition'] = condition
                                elif operation == "GROUP_BY":
                                    kwargs['group_column'] = group_column
                                    if agg_column != "Count Only":
                                        kwargs['agg_column'] = agg_column
                                        kwargs['agg_function'] = agg_function
                                elif operation == "PIVOT":
                                    kwargs['index_column'] = index_column
                                    kwargs['columns_column'] = columns_column
                                    kwargs['values_column'] = values_column
                                    kwargs['agg_function'] = agg_function
                                elif operation == "CONDITIONAL":
                                    kwargs['column'] = column
                                    kwargs['condition'] = condition
                                    kwargs['true_value'] = true_value
                                    kwargs['false_value'] = false_value
                                
                                # Perform operation
                                result = perform_advanced_statistics(df, operation, **kwargs)
                                
                                # Store result in session state
                                st.session_state['last_stats_result'] = result
                                
                            except Exception as e:
                                st.error(f"Operation failed: {str(e)}")
                
                with stats_col2:
                    st.subheader("Quick Reference")
                    
                    # Show results directly after execution without complex formatting
                    if 'last_stats_result' in st.session_state:
                        result = st.session_state['last_stats_result']
                        
                        if result['success']:
                            st.success(result['message'])
                            
                            # Simple result display
                            if isinstance(result['data'], (int, float)):
                                st.metric("Result", format_large_number(result['data']))
                            elif isinstance(result['data'], dict):
                                for key, value in result['data'].items():
                                    st.write(f"**{key}:** {value}")
                            elif hasattr(result['data'], 'shape'):  # DataFrame
                                st.dataframe(result['data'], use_container_width=True)
                        else:
                            st.error(result['message'])
                    else:
                        st.info("Execute an operation to see results here.")
                        
                        # Operation help
                        with st.expander("Operation Guide", expanded=False):
                            st.markdown("""
                            **Available Operations:**
                            
                            - **SUM**: Calculate sum of numeric column
                            - **COUNT**: Count rows or count rows matching condition (COUNTIF)
                            - **DISTINCT_COUNT**: Count unique values in column
                            - **MIN_MAX**: Find minimum and maximum values
                            - **GROUP_BY**: Group data by column and aggregate
                            - **PIVOT**: Create pivot table with aggregation
                            - **UNIQUE**: List all unique values in column
                            - **CONDITIONAL**: Apply IF/CASE logic to create new column
                            
                            **Tips:**
                            - Use COUNTIF by selecting COUNT and entering a condition
                            - PIVOT works best with categorical data
                            - Results can be saved as new datasets for further analysis
                            """)
        
        with tab4:
            create_section_header("Data Explorer", "Interactive data exploration with advanced filtering and sampling")
            
            if df is None or df.empty:
                st.error("No data available for exploration")
            else:
                # Professional data preview layout
                preview_col1, preview_col2 = st.columns([3, 1])
                
                with preview_col2:
                    # Quick filters
                    st.subheader("Quick Filters")
                    
                    # Numeric range filters
                    if numeric_cols:
                        selected_numeric = st.selectbox("Select numeric column:", ["None"] + numeric_cols, key="explorer_numeric_col")
                        if selected_numeric != "None":
                            col_data = df[selected_numeric].dropna()
                            if len(col_data) > 0:
                                min_val, max_val = float(col_data.min()), float(col_data.max())
                                if min_val != max_val:
                                    range_vals = st.slider(
                                        f"Range for {selected_numeric}:",
                                        min_val, max_val, (min_val, max_val),
                                        key="explorer_numeric_filter"
                                    )
                                    df = df[(df[selected_numeric] >= range_vals[0]) & (df[selected_numeric] <= range_vals[1])]
                    
                    # Categorical filters
                    if categorical_cols:
                        selected_categorical = st.selectbox("Select categorical column:", ["None"] + categorical_cols, key="explorer_cat_col")
                        if selected_categorical != "None":
                            unique_vals = df[selected_categorical].dropna().unique()
                            if len(unique_vals) <= 50:  # Only for reasonable number of categories
                                selected_values = st.multiselect(
                                    f"Filter {selected_categorical}:",
                                    unique_vals,
                                    key="explorer_categorical_filter"
                                )
                                if selected_values:
                                    df = df[df[selected_categorical].isin(selected_values)]
                    
                    # Display filter results
                    if len(df) < len(st.session_state.selected_df):
                        reduction = len(st.session_state.selected_df) - len(df)
                        st.info(f"Filtered out {reduction:,} rows")
                    
                    # Data export
                    if st.button("Export Filtered Data", key="export_filtered"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        export_name = f"filtered_data_{timestamp}.csv"
                        st.session_state.dfs[export_name] = df.copy()
                        if export_name not in st.session_state.active_file_tabs:
                            st.session_state.active_file_tabs.append(export_name)
                        st.success(f"Filtered data saved as {export_name}")
                        st.rerun()
                
                with preview_col1:
                    create_section_header("Data Sample", f"Showing interactive sample from {len(df):,} total records")
                    
                    # Professional display options
                    display_col1, display_col2, display_col3 = st.columns(3)
                    with display_col1:
                        sample_size = st.number_input("Sample size:", 10, min(1000, len(df)), 50, key="explorer_sample_size")
                    with display_col2:
                        sort_column = st.selectbox("Sort by:", ["None"] + list(df.columns), key="explorer_sort_col")
                    with display_col3:
                        ascending = st.checkbox("Ascending", value=True, key="explorer_ascending")
                    
                    # Apply sorting and sampling
                    display_df = df.copy()
                    if sort_column != "None":
                        display_df = display_df.sort_values(sort_column, ascending=ascending)
                    
                    if len(display_df) > sample_size:
                        display_df = display_df.head(sample_size)
                    
                    # Professional table display with summary
                    create_professional_table(
                        display_df, 
                        f"Data Sample ({len(display_df):,} of {len(df):,} records)",
                        max_rows=sample_size
                    )
                    
                    # Quick statistics
                    st.subheader("Quick Statistics")
                    if len(numeric_cols) > 0:
                        stat_col = st.selectbox("Column for statistics:", numeric_cols, key="quick_stats_col")
                        if stat_col:
                            col_data = df[stat_col].dropna()
                            if len(col_data) > 0:
                                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                with stat_col1:
                                    st.metric("Mean", f"{col_data.mean():.2f}")
                                with stat_col2:
                                    st.metric("Median", f"{col_data.median():.2f}")
                                with stat_col3:
                                    st.metric("Std Dev", f"{col_data.std():.2f}")
                                with stat_col4:
                                    st.metric("Missing", f"{df[stat_col].isnull().sum():,}")
                    
                    # Data profiling
                    with st.expander("Data Profiling", expanded=False):
                        profile_col1, profile_col2 = st.columns(2)
                        
                        with profile_col1:
                            st.write("**Column Types**")
                            dtype_counts = df.dtypes.value_counts()
                            for data_type, count in dtype_counts.items():
                                st.write(f"• {data_type}: {count} columns")
                        
                        with profile_col2:
                            st.write("**Missing Data**")
                            missing_data = df.isnull().sum()
                            missing_cols = missing_data[missing_data > 0]
                            if len(missing_cols) > 0:
                                for col, missing_count in missing_cols.head(5).items():
                                    missing_pct = (missing_count / len(df)) * 100
                                    st.write(f"• {col}: {missing_count} ({missing_pct:.1f}%)")
                            else:
                                st.write("• No missing data")
        
        with tab5:
            create_section_header("Data Cleaning Studio", "AI-powered data quality enhancement with manual overrides")
            
            if df is None or df.empty:
                st.error("No data available for cleaning")
            else:
                # Generate cleaning suggestions
                with st.spinner("Analyzing data quality..."):
                    df_hash = str(hash(str(df.shape) + str(df.columns.tolist())))
                    cleaning_suggestions = suggest_cleaning_production(df_hash)
                
                if not cleaning_suggestions:
                    st.success("✓ No immediate cleaning suggestions. Your data looks good!")
                else:
                    create_section_header("AI Cleaning Recommendations", f"Found {len(cleaning_suggestions)} optimization opportunities")
                    
                # Create professional suggestions dataframe with error handling
                    try:
                        suggestions_data = []
                        for s in cleaning_suggestions:
                            if isinstance(s, dict) and all(key in s for key in ['severity', 'column', 'description', 'type', 'impact', 'confidence']):
                                priority_emoji = '🔴 HIGH' if s['severity'] == 'high' else '🟡 MEDIUM' if s['severity'] == 'medium' else '🟢 LOW'
                                suggestions_data.append({
                                    'Priority': priority_emoji,
                                    'Column': str(s['column']),
                                    'Issue Description': str(s['description']),
                                    'Recommended Action': str(s['type']).replace('_', ' ').title(),
                                    'Business Impact': str(s['impact']).title(),
                                    'Confidence Level': str(s['confidence']).title()
                                })
                        
                        if suggestions_data:
                            suggestions_df = pd.DataFrame(suggestions_data)
                        else:
                            st.error("No valid suggestions data found")
                            suggestions_df = pd.DataFrame()
                            
                    except Exception as e:
                        st.error(f"Error creating suggestions display: {str(e)}")
                        suggestions_df = pd.DataFrame()
                    
                    # Only display table if we have valid suggestions
                    if not suggestions_df.empty:
                        create_professional_table(
                            suggestions_df,
                            f"Data Quality Assessment Results ({len(cleaning_suggestions)} recommendations)"
                        )
                    else:
                        st.warning("Unable to display data quality assessment results.")
                    
                    # Apply suggestions interface
                    st.subheader("Quick Actions")
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        selected_suggestion = st.selectbox(
                            "Select suggestion to apply:",
                            options=[f"{s['column']} - {s['description']}" for s in cleaning_suggestions],
                            key="cleaning_suggestion_selector"
                        )
                        
                        if st.button("Apply Selected", key="apply_selected_cleaning"):
                            suggestion_idx = next(
                                (i for i, s in enumerate(cleaning_suggestions)
                                 if f"{s['column']} - {s['description']}" == selected_suggestion),
                                None
                            )
                            
                            if suggestion_idx is not None:
                                with st.spinner("Applying cleaning..."):
                                    success, message = apply_cleaning_suggestion_production(cleaning_suggestions[suggestion_idx])
                                
                                if success:
                                    st.success(f"✓ {message}")
                                    # Update the main dataframe
                                    current_file = st.session_state.get('current_file_tab', list(st.session_state.dfs.keys())[0] if st.session_state.dfs else None)
                                    if current_file:
                                        st.session_state.dfs[current_file] = st.session_state.selected_df
                                    time.sleep(1)  # Brief pause for user feedback
                                    st.rerun()
                                else:
                                    st.error(f"✗ {message}")
                    
                    with action_col2:
                        if st.button("Apply High Priority", key="apply_high_priority"):
                            high_priority_suggestions = [s for s in cleaning_suggestions if s['severity'] == 'high']
                            
                            if not high_priority_suggestions:
                                st.info(" No high priority suggestions to apply.")
                            else:
                                success_count = 0
                                with st.spinner(f"Applying {len(high_priority_suggestions)} high priority suggestions..."):
                                    for suggestion in high_priority_suggestions:
                                        success, message = apply_cleaning_suggestion_production(suggestion)
                                        if success:
                                            success_count += 1
                                        else:
                                            st.warning(f"⚠️ {message}")
                                
                                if success_count > 0:
                                    st.success(f"✓ Applied {success_count} cleaning operations successfully!")
                                    # Update the main dataframe
                                    current_file = st.session_state.get('current_file_tab', list(st.session_state.dfs.keys())[0] if st.session_state.dfs else None)
                                    if current_file:
                                        st.session_state.dfs[current_file] = st.session_state.selected_df
                                    time.sleep(1)
                                    st.rerun()
                
                # Manual cleaning tools
                with st.expander("Manual Cleaning Tools", expanded=False):
                    manual_col1, manual_col2, manual_col3 = st.columns(3)
                    
                    with manual_col1:
                        clean_column = st.selectbox("Select Column:", df.columns, key="manual_clean_column")
                        clean_method = st.selectbox("Method:", CLEANING_METHODS, key="manual_clean_method")
                    
                    with manual_col2:
                        if clean_method in ["mean", "median"]:
                            custom_value = st.number_input("Custom value (optional):", value=0.0, key="custom_numeric_value")
                        elif clean_method == "mode":
                            custom_value = st.text_input("Custom value (optional):", value="", key="custom_text_value")
                        else:
                            custom_value = None
                    
                    with manual_col3:
                        if st.button("Apply Manual Cleaning", key="apply_manual_cleaning"):
                            try:
                                df_cleaned = df.copy()
                                original_missing = df_cleaned[clean_column].isnull().sum()
                                
                                if clean_method == "drop":
                                    df_cleaned = df_cleaned.drop(columns=[clean_column])
                                    success_msg = f"Dropped column '{clean_column}'"
                                
                                elif clean_method == "mean":
                                    fill_val = custom_value if custom_value != 0.0 else df_cleaned[clean_column].mean()
                                    df_cleaned[clean_column] = df_cleaned[clean_column].fillna(fill_val)
                                    success_msg = f"Filled {original_missing} values in '{clean_column}' with mean: {fill_val:.2f}"
                                
                                elif clean_method == "median":
                                    fill_val = custom_value if custom_value != 0.0 else df_cleaned[clean_column].median()
                                    df_cleaned[clean_column] = df_cleaned[clean_column].fillna(fill_val)
                                    success_msg = f"Filled {original_missing} values in '{clean_column}' with median: {fill_val:.2f}"
                                
                                elif clean_method == "mode":
                                    if custom_value:
                                        fill_val = custom_value
                                    else:
                                        mode_vals = df_cleaned[clean_column].mode()
                                        fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 'Unknown'
                                    df_cleaned[clean_column] = df_cleaned[clean_column].fillna(fill_val)
                                    success_msg = f"Filled {original_missing} values in '{clean_column}' with mode: '{fill_val}'"
                                
                                elif clean_method == "forward_fill":
                                    df_cleaned[clean_column] = df_cleaned[clean_column].ffill()
                                    success_msg = f"Forward filled {original_missing} values in '{clean_column}'"
                                
                                elif clean_method == "backward_fill":
                                    df_cleaned[clean_column] = df_cleaned[clean_column].bfill()
                                    success_msg = f"Backward filled {original_missing} values in '{clean_column}'"
                                
                                elif clean_method == "interpolate":
                                    if df_cleaned[clean_column].dtype in ['int64', 'float64']:
                                        df_cleaned[clean_column] = df_cleaned[clean_column].interpolate()
                                        success_msg = f"Interpolated {original_missing} values in '{clean_column}'"
                                    else:
                                        raise ValueError("Interpolation only works with numeric data")
                                
                                # Update dataframes
                                st.session_state.selected_df = df_cleaned
                                current_file = st.session_state.get('current_file_tab', list(st.session_state.dfs.keys())[0] if st.session_state.dfs else None)
                                if current_file:
                                    st.session_state.dfs[current_file] = df_cleaned
                                st.success(success_msg)
                                time.sleep(1)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Manual cleaning failed: {str(e)}")
        
        with tab6:
            st.header("Advanced Anomaly Detection")
            st.markdown("*Multiple detection algorithms with interactive visualization*")
            
            if not numeric_cols:
                st.warning("⚠️ No numeric columns available for anomaly detection")
            else:
                # Configuration section
                config_col1, config_col2, config_col3, config_col4 = st.columns(4)
                
                with config_col1:
                    anomaly_method = st.selectbox(
                        "Detection Method:", 
                        ANOMALY_METHODS, 
                        key="anomaly_method_prod"
                    )
                
                with config_col2:
                    selected_features = st.multiselect(
                        "Features:", 
                        numeric_cols, 
                        default=numeric_cols[:min(4, len(numeric_cols))], 
                        key="anomaly_features_prod"
                    )
                
                with config_col3:
                    # Method-specific parameters
                    if anomaly_method == "IsolationForest" and SKLEARN_AVAILABLE:
                        param_value = st.slider("Contamination Rate:", 0.01, 0.3, 0.1, key="isolation_contamination")
                        params = {"contamination": param_value}
                    elif anomaly_method in ["Z-Score", "Modified Z-Score"]:
                        param_value = st.slider("Threshold:", 1.0, 5.0, 3.0 if anomaly_method == "Z-Score" else 3.5, key="z_threshold_prod")
                        params = {"threshold": param_value}
                    else:  # IQR
                        param_value = st.slider("IQR Multiplier:", 0.5, 3.0, 1.5, key="iqr_multiplier_prod")
                        params = {"multiplier": param_value}
                
                with config_col4:
                    if st.button("Detect Anomalies", key="detect_anomalies_prod") and selected_features:
                        with st.spinner(f"Running {anomaly_method} detection..."):
                            df_hash = str(hash(str(df.shape) + str(df.columns.tolist())))
                            anomaly_result = detect_anomalies_production(df_hash, anomaly_method, params, selected_features)
                            st.session_state.anomaly_results = anomaly_result
                
                # Display results
                if 'anomaly_results' in st.session_state and st.session_state.anomaly_results and "error" not in st.session_state.anomaly_results:
                    result = st.session_state.anomaly_results
                    
                    # Summary metrics
                    st.subheader(" Detection Summary")
                    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
                    
                    with summary_col1:
                        st.metric("Method", result["method"])
                    with summary_col2:
                        st.metric("Anomalies", f"{result['outlier_count']:,}")
                    with summary_col3:
                        st.metric("Rate", f"{result.get('anomaly_rate', 0):.1f}%")
                    with summary_col4:
                        st.metric("Features", len(result['columns']))
                    with summary_col5:
                        st.metric("Time", f"{result.get('processing_time', 0):.2f}s")
                    
                    # Quality warnings
                    if result.get('quality_warnings'):
                        for warning in result['quality_warnings']:
                            st.warning(f"⚠️ {warning}")
                    
                    # Visualization
                    if len(result['columns']) >= 2 and result['outlier_count'] > 0:
                        st.subheader(" Anomaly Visualization")
                        
                        viz_col1, viz_col2 = st.columns([3, 1])
                        
                        with viz_col2:
                            x_feature = st.selectbox("X-axis:", result['columns'], key="anomaly_viz_x")
                            y_feature = st.selectbox("Y-axis:", result['columns'], index=1 if len(result['columns']) > 1 else 0, key="anomaly_viz_y")
                            
                            color_by_score = st.checkbox("Color by Score", value=True, key="color_by_score")
                            show_normal = st.checkbox("Show Normal Points", value=True, key="show_normal_points")
                        
                        with viz_col1:
                            try:
                                # Prepare visualization data
                                viz_df = df.loc[result['index']].copy() if 'index' in result else df.copy()
                                
                                # Create anomaly indicators
                                if 'outliers' in result:
                                    viz_df['is_anomaly'] = result['outliers'] == -1
                                else:
                                    # Fallback for simple threshold-based methods
                                    viz_df['is_anomaly'] = False
                                    if 'anomaly_indices' in result:
                                        viz_df.loc[result['anomaly_indices'], 'is_anomaly'] = True
                                
                                if 'anomaly_scores' in result:
                                    viz_df['anomaly_score'] = result['anomaly_scores']
                                else:
                                    viz_df['anomaly_score'] = np.where(viz_df['is_anomaly'], 1.0, 0.0)
                                
                                # Filter data if requested
                                if not show_normal:
                                    viz_df = viz_df[viz_df['is_anomaly']]
                                
                                if len(viz_df) > 0:
                                    # Create scatter plot
                                    if color_by_score and 'anomaly_scores' in result:
                                        fig_anomaly = px.scatter(
                                            viz_df, 
                                            x=x_feature, 
                                            y=y_feature,
                                            color='anomaly_score',
                                            symbol='is_anomaly',
                                            title=f"Anomaly Detection: {result['method']}",
                                            labels={'is_anomaly': 'Anomaly', 'anomaly_score': 'Anomaly Score'},
                                            color_continuous_scale="Viridis",
                                            height=500
                                        )
                                    else:
                                        fig_anomaly = px.scatter(
                                            viz_df, 
                                            x=x_feature, 
                                            y=y_feature,
                                            color='is_anomaly',
                                            size='anomaly_score',
                                            title=f"Anomaly Detection: {result['method']}",
                                            color_discrete_map={True: 'red', False: 'blue'},
                                            labels={'is_anomaly': 'Anomaly'},
                                            height=500
                                        )
                                    
                                    fig_anomaly.update_layout(
                                        showlegend=True,
                                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                                    )
                                    
                                    st.plotly_chart(fig_anomaly, use_container_width=True)
                                    
                                    # Additional insights
                                    if result['outlier_count'] > 0:
                                        st.info(f"Found {result['outlier_count']} anomalies out of {len(result.get('outliers', viz_df)):,} data points{result.get('sample_note', '')}")
                                        
                                        # Option to save anomalies
                                        if st.button(" Save Anomalies as Dataset", key="save_anomalies"):
                                            anomaly_data = viz_df[viz_df['is_anomaly']].copy()
                                            timestamp = datetime.now().strftime("%H%M%S")
                                            anomaly_name = f"anomalies_{anomaly_method.lower()}_{timestamp}.csv"
                                            st.session_state.dfs[anomaly_name] = anomaly_data
                                            if anomaly_name not in st.session_state.active_file_tabs:
                                                st.session_state.active_file_tabs.append(anomaly_name)
                                            st.success(f"Anomalies saved as {anomaly_name}")
                                            st.rerun()
                                else:
                                    st.warning("No data points to visualize.")
                                    
                            except Exception as e:
                                st.error(f"Visualization error: {str(e)}")
                                if LOGGING_ENABLED:
                                    logging.error(f"Anomaly visualization error: {e}")
                    
                    # Detailed results table
                    if result['outlier_count'] > 0:
                        with st.expander(" Detailed Results", expanded=False):
                            if 'outliers' in result and len(result['outliers']) > 0:
                                try:
                                    anomaly_indices = np.where(result['outliers'] == -1)[0]
                                    if len(anomaly_indices) > 0 and 'index' in result:
                                        # Safely get original indices and ensure they exist in the DataFrame
                                        original_indices = result['index'][anomaly_indices]
                                        valid_indices = [idx for idx in original_indices if idx in df.index]
                                        
                                        if valid_indices:
                                            detailed_anomalies = df.loc[valid_indices]
                                            st.dataframe(detailed_anomalies, use_container_width=True)
                                        else:
                                            st.warning("Anomaly indices not found in current dataset")
                                    else:
                                        st.info("No detailed anomaly data available")
                                except Exception as detail_e:
                                    st.error(f"Error displaying detailed results: {str(detail_e)}")
                                    if LOGGING_ENABLED:
                                        logging.error(f"Anomaly detail display error: {detail_e}")
                else:
                    st.info("Configure detection parameters and click 'Detect Anomalies' to analyze your data.")
            
            try:
                pass  # Additional error handling if needed
            except Exception as e:
                st.error(f"Anomaly detection failed: {str(e)}")
                if LOGGING_ENABLED:
                    logging.error(f"Anomaly detection error: {e}")
        
        with tab7:
            st.header("PowerBI-Style Visualizations")
            st.markdown("*Click sidebar to add charts*")
            
            if df is None or df.empty:
                st.error("No data available for PowerBI-style visualizations")
            else:
                # Visualization configuration panel
                st.sidebar.markdown("### Visualization Config")
                
                viz_type = st.sidebar.selectbox(
                    "Visualization Type:",
                    [
                        "Enhanced Bar Chart", "Interactive Scatter Plot", "Drill-Down Table", 
                        "Heatmap Matrix", "Multi-Series Line Chart", "Funnel Chart", 
                        "Waterfall Chart", "Treemap", "Sunburst Chart"
                    ],
                    key="powerbi_viz_type"
                )
                
                # Common configuration options
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                all_cols = df.columns.tolist()
                
                # Chart-specific configurations
                if viz_type in ["Enhanced Bar Chart", "Funnel Chart"]:
                    x_column = st.sidebar.selectbox(
                        "X-axis (Category):",
                        categorical_cols + numeric_cols,
                        key="powerbi_x_axis"
                    )
                    y_column = st.sidebar.selectbox(
                        "Y-axis (Value):",
                        numeric_cols,
                        key="powerbi_y_axis"
                    )
                    
                    color_column = st.sidebar.selectbox(
                        "Color By (optional):",
                        ["None"] + categorical_cols,
                        key="powerbi_color_column"
                    )
                    
                    # Aggregation options
                    agg_function = st.sidebar.selectbox(
                        "Aggregation:",
                        AGG_OPTIONS,
                        key="powerbi_agg_function"
                    )
                
                elif viz_type == "Interactive Scatter Plot":
                    x_column = st.sidebar.selectbox(
                        "X-axis:",
                        numeric_cols,
                        key="scatter_x_axis"
                    )
                    y_column = st.sidebar.selectbox(
                        "Y-axis:",
                        numeric_cols,
                        key="scatter_y_axis"
                    )
                    size_column = st.sidebar.selectbox(
                        "Size By (optional):",
                        ["None"] + numeric_cols,
                        key="scatter_size_column"
                    )
                    color_column = st.sidebar.selectbox(
                        "Color By (optional):",
                        ["None"] + categorical_cols + numeric_cols,
                        key="scatter_color_column"
                    )
                
                elif viz_type == "Heatmap Matrix":
                    if len(numeric_cols) >= 2:
                        selected_columns = st.sidebar.multiselect(
                            "Select Numeric Columns:",
                            numeric_cols,
                            default=numeric_cols[:min(5, len(numeric_cols))],
                            key="heatmap_columns"
                        )
                    else:
                        st.sidebar.warning("Need at least 2 numeric columns for heatmap")
                        selected_columns = numeric_cols
                
                elif viz_type == "Multi-Series Line Chart":
                    x_column = st.sidebar.selectbox(
                        "X-axis:",
                        all_cols,
                        key="line_x_axis"
                    )
                    y_columns = st.sidebar.multiselect(
                        "Y-axis (Multiple Series):",
                        numeric_cols,
                        default=[numeric_cols[0]] if numeric_cols else [],
                        key="line_y_columns"
                    )
                    group_column = st.sidebar.selectbox(
                        "Group By (optional):",
                        ["None"] + categorical_cols,
                        key="line_group_column"
                    )
                
                elif viz_type == "Treemap":
                    hierarchy_columns = st.sidebar.multiselect(
                        "Hierarchy (1-3 levels):",
                        categorical_cols,
                        key="treemap_hierarchy"
                    )
                    value_column = st.sidebar.selectbox(
                        "Value Column:",
                        numeric_cols,
                        key="treemap_value"
                    )
                
                # Interactive options
                st.sidebar.markdown("### Interactive Options")
                enable_drill_down = st.sidebar.checkbox("Enable Drill-Down", value=True, key="enable_drill_down")
                show_data_labels = st.sidebar.checkbox("Show Data Labels", value=True, key="show_data_labels")
                custom_colors = st.sidebar.checkbox("Custom Color Palette", key="custom_colors")
                
                if custom_colors:
                    color_palette = st.sidebar.selectbox(
                        "Color Palette:",
                        ["viridis", "plasma", "inferno", "magma", "cividis", "Set3", "Pastel1", "Dark2"],
                        key="color_palette"
                    )
                
                # Generate visualization button
                generate_viz = st.sidebar.button("Generate Visualization", key="generate_powerbi_viz")
                
                # Clear visualization button
                if st.sidebar.button("Clear Visualization", key="clear_powerbi_viz"):
                    if 'powerbi_viz_generated' in st.session_state:
                        del st.session_state['powerbi_viz_generated']
                    if 'powerbi_viz_config' in st.session_state:
                        del st.session_state['powerbi_viz_config']
                    st.rerun()
                
                # Store visualization state to prevent closing when interacting with controls
                if generate_viz:
                    st.session_state['powerbi_viz_generated'] = True
                    st.session_state['powerbi_viz_config'] = {
                        'viz_type': viz_type,
                        'x_column': x_column if 'x_column' in locals() else None,
                        'y_column': y_column if 'y_column' in locals() else None,
                        'agg_function': agg_function if 'agg_function' in locals() else None,
                        'color_column': color_column if 'color_column' in locals() else None,
                        'size_column': size_column if 'size_column' in locals() else None,
                        'selected_columns': selected_columns if 'selected_columns' in locals() else None,
                        'y_columns': y_columns if 'y_columns' in locals() else None,
                        'group_column': group_column if 'group_column' in locals() else None,
                        'hierarchy_columns': hierarchy_columns if 'hierarchy_columns' in locals() else None,
                        'value_column': value_column if 'value_column' in locals() else None,
                        'enable_drill_down': enable_drill_down,
                        'show_data_labels': show_data_labels,
                        'custom_colors': custom_colors,
                        'color_palette': color_palette if custom_colors else None
                    }
                
                # Check if visualization should be displayed
                if st.session_state.get('powerbi_viz_generated', False):
                    # Restore config from session state
                    config = st.session_state.get('powerbi_viz_config', {})
                    viz_type = config.get('viz_type')
                    x_column = config.get('x_column')
                    y_column = config.get('y_column')
                    agg_function = config.get('agg_function')
                    color_column = config.get('color_column')
                    size_column = config.get('size_column')
                    selected_columns = config.get('selected_columns')
                    y_columns = config.get('y_columns')
                    group_column = config.get('group_column')
                    hierarchy_columns = config.get('hierarchy_columns')
                    value_column = config.get('value_column')
                    enable_drill_down = config.get('enable_drill_down', True)
                    show_data_labels = config.get('show_data_labels', True)
                    custom_colors = config.get('custom_colors', False)
                    color_palette = config.get('color_palette')
                
                if st.session_state.get('powerbi_viz_generated', False):
                    
                    try:
                        # Main visualization area
                        viz_col1, viz_col2 = st.columns([3, 1])
                        
                        with viz_col1:
                            if viz_type == "Enhanced Bar Chart":
                                st.subheader(f"Enhanced Bar Chart: {y_column} by {x_column}")
                                
                                # Aggregate data
                                if color_column == "None":
                                    if agg_function == "None":
                                        # No aggregation - use raw data
                                        grouped = df[[x_column, y_column]].copy()
                                    else:
                                        grouped = df.groupby(x_column)[y_column].agg(agg_function).reset_index()
                                    
                                    # Create enhanced bar chart
                                    agg_title = "Raw Data" if agg_function == "None" else agg_function.title()
                                    fig = px.bar(
                                        grouped, 
                                        x=x_column, 
                                        y=y_column,
                                        title=f"{y_column.title()} ({agg_title}) by {x_column.title()}",
                                        color=y_column if not custom_colors else None,
                                        color_continuous_scale=color_palette if custom_colors else None,
                                        text=y_column if show_data_labels else None
                                    )
                                else:
                                    if agg_function == "None":
                                        # No aggregation - use raw data with color column
                                        grouped = df[[x_column, color_column, y_column]].copy()
                                    else:
                                        grouped = df.groupby([x_column, color_column])[y_column].agg(agg_function).reset_index()
                                    
                                    agg_title = "Raw Data" if agg_function == "None" else agg_function.title()
                                    fig = px.bar(
                                        grouped,
                                        x=x_column,
                                        y=y_column,
                                        color=color_column,
                                        title=f"{y_column.title()} ({agg_title}) by {x_column.title()}",
                                        color_discrete_sequence=px.colors.qualitative.Set3 if custom_colors else None,
                                        text=y_column if show_data_labels else None
                                    )
                                
                                fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                                fig.update_layout(height=500, showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                if enable_drill_down:
                                    st.subheader("🔍 Drill-Down Data")
                                    selected_category = st.selectbox(
                                        f"Select {x_column} to drill down:",
                                        grouped[x_column].unique(),
                                        key="drill_down_category"
                                    )
                                    drill_down_data = df[df[x_column] == selected_category]
                                    st.dataframe(drill_down_data, use_container_width=True)
                            
                            elif viz_type == "Interactive Scatter Plot":
                                st.subheader(f"Interactive Scatter Plot: {y_column} vs {x_column}")
                                
                                # Create scatter plot
                                fig_kwargs = {
                                    'data_frame': df,
                                    'x': x_column,
                                    'y': y_column,
                                    'title': f"{y_column.title()} vs {x_column.title()}",
                                    'hover_data': [col for col in all_cols if col not in [x_column, y_column]][:5]
                                }
                                
                                if size_column != "None":
                                    fig_kwargs['size'] = size_column
                                
                                if color_column != "None":
                                    fig_kwargs['color'] = color_column
                                    if custom_colors and color_column in numeric_cols:
                                        fig_kwargs['color_continuous_scale'] = color_palette
                                
                                fig = px.scatter(**fig_kwargs)
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif viz_type == "Heatmap Matrix" and selected_columns:
                                st.subheader("Correlation Heatmap Matrix")
                                
                                # Calculate correlation matrix
                                corr_matrix = df[selected_columns].corr()
                                
                                # Create heatmap
                                fig = px.imshow(
                                    corr_matrix,
                                    title="Correlation Matrix",
                                    color_continuous_scale=color_palette if custom_colors else "RdBu",
                                    aspect="auto",
                                    text_auto=True if show_data_labels else False
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif viz_type == "Multi-Series Line Chart" and y_columns:
                                st.subheader(f"Multi-Series Line Chart")
                                
                                if group_column == "None":
                                    # Simple multi-series line chart
                                    fig = px.line(
                                        df,
                                        x=x_column,
                                        y=y_columns,
                                        title=f"Trends over {x_column.title()}"
                                    )
                                else:
                                    # Group by category and create multiple series
                                    melted_df = df.melt(
                                        id_vars=[x_column, group_column],
                                        value_vars=y_columns,
                                        var_name='Metric',
                                        value_name='Value'
                                    )
                                    
                                    fig = px.line(
                                        melted_df,
                                        x=x_column,
                                        y='Value',
                                        color=group_column,
                                        line_dash='Metric',
                                        title=f"Trends by {group_column.title()}"
                                    )
                                
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif viz_type == "Treemap" and hierarchy_columns and value_column:
                                st.subheader("Interactive Treemap")
                                
                                # Aggregate data for treemap
                                treemap_data = df.groupby(hierarchy_columns)[value_column].sum().reset_index()
                                
                                fig = px.treemap(
                                    treemap_data,
                                    path=hierarchy_columns,
                                    values=value_column,
                                    title=f"Treemap of {value_column.title()}",
                                    color=value_column,
                                    color_continuous_scale=color_palette if custom_colors else "viridis"
                                )
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif viz_type == "Drill-Down Table":
                                st.subheader("Interactive Drill-Down Table")
                                
                                # Allow users to select grouping columns
                                group_cols = st.multiselect(
                                    "Select grouping columns:",
                                    categorical_cols + numeric_cols,
                                    default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols,
                                    key="drilldown_group_cols"
                                )
                                
                                if group_cols:
                                    # Create hierarchical grouping
                                    grouped_data = df.groupby(group_cols).size().reset_index(name='Count')
                                    
                                    # Add aggregated metrics if numeric columns exist
                                    if numeric_cols:
                                        agg_col = st.selectbox(
                                            "Metric to aggregate:",
                                            numeric_cols,
                                            key="drilldown_agg_col"
                                        )
                                        agg_func = st.selectbox(
                                            "Aggregation function:",
                                            ['sum', 'mean', 'median', 'min', 'max'],
                                            key="drilldown_agg_func"
                                        )
                                        
                                        agg_data = df.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
                                        grouped_data = grouped_data.merge(agg_data, on=group_cols)
                                    
                                    st.dataframe(grouped_data, use_container_width=True)
                                    
                                    # Interactive drill-down
                                    if enable_drill_down and len(group_cols) > 0:
                                        selected_val = st.selectbox(
                                            f"Drill down by {group_cols[0]}:",
                                            grouped_data[group_cols[0]].unique(),
                                            key="drill_table_select"
                                        )
                                        
                                        filtered_data = df[df[group_cols[0]] == selected_val]
                                        st.subheader(f"Detailed view for {group_cols[0]}: {selected_val}")
                                        st.dataframe(filtered_data, use_container_width=True)
                            
                            elif viz_type == "Funnel Chart" and len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                st.subheader("Funnel Chart")
                                
                                # Funnel charts work best with sequential stages
                                stage_col = st.selectbox(
                                    "Stage/Category Column:",
                                    categorical_cols,
                                    key="funnel_stage_col"
                                )
                                
                                value_col = st.selectbox(
                                    "Value Column:",
                                    numeric_cols,
                                    key="funnel_value_col"
                                )
                                
                                # Create funnel data
                                funnel_data = df.groupby(stage_col)[value_col].sum().reset_index()
                                funnel_data = funnel_data.sort_values(value_col, ascending=False)
                                
                                # Create funnel chart using bar chart
                                fig = px.bar(
                                    funnel_data,
                                    x=value_col,
                                    y=stage_col,
                                    orientation='h',
                                    title=f"Funnel Chart: {value_col} by {stage_col}",
                                    color=value_col,
                                    color_continuous_scale=color_palette if custom_colors else "Blues"
                                )
                                
                                # Style as funnel
                                fig.update_layout(
                                    height=500,
                                    yaxis={'categoryorder': 'total ascending'}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show conversion rates
                                if len(funnel_data) > 1:
                                    st.subheader("Conversion Rates")
                                    funnel_data['Conversion_Rate'] = (funnel_data[value_col] / funnel_data[value_col].iloc[0] * 100).round(2)
                                    st.dataframe(funnel_data[[stage_col, value_col, 'Conversion_Rate']], use_container_width=True)
                            
                            elif viz_type == "Waterfall Chart" and len(numeric_cols) > 0:
                                st.subheader("Waterfall Chart")
                                
                                waterfall_col = st.selectbox(
                                    "Select column for waterfall:",
                                    numeric_cols,
                                    key="waterfall_col"
                                )
                                
                                category_col = st.selectbox(
                                    "Category column:",
                                    categorical_cols if categorical_cols else [None],
                                    key="waterfall_category"
                                )
                                
                                if category_col:
                                    # Create waterfall data
                                    waterfall_data = df.groupby(category_col)[waterfall_col].sum().reset_index()
                                    waterfall_data = waterfall_data.sort_values(waterfall_col)
                                    
                                    # Calculate cumulative values
                                    waterfall_data['Cumulative'] = waterfall_data[waterfall_col].cumsum()
                                    waterfall_data['Start'] = waterfall_data['Cumulative'] - waterfall_data[waterfall_col]
                                    
                                    # Create waterfall visualization
                                    fig = px.bar(
                                        waterfall_data,
                                        x=category_col,
                                        y=waterfall_col,
                                        title=f"Waterfall Chart: {waterfall_col} by {category_col}",
                                        color=waterfall_col,
                                        color_continuous_scale="RdYlGn"
                                    )
                                    
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.dataframe(waterfall_data, use_container_width=True)
                            
                            elif viz_type == "Sunburst Chart" and len(categorical_cols) >= 2:
                                st.subheader("Sunburst Chart")
                                
                                # Select hierarchical columns
                                sunburst_cols = st.multiselect(
                                    "Select hierarchy (2-3 levels):",
                                    categorical_cols,
                                    default=categorical_cols[:2],
                                    key="sunburst_hierarchy"
                                )
                                
                                if len(sunburst_cols) >= 2:
                                    value_col = st.selectbox(
                                        "Value column (optional):",
                                        ["Count"] + numeric_cols,
                                        key="sunburst_value"
                                    )
                                    
                                    if value_col == "Count":
                                        sunburst_data = df.groupby(sunburst_cols).size().reset_index(name='values')
                                    else:
                                        sunburst_data = df.groupby(sunburst_cols)[value_col].sum().reset_index()
                                        sunburst_data.columns = list(sunburst_cols) + ['values']
                                    
                                    fig = px.sunburst(
                                        sunburst_data,
                                        path=sunburst_cols,
                                        values='values',
                                        title=f"Sunburst Chart: {' → '.join(sunburst_cols)}",
                                        color='values',
                                        color_continuous_scale=color_palette if custom_colors else "Viridis"
                                    )
                                    
                                    fig.update_layout(height=600)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                st.warning(f"⚠️ {viz_type} requires specific column types or combinations. Please check your data and selections:")
                                st.info("""
                                **Requirements:**
                                - **Enhanced Bar Chart, Funnel Chart**: Categorical + Numeric columns
                                - **Interactive Scatter Plot**: 2+ Numeric columns
                                - **Heatmap Matrix**: 2+ Numeric columns
                                - **Multi-Series Line Chart**: 1+ Numeric columns
                                - **Treemap**: 1+ Categorical + 1 Numeric column
                                - **Sunburst Chart**: 2+ Categorical columns
                                - **Waterfall Chart**: 1+ Numeric + 1 Categorical column
                                """)
                        
                        with viz_col2:
                            st.subheader(" Quick Stats")
                            
                            if viz_type in ["Enhanced Bar Chart", "Interactive Scatter Plot"]:
                                # Show relevant statistics
                                if y_column in numeric_cols:
                                    st.metric("Total Records", f"{len(df):,}")
                                    st.metric(f"Mean {y_column}", f"{df[y_column].mean():.2f}")
                                    st.metric(f"Max {y_column}", f"{df[y_column].max():.2f}")
                                    st.metric(f"Min {y_column}", f"{df[y_column].min():.2f}")
                            
                            # Data export options
                            st.subheader(" Export Options")
                            
                            export_format = st.selectbox(
                                "Export Format:",
                                ["CSV", "Excel", "JSON"],
                                key="export_format"
                            )
                            
                            if st.button(" Export Current View", key="export_viz_data"):
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                export_filename = f"powerbi_viz_export_{timestamp}"
                                
                                try:
                                    if export_format == "CSV":
                                        csv_data = df.to_csv(index=False)
                                        st.download_button(
                                            label=f" Download {export_filename}.csv",
                                            data=csv_data,
                                            file_name=f"{export_filename}.csv",
                                            mime="text/csv",
                                            key="download_csv_powerbi"
                                        )
                                    elif export_format == "Excel":
                                        buffer = io.BytesIO()
                                        df.to_excel(buffer, index=False, engine='openpyxl')
                                        excel_data = buffer.getvalue()
                                        st.download_button(
                                            label=f" Download {export_filename}.xlsx",
                                            data=excel_data,
                                            file_name=f"{export_filename}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key="download_excel_powerbi"
                                        )
                                    elif export_format == "JSON":
                                        json_data = df.to_json(orient='records', indent=2)
                                        st.download_button(
                                            label=f" Download {export_filename}.json",
                                            data=json_data,
                                            file_name=f"{export_filename}.json",
                                            mime="application/json",
                                            key="download_json_powerbi"
                                        )
                                except Exception as export_e:
                                    st.error(f"Export failed: {str(export_e)}")
                    
                    except Exception as e:
                        st.error(f"Visualization generation failed: {str(e)}")
                        if LOGGING_ENABLED:
                            logging.error(f"PowerBI visualization error: {e}")
        
        with tab8:
            st.header("Standard Visualizations")
            st.markdown("*Create standard charts and plots for data analysis*")
            
            if df is None or df.empty:
                st.error("No data available for visualization")
            else:
                # Visualization configuration
                viz_config_col1, viz_config_col2, viz_config_col3 = st.columns(3)
                
                with viz_config_col1:
                    chart_type = st.selectbox(
                        "Chart Type:",
                        CHART_OPTIONS,
                        key="std_chart_type"
                    )
                
                with viz_config_col2:
                    if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart"]:
                        x_column = st.selectbox(
                            "X-axis:",
                            df.columns.tolist(),
                            key="std_x_column"
                        )
                        y_column = st.selectbox(
                            "Y-axis:",
                            numeric_cols,
                            key="std_y_column"
                        ) if numeric_cols else None
                    
                    elif chart_type == "Histogram":
                        hist_column = st.selectbox(
                            "Column:",
                            numeric_cols,
                            key="hist_column"
                        ) if numeric_cols else None
                    
                    elif chart_type == "Box Plot":
                        box_column = st.selectbox(
                            "Value Column:",
                            numeric_cols,
                            key="box_column"
                        ) if numeric_cols else None
                        group_column = st.selectbox(
                            "Group By (optional):",
                            ["None"] + categorical_cols,
                            key="box_group_column"
                        )
                    
                    elif chart_type == "Pie Chart":
                        pie_column = st.selectbox(
                            "Category Column:",
                            categorical_cols,
                            key="pie_column"
                        ) if categorical_cols else None
                        pie_value = st.selectbox(
                            "Value Column (optional):",
                            ["Count"] + numeric_cols,
                            key="pie_value"
                        )
                
                with viz_config_col3:
                    # Chart styling options
                    theme = st.selectbox(
                        "Theme:",
                        THEME_OPTIONS,
                        key="chart_theme"
                    )
                    
                    color_palette = st.selectbox(
                        "Color Palette:",
                        COLOR_PALETTES,
                        key="chart_colors"
                    )
                    
                    chart_height = st.slider(
                        "Chart Height:",
                        300, 800, 500,
                        key="chart_height"
                    )
                    
                    # Data labels option
                    show_data_labels = st.checkbox(
                        "Show Data Labels",
                        value=False,
                        key="std_show_data_labels"
                    )
                
                # Generate visualization
                if st.button("Generate Chart", key="generate_std_chart", type="primary"):
                    try:
                        with st.spinner("Creating visualization..."):
                            fig = None
                            
                            if chart_type == "Scatter Plot" and x_column and y_column:
                                fig = px.scatter(
                                    df, x=x_column, y=y_column,
                                    title=f"{chart_type}: {y_column} vs {x_column}",
                                    template=theme,
                                    color_discrete_sequence=getattr(px.colors.qualitative, color_palette.replace(' ', '_'), px.colors.qualitative.Plotly),
                                    height=chart_height
                                )
                            
                            elif chart_type == "Line Chart" and x_column and y_column:
                                fig = px.line(
                                    df, x=x_column, y=y_column,
                                    title=f"{chart_type}: {y_column} over {x_column}",
                                    template=theme,
                                    height=chart_height
                                )
                            
                            elif chart_type == "Bar Chart" and x_column and y_column:
                                # Aggregate data for bar chart
                                agg_df = df.groupby(x_column)[y_column].mean().reset_index()
                                fig = px.bar(
                                    agg_df, x=x_column, y=y_column,
                                    title=f"{chart_type}: Average {y_column} by {x_column}",
                                    template=theme,
                                    color=y_column,
                                    height=chart_height,
                                    text=y_column if show_data_labels else None
                                )
                                if show_data_labels:
                                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                            
                            elif chart_type == "Histogram" and hist_column:
                                fig = px.histogram(
                                    df, x=hist_column,
                                    title=f"{chart_type}: Distribution of {hist_column}",
                                    template=theme,
                                    nbins=30,
                                    height=chart_height
                                )
                            
                            elif chart_type == "Box Plot" and box_column:
                                if group_column != "None":
                                    fig = px.box(
                                        df, y=box_column, x=group_column,
                                        title=f"{chart_type}: {box_column} by {group_column}",
                                        template=theme,
                                        height=chart_height
                                    )
                                else:
                                    fig = px.box(
                                        df, y=box_column,
                                        title=f"{chart_type}: {box_column}",
                                        template=theme,
                                        height=chart_height
                                    )
                            
                            elif chart_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
                                corr_matrix = df[numeric_cols].corr()
                                fig = px.imshow(
                                    corr_matrix,
                                    title="Correlation Heatmap",
                                    template=theme,
                                    color_continuous_scale=color_palette,
                                    height=chart_height,
                                    text_auto=True if show_data_labels else False
                                )
                            
                            elif chart_type == "Pie Chart" and pie_column:
                                if pie_value == "Count":
                                    pie_data = df[pie_column].value_counts().reset_index()
                                    pie_data.columns = [pie_column, 'count']
                                    fig = px.pie(
                                        pie_data, values='count', names=pie_column,
                                        title=f"{chart_type}: Distribution of {pie_column}",
                                        template=theme,
                                        height=chart_height
                                    )
                                else:
                                    agg_data = df.groupby(pie_column)[pie_value].sum().reset_index()
                                    fig = px.pie(
                                        agg_data, values=pie_value, names=pie_column,
                                        title=f"{chart_type}: {pie_value} by {pie_column}",
                                        template=theme,
                                        height=chart_height
                                    )
                            
                            elif chart_type == "Violin Plot" and len(numeric_cols) >= 1:
                                fig = px.violin(
                                    df, y=numeric_cols[0],
                                    title=f"{chart_type}: Distribution of {numeric_cols[0]}",
                                    template=theme,
                                    height=chart_height
                                )
                            
                            # Display the chart
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Store current chart config in session state for saving
                                st.session_state.current_chart_config = {
                                    'chart_type': chart_type,
                                    'x_column': x_column if 'x_column' in locals() else None,
                                    'y_column': y_column if 'y_column' in locals() else None,
                                    'theme': theme,
                                    'color_palette': color_palette,
                                    'height': chart_height,
                                    'show_data_labels': show_data_labels,
                                    'timestamp': datetime.now().isoformat()
                                }
                            else:
                                st.warning("Could not generate chart with current configuration. Please check your column selections.")
                    
                    except Exception as e:
                        st.error(f"Chart generation failed: {str(e)}")
                        if LOGGING_ENABLED:
                            logging.error(f"Standard visualization error: {e}")
                
                # Chart configuration management (outside the generation button flow)
                if 'current_chart_config' in st.session_state:
                    st.divider()
                    config_save_col1, config_save_col2 = st.columns([3, 1])
                    
                    with config_save_col1:
                        st.write(f"**Current Chart:** {st.session_state.current_chart_config.get('chart_type', 'None')}")
                    
                    with config_save_col2:
                        if st.button("Save Chart Config", key="save_chart_config_btn"):
                            if 'chart_configs' not in st.session_state:
                                st.session_state.chart_configs = []
                            
                            st.session_state.chart_configs.append(st.session_state.current_chart_config.copy())
                            st.success("✓ Chart configuration saved!")
                            
                            # Optional: limit the number of saved configs
                            if len(st.session_state.chart_configs) > 10:
                                st.session_state.chart_configs = st.session_state.chart_configs[-10:]
                
                # Chart gallery from saved configs
                if st.session_state.chart_configs:
                    with st.expander("🇺️ Saved Chart Gallery", expanded=False):
                        for i, config in enumerate(st.session_state.chart_configs[-5:]):  # Show last 5
                            timestamp_str = config.get('timestamp', 'Unknown time')[:16] if config.get('timestamp') else 'Unknown time'
                            st.write(f"**{config.get('chart_type', config.get('type', 'Unknown'))}** - {config.get('x_column', 'N/A')} vs {config.get('y_column', 'N/A')} ({timestamp_str})")
                            if st.button(f"Recreate Chart {i+1}", key=f"recreate_chart_{i}"):
                                st.info("Chart recreation feature coming soon!")
        
        with tab9:
            st.header("ML Studio")
            st.markdown("*Advanced machine learning model training and evaluation*")
            
            if not SKLEARN_AVAILABLE:
                st.error("Machine Learning features require scikit-learn. Please install it to use this tab.")
                st.code("pip install scikit-learn", language="bash")
            elif df is None or df.empty:
                st.error("No data available for machine learning")
            else:
                # ML workflow configuration
                ml_col1, ml_col2 = st.columns([2, 1])
                
                with ml_col1:
                    st.subheader("Model Configuration")
                    
                    # Task type selection
                    task_type = st.selectbox(
                        "Machine Learning Task:",
                        ["Regression", "Classification", "Clustering"],
                        key="ml_task_type"
                    )
                    
                    if task_type in ["Regression", "Classification"]:
                        # Supervised learning setup
                        target_column = st.selectbox(
                            "Target Column (y):",
                            numeric_cols if task_type == "Regression" else df.columns.tolist(),
                            key="ml_target_column"
                        )
                        
                        feature_columns = st.multiselect(
                            "Feature Columns (X):",
                            [col for col in numeric_cols if col != target_column],
                            default=[col for col in numeric_cols if col != target_column][:5],
                            key="ml_feature_columns"
                        )
                        
                        # Model selection
                        if task_type == "Regression":
                            model_type = st.selectbox(
                                "Model Type:",
                                ["RandomForest", "MLP", "LinearRegression"],
                                key="ml_regression_model_type"
                            )
                        else:  # Classification
                            model_type = st.selectbox(
                                "Model Type:",
                                ["RandomForest", "MLP", "LogisticRegression"],
                                key="ml_classification_model_type"
                            )
                    
                    else:  # Clustering
                        feature_columns = st.multiselect(
                            "Feature Columns:",
                            numeric_cols,
                            default=numeric_cols[:5],
                            key="ml_clustering_features"
                        )
                        
                        model_type = st.selectbox(
                            "Clustering Algorithm:",
                            ["KMeans", "MiniBatchKMeans"],
                            key="ml_clustering_type"
                        )
                        
                        n_clusters = st.slider(
                            "Number of Clusters:",
                            2, 10, 3,
                            key="ml_n_clusters"
                        )
                    
                    # Training configuration
                    train_config_col1, train_config_col2 = st.columns(2)
                    
                    with train_config_col1:
                        test_size = st.slider(
                            "Test Size (%):",
                            10, 50, 20,
                            key="ml_test_size"
                        ) / 100.0 if task_type != "Clustering" else None
                    
                    with train_config_col2:
                        random_state = st.number_input(
                            "Random State:",
                            0, 1000, 42,
                            key="ml_random_state"
                        )
                
                with ml_col2:
                    st.subheader("Model Performance")
                    
                    # Display existing trained models
                    if st.session_state.trained_models:
                        st.write("**Trained Models:**")
                        for model_name, model_info in st.session_state.trained_models.items():
                            with st.expander(f"{model_name}", expanded=False):
                                st.json(model_info.get('metrics', {}))
                    else:
                        st.info("No models trained yet")
                
                # Train model button
                if st.button("Train Model", key="train_ml_model", type="primary"):
                    if task_type == "Clustering":
                        if not feature_columns:
                            st.error("Please select feature columns for clustering")
                        else:
                            with st.spinner(f"Training {model_type} clustering model..."):
                                try:
                                    # Prepare data
                                    X = df[feature_columns].fillna(0)
                                    
                                    # Train clustering model
                                    if model_type == "KMeans":
                                        model = KMeans(n_clusters=n_clusters, random_state=random_state)
                                    else:  # MiniBatchKMeans
                                        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
                                    
                                    clusters = model.fit_predict(X)
                                    
                                    # Calculate clustering metrics
                                    silhouette_avg = silhouette_score(X, clusters)
                                    davies_bouldin = davies_bouldin_score(X, clusters)
                                    calinski_harabasz = calinski_harabasz_score(X, clusters)
                                    
                                    # Store results
                                    model_name = f"{model_type}_{datetime.now().strftime('%H%M%S')}"
                                    st.session_state.trained_models[model_name] = {
                                        'model': model,
                                        'model_type': 'clustering',
                                        'algorithm': model_type,
                                        'features': feature_columns,
                                        'n_clusters': n_clusters,
                                        'metrics': {
                                            'silhouette_score': silhouette_avg,
                                            'davies_bouldin_score': davies_bouldin,
                                            'calinski_harabasz_score': calinski_harabasz
                                        },
                                        'predictions': clusters
                                    }
                                    
                                    st.success(f"Model '{model_name}' trained successfully!")
                                    st.json({
                                        'Silhouette Score': f"{silhouette_avg:.3f}",
                                        'Davies-Bouldin Score': f"{davies_bouldin:.3f}",
                                        'Calinski-Harabasz Score': f"{calinski_harabasz:.3f}"
                                    })
                                    
                                    # Add cluster predictions to dataframe
                                    df_with_clusters = df.copy()
                                    df_with_clusters['cluster'] = clusters
                                    
                                    # Save clustered data
                                    clustered_name = f"clustered_data_{model_type.lower()}_{datetime.now().strftime('%H%M%S')}.csv"
                                    st.session_state.dfs[clustered_name] = df_with_clusters
                                    if clustered_name not in st.session_state.active_file_tabs:
                                        st.session_state.active_file_tabs.append(clustered_name)
                                    
                                    st.info(f"Data with cluster labels saved as {clustered_name}")
                                    
                                    # Visualization
                                    if len(feature_columns) >= 2:
                                        fig = px.scatter(
                                            df_with_clusters,
                                            x=feature_columns[0],
                                            y=feature_columns[1],
                                            color='cluster',
                                            title=f"Clustering Results: {model_type}",
                                            color_continuous_scale="viridis"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                except Exception as e:
                                    st.error(f"Clustering failed: {str(e)}")
                    
                    else:  # Supervised learning (Regression/Classification)
                        if not feature_columns or not target_column:
                            st.error("Please select feature columns and target column")
                        else:
                            with st.spinner(f"Training {model_type} {task_type.lower()} model..."):
                                try:
                                    # Use the production ML training function
                                    result = train_ml_model_production(
                                        df, feature_columns, target_column, 
                                        task_type, model_type
                                    )
                                    
                                    if 'error' not in result:
                                        st.success(f"Model trained successfully!")
                                        st.json(result['metrics'])
                                        
                                        # Store in session state
                                        model_name = f"{model_type}_{task_type}_{datetime.now().strftime('%H%M%S')}"
                                        st.session_state.trained_models[model_name] = result
                                        
                                        # Feature importance plot if available
                                        if 'feature_importance' in result and result['feature_importance']:
                                            try:
                                                # Create feature importance DataFrame safely
                                                feature_names = []
                                                importance_values = []
                                                
                                                for feature in feature_columns:
                                                    if feature in result['feature_importance']:
                                                        feature_names.append(str(feature))
                                                        importance_values.append(float(result['feature_importance'][feature]))
                                                
                                                if feature_names and importance_values:
                                                    importance_df = pd.DataFrame({
                                                        'feature': feature_names,
                                                        'importance': importance_values
                                                    }).sort_values('importance', ascending=False)
                                                    
                                                    fig = px.bar(
                                                        importance_df,
                                                        x='importance',
                                                        y='feature',
                                                        orientation='h',
                                                        title="Feature Importance"
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.info("Feature importance data not available for visualization")
                                            except Exception as plot_e:
                                                st.warning(f"Could not create feature importance plot: {str(plot_e)}")
                                    
                                    else:
                                        st.error(f"Training failed: {result['error']}")
                                
                                except Exception as e:
                                    st.error(f"Model training failed: {str(e)}")
                                    if LOGGING_ENABLED:
                                        logging.error(f"ML training error: {e}")
                
                # Model comparison and management
                if len(st.session_state.trained_models) > 1:
                    with st.expander(" Model Comparison", expanded=False):
                        st.subheader("Compare Model Performance")
                        
                        # Create comparison dataframe
                        comparison_data = []
                        for name, model_info in st.session_state.trained_models.items():
                            if 'metrics' in model_info:
                                row = {'Model': name, 'Type': model_info.get('algorithm', 'Unknown')}
                                row.update(model_info['metrics'])
                                comparison_data.append(row)
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                        
                        # Clear models button
                        if st.button("🗑️ Clear All Models", key="clear_ml_models"):
                            st.session_state.trained_models = {}
                            st.success("All models cleared!")
                            st.rerun()
        
        with tab10:
            st.header("SQL Query")
            st.markdown("*Execute SQL queries on your dataset*")
            
            # Check if SQL libraries are available
            if not SQL_AVAILABLE:
                st.error("⚠️ SQL functionality requires additional libraries")
                
                with st.expander(" Enable SQL Functionality", expanded=True):
                    st.markdown("""
                    **Missing Dependency:** SQLAlchemy is required for SQL operations.
                    
                    **Quick Fix:**
                    ```bash
                    pip install sqlalchemy
                    ```
                    
                    **After installation:**
                    1. Restart your Streamlit application
                    2. Return to this tab to access SQL features
                    
                    **SQL Features Available After Install:**
                    - Execute SQL queries on your dataset
                    - Create temporary tables from DataFrames  
                    - Advanced data filtering with SQL
                    - Join operations between data sources
                    """)
                    
                    if st.button(" Check SQL Availability", key="check_sql_availability"):
                        st.info("Please restart the application after installing SQLAlchemy")
                
                st.info(" **Alternative:** Use the Data Explorer tab for filtering and analysis without SQL.")
                return
            
            # SQL Query Interface
            try:
                # Generate a unique hash for the current dataframe
                current_file_hash = str(hash(str(df.shape) + str(df.columns.tolist()) + st.session_state.get('current_file_tab', '')))
                stored_file_hash = st.session_state.get('sql_file_hash', '')
                
                # Check if we need to create/refresh the database
                needs_refresh = (
                    'sql_engine' not in st.session_state or 
                    'sql_table_name' not in st.session_state or
                    current_file_hash != stored_file_hash
                )
                
                # Refresh database button
                if st.button("Refresh Database", key="refresh_sql_db"):
                    needs_refresh = True
                
                if needs_refresh:
                    with st.spinner("Setting up SQL database..."):
                        # Clear any previous state
                        if 'sql_engine' in st.session_state:
                            st.session_state.pop('sql_engine', None)
                        if 'sql_table_name' in st.session_state:
                            st.session_state.pop('sql_table_name', None)
                        
                        st.session_state.sql_file_hash = current_file_hash
                        
                        # Validate dataframe before creating database
                        if df is None or df.empty:
                            st.error("Cannot create database: DataFrame is empty or None")
                            return
                        
                        # Get current dataframe info
                        current_file_name = st.session_state.get('current_file_tab', 'current_data')
                        
                        st.info(f"Creating database from file: **{current_file_name}** ({len(df):,} rows, {len(df.columns)} columns)")
                        
                        # Pass the original file name to the function - it will sanitize it internally
                        # The function expects a dict with filename as key
                        result = create_database_from_dataframes({current_file_name: df})
                        
                        # Handle the result
                        if result.get('success', False):
                            st.session_state.sql_engine = result['engine']
                            
                            # Get the actual table name that was created by the function
                            if result.get('tables_created') and len(result['tables_created']) > 0:
                                table_info = result['tables_created'][0]
                                # Use the actual sanitized table name from the function
                                actual_table_name = table_info['table_name']
                                st.session_state.sql_table_name = actual_table_name
                            else:
                                # Fallback: manually sanitize like the function does
                                actual_table_name = re.sub(r'[^\w]', '_', str(current_file_name).split('.')[0]).lower()
                                st.session_state.sql_table_name = actual_table_name
                            
                            st.success(f"✅ Database ready! Table: `{actual_table_name}`")
                            
                            # Show database details
                            with st.expander(" Database Setup Details", expanded=False):
                                st.write("**Setup Summary:**")
                                st.write(f"• Tables Created: {result.get('total_tables', 0)}")
                                st.write(f"• Total Rows: {result.get('total_rows', 0):,}")
                                
                                if result.get('tables_created'):
                                    st.write("\n**Table Details:**")
                                    for table in result['tables_created']:
                                        st.write(f"• **{table['original_name']}** → `{table['table_name']}` ({table['rows']:,} rows, {table['columns']} cols)")
                                
                                # Test the connection
                                try:
                                    test_query = f"SELECT COUNT(*) as row_count FROM {st.session_state.sql_table_name}"
                                    test_result = pd.read_sql_query(test_query, st.session_state.sql_engine)
                                    st.write(f"\n**Connection Test:** ✅ Successfully counted {test_result.iloc[0]['row_count']:,} rows")
                                except Exception as test_e:
                                    st.write(f"\n**Connection Test:** ❌ Failed - {str(test_e)}")
                        else:
                            # Database creation failed
                            error_msg = result.get('error', 'Unknown database creation error')
                            st.error(f"❌ Database setup failed: {error_msg}")
                            
                            with st.expander("🔧 Troubleshooting", expanded=True):
                                st.write("**Common Issues:**")
                                st.write("• Ensure SQLAlchemy is installed: `pip install sqlalchemy`")
                                st.write("• Check if DataFrame is valid and not empty")
                                st.write("• Verify column names don't contain special characters")
                                st.write(f"\n**Current DataFrame Info:**")
                                st.write(f"• Shape: {df.shape}")
                                st.write(f"• Columns: {list(df.columns)[:10]}" + ("..." if len(df.columns) > 10 else ""))
                                st.write(f"• Memory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
                            return
                
                # Check if database is ready
                if 'sql_engine' not in st.session_state or 'sql_table_name' not in st.session_state:
                    st.warning("⚠️ Database not initialized. Click 'Refresh Database' to set up.")
                    return
                
                # Get the actual table name
                actual_table_name = st.session_state.sql_table_name
                
                # Enhanced debug info to track table persistence
                with st.expander("Debug Info", expanded=False):
                    st.write(f"**Current File Tab:** `{st.session_state.get('current_file_tab', 'Not set')}`")
                    st.write(f"**Session State Table Name:** `{st.session_state.get('sql_table_name', 'Not set')}`")
                    st.write(f"**SQL Engine:** {st.session_state.get('sql_engine', 'Not set')}")
                    st.write(f"**SQL File Hash:** `{st.session_state.get('sql_file_hash', 'Not set')}`")
                    
                    # Show what the sanitized name would be
                    current_file = st.session_state.get('current_file_tab', 'current_data')
                    expected_sanitized = re.sub(r'[^\w]', '_', str(current_file).split('.')[0]).lower()
                    st.write(f"**Expected Sanitized Name:** `{expected_sanitized}`")
                    
                    # Query database tables with enhanced error handling
                    try:
                        # Test basic database connectivity first
                        engine = st.session_state.sql_engine
                        st.write(f"**Engine URL:** `{engine.url}`")
                        
                        # Check if we can connect to the engine
                        with engine.connect() as conn:
                            st.write("**Engine Connection:** ✅ Successful")
                        
                        # Query tables using pandas (more reliable for in-memory DBs)
                        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
                        existing_tables = pd.read_sql_query(tables_query, engine)
                        table_list = existing_tables['name'].tolist()
                        
                        if table_list:
                            st.write(f"**Tables in Database:** {', '.join(['`' + t + '`' for t in table_list])}")
                            
                            # Test a simple count query on the first table to verify data exists
                            first_table = table_list[0]
                            try:
                                count_query = f"SELECT COUNT(*) as count FROM {first_table}"
                                count_result = pd.read_sql_query(count_query, engine)
                                row_count = count_result.iloc[0]['count']
                                st.write(f"**Table '{first_table}' Row Count:** {row_count:,} rows ✅")
                            except Exception as count_e:
                                st.write(f"**Table '{first_table}' Row Count:** ❌ Failed - {str(count_e)}")
                            
                            # Check if expected name matches
                            if expected_sanitized not in table_list:
                                st.warning(f"⚠️ Expected table `{expected_sanitized}` not found. Available: {table_list}")
                        else:
                            st.error("**Tables in Database:** ❌ No tables found! Database might be empty.")
                            
                    except Exception as e:
                        st.error(f"**Database Query Failed:** {str(e)}")
                        st.write("**Possible Issues:**")
                        st.write("• Database connection lost")
                        st.write("• In-memory database was recreated")
                        st.write("• SQLite engine reference is invalid")
                        st.info(" Try clicking 'Refresh Database' to recreate the connection.")
                
                # Show available tables and columns
                with st.expander("Available Table & Columns", expanded=True):
                    st.write(f"**Table Name:** `{actual_table_name}`")
                    st.write(f"**Columns:** {', '.join(['`' + col + '`' for col in df.columns.tolist()])}")
                    st.write(f"**Sample Data:**")
                    st.dataframe(df.head(3), use_container_width=True)
                
                # Query info bar
                st.info(f"**Active Table**: `{actual_table_name}` | **Rows**: {len(df):,} | **Columns**: {len(df.columns)}")
                
                # Default query
                default_query = f"SELECT * FROM {actual_table_name} LIMIT 10"
                
                # SQL Query input area
                sql_query = st.text_area(
                    "Enter your SQL query:",
                    value=st.session_state.get('sql_query_input', default_query),
                    height=150,
                    key="sql_query_area",
                    help=f"Table name: {actual_table_name}"
                )
                
                # Store the query in session state
                st.session_state.sql_query_input = sql_query
                
                # Quick query buttons
                st.write("**Quick Queries:**")
                query_col1, query_col2, query_col3, query_col4 = st.columns(4)
                
                with query_col1:
                    if st.button("Show All", key="sql_show_all"):
                        sql_query = f"SELECT * FROM {actual_table_name}"
                        st.session_state.sql_query_input = sql_query
                        st.rerun()
                
                with query_col2:
                    if st.button("Count Rows", key="sql_count"):
                        sql_query = f"SELECT COUNT(*) as total_rows FROM {actual_table_name}"
                        st.session_state.sql_query_input = sql_query
                        st.rerun()
                
                with query_col3:
                    if st.button("Basic Stats", key="sql_stats"):
                        if numeric_cols:
                            col = numeric_cols[0]
                            sql_query = f"SELECT AVG(`{col}`) as avg_{col}, MAX(`{col}`) as max_{col}, MIN(`{col}`) as min_{col} FROM {actual_table_name}"
                        else:
                            sql_query = f"SELECT COUNT(*) as total_rows FROM {actual_table_name}"
                        st.session_state.sql_query_input = sql_query
                        st.rerun()
                
                with query_col4:
                    if st.button("Sample Data", key="sql_sample"):
                        sql_query = f"SELECT * FROM {actual_table_name} LIMIT 5"
                        st.session_state.sql_query_input = sql_query
                        st.rerun()
                
                # Clear query button
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Clear Query", key="clear_sql_query"):
                        st.session_state.sql_query_input = default_query
                        st.rerun()
                
                # Execute query button
                if st.button("Execute Query", key="execute_sql_query", type="primary"):
                    with st.spinner("Executing SQL query..."):
                        try:
                            # Enhanced pre-execution verification
                            st.info(f"Pre-execution check: Verifying database connection and tables...")
                            
                            # First, verify the engine is still valid
                            if 'sql_engine' not in st.session_state or st.session_state.sql_engine is None:
                                st.error("SQL engine is not available!")
                                st.info("Click 'Refresh Database' to recreate the connection.")
                                return
                            
                            engine = st.session_state.sql_engine
                            
                            # Test engine connectivity
                            try:
                                with engine.connect() as test_conn:
                                    test_conn.execute(text("SELECT 1"))
                                st.success("✅ Database connection verified")
                            except Exception as conn_e:
                                st.error(f"❌ Database connection failed: {str(conn_e)}")
                                st.info(" Click 'Refresh Database' to recreate the connection.")
                                return
                            
                            # Verify tables exist using the improved method
                            try:
                                tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
                                available_tables = pd.read_sql_query(tables_query, engine)
                                available_table_names = available_tables['name'].tolist()
                                
                                st.info(f"Found {len(available_table_names)} tables: {', '.join(['`' + t + '`' for t in available_table_names])}")
                                
                                # Debug: Show what tables actually exist
                                if len(available_table_names) == 0:
                                    st.error("❌ No tables found in database after connection verification!")
                                    st.error("🔴 **Critical Issue:** Database appears empty despite successful connection.")
                                    st.info("💡 This indicates the in-memory database was recreated. Click 'Refresh Database'.")
                                    return
                                
                                # If our expected table doesn't exist, use the first available table
                                if actual_table_name not in available_table_names:
                                    if available_table_names:
                                        # Update to use the actual existing table
                                        st.warning(f"⚠️ Expected table '{actual_table_name}' not found. Using '{available_table_names[0]}' instead.")
                                        actual_table_name = available_table_names[0]
                                        st.session_state.sql_table_name = actual_table_name
                                    else:
                                        st.error(f"❌ Table '{actual_table_name}' not found and no tables available!")
                                        st.write(f"Available tables: {', '.join(['`' + t + '`' for t in available_table_names]) if available_table_names else 'None'}")
                                        st.info("💡 Click 'Refresh Database' to recreate the table.")
                                        return
                            except Exception as table_check_error:
                                st.warning(f"Could not verify tables: {str(table_check_error)}")
                            
                            # Auto-correct common table name mistakes
                            corrected_query = sql_query
                            
                            # Add the actual table name to the list of corrections
                            common_mistakes = [
                                'dailyactivity_merged', 'dailysteps_merged', 'current_data', 
                                'data', 'dataset', 'table', 'df', 'dataframe', 'my_table'
                            ]
                            
                            # Also check for the unsanitized file name
                            current_file = st.session_state.get('current_file_tab', '')
                            if current_file and current_file != actual_table_name:
                                common_mistakes.append(current_file)
                            
                            query_lower = sql_query.lower()
                            correction_made = False
                            
                            for wrong_name in common_mistakes:
                                if wrong_name.lower() in query_lower and wrong_name.lower() != actual_table_name.lower():
                                    # Use word boundaries to avoid partial replacements
                                    pattern = r'\b' + re.escape(wrong_name) + r'\b'
                                    corrected_query = re.sub(pattern, actual_table_name, corrected_query, flags=re.IGNORECASE)
                                    if corrected_query != sql_query:
                                        correction_made = True
                                        st.info(f"🔧 Auto-corrected table name: `{wrong_name}` → `{actual_table_name}`")
                            
                            if correction_made:
                                st.code(corrected_query, language='sql')
                            
                            # Execute the query
                            result = execute_sql_query(st.session_state.sql_engine, corrected_query)
                            
                            # Handle the result
                            if result['success']:
                                if result['data'] is not None and not result['data'].empty:
                                    query_result = result['data']
                                    rows_returned = len(query_result)
                                    exec_time = result.get('execution_time', 0)
                                    
                                    st.success(f"✅ Query executed successfully! {rows_returned:,} rows returned in {exec_time:.3f}s")
                                    
                                    # Display the results
                                    st.dataframe(query_result, use_container_width=True)
                                    
                                    # Option to save results
                                    save_col1, save_col2 = st.columns([3, 1])
                                    with save_col2:
                                        if st.button("Save Result", key="save_sql_result"):
                                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                            result_name = f"sql_result_{timestamp}.csv"
                                            st.session_state.dfs[result_name] = query_result
                                            if result_name not in st.session_state.active_file_tabs:
                                                st.session_state.active_file_tabs.append(result_name)
                                            st.success(f"Saved as: {result_name}")
                                            time.sleep(1)
                                            st.rerun()
                                else:
                                    # Query succeeded but returned no data
                                    exec_time = result.get('execution_time', 0)
                                    rows_affected = result.get('rows_affected', 0)
                                    st.success(f"✅ Query executed successfully in {exec_time:.3f}s! {rows_affected} rows affected.")
                            else:
                                # Query failed
                                error_msg = result.get('error', 'Unknown error occurred')
                                st.error(f"❌ Query execution failed: {error_msg}")
                                
                                # Provide helpful error guidance
                                with st.expander(" SQL Help & Troubleshooting", expanded=True):
                                    if "no such table" in error_msg.lower():
                                        st.write("**Table Not Found:**")
                                        st.write(f"• ✅ Use this table name: `{actual_table_name}`")
                                        st.write(f"•  Try refreshing the database")
                                    elif "no such column" in error_msg.lower():
                                        st.write("**Column Not Found:**")
                                        st.write("Available columns:")
                                        for col in df.columns:
                                            st.write(f"• `{col}`")
                                    else:
                                        st.write("**Common SQL Issues:**")
                                        st.write("• Check table name spelling")
                                        st.write("• Verify column names (case-sensitive)")
                                        st.write("• Use backticks for columns with spaces: `` `column name` ``")
                                        st.write("• Use single quotes for string values: `'value'`")
                                    
                                    st.write("\n**Example Queries:**")
                                    st.code(f"SELECT * FROM {actual_table_name} LIMIT 5", language='sql')
                                    st.code(f"SELECT COUNT(*) FROM {actual_table_name}", language='sql')
                                    if numeric_cols:
                                        st.code(f"SELECT AVG(`{numeric_cols[0]}`) FROM {actual_table_name}", language='sql')
                        
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {str(e)}")
                            if LOGGING_ENABLED:
                                logging.error(f"SQL execution error: {e}", exc_info=True)
                
            except Exception as e:
                st.error(f"SQL interface error: {str(e)}")
                if LOGGING_ENABLED:
                    logging.error(f"SQL tab error: {e}", exc_info=True)
        
        with tab11:
            st.header("NLP Testing")
            st.markdown("*Natural Language Processing and Text Analysis Tools*")
            
            if df is None or df.empty:
                st.error("No data available for NLP testing")
            else:
                # Check if NLP testing module is available
                if NLP_TESTING_AVAILABLE:
                    try:
                        render_nlp_test_ui(process_natural_query_production)
                    except Exception as e:
                        st.error(f"NLP module error: {str(e)}")
                        st.info("Falling back to basic text analysis features")
                        render_basic_nlp_features(df)
                else:
                    st.warning("⚠️ Advanced NLP features require additional modules")
                    
                    with st.expander(" Enable Advanced NLP Features", expanded=True):
                        st.markdown("""
                        **Missing Dependencies:** Advanced NLP features require additional libraries.
                        
                        **Quick Installation:**
                        ```bash
                        pip install nltk spacy textblob
                        python -m spacy download en_core_web_sm
                        ```
                        
                        **After installation:**
                        1. Restart your Streamlit application
                        2. Return to this tab to access full NLP features
                        
                        **NLP Features Available After Install:**
                        - Text preprocessing and cleaning
                        - Sentiment analysis
                        - Named entity recognition
                        - Text classification
                        - Topic modeling
                        - Word cloud generation
                        """)
                    
                    # Basic NLP features without external dependencies
                    render_basic_nlp_features(df)
        
        with tab12:
            st.header("Settings")
            st.markdown("*Application settings and configuration*")
            
            # Application settings
            settings_col1, settings_col2 = st.columns(2)
            
            with settings_col1:
                st.subheader("Display Settings")
                
                # Max rows to display
                max_display_rows = st.number_input(
                    "Max rows to display:",
                    min_value=10,
                    max_value=10000,
                    value=1000,
                    step=100,
                    key="max_display_rows"
                )
                
                # Decimal places
                decimal_places = st.number_input(
                    "Decimal places for numbers:",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    key="decimal_places"
                )
                
                # Dark mode toggle
                dark_mode = st.checkbox("Enable Dark Mode", key="dark_mode_setting")
                
            with settings_col2:
                st.subheader("Performance Settings")
                
                # Memory usage warnings
                memory_warnings = st.checkbox("Show memory usage warnings", value=True, key="memory_warnings")
                
                # Auto-save settings
                auto_save = st.checkbox("Auto-save query results", value=False, key="auto_save_results")
                
                # Debug mode
                #debug_mode = st.checkbox("Enable debug mode", value=False, key="debug_mode")
            
            # Export/Import settings
            st.subheader("Data Management")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.write("**Export Current Session**")
                if st.button("Export All Data", key="export_session_data"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # This would export all dataframes in session
                    st.success("Export functionality will be implemented")
            
            with export_col2:
                st.write("**Clear Session Data**")
                if st.button("Clear All Data", key="clear_session_data", help="This will clear all loaded datasets"):
                    if st.checkbox("I understand this will clear all data", key="confirm_clear"):
                        # Clear session state
                        for key in list(st.session_state.keys()):
                            if key.startswith('dfs') or key in ['active_file_tabs', 'current_file_tab']:
                                del st.session_state[key]
                        st.success("Session data cleared")
                        st.rerun()
            
            # Application info
            st.subheader("Application Information")
            st.info("""
            **Enterprise Data Analytics Platform**
            - Advanced statistical operations
            - PowerBI-style visualizations
            - Multi-file workflow management
            - SQL query interface
            - AI-powered insights
            
            Alex Alagoa Biobelemo
            """)
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"Application error: {error_msg}")
        
        # Provide specific guidance for common errors
        if error_msg == "'type'" or "'type'" in error_msg:
            st.error("🔴 **Type Error Detected**: This appears to be a variable naming conflict.")
            st.info("💡 **Debugging Information**: The error suggests a conflict with the built-in 'type' function.")
            
            with st.expander("🔧 Troubleshooting Steps", expanded=True):
                st.write("""
                **Possible Solutions:**
                1.  Refresh the page to reset the application state
                2.  Clear your browser cache and cookies
                3.  Check if any column names in your data might be causing conflicts
                4.  Try uploading a different dataset to isolate the issue
                
                **If the error persists:**
                - The issue might be with the specific dataset you're using
                - Try using a simpler dataset first to test functionality
                - Check that column names don't use Python reserved words
                """)
        
        if LOGGING_ENABLED:
            logging.error(f"Main app error: {e}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error details: {error_msg}")

def main():
    """
    Main function to run the Enterprise Data Analytics Platform
    """
    main_production()

if __name__ == "__main__":
    main()



