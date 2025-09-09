import streamlit as st
import pandas as pd
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
    st.warning("⚠️ Machine Learning features disabled: scikit-learn not available")

warnings.filterwarnings('ignore')

# --- Production Configuration ---
st.set_page_config(
    page_title="Enterprise Data Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Enterprise Data Analytics Platform"
    }
)

# --- Enhanced Constants ---
FILE_TYPES: List[str] = ["CSV", "Excel", "JSON"]
CHART_OPTIONS: List[str] = [
    "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot",
    "Correlation Heatmap", "Pie Chart", "Violin Plot", "Map View", "Anomaly Plot"
]
AGG_OPTIONS: List[str] = ['mean', 'sum', 'median', 'count', 'min', 'max', 'std', 'var']
THEME_OPTIONS: List[str] = ["plotly", "plotly_dark", "seaborn", "ggplot2", "simple_white", "presentation"]
ACCESSIBILITY_THEMES: List[str] = ["Light", "Dark", "High Contrast", "Colorblind Friendly"]
COLOR_PALETTES: List[str] = ["Viridis", "Plasma", "Inferno", "Magma", "Turbo", "Cividis", "Blues", "Greens"]
ML_MODELS: List[str] = ["RandomForest", "MLP", "IsolationForest"] if SKLEARN_AVAILABLE else []
CLEANING_METHODS: List[str] = ["mean", "median", "mode", "drop", "forward_fill", "backward_fill", "interpolate"]
ANOMALY_METHODS: List[str] = ["IsolationForest", "Z-Score", "Modified Z-Score", "IQR"] if SKLEARN_AVAILABLE else ["Z-Score", "Modified Z-Score", "IQR"]

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
            }
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
                result["feature_importance"] = dict(zip(available_x_cols, model.feature_importances_))
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
        else:
            return False, f"Unsupported cleaning operation: {suggestion['type']}"
        
        # Validation check
        if df is None or df.empty:
            return False, "Cleaning operation resulted in empty dataset"
        
        # Apply changes
        st.session_state.selected_df = df
        
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

# --- Enhanced Natural Language Processing ---
def process_natural_query_production(query: str) -> Dict[str, Any]:
    """Enhanced natural language query processing with better pattern matching."""
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
                            response["message"] += "\n⚠️ No rows match the filter criteria."
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
        
        if theme == "High Contrast":
            st.markdown("""
                <style>
                .stApp { 
                    background-color: #000000; 
                    color: #FFFFFF; 
                }
                .stSelectbox > div > div { 
                    background-color: #000000; 
                    color: #FFFFFF; 
                    border: 2px solid #FFFFFF;
                }
                .stMetric { 
                    background: linear-gradient(90deg, #FFFFFF 0%, #CCCCCC 100%);
                    color: #000000;
                    padding: 1rem; 
                    border-radius: 10px; 
                    border: 2px solid #FFFFFF;
                }
                .stButton > button {
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 2px solid #FFFFFF;
                }
                </style>
            """, unsafe_allow_html=True)
            
        elif theme == "Colorblind Friendly":
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
                st.sidebar.warning(f"⚠️ High memory usage: {memory_info['current_mb']:.0f}MB")
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
                    st.error("🚨 Critical memory usage!")
                elif memory_info['warning']:
                    st.warning("⚠️ High memory usage")
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
def main_production():
    """Production-grade main application with comprehensive error handling and monitoring."""
    try:
        # Initialize session state
        init_session_state()
        
        # Create production layout
        create_production_layout()
        
        # Header with enhanced controls
        header_col1, header_col2, header_col3, header_col4 = st.columns([3, 1, 1, 1])
        
        with header_col1:
            st.title(" Enterprise Data Analytics Platform - Production")
            st.markdown("*AI-Powered • Production-Ready • Performance-Optimized*")
        
        with header_col2:
            theme_choice = st.selectbox(
                "Theme",
                ACCESSIBILITY_THEMES,
                index=ACCESSIBILITY_THEMES.index(st.session_state.theme_preference),
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
            file_type = st.selectbox("File Type", FILE_TYPES, help="Select your data format")
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
                        df = load_data_production(file_content, uploaded_file.name, file_type)
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
        
        # Dataset selection with enhanced information
        st.subheader("Dataset Selection & Overview")
        
        selected_file = st.selectbox(
            "Choose dataset for analysis:",
            list(st.session_state.dfs.keys()),
            help="Select which dataset to analyze with enterprise features"
        )
        
        if selected_file not in st.session_state.dfs:
            st.error("Selected dataset not available")
            return
        
        df = st.session_state.dfs[selected_file]
        st.session_state.selected_df = df
        
        if df is None or df.empty:
            st.error("The selected dataset is empty.")
            return
        
        st.session_state.data_loaded = True
        df_hash = str(hash(str(df.shape) + str(df.columns.tolist())))
        
        # Enhanced dataset overview
        overview_col1, overview_col2, overview_col3, overview_col4, overview_col5, overview_col6 = st.columns(6)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        with overview_col1:
            st.metric("Rows", f"{len(df):,}")
        with overview_col2:
            st.metric("Columns", f"{len(df.columns)}")
        with overview_col3:
            st.metric("Numeric", len(numeric_cols))
        with overview_col4:
            st.metric("Categorical", len(categorical_cols))
        with overview_col5:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory", f"{memory_mb:.1f}MB")
        with overview_col6:
            quality_score = df.attrs.get('quality_score', 0)
            quality_color = "🟢" if quality_score > 80 else "🟡" if quality_score > 60 else "🔴"
            st.metric("Quality", f"{quality_score:.0f}/100", delta=quality_color)
        
        # Main application tabs with enhanced features
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "AI Assistant",
            "Analytics",
            "Data Explorer",
            "Data Cleaning",
            "Anomaly Detection",
            "Visualizations",
            "ML Studio",
            "⚙Settings"
        ])
        
        with tab1:
            st.header("Conversational AI Assistant")
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
            st.header("Enterprise Data Analytics")
            
            try:
                # Compute enhanced EDA
                eda_data = compute_eda_summary_enhanced(df_hash, df.shape)
                
                if 'error' in eda_data:
                    st.error(f"Analytics computation failed: {eda_data['error']}")
                else:
                    # Key insights section
                    st.subheader("Key Data Insights")
                    insight_col1, insight_col2 = st.columns([2, 1])
                    
                    with insight_col1:
                        for insight in eda_data.get('insights', ['No insights available'])[:4]:
                            st.info(f"{insight}")
                        
                        # Recommendations
                        if eda_data.get('recommendations'):
                            st.subheader("AI Recommendations")
                            for rec in eda_data.get('recommendations', [])[:3]:
                                st.success(f"{rec}")
                    
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
                    logging.error(f"Analytics tab error: {e}")
        
        with tab3:
            st.header("Data Explorer")
            st.markdown("*Interactive data exploration with advanced filtering*")
            
            # Enhanced data preview with filters
            preview_col1, preview_col2 = st.columns([3, 1])
            
            with preview_col2:
                # Quick filters
                st.subheader("Quick Filters")
                
                # Numeric range filters
                if numeric_cols:
                    selected_numeric = st.selectbox("Select numeric column:", ["None"] + numeric_cols)
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
                    selected_categorical = st.selectbox("Select categorical column:", ["None"] + categorical_cols)
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
            
            with preview_col1:
                st.subheader("Data Sample")
                
                # Display options
                display_col1, display_col2, display_col3 = st.columns(3)
                with display_col1:
                    sample_size = st.number_input("Sample size:", 10, min(1000, len(df)), 50)
                with display_col2:
                    sort_column = st.selectbox("Sort by:", ["None"] + list(df.columns))
                with display_col3:
                    ascending = st.checkbox("Ascending", value=True)
                
                # Apply sorting and sampling
                display_df = df.copy()
                if sort_column != "None":
                    display_df = display_df.sort_values(sort_column, ascending=ascending)
                
                if len(display_df) > sample_size:
                    display_df = display_df.head(sample_size)
                
                # Enhanced dataframe display
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
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
        
        with tab4:
            st.header("Data Cleaning Studio")
            st.markdown("*AI-powered data cleaning with manual overrides*")
            
            if st.session_state.data_loaded:
                # Generate cleaning suggestions
                with st.spinner("Analyzing data quality..."):
                    cleaning_suggestions = suggest_cleaning_production(df_hash)
                
                if not cleaning_suggestions:
                    st.success("No immediate cleaning suggestions. Your data looks good!")
                else:
                    st.subheader("AI Cleaning Suggestions")
                    
                    # Create suggestions dataframe
                    suggestions_df = pd.DataFrame([
                        {
                            'Priority': f"{'🔴' if s['severity'] == 'high' else '🟡' if s['severity'] == 'medium' else '🟢'}",
                            'Column': s['column'],
                            'Issue': s['description'],
                            'Action': s['type'].replace('_', ' ').title(),
                            'Impact': s['impact'].title(),
                            'Confidence': s['confidence'].title()
                        } for s in cleaning_suggestions
                    ])
                    
                    st.dataframe(
                        suggestions_df,
                        use_container_width=True,
                        column_config={
                            'Priority': st.column_config.TextColumn('Priority', width="small"),
                            'Impact': st.column_config.TextColumn('Impact', width="small"),
                            'Confidence': st.column_config.TextColumn('Confidence', width="small")
                        }
                    )
                    
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
                                    st.success(f"{message}")
                                    st.session_state.dfs[selected_file] = st.session_state.selected_df
                                    time.sleep(1)  # Brief pause for user feedback
                                    st.rerun()
                                else:
                                    st.error(f"{message}")
                    
                    with action_col2:
                        if st.button("Apply High Priority", key="apply_high_priority"):
                            high_priority_suggestions = [s for s in cleaning_suggestions if s['severity'] == 'high']
                            
                            if not high_priority_suggestions:
                                st.info("No high priority suggestions to apply.")
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
                                    st.success(f"Applied {success_count} cleaning operations successfully!")
                                    st.session_state.dfs[selected_file] = st.session_state.selected_df
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
                                
                                st.session_state.selected_df = df_cleaned
                                st.session_state.dfs[selected_file] = df_cleaned
                                st.success(success_msg)
                                time.sleep(1)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Manual cleaning failed: {str(e)}")
            else:
                st.info("Please load a dataset to access cleaning features.")
        
        with tab5:
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
                    if st.button("Detect Anomalies", key="detect_anomalies_prod", type="primary") and selected_features:
                        with st.spinner(f"Running {anomaly_method} detection..."):
                            anomaly_result = detect_anomalies_production(df_hash, anomaly_method, params, selected_features)
                            st.session_state.anomaly_results = anomaly_result
                
                # Display results
                if st.session_state.anomaly_results and "error" not in st.session_state.anomaly_results:
                    result = st.session_state.anomaly_results
                    
                    # Summary metrics
                    st.subheader("Detection Summary")
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
                        st.subheader("Anomaly Visualization")
                        
                        viz_col1, viz_col2 = st.columns([3, 1])
                        
                        with viz_col2:
                            x_feature = st.selectbox("X-axis:", result['columns'], key="anomaly_viz_x")
                            y_feature = st.selectbox("Y-axis:", result['columns'], index=1 if len(result['columns']) > 1 else 0, key="anomaly_viz_y")
                            
                            color_by_score = st.checkbox("Color by Score", value=True, key="color_by_score")
                            show_normal = st.checkbox("Show Normal Points", value=True, key="show_normal_points")
                        
                        with viz_col1:
                            try:
                                # Prepare visualization data
                                viz_df = st.session_state.selected_df.loc[result['index']].copy()
                                viz_df['is_anomaly'] = result['outliers'] == -1
                                viz_df['anomaly_score'] = result['anomaly_scores']
                                
                                # Filter data if requested
                                if not show_normal:
                                    viz_df = viz_df[viz_df['is_anomaly']]
                                
                                if len(viz_df) > 0:
                                    # Create scatter plot
                                    if color_by_score:
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
                                        st.info(f"Found {result['outlier_count']} anomalies out of {len(result['outliers']):,} data points{result.get('sample_note', '')}")
                                else:
                                    st.warning("No data points to visualize.")
                                    
                            except Exception as e:
                                st.error(f"Visualization error: {str(e)}")
                                if LOGGING_ENABLED:
                                    logging.error(f"Anomaly visualization error: {e}")
                    
                    # Anomaly details table
                    if result['outlier_count'] > 0:
                        with st.expander("Anomaly Details", expanded=False):
                            try:
                                anomaly_details = st.session_state.selected_df.loc[result['index']].copy()
                                anomaly_details['anomaly_score'] = result['anomaly_scores']
                                anomaly_details['is_anomaly'] = result['outliers'] == -1
                                
                                # Show only anomalies
                                anomaly_data = anomaly_details[anomaly_details['is_anomaly']].sort_values('anomaly_score', ascending=False)
                                
                                st.dataframe(
                                    anomaly_data.head(50),  # Show top 50 anomalies
                                    use_container_width=True
                                )
                                
                                # Export options
                                if st.button("Export Anomalies", key="export_anomalies"):
                                    csv_data = anomaly_data.to_csv(index=False)
                                    st.download_button(
                                        "Download Anomalies CSV",
                                        csv_data,
                                        f"anomalies_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv"
                                    )
                                    
                            except Exception as e:
                                st.error(f"Error displaying anomaly details: {str(e)}")
                
                elif st.session_state.anomaly_results and "error" in st.session_state.anomaly_results:
                    st.error(f"{st.session_state.anomaly_results['error']}")
        
        with tab6:
            st.header("Interactive Visualization Studio")
            st.markdown("*Create publication-ready visualizations with AI assistance*")
            
            # Sidebar controls for visualizations
            with st.sidebar:
                st.subheader("Visualization Controls")
                
                # AI chart suggestions
                if len(numeric_cols) > 0 or len(categorical_cols) > 0:
                    if len(numeric_cols) >= 2:
                        suggestions = ["Scatter Plot", "Correlation Heatmap", "Line Chart"]
                    elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                        suggestions = ["Bar Chart", "Box Plot", "Violin Plot"]
                    elif len(numeric_cols) >= 1:
                        suggestions = ["Histogram"]
                    else:
                        suggestions = ["Bar Chart"]
                    
                    st.info(f"AI Suggests: {', '.join(suggestions[:2])}")
                
                # Chart management
                chart_control_col1, chart_control_col2 = st.columns(2)
                
                with chart_control_col1:
                    if st.button("Add Chart", key="add_chart_viz"):
                        new_chart = {
                            "chart_type": suggestions[0] if 'suggestions' in locals() else "Bar Chart",
                            "id": len(st.session_state.chart_configs),
                            "title": f"Chart {len(st.session_state.chart_configs) + 1}"
                        }
                        st.session_state.chart_configs.append(new_chart)
                        st.rerun()
                
                with chart_control_col2:
                    if st.session_state.chart_configs and st.button("🗑️ Clear All", key="clear_charts_viz"):
                        st.session_state.chart_configs = []
                        st.rerun()
            
            # Chart creation interface
            if not st.session_state.chart_configs:
                st.info("Click 'Add Chart' in the sidebar to create your first visualization")
                
                # Quick start options
                st.subheader("Quick Start")
                quick_col1, quick_col2, quick_col3 = st.columns(3)
                
                quick_charts = [
                    ("Data Overview", "Correlation Heatmap"),
                    ("Distribution", "Histogram"),
                    ("Relationships", "Scatter Plot")
                ]
                
                for i, (label, chart_type) in enumerate(quick_charts):
                    with [quick_col1, quick_col2, quick_col3][i]:
                        if st.button(label, key=f"quick_chart_{i}"):
                            new_chart = {
                                "chart_type": chart_type,
                                "id": 0,
                                "title": f"{chart_type} - {selected_file}"
                            }
                            st.session_state.chart_configs = [new_chart]
                            st.rerun()
            
            else:
                # Multiple chart tabs
                chart_tabs = st.tabs([f"{config.get('title', f'Chart {i+1}')}" for i, config in enumerate(st.session_state.chart_configs)])
                
                for tab_idx, chart_tab in enumerate(chart_tabs):
                    with chart_tab:
                        if tab_idx < len(st.session_state.chart_configs):
                            config = st.session_state.chart_configs[tab_idx]
                            
                            # Chart configuration
                            st.subheader("Chart Configuration")
                            config_col1, config_col2, config_col3, config_col4 = st.columns(4)
                            
                            with config_col1:
                                chart_type = st.selectbox(
                                    "Chart Type:",
                                    CHART_OPTIONS,
                                    index=CHART_OPTIONS.index(config["chart_type"]) if config["chart_type"] in CHART_OPTIONS else 0,
                                    key=f"chart_type_{tab_idx}"
                                )
                                st.session_state.chart_configs[tab_idx]["chart_type"] = chart_type
                            
                            with config_col2:
                                # Smart column suggestions based on chart type
                                if chart_type in ["Scatter Plot", "Line Chart"]:
                                    x_options = numeric_cols + datetime_cols
                                    y_options = numeric_cols
                                elif chart_type == "Bar Chart":
                                    x_options = categorical_cols
                                    y_options = numeric_cols
                                elif chart_type == "Histogram":
                                    x_options = numeric_cols
                                    y_options = []
                                elif chart_type in ["Box Plot", "Violin Plot"]:
                                    x_options = categorical_cols + ["None"]
                                    y_options = numeric_cols
                                elif chart_type == "Pie Chart":
                                    x_options = categorical_cols
                                    y_options = numeric_cols
                                else:
                                    x_options = list(df.columns)
                                    y_options = numeric_cols
                                
                                x_axis = st.selectbox(
                                    "X-axis:",
                                    x_options if x_options else ["No suitable columns"],
                                    key=f"x_axis_{tab_idx}"
                                ) if x_options else None
                            
                            with config_col3:
                                y_axis = st.selectbox(
                                    "Y-axis:",
                                    y_options if y_options else ["No suitable columns"],
                                    key=f"y_axis_{tab_idx}"
                                ) if y_options else None
                            
                            with config_col4:
                                color_col = st.selectbox(
                                    "Color By:",
                                    ["None"] + categorical_cols,
                                    key=f"color_{tab_idx}"
                                ) if categorical_cols else None
                                if color_col == "None":
                                    color_col = None
                            
                            # Advanced options
                            with st.expander("Advanced Options", expanded=False):
                                adv_col1, adv_col2, adv_col3 = st.columns(3)
                                
                                with adv_col1:
                                    custom_title = st.text_input(
                                        "Title:",
                                        config.get('title', f"{chart_type} - {selected_file}"),
                                        key=f"title_{tab_idx}"
                                    )
                                    theme = st.selectbox("Theme:", THEME_OPTIONS, key=f"theme_{tab_idx}")
                                
                                with adv_col2:
                                    height = st.slider("Height (px):", 300, 1000, 600, key=f"height_{tab_idx}")
                                    width = st.slider("Width (%):", 50, 100, 100, key=f"width_{tab_idx}")
                                
                                with adv_col3:
                                    show_legend = st.checkbox("Show Legend", True, key=f"legend_{tab_idx}")
                                    show_grid = st.checkbox("Show Grid", True, key=f"grid_{tab_idx}")
                            
                            # Chart generation
                            fig = None
                            error_message = None
                            
                            try:
                                # Sample data for performance
                                plot_df = df.sample(n=min(10000, len(df)), random_state=42) if len(df) > 10000 else df
                                
                                if chart_type == "Correlation Heatmap":
                                    if len(numeric_cols) >= 2:
                                        corr_matrix = plot_df[numeric_cols[:10]].corr()  # Limit to 10 columns
                                        fig = px.imshow(
                                            corr_matrix,
                                            text_auto=True,
                                            aspect="auto",
                                            color_continuous_scale="RdBu_r",
                                            title=custom_title,
                                            height=height
                                        )
                                    else:
                                        error_message = "Need at least 2 numeric columns for correlation heatmap"
                                
                                elif chart_type == "Histogram" and x_axis:
                                    fig = px.histogram(
                                        plot_df,
                                        x=x_axis,
                                        color=color_col,
                                        title=custom_title,
                                        template=theme,
                                        height=height
                                    )
                                
                                elif chart_type == "Scatter Plot" and x_axis and y_axis:
                                    fig = px.scatter(
                                        plot_df,
                                        x=x_axis,
                                        y=y_axis,
                                        color=color_col,
                                        title=custom_title,
                                        template=theme,
                                        height=height
                                    )
                                
                                elif chart_type == "Line Chart" and x_axis and y_axis:
                                    fig = px.line(
                                        plot_df,
                                        x=x_axis,
                                        y=y_axis,
                                        color=color_col,
                                        title=custom_title,
                                        template=theme,
                                        height=height
                                    )
                                
                                elif chart_type == "Bar Chart" and x_axis and y_axis:
                                    if color_col:
                                        agg_df = plot_df.groupby([x_axis, color_col])[y_axis].mean().reset_index()
                                    else:
                                        agg_df = plot_df.groupby(x_axis)[y_axis].mean().reset_index()
                                    
                                    fig = px.bar(
                                        agg_df,
                                        x=x_axis,
                                        y=y_axis,
                                        color=color_col,
                                        title=custom_title,
                                        template=theme,
                                        height=height
                                    )
                                
                                elif chart_type == "Box Plot" and y_axis:
                                    fig = px.box(
                                        plot_df,
                                        x=x_axis if x_axis != "None" else None,
                                        y=y_axis,
                                        color=color_col,
                                        title=custom_title,
                                        template=theme,
                                        height=height
                                    )
                                
                                elif chart_type == "Violin Plot" and y_axis:
                                    fig = px.violin(
                                        plot_df,
                                        x=x_axis if x_axis != "None" else None,
                                        y=y_axis,
                                        color=color_col,
                                        title=custom_title,
                                        template=theme,
                                        height=height
                                    )
                                
                                elif chart_type == "Pie Chart" and x_axis and y_axis:
                                    agg_df = plot_df.groupby(x_axis)[y_axis].sum().reset_index()
                                    fig = px.pie(
                                        agg_df,
                                        names=x_axis,
                                        values=y_axis,
                                        title=custom_title,
                                        template=theme,
                                        height=height
                                    )
                                
                                else:
                                    error_message = "Please select appropriate columns for this chart type"
                                
                                # Apply styling
                                if fig:
                                    fig.update_layout(
                                        showlegend=show_legend,
                                        xaxis_showgrid=show_grid,
                                        yaxis_showgrid=show_grid,
                                        title_x=0.5  # Center title
                                    )
                                    
                                    # Update config title
                                    st.session_state.chart_configs[tab_idx]["title"] = custom_title
                                
                            except Exception as e:
                                error_message = f"Chart creation failed: {str(e)}"
                                if LOGGING_ENABLED:
                                    logging.error(f"Chart creation error: {error_message}")
                            
                            # Display chart or error
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{tab_idx}")
                                
                                # Chart export options
                                export_col1, export_col2 = st.columns(2)
                                with export_col1:
                                    if st.button(f"Export PNG", key=f"export_png_{tab_idx}"):
                                        st.info("Right-click the chart and select 'Download plot as png'")
                                with export_col2:
                                    if st.button(f"Export Data", key=f"export_data_{tab_idx}"):
                                        chart_data = plot_df.to_csv(index=False)
                                        st.download_button(
                                            "Download Chart Data",
                                            chart_data,
                                            f"chart_data_{tab_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            "text/csv"
                                        )
                            
                            elif error_message:
                                st.error(f"{error_message}")
                            else:
                                st.info("Configure your chart using the options above")
        
        with tab7:
            st.header("Machine Learning Studio")
            
            if not SKLEARN_AVAILABLE:
                st.warning("⚠️ Machine Learning features are not available. Please install scikit-learn to use ML capabilities.")
                st.code("pip install scikit-learn", language="bash")
                return
            
            st.markdown("*Advanced machine learning with automated model comparison*")
            
            # Check for suitable data
            if not (numeric_cols or categorical_cols):
                st.warning("⚠️ No suitable features found for machine learning")
                return
            
            ml_col1, ml_col2 = st.columns([1, 1])
            
            # Clustering Section
            with ml_col1:
                with st.container():
                    st.subheader("Advanced Clustering")
                    
                    if not numeric_cols:
                        st.warning("⚠️ No numeric columns available for clustering")
                    else:
                        cluster_col1, cluster_col2 = st.columns(2)
                        
                        with cluster_col1:
                            clustering_features = st.multiselect(
                                "Select Features:",
                                numeric_cols,
                                default=numeric_cols[:min(4, len(numeric_cols))],
                                key="clustering_features_ml"
                            )
                            
                            auto_k = st.checkbox("Auto-optimize clusters", value=True, key="auto_k_ml")
                        
                        with cluster_col2:
                            if auto_k:
                                k_clusters = st.slider("Max clusters to test:", 2, 10, 8, key="max_k_ml")
                            else:
                                k_clusters = st.slider("Number of clusters:", 2, 10, 3, key="manual_k_ml")
                            
                            preprocessing = st.checkbox("Standardize features", value=True, key="cluster_preprocess")
                        
                        if st.button("Run Clustering Analysis", key="run_clustering_ml", type="primary") and clustering_features:
                            with st.spinner("Performing clustering analysis..."):
                                try:
                                    # Prepare data
                                    cluster_data = df[clustering_features].dropna()
                                    
                                    if len(cluster_data) < 10:
                                        st.error("Not enough data points for clustering")
                                    else:
                                        # Sample for performance
                                        if len(cluster_data) > SAMPLE_SIZE_LARGE:
                                            cluster_data = cluster_data.sample(n=SAMPLE_SIZE_LARGE, random_state=42)
                                            st.info(f"Using sample of {len(cluster_data):,} rows for performance")
                                        
                                        if auto_k:
                                            # Find optimal k using silhouette score
                                            best_k = 3
                                            best_score = -1
                                            inertias = []
                                            silhouette_scores = []
                                            
                                            if preprocessing:
                                                scaler = StandardScaler()
                                                cluster_data_scaled = scaler.fit_transform(cluster_data)
                                            else:
                                                cluster_data_scaled = cluster_data.values
                                            
                                            k_range = range(2, min(k_clusters + 1, len(cluster_data) // 50))
                                            
                                            for k in k_range:
                                                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                                labels = kmeans.fit_predict(cluster_data_scaled)
                                                
                                                inertia = kmeans.inertia_
                                                inertias.append(inertia)
                                                
                                                if len(set(labels)) > 1:
                                                    sil_score = silhouette_score(cluster_data_scaled, labels)
                                                    silhouette_scores.append(sil_score)
                                                    
                                                    if sil_score > best_score:
                                                        best_score = sil_score
                                                        best_k = k
                                                else:
                                                    silhouette_scores.append(0)
                                            
                                            # Final clustering with optimal k
                                            final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                                            final_labels = final_kmeans.fit_predict(cluster_data_scaled)
                                            
                                            # Results display
                                            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                                            with result_col1:
                                                st.metric("Optimal Clusters", best_k)
                                            with result_col2:
                                                st.metric("Silhouette Score", f"{best_score:.3f}")
                                            with result_col3:
                                                st.metric("Inertia", f"{final_kmeans.inertia_:.0f}")
                                            with result_col4:
                                                st.metric("Data Points", f"{len(cluster_data):,}")
                                            
                                            # Visualization
                                            if len(clustering_features) >= 2:
                                                viz_df = cluster_data.copy()
                                                viz_df['Cluster'] = final_labels
                                                
                                                fig_cluster = px.scatter(
                                                    viz_df,
                                                    x=clustering_features[0],
                                                    y=clustering_features[1],
                                                    color='Cluster',
                                                    title=f"K-Means Clustering (k={best_k})",
                                                    height=400
                                                )
                                                
                                                st.plotly_chart(fig_cluster, use_container_width=True)
                                            
                                            # Cluster summary
                                            cluster_summary = pd.DataFrame({
                                                'Cluster': range(best_k),
                                                'Size': [np.sum(final_labels == i) for i in range(best_k)],
                                                'Percentage': [np.sum(final_labels == i) / len(final_labels) * 100 for i in range(best_k)]
                                            })
                                            
                                            st.subheader("Cluster Summary")
                                            st.dataframe(cluster_summary, use_container_width=True)
                                        
                                        else:
                                            # Simple clustering with specified k
                                            if preprocessing:
                                                scaler = StandardScaler()
                                                cluster_data_scaled = scaler.fit_transform(cluster_data)
                                            else:
                                                cluster_data_scaled = cluster_data.values
                                            
                                            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                                            labels = kmeans.fit_predict(cluster_data_scaled)
                                            
                                            # Simple results
                                            if len(clustering_features) >= 2:
                                                viz_df = cluster_data.copy()
                                                viz_df['Cluster'] = labels
                                                
                                                fig_cluster = px.scatter(
                                                    viz_df,
                                                    x=clustering_features[0],
                                                    y=clustering_features[1],
                                                    color='Cluster',
                                                    title=f"🔗 K-Means Clustering (k={k_clusters})",
                                                    height=400
                                                )
                                                
                                                st.plotly_chart(fig_cluster, use_container_width=True)
                                            
                                            st.success(f"Clustering completed with {k_clusters} clusters")
                                        
                                except Exception as e:
                                    st.error(f"Clustering failed: {str(e)}")
                                    if LOGGING_ENABLED:
                                        logging.error(f"Clustering error: {e}")
            
            # Supervised Learning Section
            with ml_col2:
                with st.container():
                    st.subheader("Supervised Learning")
                    
                    # Model configuration
                    model_col1, model_col2 = st.columns(2)
                    
                    with model_col1:
                        available_features = numeric_cols + categorical_cols
                        x_features = st.multiselect(
                            "Features (X):",
                            available_features,
                            default=numeric_cols[:min(4, len(numeric_cols))],
                            key="ml_x_features"
                        )
                        
                        algorithm = st.selectbox(
                            "Algorithm:",
                            ["RandomForest", "MLP", "AutoML Comparison"],
                            key="ml_algorithm"
                        )
                    
                    with model_col2:
                        # Target selection
                        target_options = [col for col in df.columns if col not in x_features]
                        y_target = st.selectbox("Target (Y):", target_options, key="ml_y_target")
                        
                        # Task type detection/selection
                        if y_target:
                            unique_vals = df[y_target].nunique()
                            if df[y_target].dtype in ['object', 'category'] or unique_vals <= 10:
                                default_task = "classification"
                            else:
                                default_task = "regression"
                            
                            task_type = st.selectbox(
                                "Task Type:",
                                ["regression", "classification"],
                                index=0 if default_task == "regression" else 1,
                                key="ml_task_type"
                            )
                        else:
                            task_type = "regression"
                    
                    # Advanced options
                    with st.expander("Advanced Options", expanded=False):
                        adv_col1, adv_col2, adv_col3 = st.columns(3)
                        
                        with adv_col1:
                            enable_preprocessing = st.checkbox("Auto preprocessing", value=True, key="ml_preprocessing")
                            enable_tuning = st.checkbox("Hyperparameter tuning", value=False, key="ml_tuning")
                        
                        with adv_col2:
                            if algorithm == "MLP":
                                layer1_size = st.slider("Layer 1 neurons:", 10, 200, 50, key="mlp_layer1")
                                layer2_size = st.slider("Layer 2 neurons:", 10, 200, 50, key="mlp_layer2")
                                max_iter = st.slider("Max iterations:", 100, 1000, 200, key="mlp_max_iter")
                            
                            elif algorithm == "RandomForest":
                                n_estimators = st.slider("Number of trees:", 10, 500, 100, key="rf_n_estimators")
                                max_depth = st.selectbox("Max depth:", [None, 10, 20, 50], key="rf_max_depth")
                        
                        with adv_col3:
                            test_size = st.slider("Test size (%):", 10, 40, 20, key="ml_test_size")
                            random_state = st.number_input("Random seed:", value=42, key="ml_random_state")
                    
                    # Train model
                    if st.button("Train Model", key="train_ml_model", type="primary") and x_features and y_target:
                        with st.spinner(f"Training {algorithm} model..."):
                            try:
                                # Prepare parameters
                                model_params = {}
                                
                                if algorithm == "MLP":
                                    model_params = {
                                        'hidden_layer_sizes': (layer1_size, layer2_size) if 'layer1_size' in locals() else (50, 50),
                                        'max_iter': max_iter if 'max_iter' in locals() else 200
                                    }
                                elif algorithm == "RandomForest":
                                    model_params = {
                                        'n_estimators': n_estimators if 'n_estimators' in locals() else 100,
                                        'max_depth': max_depth if 'max_depth' in locals() else None
                                    }
                                
                                # Train model
                                if algorithm == "AutoML Comparison":
                                    # Compare multiple models
                                    with st.status("Training multiple models...", expanded=True) as status:
                                        results = {}
                                        models_to_compare = ["RandomForest", "MLP"]
                                        
                                        for model_name in models_to_compare:
                                            st.write(f"Training {model_name}...")
                                            result = train_ml_model_production(
                                                df, x_features, y_target, task_type, model_name,
                                                **model_params
                                            )
                                            results[model_name] = result
                                        
                                        status.update(label="Model comparison completed!", state="complete")
                                    
                                    # Display comparison results
                                    if results:
                                        st.subheader("Model Comparison Results")
                                        
                                        comparison_data = []
                                        for model_name, result in results.items():
                                            if "error" not in result:
                                                metrics = result["metrics"]
                                                key_metric = "r2" if task_type == "regression" else "f1"
                                                
                                                comparison_data.append({
                                                    "Model": model_name,
                                                    "Key Metric": f"{metrics.get(key_metric, 0):.4f}",
                                                    "Training Time": f"{result.get('training_time', 0):.2f}s",
                                                    "Features": result.get('n_features', 0),
                                                    "Samples": result.get('n_samples', 0)
                                                })
                                        
                                        if comparison_data:
                                            comparison_df = pd.DataFrame(comparison_data)
                                            st.dataframe(comparison_df, use_container_width=True)
                                            
                                            # Best model highlight
                                            best_model = comparison_df.loc[comparison_df["Key Metric"].astype(float).idxmax(), "Model"]
                                            st.success(f"Best model: **{best_model}**")
                                            
                                            # Detailed metrics for best model
                                            best_result = results[best_model]
                                            if "error" not in best_result:
                                                st.subheader(f"{best_model} Detailed Metrics")
                                                metrics_df = pd.DataFrame([
                                                    {"Metric": k.title(), "Value": f"{v:.4f}" if isinstance(v, (int, float)) else str(v)}
                                                    for k, v in best_result["metrics"].items()
                                                ])
                                                st.dataframe(metrics_df, use_container_width=True)
                                
                                else:
                                    # Train single model
                                    result = train_ml_model_production(
                                        df, x_features, y_target, task_type, algorithm,
                                        **model_params
                                    )
                                    
                                    if "error" in result:
                                        st.error(f"{result['error']}")
                                    else:
                                        st.success(f"{algorithm} model trained successfully!")
                                        
                                        # Display metrics
                                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                        
                                        key_metric = "r2" if task_type == "regression" else "f1"
                                        
                                        with metric_col1:
                                            st.metric("Key Metric", f"{result['metrics'].get(key_metric, 0):.4f}")
                                        with metric_col2:
                                            st.metric("Training Time", f"{result.get('training_time', 0):.2f}s")
                                        with metric_col3:
                                            st.metric("Features", result.get('n_features', 0))
                                        with metric_col4:
                                            st.metric("Samples", result.get('n_samples', 0))
                                        
                                        # Detailed metrics
                                        with st.expander("Detailed Metrics", expanded=False):
                                            metrics_df = pd.DataFrame([
                                                {"Metric": k.title().replace('_', ' '), "Value": f"{v:.4f}" if isinstance(v, (int, float)) else str(v)}
                                                for k, v in result["metrics"].items()
                                            ])
                                            st.dataframe(metrics_df, use_container_width=True)
                                        
                                        # Feature importance (if available)
                                        if "feature_importance" in result:
                                            st.subheader("Feature Importance")
                                            importance_df = pd.DataFrame([
                                                {"Feature": k, "Importance": v}
                                                for k, v in result["feature_importance"].items()
                                            ]).sort_values("Importance", ascending=False)
                                            
                                            fig_importance = px.bar(
                                                importance_df.head(10),
                                                x="Importance",
                                                y="Feature",
                                                orientation='h',
                                                title="Top 10 Feature Importance",
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig_importance, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Model training failed: {str(e)}")
                                if LOGGING_ENABLED:
                                    logging.error(f"ML training error: {e}")
        
        with tab8:
            st.header("Settings & Preferences")
            st.markdown("*Customize your analytics experience*")
            
            settings_col1, settings_col2 = st.columns(2)
            
            with settings_col1:
                st.subheader("Appearance")
                
                # Theme settings
                new_theme = st.selectbox(
                    "Color Theme:",
                    ACCESSIBILITY_THEMES,
                    index=ACCESSIBILITY_THEMES.index(st.session_state.theme_preference),
                    key="settings_theme"
                )
                
                if new_theme != st.session_state.theme_preference:
                    st.session_state.theme_preference = new_theme
                    st.success("Theme updated!")
                    time.sleep(0.5)
                    st.rerun()
                
                # Performance settings
                st.subheader("Performance")
                
                perf_mode = st.selectbox(
                    "Performance Mode:",
                    ["Balanced", "Performance", "Quality"],
                    index=["balanced", "performance", "quality"].index(st.session_state.user_preferences.get('performance_mode', 'balanced')),
                    key="settings_perf_mode"
                )
                
                st.session_state.user_preferences['performance_mode'] = perf_mode.lower()
                
                if perf_mode == "Performance":
                    st.info("Optimized for speed - uses aggressive sampling")
                elif perf_mode == "Quality":
                    st.info("Optimized for accuracy - uses full datasets when possible")
                else:
                    st.info("Balanced speed and accuracy")
                
                # Data handling preferences
                auto_sample = st.checkbox(
                    "Auto-sample large datasets",
                    value=st.session_state.user_preferences.get('auto_sample', True),
                    key="settings_auto_sample"
                )
                st.session_state.user_preferences['auto_sample'] = auto_sample
                
                enable_caching = st.checkbox(
                    "Enable result caching",
                    value=st.session_state.user_preferences.get('enable_caching', True),
                    key="settings_caching"
                )
                st.session_state.user_preferences['enable_caching'] = enable_caching
            
            with settings_col2:
                st.subheader("System Information")
                
                # Memory usage
                memory_info = monitor_memory_usage()
                if 'error' not in memory_info:
                    st.metric("Current Memory Usage", f"{memory_info['current_mb']:.0f} MB")
                    st.metric("Peak Memory Usage", f"{memory_info['peak_mb']:.0f} MB")
                    st.metric("CPU Usage", f"{memory_info.get('cpu_percent', 0):.1f}%")
                
                # Session information
                session_time = time.time() - st.session_state.get('app_start_time', time.time())
                st.metric("Session Duration", f"{session_time / 60:.1f} minutes")
                st.metric("Datasets Loaded", len(st.session_state.dfs))
                st.metric("Cached Models", len(st.session_state.trained_models))
                
                # System actions
                st.subheader("System Actions")
                
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    if st.button("Clear Cache", key="settings_clear_cache"):
                        cleanup_memory()
                        st.cache_data.clear()
                        st.success("Cache cleared!")
                    
                    if st.button("Export Session", key="settings_export_session"):
                        session_data = {
                            "datasets": {k: {"shape": v.shape, "columns": list(v.columns)} for k, v in st.session_state.dfs.items()},
                            "session_duration": session_time,
                            "memory_usage": memory_info.get('current_mb', 0),
                            "user_preferences": st.session_state.user_preferences,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        session_json = json.dumps(session_data, indent=2)
                        st.download_button(
                            "Download Session Data",
                            session_json,
                            f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                
                with action_col2:
                    if st.button("Soft Reset", key="settings_soft_reset"):
                        st.session_state.chart_configs = []
                        st.session_state.filter_state = {}
                        st.session_state.chat_history = []
                        st.success("Interface reset!")
                    
                    if st.button("Full Reset", key="settings_full_reset", type="secondary"):
                        for key in list(st.session_state.keys()):
                            if key not in ['theme_preference', 'user_preferences']:
                                del st.session_state[key]
                        st.success("Full reset completed!")
                        time.sleep(1)
                        st.rerun()
                
                # Diagnostic information
                with st.expander("Diagnostic Information", expanded=False):
                    st.json({
                        "sklearn_available": SKLEARN_AVAILABLE,
                        "logging_enabled": LOGGING_ENABLED,
                        "session_state_keys": list(st.session_state.keys()),
                        "performance_limits": {
                            "max_memory_mb": MAX_MEMORY_MB,
                            "max_rows_processing": MAX_ROWS_PROCESSING,
                            "sample_size_large": SAMPLE_SIZE_LARGE
                        }
                    })
    
    except Exception as e:
        # Global error handler with recovery options
        error_msg = f"Application error: {str(e)}"
        
        if LOGGING_ENABLED:
            logging.error(f"Main application error: {error_msg}")
        
        st.error(f"{error_msg}")
        
        st.subheader("Error Recovery")
        st.markdown("Try one of these recovery options:")
        
        recovery_col1, recovery_col2, recovery_col3, recovery_col4 = st.columns(4)
        
        with recovery_col1:
            if st.button("Soft Reload", key="error_soft_reload"):
                st.rerun()
        
        with recovery_col2:
            if st.button("Clear Cache", key="error_clear_cache"):
                cleanup_memory()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
                time.sleep(1)
                st.rerun()
        
        with recovery_col3:
            if st.button("Reset Data", key="error_reset_data"):
                st.session_state.dfs = {}
                st.session_state.selected_df = None
                st.session_state.data_loaded = False
                st.success("Data reset!")
                time.sleep(1)
                st.rerun()
        
        with recovery_col4:
            if st.button("Emergency Reset", key="error_emergency_reset"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Emergency reset completed!")
                time.sleep(1)
                st.rerun()

# --- Application Entry Point ---
if __name__ == "__main__":
    try:
        # Initialize and run production application
        main_production()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        
        # Last resort recovery
        if st.button("Force Restart", key="force_restart"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

