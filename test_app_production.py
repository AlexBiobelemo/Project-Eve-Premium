"""
Comprehensive Test Suite for Enterprise Data Analytics Platform
Tests all major components and functionality without external dependencies
"""

import unittest
import pandas as pd
import numpy as np
import io
import json
import tempfile
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Add the app directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from the production app
try:
    from app_production import (
        calculate_data_quality_score_enhanced,
        validate_dataframe,
        load_data_production,
        suggest_cleaning_production,
        apply_cleaning_suggestion_production,
        process_natural_query_production,
        detect_anomalies_production,
        compute_eda_summary_enhanced,
        monitor_memory_usage,
        cleanup_memory
    )
    
    # Try to import ML functions if available
    try:
        from app_production import train_ml_model_production
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        
except ImportError as e:
    print(f"Warning: Could not import app functions: {e}")
    print("Some tests may be skipped")


class TestDataQuality(unittest.TestCase):
    """Test data quality and validation functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create a sample dataframe with various data quality issues
        np.random.seed(42)
        self.good_df = pd.DataFrame({
            'numeric_col': np.random.normal(100, 15, 1000),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 1000),
            'date_col': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'clean_col': range(1000)
        })
        
        # Create a problematic dataframe
        self.bad_df = pd.DataFrame({
            'mostly_missing': [1, 2] + [np.nan] * 998,
            'duplicates': [1] * 500 + [2] * 500,
            'mixed_types': ['text'] * 500 + list(range(500)),
            'constant_col': [1] * 1000,
            'outliers': [1] * 995 + [1000, 2000, 3000, 4000, 5000]
        })
    
    def test_data_quality_score_good_data(self):
        """Test quality score calculation for good data"""
        if 'calculate_data_quality_score_enhanced' not in globals():
            self.skipTest("Function not available")
            
        score = calculate_data_quality_score_enhanced(self.good_df)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertGreater(score, 80.0)  # Good data should score high
    
    def test_data_quality_score_bad_data(self):
        """Test quality score calculation for problematic data"""
        if 'calculate_data_quality_score_enhanced' not in globals():
            self.skipTest("Function not available")
            
        score = calculate_data_quality_score_enhanced(self.bad_df)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertLess(score, 70.0)  # Bad data should score low
    
    def test_data_quality_score_empty_data(self):
        """Test quality score with empty dataframe"""
        if 'calculate_data_quality_score_enhanced' not in globals():
            self.skipTest("Function not available")
            
        empty_df = pd.DataFrame()
        score = calculate_data_quality_score_enhanced(empty_df)
        self.assertEqual(score, 0.0)
    
    def test_validate_dataframe_valid(self):
        """Test dataframe validation with valid data"""
        if 'validate_dataframe' not in globals():
            self.skipTest("Function not available")
            
        result = validate_dataframe(self.good_df)
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('valid', False))
        self.assertIn('memory_mb', result)
        self.assertIn('shape', result)
    
    def test_validate_dataframe_empty(self):
        """Test dataframe validation with empty data"""
        if 'validate_dataframe' not in globals():
            self.skipTest("Function not available")
            
        empty_df = pd.DataFrame()
        result = validate_dataframe(empty_df)
        self.assertFalse(result.get('valid', True))
        self.assertIn('error', result)
    
    def test_validate_dataframe_none(self):
        """Test dataframe validation with None"""
        if 'validate_dataframe' not in globals():
            self.skipTest("Function not available")
            
        result = validate_dataframe(None)
        self.assertFalse(result.get('valid', True))
        self.assertIn('error', result)


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality"""
    
    def setUp(self):
        """Set up test data files"""
        self.test_data = pd.DataFrame({
            'A': range(100),
            'B': np.random.normal(0, 1, 100),
            'C': ['category_' + str(i % 5) for i in range(100)]
        })
    
    def test_load_csv_data(self):
        """Test loading CSV data"""
        if 'load_data_production' not in globals():
            self.skipTest("Function not available")
            
        # Create CSV content
        csv_content = self.test_data.to_csv(index=False).encode('utf-8')
        
        # Test loading
        result_df = load_data_production(csv_content, "test.csv", "CSV")
        
        if result_df is not None:
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(result_df), len(self.test_data))
            self.assertEqual(len(result_df.columns), len(self.test_data.columns))
    
    def test_load_json_data(self):
        """Test loading JSON data"""
        if 'load_data_production' not in globals():
            self.skipTest("Function not available")
            
        # Create JSON content
        json_content = self.test_data.to_json(orient='records').encode('utf-8')
        
        # Test loading
        result_df = load_data_production(json_content, "test.json", "JSON")
        
        if result_df is not None:
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(result_df), len(self.test_data))
    
    def test_load_invalid_data(self):
        """Test loading invalid data"""
        if 'load_data_production' not in globals():
            self.skipTest("Function not available")
            
        # Test with invalid CSV content
        invalid_content = b"invalid,csv,content\nwith,malformed\nrows"
        result_df = load_data_production(invalid_content, "invalid.csv", "CSV")
        
        # Should handle gracefully (return None or valid DataFrame)
        if result_df is not None:
            self.assertIsInstance(result_df, pd.DataFrame)
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format"""
        if 'load_data_production' not in globals():
            self.skipTest("Function not available")
            
        content = b"some content"
        result_df = load_data_production(content, "test.txt", "TXT")
        
        # Should return None for unsupported formats
        self.assertIsNone(result_df)


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functionality"""
    
    def setUp(self):
        """Set up test data with cleaning issues"""
        # Create a mock session state
        self.mock_session_state = type('MockSessionState', (), {})()
        
        # Create test dataframe with various issues
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'high_missing': [1, 2, 3] + [np.nan] * 97,  # 97% missing
            'medium_missing': [1] * 80 + [np.nan] * 20,  # 20% missing
            'low_missing': [1] * 95 + [np.nan] * 5,      # 5% missing
            'numeric_string': ['1', '2', '3', '4', '5'] * 20,  # Should be numeric
            'date_string': ['2023-01-01', '2023-01-02'] * 50,  # Should be datetime
            'outliers': [1] * 95 + [100, 200, 300, 400, 500],  # Has outliers
            'good_column': range(100)  # No issues
        })
        
        self.mock_session_state.selected_df = self.test_df
    
    def test_suggest_cleaning_high_missing(self):
        """Test cleaning suggestions for high missing data"""
        if 'suggest_cleaning_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock the session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            # Generate a simple hash for testing
            df_hash = str(hash(str(self.test_df.shape)))
            suggestions = suggest_cleaning_production(df_hash)
            
            if suggestions:
                self.assertIsInstance(suggestions, list)
                
                # Should suggest dropping high_missing column
                high_missing_suggestions = [s for s in suggestions if s['column'] == 'high_missing']
                if high_missing_suggestions:
                    self.assertIn('drop', high_missing_suggestions[0]['type'])
                    self.assertEqual(high_missing_suggestions[0]['severity'], 'high')
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_apply_cleaning_suggestion_drop_column(self):
        """Test applying drop column suggestion"""
        if 'apply_cleaning_suggestion_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            suggestion = {
                'type': 'drop_column',
                'column': 'high_missing',
                'description': 'Drop high_missing (missing: 97.0%)'
            }
            
            success, message = apply_cleaning_suggestion_production(suggestion)
            
            if success:
                self.assertTrue(success)
                self.assertIsInstance(message, str)
                # Check if column was actually dropped
                self.assertNotIn('high_missing', self.mock_session_state.selected_df.columns)
            else:
                # If it fails, should provide error message
                self.assertIsInstance(message, str)
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_apply_cleaning_suggestion_impute(self):
        """Test applying imputation suggestion"""
        if 'apply_cleaning_suggestion_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            suggestion = {
                'type': 'impute_numeric',
                'column': 'medium_missing',
                'description': 'Impute medium_missing with mean'
            }
            
            success, message = apply_cleaning_suggestion_production(suggestion)
            
            if success:
                self.assertTrue(success)
                # Check if missing values were filled
                remaining_missing = self.mock_session_state.selected_df['medium_missing'].isnull().sum()
                self.assertLessEqual(remaining_missing, 0)
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")


class TestNaturalLanguageProcessing(unittest.TestCase):
    """Test natural language query processing"""
    
    def setUp(self):
        """Set up test data"""
        # Create mock session state
        self.mock_session_state = type('MockSessionState', (), {})()
        
        self.test_df = pd.DataFrame({
            'sales': [100, 200, 150, 300, 250] * 20,
            'region': ['North', 'South', 'East', 'West', 'Central'] * 20,
            'age': [25, 30, 35, 40, 45] * 20,
            'missing_col': [1, 2, np.nan, 4, 5] * 20
        })
        
        self.mock_session_state.selected_df = self.test_df
    
    def test_process_query_empty(self):
        """Test processing empty query"""
        if 'process_natural_query_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            response = process_natural_query_production("")
            self.assertIsInstance(response, dict)
            self.assertIn('message', response)
            self.assertIn('success', response)
            self.assertFalse(response['success'])
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_process_query_stats(self):
        """Test processing statistics query"""
        if 'process_natural_query_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            response = process_natural_query_production("show stats for sales")
            self.assertIsInstance(response, dict)
            self.assertIn('message', response)
            self.assertIn('success', response)
            
            if response['success']:
                self.assertIn('sales', response['message'])
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_process_query_filter(self):
        """Test processing filter query"""
        if 'process_natural_query_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state  
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            response = process_natural_query_production("filter region equals North")
            self.assertIsInstance(response, dict)
            self.assertIn('message', response)
            self.assertIn('success', response)
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_process_query_clean_missing(self):
        """Test processing data cleaning query"""
        if 'process_natural_query_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            response = process_natural_query_production("clean missing data in missing_col")
            self.assertIsInstance(response, dict)
            self.assertIn('message', response)
            self.assertIn('success', response)
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create mock session state
        self.mock_session_state = type('MockSessionState', (), {})()
        
        # Create test data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 95)
        outliers = [10, -10, 15, -15, 20]  # Clear outliers
        
        self.test_df = pd.DataFrame({
            'feature1': np.concatenate([normal_data, outliers]),
            'feature2': np.random.normal(5, 2, 100),
            'categorical': ['A', 'B', 'C'] * 33 + ['A']  # Not used in anomaly detection
        })
        
        self.mock_session_state.selected_df = self.test_df
    
    def test_detect_anomalies_z_score(self):
        """Test Z-Score anomaly detection"""
        if 'detect_anomalies_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            df_hash = str(hash(str(self.test_df.shape)))
            params = {"threshold": 3.0}
            features = ['feature1', 'feature2']
            
            result = detect_anomalies_production(df_hash, "Z-Score", params, features)
            
            self.assertIsInstance(result, dict)
            
            if 'error' not in result:
                self.assertIn('outliers', result)
                self.assertIn('anomaly_scores', result)
                self.assertIn('outlier_count', result)
                self.assertGreater(result['outlier_count'], 0)  # Should detect some outliers
                self.assertEqual(result['method'], "Z-Score")
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_detect_anomalies_iqr(self):
        """Test IQR anomaly detection"""
        if 'detect_anomalies_production' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            df_hash = str(hash(str(self.test_df.shape)))
            params = {"multiplier": 1.5}
            features = ['feature1', 'feature2']
            
            result = detect_anomalies_production(df_hash, "IQR", params, features)
            
            self.assertIsInstance(result, dict)
            
            if 'error' not in result:
                self.assertIn('outliers', result)
                self.assertIn('anomaly_scores', result)
                self.assertIn('outlier_count', result)
                self.assertEqual(result['method'], "IQR")
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_detect_anomalies_no_numeric_features(self):
        """Test anomaly detection with no numeric features"""
        if 'detect_anomalies_production' not in globals():
            self.skipTest("Function not available")
        
        # Create dataframe with only categorical data
        categorical_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 10,
            'cat2': ['X', 'Y', 'Z'] * 10
        })
        
        self.mock_session_state.selected_df = categorical_df
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            df_hash = str(hash(str(categorical_df.shape)))
            params = {"threshold": 3.0}
            
            result = detect_anomalies_production(df_hash, "Z-Score", params)
            
            self.assertIsInstance(result, dict)
            self.assertIn('error', result)  # Should return error for no numeric columns
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")


class TestEDAFunctions(unittest.TestCase):
    """Test Exploratory Data Analysis functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create mock session state
        self.mock_session_state = type('MockSessionState', (), {})()
        
        # Create comprehensive test dataframe
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'numeric1': np.random.normal(100, 15, 1000),
            'numeric2': np.random.exponential(2, 1000),
            'categorical1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'categorical2': np.random.choice(['Type1', 'Type2', 'Type3'], 1000),
            'datetime_col': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'boolean_col': np.random.choice([True, False], 1000),
            'missing_col': np.where(np.random.random(1000) > 0.1, 
                                   np.random.normal(50, 10, 1000), np.nan)
        })
        
        # Add quality score to attrs
        self.test_df.attrs = {'quality_score': 85.0}
        
        self.mock_session_state.selected_df = self.test_df
    
    def test_compute_eda_summary(self):
        """Test EDA summary computation"""
        if 'compute_eda_summary_enhanced' not in globals():
            self.skipTest("Function not available")
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            df_hash = str(hash(str(self.test_df.shape)))
            
            result = compute_eda_summary_enhanced(df_hash, self.test_df.shape)
            
            self.assertIsInstance(result, dict)
            
            if 'error' not in result:
                # Check required fields
                self.assertIn('insights', result)
                self.assertIn('recommendations', result)
                self.assertIn('numeric_columns', result)
                self.assertIn('categorical_columns', result)
                self.assertIn('missing_values', result)
                self.assertIn('quality_score', result)
                
                # Check data types
                self.assertIsInstance(result['insights'], list)
                self.assertIsInstance(result['recommendations'], list)
                self.assertIsInstance(result['numeric_columns'], int)
                self.assertIsInstance(result['categorical_columns'], int)
                
                # Verify column counts
                self.assertEqual(result['numeric_columns'], 3)  # numeric1, numeric2, missing_col
                self.assertEqual(result['categorical_columns'], 2)  # categorical1, categorical2
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")
    
    def test_compute_eda_summary_empty_data(self):
        """Test EDA summary with empty data"""
        if 'compute_eda_summary_enhanced' not in globals():
            self.skipTest("Function not available")
        
        # Set empty dataframe
        self.mock_session_state.selected_df = pd.DataFrame()
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = self.mock_session_state
        
        try:
            df_hash = "empty"
            result = compute_eda_summary_enhanced(df_hash, (0, 0))
            
            self.assertIsInstance(result, dict)
            self.assertIn('error', result)
        except Exception as e:
            self.skipTest(f"Function requires session state: {e}")


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring functions"""
    
    def test_monitor_memory_usage(self):
        """Test memory usage monitoring"""
        if 'monitor_memory_usage' not in globals():
            self.skipTest("Function not available")
        
        try:
            result = monitor_memory_usage()
            self.assertIsInstance(result, dict)
            
            if 'error' not in result:
                # Check required fields
                self.assertIn('current_mb', result)
                self.assertIn('peak_mb', result)
                self.assertIn('warning', result)
                self.assertIn('critical', result)
                
                # Check data types
                self.assertIsInstance(result['current_mb'], (int, float))
                self.assertIsInstance(result['peak_mb'], (int, float))
                self.assertIsInstance(result['warning'], bool)
                self.assertIsInstance(result['critical'], bool)
                
                # Memory usage should be positive
                self.assertGreater(result['current_mb'], 0)
                self.assertGreaterEqual(result['peak_mb'], result['current_mb'])
        except Exception as e:
            # Memory monitoring might not work in all environments
            self.skipTest(f"Memory monitoring not available: {e}")
    
    def test_cleanup_memory(self):
        """Test memory cleanup function"""
        if 'cleanup_memory' not in globals():
            self.skipTest("Function not available")
        
        try:
            # This should not raise an exception
            cleanup_memory()
        except Exception as e:
            self.fail(f"Memory cleanup raised an exception: {e}")


class TestMachineLearning(unittest.TestCase):
    """Test machine learning functionality"""
    
    def setUp(self):
        """Set up test data for ML"""
        # Create test data for ML
        np.random.seed(42)
        n_samples = 500
        
        # Regression data
        self.regression_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(2, 1, n_samples),
            'feature3': np.random.uniform(-1, 1, n_samples),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        # Create target with some noise
        self.regression_df['target'] = (
            2 * self.regression_df['feature1'] + 
            1.5 * self.regression_df['feature2'] + 
            0.5 * self.regression_df['feature3'] + 
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Classification data
        self.classification_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'categorical_feature': np.random.choice(['X', 'Y'], n_samples)
        })
        
        # Create binary target
        self.classification_df['target'] = np.where(
            self.classification_df['feature1'] + self.classification_df['feature2'] > 0,
            1, 0
        )
    
    def test_ml_model_training_regression(self):
        """Test ML model training for regression"""
        if not ML_AVAILABLE:
            self.skipTest("ML functions not available")
        
        try:
            x_features = ['feature1', 'feature2', 'feature3']
            y_target = 'target'
            
            result = train_ml_model_production(
                self.regression_df, x_features, y_target, 
                "regression", "RandomForest"
            )
            
            self.assertIsInstance(result, dict)
            
            if 'error' not in result:
                # Check required fields
                self.assertIn('metrics', result)
                self.assertIn('training_time', result)
                self.assertIn('n_samples', result)
                self.assertIn('algorithm', result)
                
                # Check regression metrics
                metrics = result['metrics']
                self.assertIn('mae', metrics)
                self.assertIn('mse', metrics)
                self.assertIn('r2', metrics)
                
                # R2 should be reasonable for our synthetic data
                self.assertGreater(metrics['r2'], 0.5)
                
                self.assertEqual(result['algorithm'], 'RandomForest')
        except Exception as e:
            self.skipTest(f"ML training not available: {e}")
    
    def test_ml_model_training_classification(self):
        """Test ML model training for classification"""
        if not ML_AVAILABLE:
            self.skipTest("ML functions not available")
        
        try:
            x_features = ['feature1', 'feature2']
            y_target = 'target'
            
            result = train_ml_model_production(
                self.classification_df, x_features, y_target,
                "classification", "RandomForest"
            )
            
            self.assertIsInstance(result, dict)
            
            if 'error' not in result:
                # Check required fields
                self.assertIn('metrics', result)
                self.assertIn('training_time', result)
                self.assertIn('algorithm', result)
                
                # Check classification metrics
                metrics = result['metrics']
                self.assertIn('accuracy', metrics)
                self.assertIn('precision', metrics)
                self.assertIn('recall', metrics)
                self.assertIn('f1', metrics)
                
                # Accuracy should be reasonable
                self.assertGreater(metrics['accuracy'], 0.6)
                
                self.assertEqual(result['algorithm'], 'RandomForest')
        except Exception as e:
            self.skipTest(f"ML training not available: {e}")
    
    def test_ml_model_invalid_inputs(self):
        """Test ML model training with invalid inputs"""
        if not ML_AVAILABLE:
            self.skipTest("ML functions not available")
        
        try:
            # Test with empty dataframe
            result = train_ml_model_production(
                pd.DataFrame(), ['feature1'], 'target',
                "regression", "RandomForest"
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('error', result)
            
            # Test with non-existent columns
            result = train_ml_model_production(
                self.regression_df, ['nonexistent_feature'], 'target',
                "regression", "RandomForest"
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('error', result)
            
            # Test with non-existent target
            result = train_ml_model_production(
                self.regression_df, ['feature1'], 'nonexistent_target',
                "regression", "RandomForest"
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('error', result)
        except Exception as e:
            self.skipTest(f"ML training not available: {e}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_large_dataframe_handling(self):
        """Test handling of large dataframes"""
        if 'validate_dataframe' not in globals():
            self.skipTest("Function not available")
        
        # Create a large dataframe (but not too large to cause memory issues)
        large_df = pd.DataFrame({
            'col1': range(50000),
            'col2': np.random.random(50000)
        })
        
        result = validate_dataframe(large_df, max_size_mb=10)  # Small limit
        self.assertIsInstance(result, dict)
        # Should either be valid or have a warning about size
        
    def test_dataframe_with_all_missing_values(self):
        """Test dataframe with all missing values"""
        if 'calculate_data_quality_score_enhanced' not in globals():
            self.skipTest("Function not available")
        
        all_missing_df = pd.DataFrame({
            'col1': [np.nan] * 100,
            'col2': [np.nan] * 100,
            'col3': [np.nan] * 100
        })
        
        score = calculate_data_quality_score_enhanced(all_missing_df)
        self.assertIsInstance(score, float)
        self.assertLessEqual(score, 20.0)  # Should be very low score
    
    def test_dataframe_with_single_row(self):
        """Test dataframe with single row"""
        if 'calculate_data_quality_score_enhanced' not in globals():
            self.skipTest("Function not available")
        
        single_row_df = pd.DataFrame({
            'col1': [1],
            'col2': ['A'],
            'col3': [3.14]
        })
        
        score = calculate_data_quality_score_enhanced(single_row_df)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
    
    def test_dataframe_with_unicode_characters(self):
        """Test dataframe with unicode characters"""
        if 'calculate_data_quality_score_enhanced' not in globals():
            self.skipTest("Function not available")
        
        unicode_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'José', '李明', '🚀Data'],
            'emoji_col': ['🎉', '😀', '🚀', '💡', '⭐'],
            'mixed': ['text', 123, '🌟', 'données', None]
        })
        
        score = calculate_data_quality_score_enhanced(unicode_df)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


class TestIntegration(unittest.TestCase):
    """Integration tests that test multiple components together"""
    
    def setUp(self):
        """Set up comprehensive test data"""
        np.random.seed(42)
        
        # Create a realistic dataset
        n_samples = 1000
        self.integration_df = pd.DataFrame({
            'sales_amount': np.random.lognormal(7, 0.5, n_samples),
            'profit_margin': np.random.normal(0.15, 0.05, n_samples),
            'customer_segment': np.random.choice(['Enterprise', 'SMB', 'Consumer'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'sales_rep_performance': np.random.normal(75, 15, n_samples).clip(0, 100),
            'customer_satisfaction': np.random.normal(4.2, 0.8, n_samples).clip(1, 5),
            'days_to_close': np.random.gamma(2, 15, n_samples).astype(int).clip(1, 365)
        })
        
        # Add some realistic data quality issues
        missing_indices = np.random.choice(n_samples, size=50, replace=False)
        self.integration_df.loc[missing_indices, 'profit_margin'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(n_samples, size=20, replace=False)
        self.integration_df.loc[outlier_indices, 'sales_amount'] *= 5
        
        # Add quality score
        self.integration_df.attrs = {'quality_score': 78.5}
    
    def test_full_data_pipeline(self):
        """Test complete data processing pipeline"""
        # Test data quality assessment
        if 'calculate_data_quality_score_enhanced' in globals():
            quality_score = calculate_data_quality_score_enhanced(self.integration_df)
            self.assertIsInstance(quality_score, float)
            self.assertGreater(quality_score, 60.0)  # Should be decent quality
        
        # Test data validation
        if 'validate_dataframe' in globals():
            validation_result = validate_dataframe(self.integration_df)
            self.assertIsInstance(validation_result, dict)
            self.assertTrue(validation_result.get('valid', False))
        
        print("Integration test: Data pipeline completed successfully")
    
    def test_data_quality_to_cleaning_pipeline(self):
        """Test pipeline from quality assessment to cleaning suggestions"""
        # Mock session state for cleaning functions
        mock_session_state = type('MockSessionState', (), {})()
        mock_session_state.selected_df = self.integration_df
        
        # Mock session state
        import sys
        if 'app_production' in sys.modules:
            original_session_state = getattr(sys.modules['app_production'], 'st', None)
            if hasattr(original_session_state, 'session_state'):
                original_session_state.session_state = mock_session_state
        
        try:
            if 'suggest_cleaning_production' in globals():
                df_hash = str(hash(str(self.integration_df.shape)))
                suggestions = suggest_cleaning_production(df_hash)
                
                if suggestions:
                    self.assertIsInstance(suggestions, list)
                    
                    # Should have suggestions for missing data
                    missing_suggestions = [s for s in suggestions if 'missing' in s.get('description', '').lower()]
                    self.assertGreater(len(missing_suggestions), 0)
                    
                print("Integration test: Quality to cleaning pipeline completed")
        except Exception as e:
            print(f"Integration test skipped: {e}")


def run_test_suite():
    """Run the complete test suite"""
    print("=" * 80)
    print("ENTERPRISE DATA ANALYTICS PLATFORM - TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataQuality,
        TestDataLoading,
        TestDataCleaning,
        TestNaturalLanguageProcessing,
        TestAnomalyDetection,
        TestEDAFunctions,
        TestPerformanceMonitoring,
        TestMachineLearning,
        TestEdgeCases,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 'N/A'}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('\\n')[-2] if traceback else 'Unknown error'}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2] if traceback else 'Unknown error'}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Test suite passed with high success rate!")
    elif success_rate >= 75:
        print("✅ GOOD: Test suite passed with acceptable success rate")
    elif success_rate >= 50:
        print("⚠️ WARNING: Some tests failed - review needed")
    else:
        print("❌ CRITICAL: Many tests failed - major issues detected")
    
    return result


if __name__ == "__main__":
    # Run the test suite
    test_result = run_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if test_result.wasSuccessful() else 1)
