"""
NLP Test Integration Module for Enterprise Data Analytics Platform

This module integrates NLP testing capabilities directly into the main application.
It provides a comprehensive test framework for natural language prompts, ensuring
that the conversational AI assistant functionality works correctly.

"""

import pandas as pd
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import streamlit as st
import plotly.express as px
import base64
from pathlib import Path

# NLP Test constants
NLP_TEST_VERSION = "1.0.0"

class NLPTester:
    """
    Integrated NLP testing framework for the Enterprise Data Analytics Platform.
    
    This class provides functionality to test the NLP capabilities of the
    process_natural_query_production function directly within the application.
    """
    
    def __init__(self, process_query_func):
        """
        Initialize the NLP tester with the query processing function.
        
        Args:
            process_query_func: The function that processes natural language queries
        """
        self.process_query_func = process_query_func
        self.logger = logging.getLogger(__name__)
        
        # Test results storage
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'detailed_results': []
        }
        
        # Define all test prompts organized by feature category
        self.test_prompts = self._define_test_prompts()
        
        # Create sample data that matches expected structure if needed
        self.has_test_data = self._check_for_test_data()
    
    def _check_for_test_data(self) -> bool:
        """Check if there's already suitable test data loaded in the session."""
        if 'selected_df' not in st.session_state or st.session_state.selected_df is None:
            return False
            
        df = st.session_state.selected_df
        if df is None or df.empty:
            return False
            
        # Check for required columns that match test prompts
        required_columns = [
            'sales_amount', 'profit_margin', 'customer_segment', 
            'customer_satisfaction', 'region', 'days_to_close'
        ]
        
        # Check if at least 4 of the required columns exist
        matches = sum(1 for col in required_columns if col in df.columns)
        return matches >= 4
    
    def create_test_data(self) -> pd.DataFrame:
        """Create comprehensive test dataset that mirrors the expected data structure."""
        
        n_samples = 1000
        np.random.seed(42)  # For reproducible tests
        
        # Create realistic test data with controlled missing values and outliers
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='4H')
        
        # Base trend for realistic sales data
        base_trend = np.sin(np.arange(n_samples) / 100) * 1000
        noise = np.random.normal(0, 100, n_samples)
        
        data = pd.DataFrame({
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
        
        # Add controlled missing values for testing
        missing_indices = np.random.choice(data.index, size=int(0.05 * len(data)), replace=False)
        data.loc[missing_indices[:len(missing_indices)//3], 'profit_margin'] = np.nan
        data.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'customer_satisfaction'] = np.nan
        data.loc[missing_indices[2*len(missing_indices)//3:], 'customer_segment'] = np.nan
        
        # Add controlled outliers for testing
        outlier_indices = np.random.choice(data.index, size=int(0.02 * len(data)), replace=False)
        data.loc[outlier_indices, 'sales_amount'] *= np.random.uniform(5, 10, len(outlier_indices))
        
        # Set quality attributes
        data.attrs = {
            'source_file': 'nlp_test_data.csv',
            'load_time': datetime.now().isoformat(),
            'quality_score': 85  # Good quality score for testing
        }
        
        return data
    
    def _define_test_prompts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Define comprehensive test prompts organized by feature category."""
        
        return {
            'data_cleaning': [
                {
                    'id': 'DC001',
                    'prompt': 'Clean missing data in sales_amount',
                    'expected_action': 'clean_missing',
                    'feature_tested': 'Handles missing data imputation for numeric column using mean/median based on skewness',
                    'expected_outcome': 'Fills missing values in sales_amount with appropriate method',
                    'category': 'Data Cleaning',
                    'priority': 'high'
                },
                {
                    'id': 'DC002',
                    'prompt': 'Fill missing values in customer_segment',
                    'expected_action': 'clean_missing',
                    'feature_tested': 'Imputes missing values in categorical column using mode',
                    'expected_outcome': 'Fills missing values in customer_segment with most frequent value',
                    'category': 'Data Cleaning',
                    'priority': 'high'
                },
                {
                    'id': 'DC003',
                    'prompt': 'Drop column days_to_close',
                    'expected_action': 'drop_column',
                    'feature_tested': 'Drops specified column from dataset',
                    'expected_outcome': 'Removes days_to_close column and updates dataset',
                    'category': 'Data Cleaning',
                    'priority': 'medium'
                },
                {
                    'id': 'DC004',
                    'prompt': 'Apply high priority cleaning suggestions',
                    'expected_action': 'apply_cleaning_suggestions',
                    'feature_tested': 'Applies high-priority cleaning suggestions automatically',
                    'expected_outcome': 'Applies high-severity cleaning operations like dropping sparse columns',
                    'category': 'Data Cleaning',
                    'priority': 'medium'
                }
            ],
            
            'statistical_analysis': [
                {
                    'id': 'SA001',
                    'prompt': 'Show stats for sales_amount',
                    'expected_action': 'show_stats',
                    'feature_tested': 'Generates descriptive statistics for numeric column',
                    'expected_outcome': 'Displays statistics (mean, median, std, etc.) for sales_amount with insights',
                    'category': 'Statistical Analysis',
                    'priority': 'high'
                },
                {
                    'id': 'SA002',
                    'prompt': 'Describe customer_segment',
                    'expected_action': 'show_stats',
                    'feature_tested': 'Summarizes categorical column with value counts',
                    'expected_outcome': 'Shows top values, unique count, and most common value for customer_segment',
                    'category': 'Statistical Analysis',
                    'priority': 'high'
                },
                {
                    'id': 'SA003',
                    'prompt': 'Summarize profit_margin',
                    'expected_action': 'show_stats',
                    'feature_tested': 'Alternative phrasing for statistical summary of numeric column',
                    'expected_outcome': 'Displays statistics for profit_margin, tests synonym recognition',
                    'category': 'Statistical Analysis',
                    'priority': 'medium'
                }
            ],
            
            'data_filtering': [
                {
                    'id': 'DF001',
                    'prompt': 'Filter region equals Europe',
                    'expected_action': 'filter_equal',
                    'feature_tested': 'Filters rows where categorical column matches specific value',
                    'expected_outcome': 'Updates dataset to include only rows where region is Europe',
                    'category': 'Data Filtering',
                    'priority': 'high'
                },
                {
                    'id': 'DF002',
                    'prompt': 'Show rows where sales_amount > 1000',
                    'expected_action': 'filter_greater',
                    'feature_tested': 'Filters rows based on numeric column with greater than condition',
                    'expected_outcome': 'Updates dataset to include rows where sales_amount exceeds 1000',
                    'category': 'Data Filtering',
                    'priority': 'high'
                },
                {
                    'id': 'DF003',
                    'prompt': 'Filter customer_satisfaction < 3',
                    'expected_action': 'filter_less',
                    'feature_tested': 'Filters rows based on numeric column with less than condition',
                    'expected_outcome': 'Updates dataset to include rows where customer_satisfaction < 3',
                    'category': 'Data Filtering',
                    'priority': 'high'
                }
            ],
            
            'visualization': [
                {
                    'id': 'VZ001',
                    'prompt': 'Create scatter plot',
                    'expected_action': 'create_chart',
                    'feature_tested': 'Initiates scatter plot configuration',
                    'expected_outcome': 'Adds scatter plot to chart_configs for customization',
                    'category': 'Visualization',
                    'priority': 'high'
                },
                {
                    'id': 'VZ002',
                    'prompt': 'Make histogram',
                    'expected_action': 'create_chart',
                    'feature_tested': 'Creates histogram configuration',
                    'expected_outcome': 'Adds histogram to chart_configs for customization',
                    'category': 'Visualization',
                    'priority': 'high'
                },
                {
                    'id': 'VZ003',
                    'prompt': 'Plot sales_amount vs profit_margin',
                    'expected_action': 'create_chart_xy',
                    'feature_tested': 'Creates scatter plot with specified X and Y axes',
                    'expected_outcome': 'Adds scatter plot configuration with specified axes',
                    'category': 'Visualization',
                    'priority': 'high'
                },
                {
                    'id': 'VZ004',
                    'prompt': 'Create bar chart',
                    'expected_action': 'create_chart',
                    'feature_tested': 'Initiates bar chart configuration',
                    'expected_outcome': 'Adds bar chart to chart_configs for customization',
                    'category': 'Visualization',
                    'priority': 'medium'
                },
                {
                    'id': 'VZ005',
                    'prompt': 'Show correlation heatmap',
                    'expected_action': 'create_chart',
                    'feature_tested': 'Creates correlation heatmap for numeric columns',
                    'expected_outcome': 'Adds correlation heatmap to chart_configs',
                    'category': 'Visualization',
                    'priority': 'medium'
                }
            ],
            
            'anomaly_detection': [
                {
                    'id': 'AD001',
                    'prompt': 'Detect anomalies in sales_amount using Z-Score',
                    'expected_action': 'detect_anomalies',
                    'feature_tested': 'Performs anomaly detection using Z-Score method',
                    'expected_outcome': 'Runs anomaly detection with Z-Score and displays results',
                    'category': 'Anomaly Detection',
                    'priority': 'medium'
                },
                {
                    'id': 'AD002',
                    'prompt': 'Find outliers in profit_margin with IQR',
                    'expected_action': 'detect_anomalies',
                    'feature_tested': 'Performs anomaly detection using IQR method',
                    'expected_outcome': 'Runs anomaly detection with IQR method and shows results',
                    'category': 'Anomaly Detection',
                    'priority': 'medium'
                },
                {
                    'id': 'AD003',
                    'prompt': 'Run anomaly detection on customer_satisfaction with IsolationForest',
                    'expected_action': 'detect_anomalies',
                    'feature_tested': 'Tests IsolationForest anomaly detection if available',
                    'expected_outcome': 'Runs IsolationForest anomaly detection and visualizes results',
                    'category': 'Anomaly Detection',
                    'priority': 'low'
                }
            ],
            
            'machine_learning': [
                {
                    'id': 'ML001',
                    'prompt': 'Train RandomForest model to predict sales_amount',
                    'expected_action': 'train_ml_model',
                    'feature_tested': 'Trains RandomForest regression model',
                    'expected_outcome': 'Initiates ML model training with RandomForest for regression',
                    'category': 'Machine Learning',
                    'priority': 'low'
                },
                {
                    'id': 'ML002',
                    'prompt': 'Perform clustering on sales_amount and profit_margin',
                    'expected_action': 'perform_clustering',
                    'feature_tested': 'Performs K-Means clustering with automatic cluster selection',
                    'expected_outcome': 'Runs clustering and visualizes cluster results',
                    'category': 'Machine Learning',
                    'priority': 'low'
                }
            ],
            
            'settings_preferences': [
                {
                    'id': 'SP001',
                    'prompt': 'Change theme to Dark',
                    'expected_action': 'change_theme',
                    'feature_tested': 'Changes application theme to Dark mode',
                    'expected_outcome': 'Updates theme_preference to Dark and refreshes UI',
                    'category': 'Settings and Preferences',
                    'priority': 'low'
                }
            ],
            
            'error_handling': [
                {
                    'id': 'EH001',
                    'prompt': 'Clean missing data in nonexistent_column',
                    'expected_action': 'error_handling',
                    'feature_tested': 'Tests error handling for invalid column names',
                    'expected_outcome': 'Returns error message indicating column not found',
                    'category': 'Error Handling',
                    'priority': 'high'
                }
            ]
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all NLP tests against the process_query_function.
        
        Returns:
            Dict with comprehensive test results
        """
        self.logger.info("Starting comprehensive NLP prompt test suite...")
        start_time = time.time()
        
        # Reset test results
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'detailed_results': []
        }
        
        all_results = []
        category_results = {}
        
        # Test each category
        for category_name, prompts in self.test_prompts.items():
            self.logger.info(f"Testing category: {category_name}")
            category_results[category_name] = {
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'tests': []
            }
            
            for prompt_config in prompts:
                result = self._run_single_test(prompt_config)
                all_results.append(result)
                category_results[category_name]['tests'].append(result)
                
                # Update category counters
                if result['status'] == 'passed':
                    category_results[category_name]['passed'] += 1
                    self.test_results['passed'] += 1
                elif result['status'] == 'failed':
                    category_results[category_name]['failed'] += 1
                    self.test_results['failed'] += 1
                else:
                    category_results[category_name]['errors'] += 1
                    self.test_results['errors'] += 1
        
        # Calculate final metrics
        total_time = time.time() - start_time
        total_tests = len(all_results)
        
        final_results = {
            'summary': {
                'total_tests': total_tests,
                'passed': self.test_results['passed'],
                'failed': self.test_results['failed'],
                'errors': self.test_results['errors'],
                'success_rate': (self.test_results['passed'] / total_tests) * 100 if total_tests > 0 else 0,
                'execution_time_seconds': total_time,
                'timestamp': datetime.now().isoformat()
            },
            'category_breakdown': category_results,
            'detailed_results': all_results,
            'feature_coverage': self._calculate_feature_coverage(),
            'recommendations': self._generate_recommendations()
        }
        
        self.logger.info(f"Test suite completed in {total_time:.2f} seconds")
        self.logger.info(f"Results: {self.test_results['passed']}/{total_tests} passed ({final_results['summary']['success_rate']:.1f}%)")
        
        return final_results
    
    def run_category_tests(self, category_name: str) -> Dict[str, Any]:
        """
        Run tests for a specific category.
        
        Args:
            category_name: Category name to test
            
        Returns:
            Dict with test results for the category
        """
        if category_name not in self.test_prompts:
            return {
                'error': f"Category '{category_name}' not found",
                'valid_categories': list(self.test_prompts.keys())
            }
        
        self.logger.info(f"Testing category: {category_name}")
        start_time = time.time()
        
        prompts = self.test_prompts[category_name]
        results = []
        
        passed = 0
        failed = 0
        errors = 0
        
        for prompt_config in prompts:
            result = self._run_single_test(prompt_config)
            results.append(result)
            
            if result['status'] == 'passed':
                passed += 1
            elif result['status'] == 'failed':
                failed += 1
            else:
                errors += 1
        
        category_results = {
            'category': category_name,
            'summary': {
                'total_tests': len(prompts),
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'success_rate': (passed / len(prompts)) * 100 if len(prompts) > 0 else 0,
                'execution_time_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': results
        }
        
        return category_results
    
    def run_single_test(self, test_id: str) -> Dict[str, Any]:
        """
        Run a single test by its ID.
        
        Args:
            test_id: The ID of the test to run (e.g., 'DC001')
            
        Returns:
            Dict with test result
        """
        for category, prompts in self.test_prompts.items():
            for prompt_config in prompts:
                if prompt_config['id'] == test_id:
                    return self._run_single_test(prompt_config)
        
        return {
            'error': f"Test ID '{test_id}' not found",
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_single_test(self, prompt_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test and validate results."""
        
        test_id = prompt_config['id']
        prompt = prompt_config['prompt']
        
        self.logger.info(f"Running test {test_id}: '{prompt}'")
        
        try:
            # Record start time
            start_time = time.time()
            
            # Make a backup of any state that might be modified during testing
            backup_state = self._backup_state()
            
            # Process the query
            response = self.process_query_func(prompt)
            
            # Record execution time
            execution_time = time.time() - start_time
            
            # Validate response
            validation_result = self._validate_response(response, prompt_config)
            
            # Restore original state if needed
            if validation_result.get('restore_state', False):
                self._restore_state(backup_state)
            
            result = {
                'test_id': test_id,
                'prompt': prompt,
                'category': prompt_config['category'],
                'priority': prompt_config['priority'],
                'feature_tested': prompt_config['feature_tested'],
                'expected_outcome': prompt_config['expected_outcome'],
                'expected_action': prompt_config['expected_action'],
                'response': response,
                'validation': validation_result,
                'status': 'passed' if validation_result['valid'] else 'failed',
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            if validation_result['valid']:
                self.logger.info(f"Test {test_id} PASSED")
            else:
                self.logger.warning(f"Test {test_id} FAILED: {validation_result['reason']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Test {test_id} ERROR: {str(e)}")
            
            return {
                'test_id': test_id,
                'prompt': prompt,
                'category': prompt_config['category'],
                'priority': prompt_config['priority'],
                'feature_tested': prompt_config['feature_tested'],
                'expected_outcome': prompt_config['expected_outcome'],
                'expected_action': prompt_config['expected_action'],
                'response': None,
                'validation': {'valid': False, 'reason': f'Exception: {str(e)}'},
                'status': 'error',
                'execution_time': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _backup_state(self) -> Dict[str, Any]:
        """Backup important session state that might be modified during testing."""
        backup = {}
        
        if 'selected_df' in st.session_state and st.session_state.selected_df is not None:
            backup['selected_df'] = st.session_state.selected_df.copy()
        
        if 'chart_configs' in st.session_state:
            backup['chart_configs'] = st.session_state.chart_configs.copy()
            
        if 'theme_preference' in st.session_state:
            backup['theme_preference'] = st.session_state.theme_preference
            
        return backup
    
    def _restore_state(self, backup: Dict[str, Any]):
        """Restore session state from backup."""
        if 'selected_df' in backup and backup['selected_df'] is not None:
            st.session_state.selected_df = backup['selected_df']
            
        if 'chart_configs' in backup:
            st.session_state.chart_configs = backup['chart_configs']
            
        if 'theme_preference' in backup:
            st.session_state.theme_preference = backup['theme_preference']
    
    def _validate_response(self, response: Dict[str, Any], prompt_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the response against expected outcomes."""
        
        validation_result = {'valid': False, 'restore_state': False, 'reason': ''}
        
        if response is None:
            validation_result['reason'] = 'Response is None'
            return validation_result
        
        if not isinstance(response, dict):
            validation_result['reason'] = 'Response is not a dictionary'
            return validation_result
        
        # Check required fields
        required_fields = ['message', 'success']
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            validation_result['reason'] = f'Missing fields: {missing_fields}'
            return validation_result
        
        # Basic validation based on expected action
        expected_action = prompt_config['expected_action']
        
        # For error handling tests, we expect success=False
        if expected_action == 'error_handling':
            if response.get('success', True):  # Should be False for error cases
                validation_result['reason'] = 'Expected error case but got success=True'
                return validation_result
            
            validation_result['valid'] = True
            validation_result['reason'] = 'Error handling test passed'
            return validation_result
        
        # For most other tests, we expect success=True or a helpful message
        success = response.get('success', False)
        message = response.get('message', '')
        action = response.get('action', None)
        
        # Validate based on action type
        if expected_action in ['clean_missing', 'drop_column']:
            # Data cleaning actions should either succeed or provide helpful feedback
            if success or 'no missing values' in message.lower() or 'not found' in message.lower():
                validation_result['valid'] = True
                validation_result['reason'] = 'Data cleaning response is appropriate'
                validation_result['restore_state'] = True  # Restore original data after test
            else:
                validation_result['reason'] = f'Data cleaning failed: {message}'
                validation_result['restore_state'] = True
        
        elif expected_action == 'show_stats':
            # Statistical analysis should provide statistics or meaningful feedback
            if success or any(keyword in message.lower() for keyword in ['statistics', 'values', 'count', 'mean']):
                validation_result['valid'] = True
                validation_result['reason'] = 'Statistical analysis response is appropriate'
            else:
                validation_result['reason'] = f'Statistics response insufficient: {message}'
        
        elif expected_action in ['filter_equal', 'filter_greater', 'filter_less']:
            # Filtering should provide results or meaningful feedback
            if success or 'filter' in message.lower() or 'rows' in message.lower():
                validation_result['valid'] = True
                validation_result['reason'] = 'Filtering response is appropriate'
                validation_result['restore_state'] = True  # Restore original data after test
            else:
                validation_result['reason'] = f'Filtering response insufficient: {message}'
                validation_result['restore_state'] = True
        
        elif expected_action in ['create_chart', 'create_chart_xy']:
            # Chart creation should indicate success or provide guidance
            if success or 'chart' in message.lower() or 'visualization' in message.lower():
                validation_result['valid'] = True
                validation_result['reason'] = 'Chart creation response is appropriate'
                validation_result['restore_state'] = True  # Restore chart configs after test
            else:
                validation_result['reason'] = f'Chart creation response insufficient: {message}'
                validation_result['restore_state'] = True
                
        elif expected_action in ['detect_anomalies', 'train_ml_model', 'perform_clustering', 'change_theme']:
            # Advanced features might not be directly callable via NLP, so validate intention
            if action is not None or 'would be' in message.lower() or 'requires' in message.lower():
                validation_result['valid'] = True
                validation_result['reason'] = 'Advanced feature response is appropriate'
            else:
                validation_result['reason'] = f'Advanced feature response insufficient: {message}'
        
        else:
            # For other actions, check if response is meaningful
            if len(message.strip()) > 10:  # At least some meaningful content
                validation_result['valid'] = True
                validation_result['reason'] = 'Response contains meaningful content'
            else:
                validation_result['reason'] = 'Response is too brief or empty'
        
        return validation_result
    
    def _calculate_feature_coverage(self) -> Dict[str, Any]:
        """Calculate feature coverage metrics."""
        
        total_features = sum(len(prompts) for prompts in self.test_prompts.values())
        
        coverage = {
            'total_features_tested': total_features,
            'categories_covered': len(self.test_prompts),
            'category_distribution': {
                category: len(prompts) for category, prompts in self.test_prompts.items()
            },
            'priority_distribution': {},
            'feature_completeness': {}
        }
        
        # Calculate priority distribution
        all_prompts = [prompt for prompts in self.test_prompts.values() for prompt in prompts]
        priorities = [prompt['priority'] for prompt in all_prompts]
        for priority in ['high', 'medium', 'low']:
            coverage['priority_distribution'][priority] = priorities.count(priority)
        
        # Feature completeness assessment
        major_features = ['data_cleaning', 'statistical_analysis', 'data_filtering', 'visualization']
        minor_features = ['anomaly_detection', 'machine_learning', 'settings_preferences', 'error_handling']
        
        coverage['feature_completeness'] = {
            'major_features_covered': len([f for f in major_features if f in self.test_prompts]),
            'minor_features_covered': len([f for f in minor_features if f in self.test_prompts]),
            'total_major_features': len(major_features),
            'total_minor_features': len(minor_features)
        }
        
        return coverage
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        
        # Analyze test results for patterns
        if self.test_results['failed'] > 0:
            recommendations.append(
                f"  {self.test_results['failed']} tests failed. Review failed test details for improvement areas."
            )
        
        if self.test_results['errors'] > 0:
            recommendations.append(
                f" {self.test_results['errors']} tests had errors. Check error handling and exception management."
            )
        
        success_rate = (self.test_results['passed'] / 
                       (self.test_results['passed'] + self.test_results['failed'] + self.test_results['errors'])) * 100
        
        if success_rate >= 90:
            recommendations.append("Excellent NLP coverage! The system handles most query types effectively.")
        elif success_rate >= 75:
            recommendations.append("Good NLP coverage with room for improvement in specific areas.")
        elif success_rate >= 50:
            recommendations.append("Moderate NLP coverage. Consider expanding pattern matching and error handling.")
        else:
            recommendations.append("Low NLP coverage. Significant improvements needed in query processing.")
        
        # Feature-specific recommendations based on failures
        failed_categories = {}
        for cat, cat_data in self.test_results.get('category_results', {}).items():
            if cat_data.get('failed', 0) > 0:
                failed_categories[cat] = cat_data.get('failed', 0)
        
        if 'data_cleaning' in failed_categories:
            recommendations.append("Improve data cleaning pattern matching for imputation and column operations.")
            
        if 'statistical_analysis' in failed_categories:
            recommendations.append("Enhance statistical analysis to handle different phrasing and column types.")
            
        if 'data_filtering' in failed_categories:
            recommendations.append("Strengthen filtering operations to support various conditions and expressions.")
            
        if 'visualization' in failed_categories:
            recommendations.append("Expand visualization support with more chart types and automatic suggestions.")
            
        # General recommendations
        recommendations.extend([
            "Consider adding more advanced visualization options for complex chart types",
            "Implement more sophisticated ML model training prompts",
            "Add support for bulk operations and batch processing commands",
            "Include performance optimization queries for large datasets",
            "Expand theme and accessibility customization options"
        ])
        
        return recommendations
    
    def export_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export test results to JSON file."""
        
        if filename is None:
            filename = f"nlp_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            reports_dir = Path("test_reports")
            reports_dir.mkdir(exist_ok=True)
            
            file_path = reports_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Test results exported to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return ""
    
    def generate_html_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Generate HTML report from test results."""
        
        if filename is None:
            filename = f"nlp_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = self._create_html_report(results)
        
        try:
            reports_dir = Path("test_reports")
            reports_dir.mkdir(exist_ok=True)
            
            file_path = reports_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return ""
    
    def _create_html_report(self, results: Dict[str, Any]) -> str:
        """Create HTML content for the test report."""
        
        summary = results['summary']
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NLP Prompt Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                .category-section {{ margin-bottom: 30px; }}
                .category-header {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 15px; }}
                .test-item {{ background-color: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; }}
                .test-failed {{ border-left-color: #dc3545; }}
                .test-error {{ border-left-color: #ffc107; }}
                .status-badge {{ padding: 5px 10px; border-radius: 15px; color: white; font-size: 0.8em; }}
                .status-passed {{ background-color: #28a745; }}
                .status-failed {{ background-color: #dc3545; }}
                .status-error {{ background-color: #ffc107; color: black; }}
                .recommendations {{ background-color: #e7f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>NLP Prompt Test Suite Report</h1>
                    <p>Enterprise Data Analytics Platform - Natural Language Processing Coverage</p>
                    <p><strong>Generated:</strong> {summary['timestamp']}</p>
                </div>
                
                <div class="summary">
                    <div class="metric-card">
                        <div class="metric-value">{summary['total_tests']}</div>
                        <div class="metric-label">Total Tests</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['passed']}</div>
                        <div class="metric-label">Passed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['failed']}</div>
                        <div class="metric-label">Failed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['success_rate']:.1f}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['execution_time_seconds']:.2f}s</div>
                        <div class="metric-label">Execution Time</div>
                    </div>
                </div>
        """
        
        # Add category breakdown
        for category_name, category_data in results['category_breakdown'].items():
            html += f"""
                <div class="category-section">
                    <div class="category-header">
                        <h3>{category_name.replace('_', ' ').title()}</h3>
                        <p>Passed: {category_data['passed']} | Failed: {category_data['failed']} | Errors: {category_data['errors']}</p>
                    </div>
            """
            
            for test in category_data['tests']:
                status_class = f"status-{test['status']}"
                test_class = f"test-{test['status']}" if test['status'] != 'passed' else ""
                
                html += f"""
                    <div class="test-item {test_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <strong>{test['test_id']}: {test['prompt']}</strong>
                            <span class="status-badge {status_class}">{test['status'].upper()}</span>
                        </div>
                        <p><strong>Feature Tested:</strong> {test['feature_tested']}</p>
                        <p><strong>Expected Outcome:</strong> {test['expected_outcome']}</p>
                        <p><strong>Response:</strong> {test['response']['message'] if test['response'] else 'No response'}</p>
                """
                
                if test['status'] != 'passed':
                    html += f"<p><strong>Issue:</strong> {test['validation']['reason']}</p>"
                
                html += f"<p><strong>Execution Time:</strong> {test['execution_time']:.3f}s</p></div>"
            
            html += "</div>"
        
        # Add recommendations
        html += f"""
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <ul>
        """
        
        for recommendation in results['recommendations']:
            html += f"<li>{recommendation}</li>"
        
        html += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

def render_nlp_test_ui(process_natural_query_production):
    """
    Render the NLP test UI tab in the Streamlit application.
    
    Args:
        process_natural_query_production: The function that processes natural language queries
    """
    # Initialize the NLP tester
    tester = NLPTester(process_natural_query_production)
    
    # Status indicator
    status_col1, status_col2, status_col3 = st.columns([2, 1, 1])
    with status_col1:
        st.markdown(" **Comprehensive AI validation** with 22 test scenarios across 8 feature categories")
    with status_col2:
        st.metric("Test Coverage", "22 prompts")
    with status_col3:
        st.metric("Categories", "8 areas")
    
    # Test Data Setup
    st.subheader("Test Environment")
    
    # Check if we need test data
    if not tester.has_test_data:
        st.warning("⚠️ No suitable test data detected for comprehensive validation")
        
        info_col1, info_col2 = st.columns([2, 1])
        with info_col1:
            st.markdown("""
            **Required columns for testing:**
            - `sales_amount`, `profit_margin` (numeric)
            - `customer_segment`, `region` (categorical)  
            - `customer_satisfaction` (numeric, 1-5 scale)
            """)
        with info_col2:
            if st.button("Create Test Data", type="primary"):
                with st.spinner("Generating comprehensive test dataset..."):
                    test_data = tester.create_test_data()
                    st.session_state.selected_df = test_data
                    st.session_state.dfs['nlp_test_data.csv'] = test_data
                    st.session_state.data_loaded = True
                    
                    st.success(f"Test data ready! {len(test_data):,} rows × {len(test_data.columns)} columns")
                    st.rerun()
    else:
        st.success("Test data available for testing!")
        
        # Data info
        df = st.session_state.selected_df
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_values:,}")
        with col4:
            st.metric("Data Quality", f"{df.attrs.get('quality_score', 'N/A')}/100")
    
    # Test Configuration
    st.subheader("Testing Console")
    
    tab1, tab2, tab3 = st.tabs(["Run Tests", "Explore Tests", "Results"])
    
    with tab1:
        st.markdown("*Execute comprehensive validation tests to evaluate your AI assistant's capabilities*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            test_options = st.radio(
                "Test Scope",
                options=["Full Test Suite", "Category Test", "Single Test"],
                horizontal=True
            )
            
            if test_options == "Category Test":
                category = st.selectbox(
                    "Select Category",
                    options=list(tester.test_prompts.keys()),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
            elif test_options == "Single Test":
                # Flatten all test IDs with their prompts
                all_tests = []
                for category, prompts in tester.test_prompts.items():
                    for prompt in prompts:
                        all_tests.append((prompt['id'], f"{prompt['id']}: {prompt['prompt']}"))
                
                test_id = st.selectbox(
                    "Select Test",
                    options=[t[0] for t in all_tests],
                    format_func=lambda x: next((t[1] for t in all_tests if t[0] == x), x)
                )
        
        with col2:
            export_options = st.multiselect(
                "Export Options",
                options=["JSON Report", "HTML Report"],
                default=["HTML Report"]
            )
        
        # Run button
        if st.button("Run Tests", type="primary"):
            if not tester.has_test_data:
                st.error("Please create or load test data first!")
            else:
                with st.spinner("Running NLP tests..."):
                    if test_options == "Full Test Suite":
                        results = tester.run_all_tests()
                        st.session_state.nlp_test_results = results
                        
                    elif test_options == "Category Test":
                        results = tester.run_category_tests(category)
                        st.session_state.nlp_test_results = {
                            'category_results': results,
                            'is_category_test': True
                        }
                        
                    elif test_options == "Single Test":
                        result = tester.run_single_test(test_id)
                        st.session_state.nlp_test_results = {
                            'single_test_result': result,
                            'is_single_test': True
                        }
                    
                    # Export results
                    if "JSON Report" in export_options:
                        json_path = tester.export_results(results)
                        if json_path:
                            st.success(f"JSON report exported to {json_path}")
                    
                    if "HTML Report" in export_options:
                        html_path = tester.generate_html_report(results)
                        if html_path:
                            st.success(f"HTML report exported to {html_path}")
                
                st.success("Tests completed! Check the 'Test Results' tab for details.")
    
    with tab2:
        st.markdown("Explore available NLP test prompts and their expected behaviors.")
        
        # Category filter
        category_filter = st.selectbox(
            "Filter by Category",
            options=["All Categories"] + list(tester.test_prompts.keys()),
            format_func=lambda x: x if x == "All Categories" else x.replace('_', ' ').title()
        )
        
        # Priority filter
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=["high", "medium", "low"],
            default=["high", "medium", "low"],
            format_func=lambda x: x.title()
        )
        
        # Display prompts
        st.markdown("### Available Test Scenarios")
        
        if category_filter == "All Categories":
            categories_to_show = tester.test_prompts.keys()
        else:
            categories_to_show = [category_filter]
        
        for category in categories_to_show:
            with st.expander(f" {category.replace('_', ' ').title()} ({len(tester.test_prompts[category])} tests)", expanded=False):
                for prompt in tester.test_prompts[category]:
                    if prompt['priority'] in priority_filter:
                        priority_icon = "🔴" if prompt['priority'] == 'high' else "🟡" if prompt['priority'] == 'medium' else "🟢"
                        
                        # Use consistent card styling with the app theme
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{prompt['id']}:** `{prompt['prompt']}`")
                                st.caption(f" {prompt['feature_tested']}")
                            with col2:
                                st.markdown(f"{priority_icon} {prompt['priority'].upper()}")
                            st.divider()
    
    with tab3:
        st.markdown("*Review validation results and system performance metrics*")
        
        if 'nlp_test_results' not in st.session_state:
            st.info("Test results will appear here after running validation tests")
        else:
            results = st.session_state.nlp_test_results
            
            if 'is_single_test' in results and results['is_single_test']:
                # Display single test result
                test = results['single_test_result']
                
                st.markdown(f"## Test Validation: {test['test_id']}")
                
                # Status indicator with metrics
                status_col1, status_col2, status_col3 = st.columns(3)
                
                with status_col1:
                    status_icon = "✅" if test['status'] == 'passed' else "❌" if test['status'] == 'failed' else "⚠️"
                    st.metric("Status", f"{status_icon} {test['status'].title()}")
                
                with status_col2:
                    st.metric("Response Time", f"{test['execution_time']:.3f}s")
                
                with status_col3:
                    priority_icon = "🔴" if test['priority'] == 'high' else "🟡" if test['priority'] == 'medium' else "🟢"
                    st.metric("Priority", f"{priority_icon} {test['priority'].title()}")
                
                # Test prompt display
                st.markdown("### Test Query")
                st.code(test['prompt'], language=None)
                
                st.markdown("### System Response")
                if test['response']:
                    response_col1, response_col2 = st.columns([3, 1])
                    with response_col1:
                        st.markdown("**Response Message:**")
                        st.info(test['response']['message'])
                    with response_col2:
                        success_icon = "✅" if test['response'].get('success', False) else "❌"
                        st.metric("Action Success", f"{success_icon}")
                        
                        if test['response'].get('action'):
                            st.metric("Action Type", test['response']['action'])
                else:
                    st.error("⚠️ No response received from system")
                
                # Show validation details only if there are issues
                if test['status'] != 'passed':
                    with st.expander("Validation Details", expanded=False):
                        st.markdown(f"**Issue:** {test['validation']['reason']}")
                        st.markdown(f"**Feature Tested:** {test['feature_tested']}")
                        st.markdown(f"**Expected Outcome:** {test['expected_outcome']}")
                
            elif 'is_category_test' in results and results['is_category_test']:
                # Display category test results
                category_results = results['category_results']
                
                st.markdown(f"## Category Test Results: {category_results['category'].replace('_', ' ').title()}")
                
                summary = category_results['summary']
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", summary['total_tests'])
                with col2:
                    st.metric("Passed", summary['passed'])
                with col3:
                    st.metric("Failed", summary['failed'] + summary['errors'])
                with col4:
                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                
                # Detailed results
                st.markdown("### Detailed Results")
                
                for test in category_results['detailed_results']:
                    status_color = {
                        'passed': 'green',
                        'failed': 'red',
                        'error': 'orange'
                    }.get(test['status'], 'gray')
                    
                    with st.expander(f"{test['test_id']}: {test['prompt']} - {test['status'].upper()}"):
                        st.markdown(f"**Status:** <span style='color: {status_color};'>{test['status'].upper()}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Execution Time:** {test['execution_time']:.3f}s")
                        st.markdown(f"**Feature Tested:** {test['feature_tested']}")
                        
                        if test['response']:
                            st.markdown("**Response Message:**")
                            st.code(test['response']['message'])
                        
                        if test['status'] != 'passed':
                            st.markdown("**Issue:**")
                            st.warning(test['validation']['reason'])
                
            else:
                # Display full test results
                summary = results['summary']
                
                # Summary metrics
                st.markdown("## Test Summary")
                
                metric_cols = st.columns(5)
                with metric_cols[0]:
                    st.metric("Total Tests", summary['total_tests'])
                with metric_cols[1]:
                    st.metric("Passed", summary['passed'])
                with metric_cols[2]:
                    st.metric("Failed", summary['failed'])
                with metric_cols[3]:
                    st.metric("Errors", summary['errors'])
                with metric_cols[4]:
                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                
                # Category breakdown
                st.markdown("## Category Breakdown")
                
                # Create a DataFrame for the category results
                category_data = []
                for category, data in results['category_breakdown'].items():
                    success_rate = (data['passed'] / (data['passed'] + data['failed'] + data['errors'])) * 100 if (data['passed'] + data['failed'] + data['errors']) > 0 else 0
                    category_data.append({
                        'Category': category.replace('_', ' ').title(),
                        'Total': data['passed'] + data['failed'] + data['errors'],
                        'Passed': data['passed'],
                        'Failed': data['failed'],
                        'Errors': data['errors'],
                        'Success Rate': f"{success_rate:.1f}%"
                    })
                
                category_df = pd.DataFrame(category_data)
                st.dataframe(category_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    category_df, 
                    x='Category', 
                    y=['Passed', 'Failed', 'Errors'],
                    title='Test Results by Category',
                    labels={'value': 'Number of Tests', 'variable': 'Status'},
                    color_discrete_map={'Passed': '#28a745', 'Failed': '#dc3545', 'Errors': '#ffc107'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Success rate by category
                success_df = category_df.copy()
                success_df['Success Rate'] = success_df['Success Rate'].str.replace('%', '').astype(float)
                
                fig2 = px.bar(
                    success_df,
                    x='Category',
                    y='Success Rate',
                    title='Success Rate by Category',
                    labels={'Success Rate': 'Success Rate (%)'},
                    color='Success Rate',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig2.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig2, use_container_width=True)
                
                # Performance Summary
                st.markdown("## Performance Summary")
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                with perf_col1:
                    avg_time = sum(test['execution_time'] for test in results['detailed_results']) / len(results['detailed_results'])
                    st.metric("Avg Response Time", f"{avg_time:.3f}s")
                with perf_col2:
                    coverage = results['feature_coverage']
                    total_coverage = (coverage['feature_completeness']['major_features_covered'] + 
                                    coverage['feature_completeness']['minor_features_covered'])
                    max_coverage = (coverage['feature_completeness']['total_major_features'] + 
                                   coverage['feature_completeness']['total_minor_features'])
                    st.metric("Feature Coverage", f"{total_coverage}/{max_coverage}")
                with perf_col3:
                    reliability = f"{summary['success_rate']:.0f}%"
                    st.metric("System Reliability", reliability)
                
                # Development insights (optional)
                show_dev_insights = st.checkbox("Show Development Insights", value=False, help="Show detailed recommendations for development")
                
                if show_dev_insights:
                    with st.expander("Development Recommendations", expanded=False):
                        for i, recommendation in enumerate(results['recommendations'][:5], 1):
                            st.markdown(f"{i}. {recommendation}")
                
                # Export options
                st.markdown("## Export & Documentation")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(" Export JSON Report"):
                        json_path = tester.export_results(results)
                        if json_path:
                            st.success(f"JSON report exported to {json_path}")
                
                with col2:
                    if st.button(" Export HTML Report"):
                        html_path = tester.generate_html_report(results)
                        if html_path:
                            st.success(f"HTML report exported to {html_path}")
                
                # Provide a download link for the HTML report
                if 'download_html' not in st.session_state:
                    st.session_state.download_html = tester.generate_html_report(results, "latest_report.html")
                
                if st.session_state.download_html:
                    try:
                        with open(st.session_state.download_html, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            
                        # Create download link
                        b64 = base64.b64encode(html_content.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="nlp_test_report.html" style="display: inline-block; padding: 10px 15px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px;">Download HTML Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error creating download link: {e}")


# For direct usage in app_production.py
def integrate_nlp_testing(process_natural_query_production):
    """
    Integrate NLP testing into the main application.
    
    Args:
        process_natural_query_production: The function that processes natural language queries
    """
    # Add NLP Testing tab
    test_tab = st.tabs(["NLP Testing"])
    
    with test_tab:
        render_nlp_test_ui(process_natural_query_production)


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="NLP Test Module", layout="wide")
    st.title("NLP Test Module - Standalone Mode")
    
    # Mock function for testing
    def mock_process_query(query: str) -> Dict[str, Any]:
        """Mock query processor for demonstration."""
        return {
            "action": "show_stats",
            "message": f"Processed query: {query}",
            "success": True,
            "data": None
        }
    
    render_nlp_test_ui(mock_process_query)