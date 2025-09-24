# ProjectEve Analytics Platform Features

The **ProjectEve Analytics Platform** is a production-grade data analytics application built with Streamlit, designed for multi-file workflows, advanced statistical operations, visualizations, machine learning, SQL querying, and natural language processing (NLP). This document outlines its features, organized by the application's 12 tabs.

## Table of Contents

- [AI Assistant (Tab 1)](#ai-assistant-tab-1)
- [Analytics (Tab 2)](#analytics-tab-2)
- [Advanced Statistics (Tab 3)](#advanced-statistics-tab-3)
- [Data Explorer (Tab 4)](#data-explorer-tab-4)
- [Data Cleaning (Tab 5)](#data-cleaning-tab-5)
- [Anomaly Detection (Tab 6)](#anomaly-detection-tab-6)
- [PowerBI-Style Visualizations (Tab 7)](#powerbi-style-visualizations-tab-7)
- [Standard Visualizations (Tab 8)](#standard-visualizations-tab-8)
- [ML Studio (Tab 9)](#ml-studio-tab-9)
- [SQL Query (Tab 10)](#sql-query-tab-10)
- [NLP Testing (Tab 11)](#nlp-testing-tab-11)
- [Settings (Tab 12)](#settings-tab-12)
- [General Features](#general-features)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

## AI Assistant (Tab 1)

The AI Assistant tab leverages AI to provide insights and process natural language queries.

- **Natural Language Query Processing**: Users can input queries like "What is the average sales by region?" which are processed by `process_natural_query_production` to generate insights or SQL queries.
- **Automated Insights**: Generates summaries, trends, and key metrics for the selected dataset.
- **Query Suggestions**: Provides suggested queries based on dataset columns and data types.
- **Error Handling**: Displays user-friendly error messages for invalid queries with fallback to basic analysis.

## Analytics (Tab 2)

The Analytics tab offers quick statistical insights and visualizations.

- **Descriptive Statistics**: Shows mean, median, min, max, and other statistics for numeric columns.
- **Data Distribution**: Visualizes distributions using histograms or box plots.
- **Correlation Analysis**: Displays correlation matrices for numeric columns.
- **Quick Filters**: Enables filtering by column values or ranges for rapid exploration.

## Advanced Statistics (Tab 3)

The Advanced Statistics tab supports sophisticated statistical operations with a secure conditional operation.

- **Statistical Operations**:
  - Computes mean, median, mode, standard deviation, variance, percentiles, and z-scores.
  - Conditional operations (e.g., "if column > value, set to X, else Y") using `np.where` instead of `eval()` for security.
- **Input Validation**: Ensures conditional operation inputs follow formats like `> 100` or `== 'value'`.
- **Result Visualization**: Presents results in tables or charts, with options to save as new datasets.
- **Error Feedback**: Provides detailed error messages for invalid inputs.

## Data Explorer (Tab 4)

The Data Explorer tab enables interactive dataset exploration.

- **Interactive Data Table**: Displays DataFrames with sorting, filtering, and pagination.
- **Column Selection**: Allows users to select or hide specific columns.
- **Search Functionality**: Supports searching for values or patterns within the dataset.
- **Data Preview**: Shows a sample (e.g., first 10 rows) with customizable row limits.

## Data Cleaning (Tab 5)

The Data Cleaning tab provides tools for preprocessing datasets.

- **Missing Value Handling**: Fills missing values (mean, median, mode, or custom) or drops rows/columns.
- **Data Type Conversion**: Converts columns to appropriate types (e.g., string to numeric, datetime).
- **Duplicate Removal**: Removes duplicate rows based on selected columns.
- **Outlier Handling**: Detects and optionally removes or flags outliers using statistical methods.
- **Save Cleaned Data**: Saves cleaned datasets to `st.session_state.dfs`.

## Anomaly Detection (Tab 6)

The Anomaly Detection tab identifies unusual data patterns.

- **Statistical Anomaly Detection**: Uses z-scores or IQR for outlier detection in numeric columns.
- **ML-Based Detection**: Supports isolation forest or DBSCAN algorithms (requires scikit-learn).
- **Visualization**: Highlights anomalies in scatter plots or tables.
- **Export Anomalies**: Saves anomalous data points as a separate dataset.

## PowerBI-Style Visualizations (Tab 7)

The PowerBI-Style Visualizations tab offers advanced, interactive visualizations.

- **Chart Types**:
  - Bar, line, area, pie, and sunburst charts.
  - Heatmap matrices for correlations.
- **Interactive Features**: Supports zoom, pan, and hover tooltips.
- **Customization**: Allows configuration of themes, colors, and chart sizes.
- **Save Visualizations**: Stores chart configurations in `st.session_state.chart_configs`.

## Standard Visualizations (Tab 8)

The Standard Visualizations tab provides common visualizations with simple setup.

- **Chart Types**:
  - Scatter plots, bar charts, line charts, histograms, box plots, and violin plots.
- **Quick Setup**: Easy selection of columns and chart types.
- **Export Options**: Saves charts as PNG or SVG files.
- **Chart Recreation**: Placeholder for recreating saved charts (to be implemented).

## ML Studio (Tab 9)

The ML Studio tab facilitates machine learning tasks.

- **Model Training**: Supports classification, regression, and clustering (e.g., Random Forest, SVM, K-Means).
- **Hyperparameter Tuning**: Basic options for tuning model parameters.
- **Model Evaluation**: Displays metrics (accuracy, F1-score, RMSE) and confusion matrices.
- **Save Models**: Stores trained models in `st.session_state.trained_models`.

## SQL Query (Tab 10)

The SQL Query tab enables SQL queries on datasets using an in-memory SQLite database.

- **Database Creation**: Creates SQLite tables from DataFrames via `create_database_from_dataframes`.
- **Query Interface**: Text area for custom SQL queries with table name auto-correction.
- **Quick Queries**:
  - Show All: `SELECT * FROM table_name`
  - Count Rows: `SELECT COUNT(*) FROM table_name`
  - Basic Stats: `SELECT AVG(column), MAX(column), MIN(column) FROM table_name`
  - Sample Data: `SELECT * FROM table_name LIMIT 5`
- **Debug Information**: Shows table names, row counts, and connection status.
- **Result Handling**: Limits results to 10,000 rows to manage memory.
- **Save Results**: Saves query results to `st.session_state.dfs`.
- **Error Handling**: Provides troubleshooting for errors like "no such table" or "no such column".

## NLP Testing (Tab 11)

The NLP Testing tab offers text analysis tools with a fallback for basic features.

- **Advanced NLP Features** (requires `nlp_test_integration` module):
  - Sentiment Analysis: Uses TextBlob for text sentiment scoring.
  - Word Frequency: Counts words (excluding stopwords) using NLTK.
  - Named Entity Recognition (NER): Identifies entities using spaCy.
  - Topic Modeling: Placeholder for future implementation.
- **Basic NLP Features** (fallback):
  - Word frequency analysis using `Counter` for text columns.
  - Displays top words in a bar chart.
- **Dependency Guidance**: Instructs users to install NLTK, spaCy, and TextBlob.
- **Performance**: Limits analysis to 1,000 rows for efficiency.

## Settings (Tab 12)

The Settings tab manages application configuration and data.

- **Display Settings**:
  - Max Rows to Display: Sets table row limits (10–10,000).
  - Decimal Places: Configures numeric precision (1–10).
  - Dark Mode: Toggles dark mode UI.
- **Performance Settings**:
  - Memory Usage Warnings: Enables/disables memory usage alerts.
  - Auto-Save Query Results: Saves SQL query results automatically.
  - Debug Mode: Enables detailed debugging output.
- **Data Management**:
  - Export Current Session: Exports all DataFrames in `st.session_state.dfs` as a ZIP of CSVs, with memory checks.
  - Clear Session Data: Clears all datasets with a confirmation checkbox.
- **Application Information**: Displays platform details and author information.

## General Features

These features enhance usability and robustness across all tabs.

- **Multi-File Workflow**:
  - Manages multiple datasets in `st.session_state.dfs`.
  - Tab-based interface for switching datasets (`st.session_state.active_file_tabs`).
  - Supports CSV, Excel, and other file uploads.
- **Session State Management**:
  - Persists DataFrames, chart configurations, models, and SQL table names.
  - Ensures seamless state updates across tabs.
- **Error Handling**:
  - Wraps application in a `try-except` block for error catching.
  - Handles type errors, syntax errors, and SQL errors with user-friendly messages.
  - Logs errors to `logs/enterprise_app_*.log` and `logs/enterprise_errors_*.log` when enabled.
- **Performance Monitoring**:
  - Uses `psutil` for memory usage checks during export and SQL queries.
  - Limits large operations to 10,000 rows.
- **Custom Styling**:
  - Applies professional CSS via `load_custom_css`.
  - Consistent UI with headers, expanders, and info panels.
- **AI-Powered Insights**:
  - Integrated for automated analysis and query suggestions.
  - Supports natural language queries.
- **Scalability**:
  - Uses in-memory SQLite for SQL queries to minimize disk I/O.

## Future Enhancements

- **Chart Recreation**: Implement chart recreation in Standard Visualizations.
- **Advanced NLP**: Add topic modeling and word cloud generation in NLP Testing.
- **Session Import**: Enable importing session data from ZIP files.
- **Persistent Settings**: Save settings to a config file.
- **Undo Functionality**: Add undo/redo for data cleaning and other operations.

## Author

Alex Alagoa Biobelemo

[LinkedIn](https://www.linkedin.com/in/alex-alagoa-biobelemo)