# ProjectEve Analytics Platform User Guide

The **ProjectEve Analytics Platform** is a powerful, Streamlit-based data analytics application designed for multi-file workflows, advanced statistical analysis, machine learning, SQL querying, natural language processing (NLP), and interactive visualizations. This user guide provides step-by-step instructions to set up, navigate, and use the platform effectively, whether you're a data analyst, researcher, or business user.

## Table of Contents

- Overview
- System Requirements
- Installation
- Getting Started
- Navigating the Application
  - AI Assistant (Tab 1)
  - Analytics (Tab 2)
  - Advanced Statistics (Tab 3)
  - Data Explorer (Tab 4)
  - Data Cleaning (Tab 5)
  - Anomaly Detection (Tab 6)
  - PowerBI-Style Visualizations (Tab 7)
  - Standard Visualizations (Tab 8)
  - ML Studio (Tab 9)
  - SQL Query (Tab 10)
  - NLP Testing (Tab 11)
  - Settings (Tab 12)
- Troubleshooting
- Author

## Overview

The ProjectEve Analytics Platform enables users to:

- Upload and manage multiple datasets (CSV, Excel, etc.).
- Perform statistical analysis, machine learning, and NLP.
- Execute SQL queries on datasets.
- Create interactive visualizations.
- Export and manage session data.

The application is organized into 12 tabs, each focusing on specific functionalities, with a user-friendly interface built using Streamlit.

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Browser**: Chrome, Firefox, or Edge for the Streamlit web interface
- **Dependencies**:
  - Required: `streamlit`, `pandas`, `numpy`, `psutil`
  - Optional (for full functionality): `sqlalchemy`, `nltk`, `spacy`, `textblob`, `scikit-learn`
  - Disk Space: \~500 MB for Python environment and dependencies
  - Memory: Minimum 4 GB RAM (8 GB recommended for large datasets)

## Installation

Follow these steps to set up the ProjectEve Analytics Platform locally.

1. **Install Python**:

   - Download and install Python 3.8+ from python.org.
   - Verify installation:

     ```bash
     python --version
     ```

2. **Clone or Download the Repository**:

   - If using GitHub, clone the repository:

     ```bash
     git clone https://github.com/AlexBiobelemo/Project-Eve-Premium
     cd projecteve
     ```
   - Alternatively, download the project files to a local directory (e.g., `C:\Users\YourName\PycharmProjects\ProjectEve`).

3. **Create a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Required Dependencies**:

   ```bash
   pip install streamlit pandas numpy psutil
   ```

5. **Install Optional Dependencies** (for SQL, NLP, and ML features):

   ```bash
   pip install sqlalchemy nltk spacy textblob scikit-learn
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords
   ```

6. **Run the Application**:

   ```bash
   streamlit run appbackup2.py
   ```

   - This opens the app in your default browser at `http://localhost:8501`.

## Getting Started

1. **Launch the App**:

   - Run the command above. The app will load in your browser.
   - If the browser doesn’t open automatically, navigate to `http://localhost:8501`.

2. **Upload a Dataset**:

   - Use the **System Controls** section (top of the app) to upload CSV or Excel files.
   - Files are stored in the session and accessible across tabs via the file tabs interface.

3. **Navigate Tabs**:

   - Use the sidebar or top navigation to switch between the 12 tabs (AI Assistant, Analytics, etc.).
   - Select a dataset from the file tabs to work with it in any tab.

4. **Configure Settings**:

   - Go to the **Settings** tab to adjust display options, performance settings, or export session data.

## Navigating the Application

### AI Assistant (Tab 1)

Analyze datasets using natural language queries and automated insights.

- **Steps**:

  1. Select a dataset from the file tabs.
  2. Enter a query (e.g., "Show average sales by region") in the text input.
  3. View automated insights or query results displayed as tables or charts.
  4. Use suggested queries for quick analysis.

- **Note**: Full natural language processing is under development. Basic insights are available.

### Analytics (Tab 2)

Perform quick statistical analysis and visualize data distributions.

- **Steps**:
  1. Select a dataset.
  2. View descriptive statistics (mean, median, min, max) for numeric columns.
  3. Generate histograms or box plots for distributions.
  4. Apply filters to explore subsets of the data.
  5. Check correlation matrices for numeric columns.

### Advanced Statistics (Tab 3)

Conduct advanced statistical operations, including secure conditional logic.

- **Steps**:

  1. Select a dataset and choose an operation (e.g., mean, conditional).
  2. For conditional operations:
     - Select a column (e.g., "Sales").
     - Enter a condition (e.g., `> 100`, `== 'value'`).
     - Specify true/false values (e.g., 1 for true, 0 for false).
  3. View results in a table or chart.
  4. Save results as a new dataset.

- **Tip**: Ensure conditions use valid operators (`>`, `<`, `==`, `!=`, `>=`, `<=`).

### Data Explorer (Tab 4)

Interactively explore datasets.

- **Steps**:
  1. Select a dataset.
  2. Use the interactive table to sort, filter, or search data.
  3. Select specific columns to display.
  4. Adjust the row limit for previews (default: 10 rows).

### Data Cleaning (Tab 5)

Clean and preprocess datasets.

- **Steps**:
  1. Select a dataset.
  2. Handle missing values:
     - Choose fill method (mean, median, mode, or custom value).
     - Or drop rows/columns with missing values.
  3. Convert column data types (e.g., string to numeric).
  4. Remove duplicate rows based on selected columns.
  5. Detect and handle outliers using statistical methods.
  6. Save the cleaned dataset to the session.

### Anomaly Detection (Tab 6)

Identify unusual patterns in data.

- **Steps**:

  1. Select a dataset.
  2. Choose detection method (statistical or ML-based, e.g., isolation forest).
  3. View anomalies in a scatter plot or table.
  4. Export anomalous data points as a new dataset.

- **Note**: ML-based detection requires `scikit-learn`.

### PowerBI-Style Visualizations (Tab 7)

Create advanced, interactive visualizations.

- **Steps**:
  1. Select a dataset and chart type (bar, line, area, pie, sunburst, heatmap).
  2. Customize colors, themes, and sizes.
  3. Interact with charts (zoom, pan, hover for tooltips).
  4. Save chart configurations for reuse.

### Standard Visualizations (Tab 8)

Generate common visualizations with simple setup.

- **Steps**:

  1. Select a dataset and chart type (scatter, bar, line, histogram, box plot, violin plot).
  2. Choose columns for the x-axis, y-axis, or categories.
  3. Export charts as PNG or SVG files.

- **Note**: Chart recreation is under development.

### ML Studio (Tab 9)

Train and evaluate machine learning models.

- **Steps**:

  1. Select a dataset and model type (classification, regression, clustering).
  2. Choose an algorithm (e.g., Random Forest, SVM, K-Means).
  3. Adjust basic hyperparameters (e.g., number of trees).
  4. Train the model and view metrics (accuracy, F1-score, RMSE).
  5. Save the trained model for reuse.

- **Note**: Requires `scikit-learn`.

### SQL Query (Tab 10)

Execute SQL queries on datasets using an in-memory SQLite database.

- **Steps**:

  1. Select a dataset. The app creates a SQLite table automatically.
  2. View the table name and columns in the "Available Table & Columns" expander.
  3. Enter a SQL query in the text area (e.g., `SELECT * FROM table_name LIMIT 10`).
  4. Use quick query buttons:
     - **Show All**: View all rows.
     - **Count Rows**: Count total rows.
     - **Basic Stats**: Calculate average, max, min for a numeric column.
     - **Sample Data**: View first 5 rows.
  5. Click **Execute Query** to run the query.
  6. Save results as a new dataset using the **Save Result** button.

- **Tips**:

  - Use backticks for column names with spaces (e.g., `column name`).
  - Check debug info for table names and connection status.
  - Refresh the database if errors occur (e.g., "no such table").

### NLP Testing (Tab 11)

Analyze text data with NLP tools.

- **Steps**:

  1. Select a dataset with text columns.
  2. Choose a text column and analysis type:
     - **Sentiment Analysis**: View sentiment scores (requires `textblob`).
     - **Word Frequency**: See top words (excludes stopwords).
     - **Named Entity Recognition**: Identify entities (requires `spacy`).
     - **Topic Modeling**: Not yet available.
  3. Click **Run Analysis** to view results (tables or charts).
  4. If advanced NLP is unavailable, basic word frequency analysis is provided.

- **Note**: Install `nltk`, `spacy`, and `textblob` for full functionality:

  ```bash
  pip install nltk spacy textblob
  python -m spacy download en_core_web_sm
  python -m nltk.downloader punkt stopwords
  ```

### Settings (Tab 12)

Configure the application and manage data.

- **Steps**:

  1. **Display Settings**:
     - Set **Max Rows to Display** (10–10,000).
     - Set **Decimal Places** for numbers (1–10).
     - Enable **Dark Mode** for the UI.
  2. **Performance Settings**:
     - Enable **Memory Usage Warnings** to monitor resource usage.
     - Enable **Auto-Save Query Results** to save SQL results automatically.
     - Enable **Debug Mode** for detailed logs.
  3. **Data Management**:
     - **Export Current Session**: Click **Export All Data** to download all datasets as a ZIP file of CSVs.
     - **Clear Session Data**: Click **Clear All Data** and confirm to reset the session.
  4. View **Application Information** for platform details.

- **Tip**: Ensure sufficient memory (1 GB free) before exporting large datasets.

## Troubleshooting

- **SQL Errors** (e.g., "no such table"):

  - Click **Refresh Database** in the SQL Query tab.
  - Install `sqlalchemy`:

    ```bash
    pip install sqlalchemy
    ```
  - Check the table name in the "Available Table & Columns" expander.

- **NLP Errors** (e.g., `KeyError: 'nlp_test_integration'`):

  - Ensure `nlp_test_integration.py` is in the project directory.
  - Install NLP dependencies (see NLP Testing section).
  - Restart the app after installation.

- **Memory Issues**:

  - Enable **Memory Usage Warnings** in the Settings tab.
  - Reduce dataset size or limit query results (10,000 rows max).

- **Syntax Errors**:

  - If you see `SyntaxError: expected 'except' or 'finally' block`, ensure the latest `appbackup2.py` (with fixed `main_production()`) is used.
  - Check logs in `logs/enterprise_app_*.log` or `logs/enterprise_errors_*.log`.

## Author

Alex Alagoa Biobelemo

LinkedIn

---

*Last Updated: September 24, 2025*