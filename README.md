# Enterprise Data Analytics Platform - Production Ready

A comprehensive, AI-powered data analytics platform built with Streamlit, featuring advanced machine learning, natural language query processing, and enterprise-grade performance optimization.

## Features

### Core Analytics
- **AI-Powered Data Analysis**: Natural language queries for data exploration
- **Advanced Machine Learning**: Random Forest, Neural Networks, and AutoML comparison
- **Intelligent Data Cleaning**: Automated suggestions with manual overrides
- **Anomaly Detection**: Multiple algorithms (Z-Score, IQR, Isolation Forest)
- **Interactive Visualizations**: 10+ chart types with publication-ready quality

### Production Features
- **Memory Optimization**: Smart sampling for large datasets (4GB RAM optimized)
- **Error Recovery**: Comprehensive exception handling with recovery options
- **Performance Monitoring**: Real-time memory and CPU usage tracking
- **Enterprise Logging**: Detailed logging for debugging and audit trails
- **Data Validation**: Comprehensive data quality scoring and validation

### User Experience
- **Accessibility Themes**: High contrast, colorblind friendly options
- **Responsive Design**: Works on various screen sizes
- **Multi-file Processing**: Batch upload and processing capabilities
- **Export Features**: Data, charts, and reports in multiple formats

## System Requirements

- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for dependencies
- **OS**: Windows, macOS, or Linux

## Installation

### 1. Clone or Download
```bash
# If you have git
git clone https://github.com/AlexBiobelemo/Project-Eve-Premium

# Or download the files directly
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install pandas numpy plotly streamlit scikit-learn seaborn matplotlib scipy psutil openpyxl

# Or install from requirements (if available)
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Run basic tests to check everything is working
python test_basic.py
```

## Quick Start```

### Run the Production Version (Recommended)
```bash
streamlit run app_production.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### Getting Started
1. **Upload Data**: Click "Upload Data Files" or use "Load Demo Data"
2. **Explore**: Use the AI Assistant tab for natural language queries
3. **Analyze**: Review insights in the Analytics tab
4. **Clean**: Apply cleaning suggestions in the Data Cleaning tab
5. **Visualize**: Create charts in the Visualizations tab
6. **Model**: Train ML models in the ML Studio tab

### Natural Language Queries
Ask questions in plain English:
- "Show stats for sales_amount"
- "Clean missing data in customer_satisfaction"
- "Filter region equals North America"
- "Create scatter plot"
- "Find outliers in profit_margin"

### Data Cleaning
- **Automatic**: Review AI suggestions and apply with one click
- **Manual**: Use custom cleaning methods for specific columns
- **Batch**: Apply high-priority suggestions all at once

### Machine Learning
- **Clustering**: Automatic K-means with optimal cluster detection
- **Supervised Learning**: Random Forest and Neural Network models
- **AutoML**: Compare multiple models automatically
- **Feature Engineering**: Automated feature creation

## Configuration

### Performance Settings
- **Performance Mode**: Optimized for speed with aggressive sampling
- **Quality Mode**: Uses full datasets when possible
- **Balanced Mode**: Default setting balancing speed and accuracy

### Memory Management
- **Auto-sampling**: Large datasets automatically sampled for performance
- **Cache Management**: Results cached with automatic cleanup
- **Memory Monitoring**: Real-time usage tracking with warnings

### Accessibility
- **Themes**: Light, Dark, High Contrast, Colorblind Friendly
- **Navigation**: Keyboard-friendly interface
- **Performance**: Optimized for low-spec hardware

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError" when running**
```bash
pip install [missing-package-name]
```

**2. High memory usage**
- Enable "Performance Mode" in Settings
- Use smaller data samples
- Clear cache regularly

**3. Slow performance**
- Check memory usage in sidebar
- Use "Clean Memory" button
- Restart the application

**4. Data loading failures**
- Check file format (CSV, Excel, JSON supported)
- Verify file encoding
- Try smaller file sizes first

### Recovery Options
If the application becomes unresponsive:
1. **Soft Reload**: Refreshes the interface
2. **Clear Cache**: Removes cached data
3. **Reset Data**: Clears all loaded datasets
4. **Emergency Reset**: Full application reset

## Testing

### Basic Tests (No dependencies required)
```bash
python test_basic.py
```

### Full Test Suite (Requires dependencies)
```bash
python test_app_production.py
```

### Test Coverage
- Data quality and validation
- Data loading and processing
- Natural language processing
- Anomaly detection algorithms
- Machine learning functionality
- Performance monitoring
- Error handling

## Performance Benchmarks

### Processing Speeds (Typical)
- **Data Loading**: 1-5 MB/s depending on format
- **Quality Analysis**: 100K rows in < 5 seconds
- **ML Training**: Sub-20 second model training
- **Anomaly Detection**: 50K rows in < 3 seconds

### Memory Usage
- **Base Application**: ~50-100MB
- **With Data (100K rows)**: ~200-300MB
- **ML Training**: +100-200MB temporary
- **Large Dataset (1M rows)**: Auto-sampled to stay under 1GB

## Security Features

- **Input Validation**: All user inputs validated and sanitized
- **File Type Validation**: Only approved file types accepted
- **Size Limits**: Files limited to reasonable sizes
- **Error Isolation**: Errors contained without affecting other functions
- **Logging**: Security events logged for audit

## Deployment

### Local Development
```bash
streamlit run app_production.py --server.port 8501
```

### Production Deployment
```bash
streamlit run app_production.py --server.headless true --server.port 8080
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app_production.py", "--server.headless", "true"]
```

## Changelog

### Premium (Production)
- Complete error handling and recovery systems
- Performance monitoring and memory management
- Enhanced data validation and quality scoring
- Production-grade logging and debugging
- Accessibility improvements and themes
- Comprehensive test suite
- Security enhancements


## Contributing

1. Test your changes with `python test_basic.py`
2. Ensure all functions have docstrings
3. Add error handling for new features
4. Update tests for new functionality
5. Follow the existing code style

## License

This project is provided as-is for educational and development purposes.

## Support

### Getting Help
1. Check the troubleshooting section above
2. Review the error logs in the `logs/` directory
3. Use the diagnostic information in Settings > System Information
4. Try the recovery options in the error handler

### Error Reporting
When reporting issues, include:
- Error message and stack trace
- System information (OS, Python version, RAM)
- Data characteristics (size, format, columns)
- Steps to reproduce

---

## Next Steps

After installation:

1. **Start with Demo Data** - Click "Load Demo Data" to try all features
2. **Explore AI Assistant** - Ask natural language questions about your data
3. **Try Data Cleaning** - See automated suggestions for data improvement
4. **Create Visualizations** - Build interactive charts and dashboards
5. **Train ML Models** - Experience AutoML with model comparison
6. **Monitor Performance** - Check system resources in real-time


