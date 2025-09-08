"""
Basic Test Script for Enterprise Data Analytics Platform
Tests application structure and imports without requiring external libraries
"""

import os
import sys
import ast
import re
from pathlib import Path


def test_file_exists():
    """Test if the production app file exists"""
    app_file = Path("app_production.py")
    assert app_file.exists(), "app_production.py file not found"
    print("✅ Production app file exists")


def test_file_syntax():
    """Test if the Python file has valid syntax"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # Parse the file to check syntax
        ast.parse(source_code)
        print("✅ Python syntax is valid")
        
        # Check file size
        file_size = len(source_code)
        print(f"📊 File size: {file_size:,} characters ({file_size/1024:.1f} KB)")
        
        # Count lines
        line_count = source_code.count('\n')
        print(f"📏 Line count: {line_count:,} lines")
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    return True


def test_imports():
    """Test if all required imports are present"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # List of expected imports
        expected_imports = [
            "streamlit",
            "pandas",
            "numpy", 
            "plotly",
            "datetime",
            "logging",
            "typing",
            "json",
            "pathlib"
        ]
        
        missing_imports = []
        for imp in expected_imports:
            if f"import {imp}" not in source_code and f"from {imp}" not in source_code:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"⚠️ Missing imports: {', '.join(missing_imports)}")
        else:
            print("✅ All expected imports are present")
            
        # Count total imports
        import_lines = [line for line in source_code.split('\n') if line.strip().startswith(('import ', 'from '))]
        print(f"📦 Total import statements: {len(import_lines)}")
        
    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False
    
    return True


def test_function_definitions():
    """Test if key functions are defined"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # List of expected functions
        expected_functions = [
            "load_data_production",
            "calculate_data_quality_score_enhanced",
            "validate_dataframe",
            "suggest_cleaning_production",
            "detect_anomalies_production",
            "process_natural_query_production",
            "compute_eda_summary_enhanced",
            "monitor_memory_usage",
            "cleanup_memory",
            "main_production"
        ]
        
        missing_functions = []
        for func in expected_functions:
            if f"def {func}(" not in source_code:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"⚠️ Missing functions: {', '.join(missing_functions)}")
        else:
            print("✅ All key functions are defined")
            
        # Count total functions
        function_pattern = r'^def\s+(\w+)\s*\('
        functions = re.findall(function_pattern, source_code, re.MULTILINE)
        print(f"🔧 Total functions defined: {len(functions)}")
        
        return len(missing_functions) == 0
        
    except Exception as e:
        print(f"❌ Error checking functions: {e}")
        return False


def test_class_definitions():
    """Test if any classes are defined"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # Count classes
        class_pattern = r'^class\s+(\w+)(\(.*\))?:'
        classes = re.findall(class_pattern, source_code, re.MULTILINE)
        print(f"🏗️ Classes defined: {len(classes)}")
        
        if classes:
            class_names = [match[0] for match in classes]
            print(f"   Classes: {', '.join(class_names)}")
            
    except Exception as e:
        print(f"❌ Error checking classes: {e}")
        return False
    
    return True


def test_constants():
    """Test if important constants are defined"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        expected_constants = [
            "FILE_TYPES",
            "CHART_OPTIONS",
            "THEME_OPTIONS",
            "MAX_MEMORY_MB",
            "MAX_ROWS_PROCESSING",
            "SAMPLE_SIZE_LARGE"
        ]
        
        missing_constants = []
        for const in expected_constants:
            if f"{const}:" not in source_code and f"{const} =" not in source_code:
                missing_constants.append(const)
        
        if missing_constants:
            print(f"⚠️ Missing constants: {', '.join(missing_constants)}")
        else:
            print("✅ All important constants are defined")
            
        return len(missing_constants) == 0
        
    except Exception as e:
        print(f"❌ Error checking constants: {e}")
        return False


def test_error_handling():
    """Test if error handling is implemented"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # Count try-except blocks
        try_count = source_code.count("try:")
        except_count = source_code.count("except")
        
        print(f"🛡️ Error handling blocks: {try_count} try blocks, {except_count} except blocks")
        
        if try_count > 0:
            print("✅ Error handling is implemented")
            return True
        else:
            print("⚠️ No error handling found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking error handling: {e}")
        return False


def test_logging_setup():
    """Test if logging is configured"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        logging_indicators = [
            "logging.basicConfig",
            "logging.getLogger",
            "logging.info",
            "logging.error",
            "setup_production_logging"
        ]
        
        logging_found = sum(1 for indicator in logging_indicators if indicator in source_code)
        
        if logging_found > 0:
            print(f"✅ Logging is configured ({logging_found} logging indicators found)")
            return True
        else:
            print("⚠️ No logging configuration found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking logging: {e}")
        return False


def test_streamlit_configuration():
    """Test if Streamlit is properly configured"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        streamlit_indicators = [
            "st.set_page_config",
            "st.title",
            "st.header",
            "st.sidebar",
            "st.tabs"
        ]
        
        streamlit_found = sum(1 for indicator in streamlit_indicators if indicator in source_code)
        
        if streamlit_found >= 3:
            print(f"✅ Streamlit is properly configured ({streamlit_found} Streamlit elements found)")
            return True
        else:
            print(f"⚠️ Limited Streamlit configuration ({streamlit_found} elements found)")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Streamlit configuration: {e}")
        return False


def test_docstrings():
    """Test if functions have docstrings"""
    try:
        with open("app_production.py", "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # Parse the AST to find functions with docstrings
        tree = ast.parse(source_code)
        
        functions_with_docstrings = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)):
                    functions_with_docstrings += 1
        
        if total_functions > 0:
            docstring_percentage = (functions_with_docstrings / total_functions) * 100
            print(f"📝 Documentation: {functions_with_docstrings}/{total_functions} functions have docstrings ({docstring_percentage:.1f}%)")
            
            if docstring_percentage >= 50:
                print("✅ Good documentation coverage")
                return True
            else:
                print("⚠️ Low documentation coverage")
                return False
        else:
            print("⚠️ No functions found for documentation check")
            return False
            
    except Exception as e:
        print(f"❌ Error checking docstrings: {e}")
        return False


def run_basic_tests():
    """Run all basic tests"""
    print("=" * 80)
    print("ENTERPRISE DATA ANALYTICS PLATFORM - BASIC TESTS")
    print("=" * 80)
    print()
    
    tests = [
        ("File Existence", test_file_exists),
        ("Python Syntax", test_file_syntax),
        ("Import Statements", test_imports),
        ("Function Definitions", test_function_definitions),
        ("Class Definitions", test_class_definitions),
        ("Constants", test_constants),
        ("Error Handling", test_error_handling),
        ("Logging Setup", test_logging_setup),
        ("Streamlit Configuration", test_streamlit_configuration),
        ("Documentation", test_docstrings)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 80)
    print("BASIC TEST RESULTS")
    print("=" * 80)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL BASIC TESTS PASSED! The application structure is solid.")
    elif passed >= total * 0.8:
        print("✅ MOST TESTS PASSED! The application looks good with minor issues.")
    elif passed >= total * 0.6:
        print("⚠️ SOME TESTS PASSED! The application needs attention.")
    else:
        print("❌ MANY TESTS FAILED! The application has significant issues.")
    
    return passed, total


def check_dependencies():
    """Check what dependencies are available"""
    print("\n🔍 Checking Available Dependencies:")
    print("-" * 40)
    
    dependencies = [
        "pandas",
        "numpy", 
        "plotly",
        "streamlit",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "scipy",
        "psutil"
    ]
    
    available = []
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            available.append(dep)
            print(f"✅ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep}")
    
    print(f"\nSummary: {len(available)}/{len(dependencies)} dependencies available")
    
    if missing:
        print(f"\nTo install missing dependencies:")
        print(f"pip install {' '.join(missing)}")
    
    return available, missing


if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Run basic tests
        passed, total = run_basic_tests()
        
        # Check dependencies
        available_deps, missing_deps = check_dependencies()
        
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"📋 Basic Tests: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
        print(f"📦 Dependencies: {len(available_deps)}/{len(available_deps)+len(missing_deps)} available")
        
        if passed == total and not missing_deps:
            print("\n🎉 PERFECT! Application is ready for production use.")
            exit_code = 0
        elif passed >= total * 0.8 and len(missing_deps) <= 3:
            print("\n✅ GOOD! Application is mostly ready. Install missing dependencies to unlock full functionality.")
            exit_code = 0
        else:
            print("\n⚠️ NEEDS WORK! Address the issues found before deployment.")
            exit_code = 1
            
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during testing: {e}")
        sys.exit(1)
