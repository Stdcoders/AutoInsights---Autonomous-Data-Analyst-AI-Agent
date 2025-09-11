from state import ReportState
import pandas as pd
import numpy as np
from datetime import datetime
import re

def data_cleaning_analysis_node(state: ReportState) -> ReportState:
    """
    This node performs comprehensive data cleaning and basic analysis on the ingested datasets.
    It updates the state with cleaned data and analysis results.
    """

    if 'processed_tables' not in state:
        raise ValueError("No processed tables found in state.")

    processed_tables = state['processed_tables']

    # Initialize cleaned tables and analysis results
    state['cleaned_tables'] = {}
    state['analysis_results'] = {}
    state['cleaning_report'] = {}

    for dataset_name, df in processed_tables.items():
        print(f"Cleaning dataset: {dataset_name}")
        
        # Initialize cleaning report for this dataset
        cleaning_report = {
            'original_shape': df.shape,
            'missing_values_before': df.isnull().sum().to_dict(),
            'duplicates_removed': 0,
            'outliers_handled': {},
            'columns_cleaned': {},
            'data_type_changes': {}
        }
        
        # Create a copy for cleaning
        cleaned_df = df.copy()
        
        # 1. Handle Missing Values Intelligently
        cleaned_df = handle_missing_values(cleaned_df, cleaning_report)
        
        # 2. Remove Duplicates
        duplicates_before = cleaned_df.duplicated().sum()
        cleaned_df = cleaned_df.drop_duplicates()
        cleaning_report['duplicates_removed'] = duplicates_before
        
        # 3. Standardize Data Types
        cleaned_df = standardize_data_types(cleaned_df, cleaning_report)
        
        # 4. Clean String Columns
        cleaned_df = clean_string_columns(cleaned_df, cleaning_report)
        
        # 5. Handle Outliers
        cleaned_df = handle_outliers(cleaned_df, cleaning_report)
        
        # 6. Validate Data Ranges (domain-specific examples)
        cleaned_df = validate_data_ranges(cleaned_df, cleaning_report)
        
        # 7. Feature Engineering (optional)
        cleaned_df = create_new_features(cleaned_df, cleaning_report)
        
        # Update cleaning report
        cleaning_report['final_shape'] = cleaned_df.shape
        cleaning_report['missing_values_after'] = cleaned_df.isnull().sum().to_dict()
        cleaning_report['cleaning_summary'] = generate_cleaning_summary(cleaned_df, df)
        
        # Perform Analysis
        analysis_results = perform_comprehensive_analysis(cleaned_df)
        
        # Store results in state
        state['cleaned_tables'][dataset_name] = cleaned_df
        state['analysis_results'][dataset_name] = analysis_results
        state['cleaning_report'][dataset_name] = cleaning_report
        
        print(f"Dataset '{dataset_name}' cleaned and analyzed. Removed {duplicates_before} duplicates.")
        print(f"Final shape: {cleaned_df.shape}")

    return state

def handle_missing_values(df, report):
    """Handle missing values with column-specific strategies"""
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            original_missing = df[column].isnull().sum()
            
            if df[column].dtype in ['int64', 'float64']:
                # For numerical columns: use median (less sensitive to outliers)
                fill_value = df[column].median()
                df[column] = df[column].fillna(fill_value)
                report['columns_cleaned'][column] = f'Filled {original_missing} missing values with median: {fill_value}'
                
            elif df[column].dtype == 'object':
                # For categorical columns: use mode or 'Unknown'
                if not df[column].mode().empty:
                    fill_value = df[column].mode()[0]
                else:
                    fill_value = 'Unknown'
                df[column] = df[column].fillna(fill_value)
                report['columns_cleaned'][column] = f'Filled {original_missing} missing values with mode: {fill_value}'
                
            elif 'date' in column.lower() or 'time' in column.lower():
                # For date columns: use forward fill or specific date
                df[column] = df[column].fillna(method='ffill')
                report['columns_cleaned'][column] = f'Filled {original_missing} missing dates with forward fill'
    
    return df

def standardize_data_types(df, report):
    """Convert columns to appropriate data types"""
    for column in df.columns:
        original_dtype = str(df[column].dtype)
        
        # Try to convert to datetime if column name suggests date/time
        if any(keyword in column.lower() for keyword in ['date', 'time', 'timestamp']):
            try:
                df[column] = pd.to_datetime(df[column], errors='coerce')
                report['data_type_changes'][column] = f'Converted from {original_dtype} to datetime'
            except:
                pass
        
        # Convert obvious numeric columns stored as strings
        elif df[column].dtype == 'object' and df[column].str.isnumeric().all():
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                report['data_type_changes'][column] = f'Converted from {original_dtype} to numeric'
            except:
                pass
    
    return df

def clean_string_columns(df, report):
    """Clean and standardize string columns"""
    for column in df.select_dtypes(include=['object']).columns:
        # Remove extra whitespace
        df[column] = df[column].str.strip()
        
        # Standardize case (title case for names, upper for codes)
        if any(keyword in column.lower() for keyword in ['name', 'title', 'description']):
            df[column] = df[column].str.title()
        elif any(keyword in column.lower() for keyword in ['code', 'id', 'abbreviation']):
            df[column] = df[column].str.upper()
        
        # Remove special characters (keep only alphanumeric and spaces)
        df[column] = df[column].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)) if pd.notnull(x) else x)
        
        report['columns_cleaned'][column] = 'Standardized text formatting'
    
    return df

def handle_outliers(df, report, method='iqr', threshold=3):
    """Handle outliers using IQR or Z-score method"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for column in numerical_cols:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            if not outliers.empty:
                # Cap outliers instead of removing
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                report['outliers_handled'][column] = f'Capped {len(outliers)} outliers using IQR method'
    
    return df

def validate_data_ranges(df, report):
    """Validate data against expected ranges"""
    validation_rules = {
        'age': (0, 120),
        'salary': (0, 1000000),
        'rating': (1, 5),
        # Add more domain-specific rules here
    }
    
    for column, (min_val, max_val) in validation_rules.items():
        if column in df.columns:
            invalid_count = len(df[(df[column] < min_val) | (df[column] > max_val)])
            if invalid_count > 0:
                # Clip values to valid range
                df[column] = df[column].clip(min_val, max_val)
                report['columns_cleaned'][column] = f'Clipped {invalid_count} values to valid range [{min_val}, {max_val}]'
    
    return df

def create_new_features(df, report):
    """Create new features from existing data"""
    # Example: Extract year from date columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            new_col_name = f'{col}_year'
            df[new_col_name] = df[col].dt.year
            report['columns_cleaned'][new_col_name] = 'Created new feature from date'
    
    return df

def generate_cleaning_summary(cleaned_df, original_df):
    """Generate a summary of cleaning operations"""
    summary = {
        'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
        'columns_remaining': cleaned_df.shape[1],
        'total_missing_values_removed': original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum(),
        'data_quality_score': calculate_data_quality_score(cleaned_df)
    }
    return summary

def calculate_data_quality_score(df):
    """Calculate a simple data quality score (0-100)"""
    completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    uniqueness = 1 - (df.duplicated().sum() / df.shape[0])
    return round((completeness * 0.7 + uniqueness * 0.3) * 100, 2)

def perform_comprehensive_analysis(df):
    """Perform comprehensive statistical analysis"""
    analysis = {
        'descriptive_stats': df.describe().to_dict(),
        'correlation_matrix': df.select_dtypes(include=[np.number]).corr().to_dict(),
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'missing_values': df.isnull().sum().to_dict()
    }
    return analysis
