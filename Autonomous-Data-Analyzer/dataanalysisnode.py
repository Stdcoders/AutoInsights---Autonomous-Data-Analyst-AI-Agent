from state import ReportState
import pandas as pd

def data_cleaning_analysis_node(state: ReportState) -> ReportState:
    """
    This node performs data cleaning and basic analysis on the ingested datasets.
    It updates the state with cleaned data and analysis results.
    """

    if 'processed_tables' not in state:
        raise ValueError("No processed tables found in state.")

    processed_tables = state['processed_tables']

    # Initialize cleaned tables and analysis results
    state['cleaned_tables'] = {}
    state['analysis_results'] = {}

    for dataset_name, df in processed_tables.items():
        print(f"Cleaning dataset: {dataset_name}")

        # Example Cleaning: Drop rows with missing values
        cleaned_df = df.dropna()

        # Example Analysis: Get descriptive statistics
        analysis = cleaned_df.describe().to_dict()

        # Store results in state
        state['cleaned_tables'][dataset_name] = cleaned_df
        state['analysis_results'][dataset_name] = analysis

        print(f"Dataset '{dataset_name}' cleaned and analyzed.")

    return state
