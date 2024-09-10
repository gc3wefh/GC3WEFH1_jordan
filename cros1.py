import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import geopandas as gpd

# Title of the app
st.title("Dataset Analysis with Streamlit")

# Define dataset paths
datasets = {
    'indicator_df': 'jordan_indicadors.csv',
    'governorate_csv': 'dataset/Average Household Size in Jordan/governorate.csv',
    'country_csv': 'dataset/Average Household Size in Jordan/country.csv',
    'para_csv': 'dataset/Average Household Size in Jordan/para.csv',
    'spi_data': '170/SPI_JMD_data_corrected_long_format.csv',
    'healthcare_facilities_csv': 'dataset/Healthcare Facilities in Jordan/healthcare.csv',
    'governorates_jordan_csv': 'dataset/Jordan Health/Governorates_jordan.csv',
    'hospitals_csv': 'dataset/Jordan Health/Hospitals.csv',
    'governorate_boundaries_csv': 'dataset/Jordan Boundaries/governorate.csv',
    'country_boundaries_csv': 'dataset/Jordan Boundaries/country.csv',
    'health_activities_csv': 'dataset/Jordan Health Map/JCAP.csv',
    'country_schema_csv': 'datasets/country_schema.csv',
    'admin_shapefile': 'jordan_admin_regions.shp'
}

# Load dataset functions
def load_csv_dataset(path):
    return pd.read_csv(path)

def load_geodataframe(path):
    return gpd.read_file(path)

# Dropdown to select multiple datasets from predefined datasets
st.subheader("Select datasets for analysis")
dataset_choices = st.multiselect("Choose Datasets", list(datasets.keys()), default=[])

# Upload a custom CSV file
uploaded_files = st.file_uploader("Or upload CSV files", type="csv", accept_multiple_files=True)

# Initialize variables to store datasets
dfs = {}

# Load selected datasets
if dataset_choices:
    for choice in dataset_choices:
        dfs[choice] = load_csv_dataset(datasets[choice])

# Load uploaded datasets
if uploaded_files:
    for file in uploaded_files:
        dfs[file.name] = pd.read_csv(file)

# Display dataset columns and allow variable selection across datasets
if dfs:
    all_columns = {}
    
    # Display all variables in each selected dataset
    for dataset_name, df in dfs.items():
        st.subheader(f"All Variables in {dataset_name}")
        columns = df.columns.tolist()
        all_columns[dataset_name] = columns
        st.write(columns)

    # Allow user to select variables across datasets
    st.subheader("Select variables across datasets")
    selected_columns = {}

    for dataset_name, columns in all_columns.items():
        selected_vars = st.multiselect(f"Select variables from {dataset_name}", columns)
        if selected_vars:
            selected_columns[dataset_name] = selected_vars

    # Merge the selected columns into a single DataFrame for analysis
    if selected_columns:
        merged_df = pd.DataFrame()

        for dataset_name, selected_vars in selected_columns.items():
            df_selected = dfs[dataset_name][selected_vars]
            merged_df = pd.concat([merged_df, df_selected], axis=1)

        st.write("Merged DataFrame from Selected Variables:")
        st.write(merged_df)

        # Proceed with analysis (scatter matrix, descriptive statistics, etc.)
        numeric_vars = [col for col in merged_df.columns if merged_df[col].dtype in [np.float64, np.int64]]

        if numeric_vars:
            st.subheader("Scatter Matrix")
            selected_vars = st.multiselect("Select up to 5 variables for scatter matrix", numeric_vars, numeric_vars[:min(5, len(numeric_vars))])

            if selected_vars:
                if len(selected_vars) > 1:  # Ensure at least 2 variables are selected for scatter matrix
                    scatter_matrix_df = merged_df[selected_vars].dropna()  # Drop missing values
                    
                    # Ensure there are at least 2 valid elements in the dataset
                    if scatter_matrix_df.shape[0] > 1:
                        scatter_matrix_fig = pd.plotting.scatter_matrix(scatter_matrix_df, alpha=0.2, figsize=(10, 10), diagonal='kde')
                        for ax in scatter_matrix_fig.ravel():
                            ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=90)
                            ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)
                        st.pyplot(plt)
                    else:
                        st.error("Not enough valid data points. Please ensure there are multiple valid rows of data.")
                else:
                    st.error("Please select at least two variables for the scatter matrix.")

            # Scatterplot with regression line
            st.subheader("Scatterplot with Regression Line")
            x_var = st.selectbox("Select X variable", numeric_vars)
            y_var = st.selectbox("Select Y variable", numeric_vars)

            if x_var and y_var:
                if x_var != y_var:
                    fig, ax = plt.subplots()
                    sb.regplot(x=merged_df[x_var], y=merged_df[y_var], scatter_kws={"s": 20, "color": "red", "alpha": 0.2}, ax=ax)
                    ax.set(xlabel=x_var, ylabel=y_var)
                    st.pyplot(fig)
                else:
                    st.error("X and Y variables must be different!")

            # Descriptive statistics
            st.subheader("Descriptive Statistics")
            selected_stat_var = st.selectbox("Select a variable to calculate statistics", numeric_vars)

            if selected_stat_var:
                var = merged_df[selected_stat_var].dropna()  # Drop missing values
                if len(var) > 1:  # Ensure there are at least 2 valid data points
                    vmean = var.mean()
                    vsd = var.std()
                    vskew = var.skew()
                    vvar = var.var()

                    st.write(f"**Mean:** {vmean}")
                    st.write(f"**Standard Deviation:** {vsd}")
                    st.write(f"**Variance:** {vvar}")
                    st.write(f"**Skew:** {vskew}")

                    # Histogram
                    fig, ax = plt.subplots()
                    var.hist(ax=ax)
                    ax.axvline(vmean, color='red', linestyle='dashed', linewidth=2)
                    ax.set_xlabel(selected_stat_var)
                    st.pyplot(fig)
                else:
                    st.error("Not enough valid data points to calculate statistics.")
else:
    st.write("Please select at least one dataset or upload CSV files to proceed.")
