import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import geopandas as gpd
from pandasai.llm import GoogleGemini
from pandasai import SmartDataframe
import os

# Title of the app
st.title("Dataset Analysis with Streamlit and Google Gemini LLM")

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

# LLM integration (Google Gemini setup)
gemini_api_key = os.environ.get('gemini')  # Ensure your API key is stored in the environment variables
llm = GoogleGemini(api_key=gemini_api_key)

def generate_llm_response(dataFrame, prompt):
    pandas_agent = SmartDataframe(dataFrame, config={"llm": llm})
    answer = pandas_agent.chat(prompt)
    return answer

# Dropdown to select a dataset from predefined datasets
st.subheader("Select a predefined dataset or upload your own")
dataset_choice = st.selectbox("Choose a Dataset", ["None"] + list(datasets.keys()))

# Upload a custom CSV file
uploaded_file = st.file_uploader("Or upload a CSV file", type="csv")

# Load selected dataset
if dataset_choice != "None":
    # Load from predefined datasets
    if dataset_choice in datasets:
        df = load_csv_dataset(datasets[dataset_choice])
else:
    # Load from uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Proceed if a dataset is loaded
if 'df' in locals():
    # Display all variables in the dataset
    st.subheader("All Variables in the Dataset:")
    st.write(df.columns.tolist())

    # Filter for numeric variables
    numeric_vars = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    st.subheader("Numeric Variables:")
    st.write(numeric_vars)

    # Select numeric variables to visualize in a scatter matrix
    st.subheader("Scatter Matrix")
    selected_vars = st.multiselect("Select up to 5 variables for scatter matrix", numeric_vars, numeric_vars[:5])

    if selected_vars:
        scatter_matrix_df = df[selected_vars]
        scatter_matrix_fig = pd.plotting.scatter_matrix(scatter_matrix_df, alpha=0.2, figsize=(10, 10), diagonal='kde')
        for ax in scatter_matrix_fig.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=90)
            ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)
        st.pyplot(plt)

    # Scatterplot with regression line
    st.subheader("Scatterplot with Regression Line")
    x_var = st.selectbox("Select X variable", numeric_vars)
    y_var = st.selectbox("Select Y variable", numeric_vars)

    if x_var and y_var:
        if x_var != y_var:
            fig, ax = plt.subplots()
            sb.regplot(x=df[x_var], y=df[y_var], scatter_kws={"s": 20, "color": "red", "alpha": 0.2}, ax=ax)
            ax.set(xlabel=x_var, ylabel=y_var)
            st.pyplot(fig)
        else:
            st.error("X and Y variables must be different!")

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    selected_stat_var = st.selectbox("Select a variable to calculate statistics", numeric_vars)

    if selected_stat_var:
        var = df[selected_stat_var]
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

    # **LLM Interaction Section**
    st.subheader("Ask Your Data Questions")
    prompt = st.text_input("Type a question about the data", "")
    
    if prompt:
        response = generate_llm_response(df, prompt)
        st.write(response)

else:
    st.write("Please select a dataset or upload a CSV file to proceed.")
