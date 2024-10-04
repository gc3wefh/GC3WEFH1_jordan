import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import geopandas as gpd
import sweetviz as sv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from pandasai.llm import GoogleGemini
from pandasai import SmartDataframe
import os

# LLM integration (Google Gemini setup)
gemini_api_key = os.environ.get('gemini')  # Ensure your API key is stored as an environment variable
llm = GoogleGemini(api_key=gemini_api_key)

def generate_llm_response(dataFrame, prompt):
    pandas_agent = SmartDataframe(dataFrame, config={"llm": llm})
    answer = pandas_agent.chat(prompt)
    return answer

# Title of the app
st.title("ML Readiness App with Enhanced Report Sizes and LLM Integration")

# Inject custom CSS for a larger button
st.markdown(
    """
    <style>
    .stButton > button {
        font-size: 20px;
        padding: 15px 30px;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Predefined dataset paths
datasets = {
    'indicator_df': 'jordan_indicadors.csv',
    'governorate_csv': 'dataset/Average Household Size in Jordan/governorate.csv',
    'country_csv': 'dataset/Average Household Size in Jordan/country.csv',
    'spi_data': '170/SPI_JMD_data_corrected_long_format.csv',
    'healthcare_facilities_csv': 'dataset/Healthcare Facilities in Jordan/healthcare.csv',
    'governorates_jordan_csv': 'dataset/Jordan Health/Governorates_jordan.csv',
    'hospitals_csv': 'dataset/Jordan Health/Hospitals.csv',
    'governorate_boundaries_csv': 'dataset/Jordan Boundaries/governorate.csv',
    'country_boundaries_csv': 'dataset/Jordan Boundaries/country.csv'
}

# **How to Pick Datasets for Analysis Section**
st.subheader("How to Pick Datasets for Analysis")
st.write("""
    Below are descriptions of each dataset to help you decide which one is suitable for your analysis:
    1. **jordan_indicadors.csv**: This dataset contains various socio-economic indicators for Jordan.
    2. **governorate.csv**: Contains data about household sizes, population densities, and more at the governorate level in Jordan.
    3. **country.csv**: National-level data for Jordan, similar to the governorate dataset but aggregated for the entire country.
    4. **spi_data.csv**: A dataset on Social Progress Index (SPI) metrics collected for Jordan.
    5. **healthcare_facilities.csv**: Information about healthcare facilities in Jordan, including capacity and services provided.
    6. **governorates_jordan.csv**: Population statistics for governorates in Jordan, including gender distribution and other demographic data.
    7. **hospitals.csv**: Details about hospital facilities, bed counts, and patient admission rates in Jordan.
    8. **governorate_boundaries.csv**: Geographical boundaries of Jordan’s governorates.
    9. **country_boundaries.csv**: Geographical boundaries of the entire country of Jordan.
    
    Use this information to select datasets based on your analysis needs, such as demographic analysis, healthcare facility assessment, or geographic data exploration.
""")

# 1. Dataset Selection and Upload
st.subheader("Select predefined datasets or upload your own")
selected_datasets = st.multiselect("Choose Datasets", list(datasets.keys()))
uploaded_files = st.file_uploader("Or upload CSV files", type="csv", accept_multiple_files=True)

# Load datasets
dataframes = {}

def load_csv_dataset(path):
    return pd.read_csv(path)

def load_geodataframe(path):
    return gpd.read_file(path)

# Load selected predefined datasets
for dataset_choice in selected_datasets:
    dataframes[dataset_choice] = load_csv_dataset(datasets[dataset_choice])

# Load uploaded CSV files
if uploaded_files:
    for uploaded_file in uploaded_files:
        dataframes[uploaded_file.name] = pd.read_csv(uploaded_file)

# Proceed if any dataset is loaded
if dataframes:
    # Basic information about each dataset
    for name, df in dataframes.items():
        st.subheader(f"Basic Dataset Information for {name}")
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        st.write("Data Types:")
        st.dataframe(df.dtypes)
        st.write("Missing Values:")
        st.dataframe(df.isnull().sum())

        # Sweetviz report generation
        if st.button(f"Generate Sweetviz Report for {name}"):
            report = sv.analyze(df)
            report.show_html(f'{name}_sweetviz_report.html')
            st.markdown(f"[View Sweetviz Report](./{name}_sweetviz_report.html)")

        # LLM description for numeric variables
        numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_vars:
            st.subheader(f"LLM Insights on Relationships in {name}")
            x_var = st.selectbox("Select X variable", numeric_vars, key=f"{name}_x")
            y_var = st.selectbox("Select Y variable", numeric_vars, key=f"{name}_y")

            if x_var and y_var and x_var != y_var:
                prompt = f"Describe the relationship between {x_var} and {y_var} in {name} dataset."
                llm_response = generate_llm_response(df, prompt)
                st.write("### LLM-Generated Description:")
                st.write(llm_response)

            # Scatterplot with regression line
            if x_var and y_var and x_var != y_var:
                fig, ax = plt.subplots()
                sb.regplot(x=df[x_var], y=df[y_var], ax=ax)
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                st.pyplot(fig)

        # Data preparation steps
        st.subheader(f"Data Preparation for {name}")
        working_df = df.copy()

        # Handling Missing Values
        st.write("Handling Missing Values...")
        imputer_num = SimpleImputer(strategy='mean')
        imputer_cat = SimpleImputer(strategy='most_frequent')
        working_df[numeric_vars] = imputer_num.fit_transform(working_df[numeric_vars])
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        working_df[categorical_features] = imputer_cat.fit_transform(working_df[categorical_features])
        st.write("Missing values handled.")

        # Encoding Categorical Variables
        if categorical_features:
            working_df = pd.get_dummies(working_df, columns=categorical_features, drop_first=True)
            st.write("Categorical variables encoded.")

        # Feature Scaling
        st.write("Scaling Features...")
        scaler = StandardScaler()
        working_df[numeric_vars] = scaler.fit_transform(working_df[numeric_vars])
        st.write("Features scaled.")

        # Removing Low Variance Features
        selector = VarianceThreshold(threshold=0.1)
        df_high_variance = selector.fit_transform(working_df)
        selected_features = working_df.columns[selector.get_support()]
        working_df = pd.DataFrame(df_high_variance, columns=selected_features)
        st.write("Low variance features removed.")

        # Handling Outliers
        st.write("Handling Outliers...")
        z_scores = np.abs(stats.zscore(working_df))
        working_df = working_df[(z_scores < 3).all(axis=1)]
        st.write(f"Outliers removed; new dataset shape: {working_df.shape}")

        # Final dataset and download
        st.subheader("Final Prepared Dataset")
        st.dataframe(working_df.head())
        st.download_button(
            label="Download Prepared Dataset",
            data=working_df.to_csv(index=False),
            file_name=f"{name}_prepared_dataset.csv",
            mime="text/csv"
        )
else:
    st.write("Please select a dataset or upload a CSV file to proceed.")
