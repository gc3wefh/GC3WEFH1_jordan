import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import geopandas as gpd
import sweetviz as sv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import os
from pandasai.llm import GoogleGemini
from pandasai import SmartDataframe

# LLM integration (Google Gemini setup)
gemini_api_key = os.environ.get('gemini')  # Ensure your API key is stored as an environment variable
llm = GoogleGemini(api_key=gemini_api_key)

def generate_llm_response(dataFrame, prompt):
    pandas_agent = SmartDataframe(dataFrame, config={"llm": llm})
    answer = pandas_agent.chat(prompt)
    return answer

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

# Allow user to select multiple datasets
st.subheader("Select predefined datasets or upload your own")
selected_datasets = st.multiselect("Choose Datasets", list(datasets.keys()))

# Upload custom CSV files
uploaded_files = st.file_uploader("Or upload CSV files", type="csv", accept_multiple_files=True)

# Dictionary to store dataframes
dataframes = {}

# Load selected predefined datasets
for dataset_choice in selected_datasets:
    dataframes[dataset_choice] = load_csv_dataset(datasets[dataset_choice])

# Load uploaded CSV files
if uploaded_files:
    for uploaded_file in uploaded_files:
        dataframes[uploaded_file.name] = pd.read_csv(uploaded_file)

# Proceed if any dataset is loaded
if dataframes:
    # Combine numeric variables from all datasets
    all_numeric_vars = {}
    for name, df in dataframes.items():
        numeric_vars = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
        all_numeric_vars[name] = numeric_vars

    # Flatten and present the combined list of numeric variables for selection
    st.subheader("Numeric Variables Across Datasets:")
    variable_options = [(f"{name} - {var}", name, var) for name, vars_list in all_numeric_vars.items() for var in vars_list]

    # **LLM Interaction Section**
    st.subheader("Ask LLM to Describe the Relationship Between Two Variables")

    # Select numeric variables for LLM description
    x_var = st.selectbox("Select X variable", numeric_vars)
    y_var = st.selectbox("Select Y variable", numeric_vars)
    
    if x_var and y_var and x_var != y_var:
        # Calculate descriptive statistics for relationship
        corr = df[[x_var, y_var]].corr().iloc[0, 1]
        x_mean, y_mean = df[x_var].mean(), df[y_var].mean()
        x_sd, y_sd = df[x_var].std(), df[y_var].std()

        # Combine the relationship analysis with distribution, spikes, and outliers in the prompt
        prompt = (
            f"Variable X represents {x_var} and Variable Y represents {y_var}. "
            f"The Pearson correlation coefficient between them is {corr:.2f}. "
            f"Variable X has a mean of {x_mean:.2f} and a standard deviation of {x_sd:.2f}, "
            f"while Variable Y has a mean of {y_mean:.2f} and a standard deviation of {y_sd:.2f}. "
            f"Can you describe the relationship between these variables? "
            f"Please do not generate any images, code, or visual outputs, and only provide a text-based analysis."
        )

        # Generate LLM response
        llm_response = generate_llm_response(df, prompt)
        
        # Display LLM response
        st.write("### LLM-Generated Description:")
        st.write(llm_response)

    # Scatterplot with regression line
    st.subheader("Scatterplot with Regression Line")

    if x_var and y_var and x_var != y_var:
        fig, ax = plt.subplots()
        sb.regplot(x=df[x_var], y=df[y_var], scatter_kws={"s": 20, "color": "red", "alpha": 0.2}, ax=ax)
        ax.set(xlabel=x_var, ylabel=y_var)
        st.pyplot(fig)
    else:
        st.error("X and Y variables must be different!")

    # **Machine Learning Readiness Analysis**
    st.subheader("Machine Learning Readiness Analysis")

    # Impute missing values
    st.write("Handling Missing Values...")
    imputer_num = SimpleImputer(strategy='mean')
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_features] = imputer_num.fit_transform(df[numerical_features])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])
    st.write("Missing values handled.")

    # Encode categorical variables
    st.write("Encoding Categorical Variables...")
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    st.write("Categorical variables encoded.")

    # Scale features
    st.write("Scaling Features...")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    st.write("Features scaled.")

    # Remove low variance features
    st.write("Removing Low Variance Features...")
    selector = VarianceThreshold(threshold=0.1)
    df_high_variance = selector.fit_transform(df_scaled)
    df_high_variance = pd.DataFrame(df_high_variance, columns=df_scaled.columns[selector.get_support()])
    st.write("Low variance features removed.")

    # Handle outliers
    st.write("Handling Outliers...")
    z_scores = np.abs(stats.zscore(df_high_variance))
    df_no_outliers = df_high_variance[(z_scores < 3).all(axis=1)]
    st.write(f"Outliers removed; new dataset shape: {df_no_outliers.shape}")

    # Display the final prepared dataset
    st.write("Final Prepared Dataset:")
    st.write(df_no_outliers.head())

    # Save the prepared dataset
    df_no_outliers.to_csv('prepared_dataset.csv', index=False)
    st.write("Prepared dataset saved as 'prepared_dataset.csv'.")

    # **Generate LLM explanation for machine learning readiness**
    prompt = (
        "The dataset has been preprocessed with imputation, encoding, scaling, and outlier removal. "
        "Can you summarize the steps and explain how this improves the dataset's readiness for machine learning?"
    )
    llm_response_readiness = generate_llm_response(df, prompt)
    st.write("### LLM-Generated Summary for ML Readiness:")
    st.write(llm_response_readiness)

else:
    st.write("Please select a dataset or upload a CSV file to proceed.")
