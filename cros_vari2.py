import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pandasai.llm import GoogleGemini
from pandasai import SmartDataframe
import os

# LLM integration (Google Gemini setup)
gemini_api_key = os.environ.get('gemini')  # Ensure your API key is stored as an environment variable
llm = GoogleGemini(api_key=gemini_api_key, model = "models/gemini-1.5-pro")

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

# Dropdown to select multiple datasets from predefined datasets
st.subheader("Select predefined datasets or upload your own")
selected_datasets = st.multiselect("Choose Datasets", list(datasets.keys()))

# Upload custom CSV files
uploaded_files = st.file_uploader("Or upload CSV files", type="csv", accept_multiple_files=True)

# Dictionary to hold dataframes
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
    
    # Select variables for analysis
    st.subheader("Select Variables for Analysis (Across Datasets):")
    x_var_info = st.selectbox("Select X variable", variable_options, format_func=lambda x: x[0])
    y_var_info = st.selectbox("Select Y variable", variable_options, format_func=lambda x: x[0])

    # Unpack selected variables
    x_dataset, x_var = x_var_info[1], x_var_info[2]
    y_dataset, y_var = y_var_info[1], y_var_info[2]

    # Handle different datasets separately
    df_x = dataframes[x_dataset]
    df_y = dataframes[y_dataset]

    # Calculate descriptive statistics for X and Y separately
    x_mean, y_mean = df_x[x_var].mean(), df_y[y_var].mean()
    x_sd, y_sd = df_x[x_var].std(), df_y[y_var].std()

    # Create prompt for LLM with variables from different datasets
    prompt = (
        f"Variable X represents {x_var} from the {x_dataset} dataset, "
        f"and Variable Y represents {y_var} from the {y_dataset} dataset. "
        f"Variable X has a mean of {x_mean:.2f} and a standard deviation of {x_sd:.2f}, "
        f"while Variable Y has a mean of {y_mean:.2f} and a standard deviation of {y_sd:.2f}. "
        f"Can you describe the relationship between these variables, even though they come from different datasets?"
    )

    # Generate LLM response
    llm_response = generate_llm_response(df_x, prompt)
    
    # Display LLM response
    st.write("### LLM-Generated Description:")
    st.write(llm_response)

    # Scatterplot with variables from different datasets
    st.subheader("Scatterplot with Variables from Different Datasets")
    fig, ax = plt.subplots()

    ax.scatter(df_x[x_var], df_y[y_var], c="blue", alpha=0.5)
    ax.set(xlabel=x_var, ylabel=y_var)
    
    st.pyplot(fig)

else:
    st.write("Please select datasets or upload CSV files to proceed.")
