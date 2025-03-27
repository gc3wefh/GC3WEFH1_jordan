import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import geopandas as gpd
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

    # Scatter Matrix
    st.subheader("Scatter Matrix")
    selected_vars = st.multiselect("Select up to 5 variables for scatter matrix", variable_options)

    if selected_vars:
        selected_vars_names = [var[2] for var in selected_vars]
        selected_datasets_for_vars = [var[1] for var in selected_vars]

        # Get the common dataset (ensure the variables are from the same dataset for scatter matrix)
        if len(set(selected_datasets_for_vars)) == 1:
            df = dataframes[selected_datasets_for_vars[0]]
            scatter_matrix_df = df[selected_vars_names]
            scatter_matrix_fig = pd.plotting.scatter_matrix(scatter_matrix_df, alpha=0.2, figsize=(10, 10), diagonal='kde')
            for ax in scatter_matrix_fig.ravel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=90)
                ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)
            st.pyplot(plt)
        else:
            st.error("Scatter matrix can only be generated for variables from the same dataset.")

    # Scatterplot with regression line and LLM description
    st.subheader("Scatterplot with Regression Line and LLM Analysis")
    x_var_info = st.selectbox("Select X variable", variable_options, format_func=lambda x: x[0])
    y_var_info = st.selectbox("Select Y variable", variable_options, format_func=lambda x: x[0])

    x_dataset, x_var = x_var_info[1], x_var_info[2]
    y_dataset, y_var = y_var_info[1], y_var_info[2]

    if x_dataset == y_dataset:
        df = dataframes[x_dataset]
        if x_var != y_var:
            # Descriptive stats for LLM prompt
            corr = df[[x_var, y_var]].corr().iloc[0, 1]
            x_mean, y_mean = df[x_var].mean(), df[y_var].mean()
            x_sd, y_sd = df[x_var].std(), df[y_var].std()

            # Create LLM prompt
            prompt = (
                f"Variable X represents {x_var} and Variable Y represents {y_var}. "
                f"The Pearson correlation coefficient between them is {corr:.2f}. "
                f"Variable X has a mean of {x_mean:.2f} and a standard deviation of {x_sd:.2f}, "
                f"while Variable Y has a mean of {y_mean:.2f} and a standard deviation of {y_sd:.2f}. "
                f"Can you describe the relationship between these variables?"
            )

            # Generate LLM response
            llm_response = generate_llm_response(df, prompt)
            st.write("### LLM-Generated Description:")
            st.write(llm_response)

            # Plot scatterplot
            fig, ax = plt.subplots()
            sb.regplot(x=df[x_var], y=df[y_var], scatter_kws={"s": 20, "color": "red", "alpha": 0.2}, ax=ax)
            ax.set(xlabel=x_var, ylabel=y_var)
            st.pyplot(fig)
        else:
            st.error("X and Y variables must be different!")
    else:
        # Allow scatterplot between variables from different datasets
        df_x, df_y = dataframes[x_dataset], dataframes[y_dataset]

        # Descriptive stats for LLM prompt
        x_mean, y_mean = df_x[x_var].mean(), df_y[y_var].mean()
        x_sd, y_sd = df_x[x_var].std(), df_y[y_var].std()

        # Create LLM prompt
        prompt = (
            f"Variable X represents {x_var} from {x_dataset} and Variable Y represents {y_var} from {y_dataset}. "
            f"Variable X has a mean of {x_mean:.2f} and a standard deviation of {x_sd:.2f}, "
            f"while Variable Y has a mean of {y_mean:.2f} and a standard deviation of {y_sd:.2f}. "
            f"Can you describe the relationship between these variables from different datasets?"
        )

        # Generate LLM response
        llm_response = generate_llm_response(df_x, prompt)
        st.write("### LLM-Generated Description:")
        st.write(llm_response)

        # Plot scatterplot with variables from different datasets
        fig, ax = plt.subplots()
        ax.scatter(df_x[x_var], df_y[y_var], c="blue", alpha=0.5)
        ax.set(xlabel=x_var, ylabel=y_var)
        st.pyplot(fig)

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    stat_var_info = st.selectbox("Select a variable to calculate statistics", variable_options, format_func=lambda x: x[0])
    stat_dataset, stat_var = stat_var_info[1], stat_var_info[2]

    if stat_var:
        df = dataframes[stat_dataset]
        var = df[stat_var]
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
        ax.set_xlabel(stat_var)
        st.pyplot(fig)

else:
    st.write("Please select datasets or upload CSV files to proceed.")
