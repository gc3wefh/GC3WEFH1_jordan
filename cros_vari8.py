import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pandasai.llm import GoogleGemini
from pandasai import SmartDataframe, SmartDatalake
import os

# LLM integration (Google Gemini setup)
gemini_api_key = os.environ.get('gemini')  # Ensure your API key is stored as an environment variable
llm = GoogleGemini(api_key=gemini_api_key)

# Title of the app
st.title("Cross-Dataset Analysis with Streamlit and Google Gemini LLM")

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

# Dropdown to select multiple datasets from predefined datasets
st.subheader("Select one or more predefined datasets")
selected_datasets = st.multiselect("Choose Datasets", list(datasets.keys()))

# Upload a custom CSV file for individual dataset analysis
uploaded_file = st.file_uploader("Or upload a CSV file for single-dataset analysis", type="csv")

# Load selected datasets and store them in a dictionary
loaded_datasets = {name: load_csv_dataset(datasets[name]) for name in selected_datasets}

# Initialize SmartDatalake if multiple datasets are selected
if len(loaded_datasets) > 1:
   lake = SmartDatalake(list(loaded_datasets.values()), config={"llm": llm})

# **Single Dataset Analysis Section**
if uploaded_file or len(selected_datasets) == 1:
   if uploaded_file:
       df = pd.read_csv(uploaded_file)
   elif len(selected_datasets) == 1:
       df = load_csv_dataset(datasets[selected_datasets[0]])

   # Filter for numeric variables
   numeric_vars = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]

   # LLM Interaction Section for Single Dataset Analysis
   st.subheader("Ask LLM to Describe the Relationship Between Two Variables in Single Dataset")

   # Select numeric variables for LLM description
   x_var = st.selectbox("Select X variable", numeric_vars)
   y_var = st.selectbox("Select Y variable", numeric_vars)
  
   if x_var and y_var and x_var != y_var:
       corr = df[[x_var, y_var]].corr().iloc[0, 1]
       x_mean, y_mean = df[x_var].mean(), df[y_var].mean()
       x_sd, y_sd = df[x_var].std(), df[y_var].std()
       prompt = (
           f"Variable X is {x_var}, and Variable Y is {y_var}. "
           f"The Pearson correlation coefficient between them is {corr:.2f}. "
           f"Variable X has a mean of {x_mean:.2f} and a standard deviation of {x_sd:.2f}, "
           f"and Variable Y has a mean of {y_mean:.2f} and a standard deviation of {y_sd:.2f}. "
           f"Please analyze the relationship between these variables."
       )
       llm_response = SmartDataframe(df, config={"llm": llm}).chat(prompt)
       st.write("### LLM-Generated Description:")
       st.write(llm_response)

# **Cross-Dataset Analysis Section**
if len(loaded_datasets) > 1:
   st.subheader("Cross-Dataset Analysis with SmartDatalake")

   # Option to merge on a common column
   common_column = st.selectbox("Select a common column for merging (optional)", ['Station_ID', 'region', 'governorate', None], index=3)
   
   # Check if common column exists in all selected datasets
   if common_column and all(common_column in df.columns for df in loaded_datasets.values()):
       merged_df = pd.concat([df.set_index(common_column) for df in loaded_datasets.values()], axis=1, join='inner').reset_index()
       st.write(f"Merged DataFrame based on {common_column}:")
       st.write(merged_df)
   else:
       st.write("No common column selected or available in all datasets. Proceeding with separate analysis.")

   # Prompt for LLM to analyze across datasets
   cross_dataset_prompt = st.text_input("Enter a question to ask across the selected datasets:")

   if cross_dataset_prompt:
       # Generate LLM response for cross-dataset analysis
       cross_dataset_response = lake.chat(cross_dataset_prompt)
       st.write("### LLM-Generated Cross-Dataset Analysis:")
       st.write(cross_dataset_response)

else:
   st.write("Please select more than one dataset for cross-dataset analysis.")

# Scatterplot with Regression Line for Single Dataset
st.subheader("Scatterplot with Regression Line")

if 'df' in locals() and x_var and y_var and x_var != y_var:
   fig, ax = plt.subplots()
   sb.regplot(x=df[x_var], y=df[y_var], scatter_kws={"s": 20, "color": "red", "alpha": 0.2}, ax=ax)
   ax.set(xlabel=x_var, ylabel=y_var)
   st.pyplot(fig)

# Descriptive Statistics
st.subheader("Descriptive Statistics for Single Dataset")

if 'df' in locals():
   selected_stat_var = st.selectbox("Select a variable to calculate statistics", numeric_vars)
   if selected_stat_var:
       var = df[selected_stat_var]
       st.write(f"**Mean:** {var.mean()}")
       st.write(f"**Standard Deviation:** {var.std()}")
       st.write(f"**Variance:** {var.var()}")
       st.write(f"**Skew:** {var.skew()}")

       # Histogram
       fig, ax = plt.subplots()
       var.hist(ax=ax)
       ax.axvline(var.mean(), color='red', linestyle='dashed', linewidth=2)
       ax.set_xlabel(selected_stat_var)
       st.pyplot(fig)

# Prompt-based descriptive statistics for single dataset
if 'df' in locals() and selected_stat_var:
   stats_prompt = (
       f"The selected variable is {selected_stat_var}. It has a mean of {var.mean():.2f}, "
       f"a standard deviation of {var.std():.2f}, a variance of {var.var():.2f}, and a skewness of {var.skew():.2f}. "
       f"Please provide insights on the distribution of this variable, including any outliers or unusual values."
   )
   llm_response_stats = SmartDataframe(df, config={"llm": llm}).chat(stats_prompt)
   st.write("### LLM-Generated Description for Descriptive Statistics:")
   st.write(llm_response_stats)
