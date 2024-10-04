import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import geopandas as gpd
from pandasai.llm import GoogleGemini
from pandasai import SmartDataframe
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import os


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

    # Select X and Y variables across datasets
    selected_x = st.selectbox("Select X variable", variable_options, format_func=lambda x: x[0])
    selected_y = st.selectbox("Select Y variable", variable_options, format_func=lambda x: x[0])

    # Parse dataset and variable names for X and Y
    if selected_x and selected_y:
        dataset_x, x_var = selected_x[1], selected_x[2]
        dataset_y, y_var = selected_y[1], selected_y[2]
        
        # Fetch respective dataframes
        df_x = dataframes[dataset_x]
        df_y = dataframes[dataset_y]

        # Perform LLM analysis only if X and Y are from the same dataset
        if dataset_x == dataset_y and x_var != y_var:
            # Descriptive statistics for the relationship
            corr = df_x[[x_var, y_var]].corr().iloc[0, 1]
            x_mean, y_mean = df_x[x_var].mean(), df_x[y_var].mean()
            x_sd, y_sd = df_x[x_var].std(), df_x[y_var].std()

            # Combine statistics in a prompt for LLM
            prompt = (
                f"Variable X represents {x_var} and Variable Y represents {y_var}. "
                f"The Pearson correlation coefficient between them is {corr:.2f}. "
                f"Variable X has a mean of {x_mean:.2f} and a standard deviation of {x_sd:.2f}, "
                f"while Variable Y has a mean of {y_mean:.2f} and a standard deviation of {y_sd:.2f}. "
                f"Can you describe the relationship between these variables? "
                f"Please do not generate any images, code, or visual outputs, and only provide a text-based analysis."
            )

            # Generate LLM response
            llm_response = generate_llm_response(df_x, prompt)

            # Display LLM response
            st.write("### LLM-Generated Description:")
            st.write(llm_response)

        # Scatterplot with regression line
        st.subheader("Scatterplot with Regression Line")

        if dataset_x == dataset_y and x_var != y_var:
            fig, ax = plt.subplots()
            sb.regplot(x=df_x[x_var], y=df_x[y_var], scatter_kws={"s": 20, "color": "red", "alpha": 0.2}, ax=ax)
            ax.set(xlabel=x_var, ylabel=y_var)
            st.pyplot(fig)
        else:
            st.error("X and Y variables must be different!")

    # **Scatter Matrix Section with LLM Analysis**
    st.subheader("Scatter Matrix Across Datasets")
    selected_vars = st.multiselect("Select up to 5 variables for scatter matrix", variable_options)

    if selected_vars:
        # Prepare a combined DataFrame for selected variables across datasets
        combined_data = pd.DataFrame()
        for selected in selected_vars:
            dataset_name, variable_name = selected[1], selected[2]
            df_selected = dataframes[dataset_name][variable_name]
            combined_data[f"{dataset_name} - {variable_name}"] = df_selected

        # Check for valid data
        if not combined_data.empty and combined_data.shape[1] > 1:
            scatter_matrix_fig = pd.plotting.scatter_matrix(combined_data.dropna(), alpha=0.2, figsize=(10, 10), diagonal='hist')
            for ax in scatter_matrix_fig.ravel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=90)
                ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)
            st.pyplot(plt)

            # LLM analysis for the scatter matrix
            prompt = (
                f"A scatter matrix has been generated for the variables {', '.join([f'{var}' for var in selected_vars])}. "
                f"Please provide insights on the relationships between these variables. "
                f"Are there any strong correlations, clusters, or patterns? "
                f"Also, can you determine if these variables are ready for machine learning? "
            )

            # Generate LLM response for scatter matrix analysis
            llm_response_scatter_matrix = generate_llm_response(combined_data, prompt)

            # Display LLM response
            st.write("### LLM-Generated Analysis for Scatter Matrix:")
            st.write(llm_response_scatter_matrix)
        else:
            st.error("The selected variables do not contain enough valid data.")
    else:
        st.error("Please select variables from at least one dataset.")

    # **Machine Learning Readiness Assessment**
    st.subheader("Machine Learning Readiness Assessment")

    # Allow selection of one dataset for ML readiness
    selected_ml_dataset = st.selectbox("Select a dataset for ML readiness", list(dataframes.keys()))

    if selected_ml_dataset:
        df_ml = dataframes[selected_ml_dataset]

        # 1. Missing Values
        missing_percent = df_ml.isnull().mean() * 100
        st.write("### Percentage of Missing Values per Feature:")
        st.write(missing_percent)

        # Impute missing values for numerical features with mean
        numerical_features = df_ml.select_dtypes(include=[np.number]).columns.tolist()
        imputer_num = SimpleImputer(strategy='mean')
        df_ml[numerical_features] = imputer_num.fit_transform(df_ml[numerical_features])

        # Impute missing values for categorical features with the most frequent value
        categorical_features = df_ml.select_dtypes(include=['object', 'category']).columns.tolist()
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_ml[categorical_features] = imputer_cat.fit_transform(df_ml[categorical_features])

        st.write("Missing values handled using imputation.")

        # 2. Encoding Categorical Variables
        st.write("### Encoding Categorical Variables")
        df_encoded = pd.get_dummies(df_ml, columns=categorical_features, drop_first=True)
        st.write("Categorical variables encoded using OneHotEncoder.")

        # 3. Feature Scaling
        st.write("### Feature Scaling")
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
        st.write("Features scaled using StandardScaler.")

        # 4. Removing Low Variance Features
        st.write("### Removing Low Variance Features")
        selector = VarianceThreshold(threshold=0.1)
        df_high_variance = selector.fit_transform(df_scaled)
        df_high_variance = pd.DataFrame(df_high_variance, columns=df_scaled.columns[selector.get_support()])
        st.write("Low variance features removed.")

        # 5. Handling Outliers
        st.write("### Handling Outliers")
        z_scores = np.abs(stats.zscore(df_high_variance))
        df_no_outliers = df_high_variance[(z_scores < 3).all(axis=1)]
        st.write(f"Outliers removed. New dataset shape: {df_no_outliers.shape}")

        st.write("### Final Prepared Dataset")
        st.write(df_no_outliers.head())

        # Save the prepared dataset
        df_no_outliers.to_csv('prepared_dataset.csv', index=False)
        st.write("Prepared dataset saved as 'prepared_dataset.csv'.")
else:
    st.write("Please select a dataset or upload a CSV file to proceed.")
