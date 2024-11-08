"""Machine Learning Readiness Assessment on CSV files using Streamlit.

Displays current properties of a CSV dataset and performs various functions on
given CSV file in order to make it Machine Learning ready, giving the option to 
download the resulting transformed CSV file. Current implementation includes:
 1. Handling Missing Values
 2. Encoding Categorical Variables
 3. Feature Scaling
 4. Removal of Low Variance Features
 5. Handling Outliers

This script is used by Streamlit to provide a web platform for assessment.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sweetviz as sv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import tempfile
from pandasai.llm import GoogleGemini
from pandasai import SmartDataframe
import os
from dotenv import load_dotenv

load_dotenv()  # Load env vars from .env file

# Set the api key from environment variable
gemini_api_key = os.environ.get('gemini')  # Make sure your API key is stored as an environment variable

# LLM Integration
llm = GoogleGemini(api_key=gemini_api_key)

def generate_llm_response(dataFrame, prompt):
    pandas_agent = SmartDataframe(dataFrame, config={"llm": llm})
    answer = pandas_agent.chat(prompt)
    return answer

@st.cache_data
def load_data(file):
    """Loads a CSV file using pandas
    
    Loads a CSV file into a pandas DataFrame and returns the DataFrame.

    Args:
        file: CSV file.

    Returns:
        DataFrame representation of input CSV file.
    """
    df = pd.read_csv(file)
    return df

def downcast_numeric_columns(df : pd.DataFrame) -> pd.DataFrame:
    """Downcasts numerical columns within a given DataFrame
    
    Selects columns that are of type int or float and downcasts them to the
    smallest numerical dtype possible in a copy of the input dataframe.

    Args:
        df: A pandas Dataframe

    Returns:
        A dataframe with all relevants column values downcasted.
    """
    downcasted_df: pd.DataFrame = df.copy()  # Deepcopy as Dataframe is mutable
    # Downcast integer columns
    for col in downcasted_df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    # Downcast float columns
    for col in downcasted_df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return downcasted_df

def show_dtypes_and_missing_vals(df: pd.DataFrame) -> pd.DataFrame:
    """Shows data types and missing values in a Dataframe

    Takes an input Dataframe and extracts the datatypes found and corresponding
    missing values. Also provides a percentage for missing values.

    Args:
        df: Dataframe to get datatypes and missing values for

    Returns:
        A dataframe corresponding to the input dataframe's datatypes, number of
        missing values, and the percentage of missing values.
    """
    # Calculate missing values per column
    missing_values = df.isnull().sum()
    missing_data_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': missing_values,
        'Missing Percentage (%)': (missing_values / len(df)) * 100
    })
    return missing_data_df

@st.cache_data
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical columns using OneHotEncoder

    Uses OneHotEncoder to encode categorical data into 1s and 0s, represented in
    sparse data. Returns a new Dataframe that has original non-categorical data
    and new encoded categorical data.

    Args:
        df: Input Dataframe with categorical data columns.
    
    Returns:
        New dataframe with encoded categorical data.
    """
    encoder = OneHotEncoder(sparse_output=True)

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    encoded_arr = encoder.fit_transform(df[categorical_columns])
    encoded_categorical_df = pd.DataFrame.sparse.from_spmatrix(
        encoded_arr,
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    non_categorical_df = df.drop(categorical_columns, axis=1)

    encoded_df = pd.concat([non_categorical_df, encoded_categorical_df], axis=1)

    return encoded_df

@st.cache_data
def get_df_density(df : pd.DataFrame) -> float:
    """Get density of a Dataframe that contains mixed regular and sparse data.

    Gets the number of non-sparse elements and returns the density as a
    percentage in decimal form.

    Args:
        df: Input Dataframe with sparse data.

    Returns:
        decimal representation of density percentage.
    """
    total_elems = df.size
    non_sparse_elems = 0

    for col in df.columns:
        if isinstance(df[col].dtype, pd.SparseDtype):
            non_sparse_elems += df[col].sparse.density * df[col].size
        else:
            non_sparse_elems += df[col].count()
    
    return float(non_sparse_elems / total_elems)

@st.cache_data
def get_top_n_rows(df: pd.DataFrame, top_n : int) -> pd.DataFrame:
    """Get the top N rows of a Dataframe.

    Creates a subset of the original Dataframe for memory saving purposes.
    Converts sparse columns into dense format and then returns the top N rows.

    Args:
        df: Input DataFrame that can contain sparse data.
        top_n: number of rows from the top of the original DataFrame.

    Returns:
        a DataFrame that is a subset of the input dataframe with only top_n rows.
    """
    # Create subset of input DataFrame to be returned.
    df_topN = df.head(top_n)

     # Identify sparse columns
    sparse_cols = [
        col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)
    ]

    # Convert each sparse column to dense and explicitly cast to float or int
    if sparse_cols:
        dense_values = df_topN[sparse_cols].sparse.to_dense()
         # Assign back to the original column
        df_topN[sparse_cols] = dense_values.astype(np.float64)

    # Verify that there are no sparse columns left
    remaining_sparse_cols = [
        col for col in df_topN.columns if isinstance(df_topN[col].dtype, pd.SparseDtype)
    ]
    if len(remaining_sparse_cols) > 0:
        print(f"Warning: Sparse columns still present: {remaining_sparse_cols}")

    return df_topN

def scaler_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Scales a DataFrame using StandardScaler
    
    A DataFrame that may have mixed sparse and regular columns is scaled.
    Scaling can only be done on non-categorical columns i.e. non-sparse columns

    Args:
        df: Input DataFrame that may have mixed sparse and regular column types.

    Returns:
        New DataFrame that is a scaled version of the input DataFrame.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    # Get non-categorical columns
    non_categorical_cols = [col for col in df.columns if not isinstance(df[col].dtype, pd.SparseDtype)]
    df_scaled[non_categorical_cols] = scaler.fit_transform(df_scaled[non_categorical_cols])
    return df_scaled

def describe_numerical(df: pd.DataFrame) -> pd.DataFrame:
    print("Printing df", df)
    numerical_cols = [col for col in df.columns if not isinstance(df[col].dtype, pd.SparseDtype)]
    print("Printing df[numericals_cols]", df[numerical_cols])
    return df[numerical_cols].describe()

def main():
    # Define a Streamlit app
    st.title("ML Readiness App with Key Feature Insights")

    # 1. File Upload
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is None:
        st.write("No data to show. Upload a file to generate data.")
    else:
        # 1. Load the dataset
        df = load_data(uploaded_file)
        st.success(f"Dataset '{uploaded_file.name}' uploaded successfully.")
        
        # 2. Basic Dataset Information (display shape)
        st.subheader("Basic Dataset Information")
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        # 3. Data Types and Missing Values
        st.subheader("Data Types and Missing Values")

        st.write("Below is the data type of each column and the number of missing "
                "values for each column of the original dataset:")

        missing_data_original = show_dtypes_and_missing_vals(df)

        # Show missing values in a more detailed and clear way
        st.dataframe(missing_data_original.style.format(
            {"Missing Percentage (%)": "{:.2f}"}
        ).background_gradient(
            cmap='Reds', subset=["Missing Values"]
        ))

        # Apply the downcast function to the DataFrame to save memory
        st.write("Same data after downcasting numerical types to save memory:")

        downcasted_df = downcast_numeric_columns(df)
        missing_data_df = show_dtypes_and_missing_vals(downcasted_df)

        # Show missing values in a more detailed and clear way
        st.dataframe(missing_data_df.style.format(
            {"Missing Percentage (%)": "{:.2f}"}
        ).background_gradient(
            cmap='Reds', subset=["Missing Values"]
        ))

        # 4. Statistical Summary
        st.subheader("Statistical Summary of Numerical Features")
        st.write(df.describe())

        # 5. Distribution of Numerical Features
        st.subheader("Distribution of Numerical Features")
        st.write("Histograms for Numerical Features:")
        fig, ax = plt.subplots(figsize=(15, 10))
        df.hist(ax=ax)
        st.pyplot(fig)

        # 6. Categorical Feature Analysis
        st.subheader("Categorical Feature Analysis")
        categorical_features = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        st.write("Categorical Features:")
        st.write(categorical_features)

        # Display value counts for categorical features
        for col in categorical_features:
            st.write(f"\nValue counts for {col}:")
            st.write(df[col].value_counts())

        # 7. Generate a Sweetviz report
        if st.button("Generate Sweetviz Report"):
            report = sv.analyze(df)

            # Use a temporary file to store the Sweetviz report
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".html"
            ) as tmp_file:
                report.show_html(tmp_file.name)
                st.success("Sweetviz report generated successfully.")
                
                # Display a download button for the report
                with open(tmp_file.name, "rb") as file:
                    btn = st.download_button(
                        label="Download Sweetviz Report",
                        data=file,
                        file_name="sweetviz_report.html",
                        mime="text/html"
                    )

        # Initialize the working DataFrame for modifications
        working_df = downcasted_df.copy()

        # 8. Data Preparation Steps
        st.subheader("Step-by-Step Data Preparation")

        ## 8.1 Handling Missing Values
        st.subheader("Step 1: Handling Missing Values")

        # Combine the relationship analysis with distribution, spikes, and outliers in the prompt
        prompt = (
            "Numerical features contain quantitative values that represent some "
            "form of measurement, count, or value and can be mathematically manipulated. "
            "They commonly include integers and floating-point numbers. Some "
            "examples are continous data, like age, height, and temperature, or "
            "discrete data, like number of items sold or scores."
            "Categorical features contain values that represent categories or "
            "groups and often have no intrinsic mathematical meaning. They are "
            "usually represented as strings or integers that act as labels. They "
            "can be nominal, meaning categories without a natural order (e.g., "
            "gender, city name) or ordinal, meaning categories with a meaningful "
            "order but no fixed distance between categories (e.g., education "
            "levels like \"High School\", \"Bachelor's\", \"Master's\". Examples "
            "include binary data, like \"yes\"/\"no\" or \"true\"/\"false\", or "
            "multiclass categories, such as car brands, types of cuisine, or countries.)"
            "Impute missing values for numerical features with the mean and "
            "impute missing values for categorical features with the most frequent value."
            "Return the dataframe after imputing missing values."
        )

        st.write("##### LLM Prompt")
        st.write(prompt)

        st.write('##### Dataframe before LLM')
        st.write(working_df)
        # Generate LLM response
        llm_response = generate_llm_response(working_df, prompt)
        
        # Display LLM response
        st.write("### LLM-Generated Description:")
        st.write(llm_response)

        # ### Impute missing values for numerical features with mean
        # numerical_features = working_df.select_dtypes(
        #     include=[np.number]
        # ).columns.tolist()
        # imputer_num = SimpleImputer(strategy='mean')
        # working_df[numerical_features] = imputer_num.fit_transform(working_df[numerical_features])

        # ### Impute missing values for categorical features with the most frequent value
        # categorical_columns = working_df.select_dtypes(include=['object', 'category']).columns
        # categorical_features = categorical_columns.tolist()
        # imputer_cat = SimpleImputer(strategy='most_frequent')
        # working_df[categorical_features] = imputer_cat.fit_transform(working_df[categorical_features])

        # st.write(
        #     "Missing values handled. Here's a preview of the updated dataset "
        #     f"{working_df.shape}:"
        # )
        # st.write(working_df.head())

        # 8.2 Encoding Categorical Variables
        st.subheader("Step 2: Encoding Categorical Variables")

        # df_string = working_df.to_string()
        # # Define categorical features
        # prompt = (
        #     "Identify and encode all categorical columns in the DataFrame, including those "
        #     "with 'ID' in their names or that represent identifiers (even if stored as float64). "
        #     "For columns with fewer unique values than 10% of the total rows, assume they are categorical. "
        #     f"Apply 'One Hot' encoding. "
        #     # "Encode only the categorical features in the dataframe. Select the best encoding "
        #     # "method where possible. In general, the best encoding methods are as "
        #     # "follows: For nominal, few unique categories, use one-hot encoding. "
        #     # "For nominal, many unique categories, use target encoding or frequency "
        #     # "encoding. For ordinal features, use ordinal or label encoding. "
        #     # "Return the dataframe after encoding the categorical features."
        # )

        # st.write("##### LLM Prompt")
        # st.write(prompt)

        # # Generate LLM response
        # llm_response = generate_llm_response(working_df, prompt)
        
        # # Display LLM response
        # st.write("### LLM-Generated Description:")
        # st.write(llm_response)

        # categorical_cols = llm_response
        # st.write(categorical_cols)
        # st.write("##### Listing categorical cols")
        # for col in categorical_cols:
        #     st.write(col)

        # st.write(working_df)

        exit()

        working_df = encode_categorical(working_df)

        st.write("Categorical variables encoded.")
        st.write("Non-zero elements: ", working_df.notna().sum().sum())
        st.write("Density:", get_df_density(working_df))


        top_n = 5
        top_values = get_top_n_rows(working_df, top_n)
        st.write("Top", top_n, "rows:")
        st.dataframe(top_values)

        # 8.3 Feature Scaling
        st.subheader("Step 3: Scaling Features")

        # Before scaling, show the mean and standard deviation of numerical columns
        st.write("Feature Statistics Before Scaling:")
        print(working_df.dtypes)
        st.write(describe_numerical(working_df))

        working_df = scaler_transform(working_df)

        # After scaling, show the mean and standard deviation
        st.write("Feature Statistics After Scaling (Mean ~0, Std Dev ~1):")
        st.write(describe_numerical(working_df))

        st.write("Features scaled. Here's the updated dataset:")
        top_values = get_top_n_rows(working_df, top_n)
        st.write("Top", top_n, "rows:")
        st.dataframe(top_values)

        # 8.4 Removing Low Variance Features
        st.subheader("Step 4: Removing Low Variance Features")
        selector = VarianceThreshold(threshold=0.1)
        df_high_variance = selector.fit_transform(working_df)

        # Select the features with high variance
        selected_features = working_df.columns[selector.get_support()]
        working_df = pd.DataFrame(df_high_variance, columns=selected_features)

        st.write("Low variance features removed. Here's the updated dataset:")
        st.write(working_df.head())

        # 8.5 Handling Outliers
        st.subheader("Step 5: Handling Outliers")
        z_scores = np.abs(stats.zscore(working_df))

        st.write("Z-scores of Features (outliers detected if Z > 3):")
        st.write(pd.DataFrame(z_scores, columns=working_df.columns).head())

        # Remove outliers based on Z-score threshold
        working_df = working_df[(z_scores < 3).all(axis=1)]

        st.write(f"Outliers removed; new dataset shape: {working_df.shape}")
        st.write(working_df.head())

        # 9. Final prepared dataset
        st.subheader("Final Prepared Dataset")
        st.write(working_df.head())

        # Save the prepared dataset to a CSV file
        st.download_button(
            label="Download Prepared Dataset",
            data=working_df.to_csv(index=False),
            file_name="prepared_dataset.csv",
            mime="text/csv"
        )

if __name__ == '__main__':
    main() 