"""Machine Learning Readiness Assessment on CSV files using Streamlit.

File: ML_readiness_Gemini.py
Author: Alex Nguyen

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
from openai import OpenAI
import os
from dotenv import load_dotenv
from operator import itemgetter
import time
import google.generativeai as genai
import json


load_dotenv(override=True)  # Load env vars from .env file

# Set the api key from environment variable
api_key = os.environ.get('openai')  # Make sure your API key is stored as an environment variable
print(api_key)

client = OpenAI(
    api_key=api_key,
)

# LLM Integration
#genai.configure(api_key=api_key)
#model = genai.GenerativeModel("gemini-1.5-flash")
#llm = OpenAI(api_token=api_key)

def generate_llm_response(df: pd.DataFrame, prompt: str):
    # df_json = df.to_json()
    # p = f"""
    #     The following is a dataset in JSON format:\n
    #     {df_json}\n
    #     {prompt}
    #     """
    # print("Printing prompt...\n")
    # print(p)
    # response = model.generate_content(p)
    # return response.text
    #answer = pandas_agent.chat(prompt)
    #pandas_agent = PandasAI(llm, conversational=False)
    df_json = df.to_json()
    query = f"""The following is a dataset in JSON format:\n
            {df_json}\n\n
            {prompt}
            """
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for analyzing data."},
                {"role": "user", "content": query}
            ]
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")

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
        downcasted_df[col] = pd.to_numeric(df[col], downcast='integer')
    # Downcast float columns
    for col in downcasted_df.select_dtypes(include=['float']).columns:
        downcasted_df[col] = pd.to_numeric(df[col], downcast='float')
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
        'Missing Percentage (%)': (missing_values / len(df)) * 100,
        'General Info': df.info()
    })
    return missing_data_df

@st.cache_data
def encode_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Encodes categorical columns using OneHotEncoder

    Uses OneHotEncoder to encode categorical data into 1s and 0s, represented in
    sparse data. Returns a new Dataframe that has original non-categorical data
    and new encoded categorical data. Designed to encode one column at a time.

    Args:
        df: Input Dataframe with categorical data columns.
        column: Name of column in dataframe that is to be encoded
    
    Returns:
        New dataframe with encoded categorical data.
    """
    encoder = OneHotEncoder(sparse_output=True)

    encoded_arr = encoder.fit_transform(df[column].values.reshape(-1,1))
    encoded_categorical_df = pd.DataFrame.sparse.from_spmatrix(
        encoded_arr,
        columns=encoder.get_feature_names_out([column])
    )
    non_categorical_df = df.drop(column, axis=1)

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
        col for col in df_topN.columns if isinstance(df_topN[col].dtype, pd.SparseDtype)
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

def scaler_transform(df: pd.DataFrame, numerical_columns: list[str]) -> pd.DataFrame:
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
    for col in numerical_columns:
        df_scaled[f"{col}_scaled"] = scaler.fit_transform(df_scaled[[col]])
    return df_scaled

def describe_numerical(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    print("Printing df", df)
    #numerical_cols = ["AgeatDiagnosis"]
    print("Printing df[numericals_cols]", df[numerical_cols])
    return df[numerical_cols].describe()

def main():
    print("Printing API Key: ", api_key)

    start = time.perf_counter()
    # Define a Streamlit app
    st.title("ML Readiness App with Key Feature Insights")

    # 1. File Upload
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is None:
        st.write("No data to show. Upload a file to generate data.")
    else:
        st.write("Loading dataset...")
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

        st.write("Dataframe in JSON format:")
        st.write(df.head().to_json())

        # Identify numerical and categorical columns in the prompt
        prompt = (
            """
            You are tasked with identifying the column types of a given DataFrame. Analyze the provided DataFrame schema and classify each column as either numerical or categorical, excluding any ID-related columns.

            Classification rules:

                Columns with purely numerical data (including continuous or sparse types) should be labeled as numerical.
                Columns with non-numerical data or discrete categories should be labeled as categorical.
                Exclude columns that are clearly ID-related (e.g., PatientID, DiagnosisID, HealthFacilityID, or similar naming patterns containing "ID" or implying unique identifiers).
                Return the result as a strict JSON object with the following structure:
                { 
                    "categorical_columns": ["column1", "column2", ... , "columnx"], 
                    "numerical_columns": ["column1", "column2", ... , "columnx"]
                } 
                Return only the JSON object without any explanations or commentary.
            """
        )
        st.write("##### LLM Prompt")
        st.write(prompt)

        # st.write('##### Dataframe before LLM')
        # st.write(working_df)
        # Generate LLM response
        llm_response = generate_llm_response(df.head(), prompt)
        
        # Display LLM response
        st.write("### LLM-Generated Description:")
        st.write(llm_response)
        print(llm_response.strip("```json\n").strip("```"))

        st.write("End of LLM-Generated Description")
        response_json = json.loads(llm_response.strip("```json\n").strip("```"))

        numerical_cols = response_json["numerical_columns"]
        categorical_cols = response_json["categorical_columns"]

        st.write(f"Numerical columns: {numerical_cols}")
        st.write(f"Categorical columns: {categorical_cols}")

        

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
        working_df = df.copy()

        # 8. Data Preparation Steps
        st.subheader("Step-by-Step Data Preparation")

        ## 8.1 Handling Missing Values
        st.subheader("Step 1: Handling Missing Values")

        # # Combine the relationship analysis with distribution, spikes, and outliers in the prompt
        # prompt = (
        # )

        # st.write("##### LLM Prompt")
        # st.write(prompt)

        # st.write('##### Dataframe before LLM')
        # st.write(working_df)
        # # Generate LLM response
        # llm_response = generate_llm_response(working_df, prompt)
        
        # # Display LLM response
        # st.write("### LLM-Generated Description:")
        # st.write(llm_response)

        # END SKIP

        ### Impute missing values for numerical features with mean
        # HARD CODED
        #numerical_cols = ["AgeatDiagnosis"]
        imputer_num = SimpleImputer(strategy='mean')
        working_df[numerical_cols] = imputer_num.fit_transform(working_df[numerical_cols])

        ### Impute missing values for categorical features with the most frequent value
        # HARD CODED
        #categorical_cols = ["Gender", "Governorate", "Diagnosis", "DateTimeDiagnosisEntered", "PatientAllergy", "HealthFacilityType", "HealthFacility"]

        imputer_cat = SimpleImputer(strategy='most_frequent')
        working_df[categorical_cols] = imputer_cat.fit_transform(working_df[categorical_cols])

        st.write(
            "Missing values handled. Here's a preview of the updated dataset "
            f"{working_df.shape}:"
        )
        st.write(working_df.head())

        downcasted_df = downcast_numeric_columns(df)
        missing_data_df = show_dtypes_and_missing_vals(downcasted_df)

        # Show missing values in a more detailed and clear way
        st.dataframe(missing_data_df.style.format(
            {"Missing Percentage (%)": "{:.2f}"}
        ).background_gradient(
            cmap='Reds', subset=["Missing Values"]
        ))

        # 8.2 Encoding Categorical Variables
        st.subheader("Step 2: Encoding Categorical Variables")

        # df_string = working_df.to_string()
        # Define categorical features
        # prompt = (
        #     """
        #     Numerical data is a type of data that expresses information in the 
        #     form of numbers. Categorical data is a type of data that is used to 
        #     group information with similar characteristics. IDs are not 
        #     considered numerical data nor categorical data. Identify all 
        #     columns containing categorical data and all columns containing 
        #     numerical data in the dataframe and return them as a JSON list. The 
        #     response should be in the format of: 
        #     { 
        #         "categorical_columns": ["column1", "column2", ... , "columnx"], 
        #         "numerical_columns": ["column1", "column2", ... , "columnx"]
        #     } 
        #     and should not include anything else.
        #     """
        # )

        # st.write("##### LLM Prompt")
        # st.write(prompt)

        # # Generate LLM response
        # llm_response = generate_llm_response(working_df, prompt)
        
        # # Display LLM response
        # st.write("### LLM-Generated Description:")
        # st.write(llm_response)

        # # categorical_cols = llm_response
        # # st.write(categorical_cols)
        # # st.write("##### Listing categorical cols")
        # # for col in categorical_cols:
        # #     st.write(col)

        # st.write("End of LLM-Generated answer")
        # st.write(working_df)

        # Record categorical columns
        # categorical_cols = llm_response["categorical_columns"].to_list()
        
        # # Define numerical features
        # prompt = (
        #     "Identify all numerical columns in the "
        #     "DataFrame and return them as a JSON list. The response should be in"
        #     " the format: "
        #     "{"
        #         "\"numerical_columns\": [\"column1\", \"column2\"],"
        #     "}."
        # )

        # st.write("##### LLM Prompt")
        # st.write(prompt)

        # # Generate LLM response
        # llm_response = generate_llm_response(working_df, prompt)
        
        # # Display LLM response
        # st.write("### LLM-Generated Description:")
        # st.write(llm_response)

        # # categorical_cols = llm_response
        # # st.write(categorical_cols)
        # # st.write("##### Listing categorical cols")
        # # for col in categorical_cols:
        # #     st.write(col)

        # st.write("End of LLM-Generated answer")

        # Record numerical columns
        # numerical_cols = llm_response["numerical_columns"].to_list()

        print("Number of categorical cols: ", len(categorical_cols))
        for ind, obj in enumerate(categorical_cols):
            while working_df[categorical_cols[ind]].nunique() > (0.3 * len(working_df)):
                categorical_cols.remove(categorical_cols[ind])
            print("Index: ", ind)
            print(categorical_cols[ind])
            print(working_df[categorical_cols[ind]].nunique())

            categorical_cols[ind] = (categorical_cols[ind], working_df[categorical_cols[ind]].nunique())

        print("Number of categorical cols: ", len(categorical_cols))

        for pair in categorical_cols:
            print(pair)
            print("Type of nunique(): ", type(pair[1]))

        # Record numerical columns


        # Sort categorical columns by number of unique entries
        sorted_categorical_cols = sorted(
            categorical_cols,
            key=itemgetter(1),
            reverse=False
        )

        for x in range(5):
            working_df = encode_categorical(working_df, sorted_categorical_cols[x][0])


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
        st.write("Numerical Feature Statistics Before Scaling:")
        print(working_df.dtypes)
        st.write(describe_numerical(working_df, numerical_cols))

        working_df = scaler_transform(working_df, numerical_cols)

        # After scaling, show the mean and standard deviation
        st.write("Numerical Feature Statistics After Scaling (Mean ~0, Std Dev ~1):")
        st.write(describe_numerical(working_df, numerical_cols))

        st.write("Features scaled to standard. Here's the updated dataset:")
        top_values = get_top_n_rows(working_df, top_n)
        st.write("Top", top_n, "rows:")
        st.dataframe(top_values)

        # 8.4 Removing Low Variance Features
        st.subheader("Step 4: Removing Low Variance Features")

        encoded_categorical_columns = [col for col in working_df.columns if isinstance(working_df[col].dtype, pd.SparseDtype)]

        print(working_df.dtypes)
        print(encoded_categorical_columns)

        ignored_cols = [col for col in working_df.columns if (col not in encoded_categorical_columns and col not in numerical_cols)]
        print("ignored columns: ", ignored_cols)

        selector = VarianceThreshold(threshold=0.1)

        # Apply variance selector to numeric data only
        numeric_data = working_df[numerical_cols]
        numeric_high_variance = selector.fit_transform(numeric_data)
        numeric_high_variance_df = pd.DataFrame(
            numeric_high_variance,
            columns=numeric_data.columns[selector.get_support()]
        )
        # Recombine with categorical data
        df_high_variance = pd.concat([working_df[ignored_cols], numeric_high_variance_df, working_df[encoded_categorical_columns]], axis=1)

        st.write("Low variance features removed. Here's the updated dataset:")
        top_values = get_top_n_rows(df_high_variance, top_n)
        st.write("Top", top_n, "rows:")
        st.dataframe(top_values)

        # 8.5 Handling Outliers
        st.subheader("Step 5: Handling Outliers")

        # Calculate Z scores of numerical columns and taking the absolute value of them, storing into a dataframe
        z_scores = np.abs((df_high_variance[numerical_cols]).apply(stats.zscore))

        st.write("Z score: ", z_scores)

        st.write("Z-scores of Features (outliers detected if Z > 3):")
        st.write(pd.DataFrame(z_scores, columns=numerical_cols).head())

        numerical_data_no_outliers = df_high_variance[numerical_cols][(z_scores < 3).all(axis=1)]

        df_final = pd.concat([df_high_variance[ignored_cols], numerical_data_no_outliers, df_high_variance[encoded_categorical_columns]], axis=1)

        st.write(f"Outliers removed; new dataset shape: {df_final.shape}")
        top_values = get_top_n_rows(df_final, top_n)
        st.write("Top", top_n, "rows:")
        st.dataframe(top_values)

        # 9. Final prepared dataset
        st.subheader("Final Prepared Dataset")
        top_values = get_top_n_rows(df_final, top_n)
        st.write("Top", top_n, "rows:")
        st.dataframe(top_values)

        # Save the prepared dataset to a CSV file
        st.download_button(
            label="Download Prepared Dataset",
            data=working_df.to_csv(index=False),
            file_name="prepared_dataset.csv",
            mime="text/csv"
        )

        end = time.perf_counter()
        runtime = end - start
        print(f"Runtime: {runtime // 60} minutes {runtime % 60} seconds")

if __name__ == '__main__':
    main() 