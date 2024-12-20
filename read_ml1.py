import streamlit as st
import pandas as pd
import numpy as np
import sweetviz as sv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy import stats

# Define a Streamlit app
st.title("ML Readiness App with Sweetviz - Sequential Steps")

# 1. File Upload
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset '{uploaded_file.name}' uploaded successfully.")
    st.write("Original Dataset:")
    st.write(df.head())

    # Generate a Sweetviz report
    if st.button("Generate Sweetviz Report"):
        report = sv.analyze(df)
        report.show_html('sweetviz_report.html')
        st.markdown("[View Sweetviz Report](./sweetviz_report.html)")

    # 2. Data Preparation Steps in Sequence
    working_df = df.copy()

    # 2.1 Handling Missing Values
    st.subheader("Step 1: Handling Missing Values")
    numerical_features = working_df.select_dtypes(include=[np.number]).columns.tolist()
    imputer_num = SimpleImputer(strategy='mean')
    working_df[numerical_features] = imputer_num.fit_transform(working_df[numerical_features])

    categorical_features = working_df.select_dtypes(include=['object', 'category']).columns.tolist()
    imputer_cat = SimpleImputer(strategy='most_frequent')
    working_df[categorical_features] = imputer_cat.fit_transform(working_df[categorical_features])

    st.write("Missing values handled. Here's the updated dataset:")
    st.write(working_df.head())

    # 2.2 Encoding Categorical Variables
    st.subheader("Step 2: Encoding Categorical Variables")
    working_df = pd.get_dummies(working_df, columns=categorical_features, drop_first=True)
    st.write("Categorical variables encoded. Here's the updated dataset:")
    st.write(working_df.head())

    # 2.3 Feature Scaling
    st.subheader("Step 3: Scaling Features")
    scaler = StandardScaler()
    working_df = pd.DataFrame(scaler.fit_transform(working_df), columns=working_df.columns)
    st.write("Features scaled. Here's the updated dataset:")
    st.write(working_df.head())

    # 2.4 Removing Low Variance Features
    st.subheader("Step 4: Removing Low Variance Features")
    selector = VarianceThreshold(threshold=0.1)
    df_high_variance = selector.fit_transform(working_df)
    
    # Select the features that have higher variance
    selected_features = working_df.columns[selector.get_support()]
    working_df = pd.DataFrame(df_high_variance, columns=selected_features)
    
    st.write("Low variance features removed. Here's the updated dataset:")
    st.write(working_df.head())

    # 2.5 Handling Outliers
    st.subheader("Step 5: Handling Outliers")
    z_scores = np.abs(stats.zscore(working_df))
    working_df = working_df[(z_scores < 3).all(axis=1)]
    
    st.write(f"Outliers removed; new dataset shape: {working_df.shape}")
    st.write(working_df.head())

    # 3. Final prepared dataset
    st.subheader("Final Prepared Dataset")
    st.write(working_df.head())

    # Save the prepared dataset to a CSV file
    st.download_button(
        label="Download Prepared Dataset",
        data=working_df.to_csv(index=False),
        file_name="prepared_dataset.csv",
        mime="text/csv"
    )
