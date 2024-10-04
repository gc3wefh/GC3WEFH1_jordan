import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sweetviz as sv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy import stats

# Define a Streamlit app
st.title("ML Readiness App with Key Feature Insights")

# 1. File Upload
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset '{uploaded_file.name}' uploaded successfully.")
    
    # 2. Basic Dataset Information
    st.subheader("Basic Dataset Information")
    
    # Display dataset shape
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Display data types and missing values
    st.subheader("Data Types and Missing Values")
    st.write("Data Types:")
    st.write(df.dtypes)

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # 3. Statistical Summary
    st.subheader("Statistical Summary of Numerical Features")
    st.write(df.describe())

    # 4. Distribution of Numerical Features
    st.subheader("Distribution of Numerical Features")
    st.write("Histograms for Numerical Features:")
    fig, ax = plt.subplots(figsize=(15, 10))
    df.hist(ax=ax)
    st.pyplot(fig)

    # 5. Categorical Feature Analysis
    st.subheader("Categorical Feature Analysis")
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    st.write("Categorical Features:")
    st.write(categorical_features)

    # Display value counts for categorical features
    for col in categorical_features:
        st.write(f"\nValue counts for {col}:")
        st.write(df[col].value_counts())

    # Generate a Sweetviz report
    if st.button("Generate Sweetviz Report"):
        report = sv.analyze(df)
        report.show_html('sweetviz_report.html')
        st.markdown("[View Sweetviz Report](./sweetviz_report.html)")

    # Initialize the working DataFrame for modifications
    working_df = df.copy()

    # 6. Data Preparation Steps
    st.subheader("Step-by-Step Data Preparation")

    # 6.1 Handling Missing Values
    st.subheader("Step 1: Handling Missing Values")
    numerical_features = working_df.select_dtypes(include=[np.number]).columns.tolist()
    imputer_num = SimpleImputer(strategy='mean')
    working_df[numerical_features] = imputer_num.fit_transform(working_df[numerical_features])

    categorical_features = working_df.select_dtypes(include=['object', 'category']).columns.tolist()
    imputer_cat = SimpleImputer(strategy='most_frequent')
    working_df[categorical_features] = imputer_cat.fit_transform(working_df[categorical_features])

    st.write("Missing values handled. Here's the updated dataset:")
    st.write(working_df.head())

    # 6.2 Encoding Categorical Variables
    st.subheader("Step 2: Encoding Categorical Variables")
    working_df = pd.get_dummies(working_df, columns=categorical_features, drop_first=True)
    st.write("Categorical variables encoded. Here's the updated dataset:")
    st.write(working_df.head())

    # 6.3 Feature Scaling
    st.subheader("Step 3: Scaling Features")
    scaler = StandardScaler()

    # Before scaling, show the mean and standard deviation of numerical columns
    st.write("Feature Statistics Before Scaling:")
    st.write(working_df.describe())

    working_df = pd.DataFrame(scaler.fit_transform(working_df), columns=working_df.columns)

    # After scaling, show the mean and standard deviation
    st.write("Feature Statistics After Scaling (Mean ~0, Std Dev ~1):")
    st.write(working_df.describe())

    st.write("Features scaled. Here's the updated dataset:")
    st.write(working_df.head())

    # 6.4 Removing Low Variance Features
    st.subheader("Step 4: Removing Low Variance Features")
    selector = VarianceThreshold(threshold=0.1)
    df_high_variance = selector.fit_transform(working_df)

    # Select the features with high variance
    selected_features = working_df.columns[selector.get_support()]
    working_df = pd.DataFrame(df_high_variance, columns=selected_features)

    st.write("Low variance features removed. Here's the updated dataset:")
    st.write(working_df.head())

    # 6.5 Handling Outliers
    st.subheader("Step 5: Handling Outliers")
    z_scores = np.abs(stats.zscore(working_df))

    st.write("Z-scores of Features (outliers detected if Z > 3):")
    st.write(pd.DataFrame(z_scores, columns=working_df.columns).head())

    # Remove outliers based on Z-score threshold
    working_df = working_df[(z_scores < 3).all(axis=1)]

    st.write(f"Outliers removed; new dataset shape: {working_df.shape}")
    st.write(working_df.head())

    # 7. Final prepared dataset
    st.subheader("Final Prepared Dataset")
    st.write(working_df.head())

    # Save the prepared dataset to a CSV file
    st.download_button(
        label="Download Prepared Dataset",
        data=working_df.to_csv(index=False),
        file_name="prepared_dataset.csv",
        mime="text/csv"
    )
