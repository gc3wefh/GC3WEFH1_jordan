import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Title of the app
st.title("Dataset Analysis with Streamlit")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display all variables in the dataset
    st.subheader("All Variables in the Dataset:")
    st.write(df.columns.tolist())

    # Filter for numeric variables
    numeric_vars = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    st.subheader("Numeric Variables:")
    st.write(numeric_vars)

    # Select numeric variables to visualize in a scatter matrix
    st.subheader("Scatter Matrix")
    selected_vars = st.multiselect("Select up to 5 variables for scatter matrix", numeric_vars, numeric_vars[:5])

    if selected_vars:
        scatter_matrix_df = df[selected_vars]
        scatter_matrix_fig = pd.plotting.scatter_matrix(scatter_matrix_df, alpha=0.2, figsize=(10, 10), diagonal='kde')
        for ax in scatter_matrix_fig.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=90)
            ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)
        st.pyplot(plt)

    # Scatterplot with regression line
    st.subheader("Scatterplot with Regression Line")
    x_var = st.selectbox("Select X variable", numeric_vars)
    y_var = st.selectbox("Select Y variable", numeric_vars)

    if x_var and y_var:
        if x_var != y_var:
            fig, ax = plt.subplots()
            sb.regplot(x=df[x_var], y=df[y_var], scatter_kws={"s": 20, "color": "red", "alpha": 0.2}, ax=ax)
            ax.set(xlabel=x_var, ylabel=y_var)
            st.pyplot(fig)
        else:
            st.error("X and Y variables must be different!")

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    selected_stat_var = st.selectbox("Select a variable to calculate statistics", numeric_vars)

    if selected_stat_var:
        var = df[selected_stat_var]
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
        ax.set_xlabel(selected_stat_var)
        st.pyplot(fig)
