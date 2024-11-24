import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Datafluencers Studio",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Datafluencers Studio: Data Dashboard")
st.sidebar.header("Step-by-Step Process")
step = st.sidebar.selectbox("Choose a Step", ["Upload File", "Process Data", "Visualize Data"])

base_url = "http://127.0.0.1:5000"

if step == "Process Data":
    st.header("🛠️ Process Data")
    uploaded_file = st.file_uploader("Upload CSV File for Processing", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data", df.head())

        selected_columns = st.multiselect("Select Columns to Clean", df.columns, default=df.columns.tolist())
        if st.button("✨ Clean Data"):
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)

            response = requests.post(
                f"{base_url}/process",
                json={"data": df.to_dict(orient="records"), "columns": selected_columns}
            )
            if response.status_code == 200:
                cleaned_data = response.json()["cleaned_data"]
                cleaned_columns = response.json()["columns"]
                df_cleaned = pd.DataFrame(cleaned_data, columns=cleaned_columns)

                st.success("Data cleaned successfully!")
                st.write("🧹 Cleaned Data", df_cleaned.head())
                st.download_button("⬇️ Download Cleaned Data", df_cleaned.to_csv(index=False), "cleaned_data.csv")
            else:
                st.error("Error in data cleaning.")

elif step == "Visualize Data":
    st.header("📊 Visualize Data")
    uploaded_file = st.file_uploader("Upload CSV File for Visualization", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data", df.head())

        selected_columns = st.multiselect("Select Columns to Visualize", df.columns, default=df.columns[:2])
        chart_type = st.selectbox("📈 Choose Chart Type", ["Polar Area Chart", "Area Chart", "Radar Chart", "Stacked Bar Chart", "Scatter Plot"])

        if len(selected_columns) >= 2:
            if chart_type == "Polar Area Chart":
                fig = px.line_polar(df, r=selected_columns[1], theta=selected_columns[0], line_close=True)
            elif chart_type == "Area Chart":
                fig = px.area(df, x=selected_columns[0], y=selected_columns[1])
            elif chart_type == "Radar Chart":
                fig = px.line_polar(df, r=selected_columns[1], theta=selected_columns[0], line_close=True)
            elif chart_type == "Stacked Bar Chart":
                if len(selected_columns) >= 3:
                    fig = px.bar(df, x=selected_columns[0], y=selected_columns[1], color=selected_columns[2], barmode="stack")
                else:
                    st.error("Stacked Bar Chart requires at least 3 selected columns.")
                    fig = None
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], color=selected_columns[2] if len(selected_columns) > 2 else None)

            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Please select at least 2 columns for visualization.")
