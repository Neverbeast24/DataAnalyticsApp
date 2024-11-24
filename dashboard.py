import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.title("Data Dashboard Application")

st.sidebar.header("Step-by-Step Process")
step = st.sidebar.selectbox("Choose a Step", ["Upload File", "Process Data", "Visualize Data"])

base_url = "http://127.0.0.1:5000"

if step == "Upload File":
    st.header("Upload File")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:
        response = requests.post(f"{base_url}/upload", files={"file": uploaded_file.getvalue()})
        if response.status_code == 200:
            st.success("File uploaded successfully!")
            st.write("Columns:", response.json().get('columns', []))
        else:
            st.error(response.json().get('error', 'An error occurred'))

elif step == "Process Data":
    st.header("Process Data")
    uploaded_file = st.file_uploader("Upload CSV File for Processing", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data", df.head())
        
        if st.button("Clean Data"):
            response = requests.post(f"{base_url}/process", json=df.to_dict(orient='records'))
            if response.status_code == 200:
                df_cleaned = pd.DataFrame(response.json())
                st.write("Cleaned Data", df_cleaned.head())
                st.download_button("Download Cleaned Data", df_cleaned.to_csv(index=False), "cleaned_data.csv")
            else:
                st.error("Error in data processing")

elif step == "Visualize Data":
    st.header("Visualize Data")
    uploaded_file = st.file_uploader("Upload CSV File for Visualization", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data", df.head())

        chart_type = st.selectbox("Choose Chart Type", ["Polar Area Chart", "Area Chart", "Radar Chart", "Stacked Bar Chart", "Scatter Plot"])

        if chart_type == "Polar Area Chart":
            fig = px.line_polar(df, r=df[df.columns[1]], theta=df[df.columns[0]], line_close=True)
            st.plotly_chart(fig)
        elif chart_type == "Area Chart":
            fig = px.area(df, x=df.columns[0], y=df.columns[1])
            st.plotly_chart(fig)
        elif chart_type == "Radar Chart":
            fig = px.line_polar(df, r=df[df.columns[1]], theta=df[df.columns[0]], line_close=True)
            st.plotly_chart(fig)
        elif chart_type == "Stacked Bar Chart":
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], color=df.columns[2], barmode="stack")
            st.plotly_chart(fig)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[2])
            st.plotly_chart(fig)
