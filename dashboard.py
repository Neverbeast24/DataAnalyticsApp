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

# Callback functions for mutual exclusivity
def select_all_callback():
    st.session_state["deselect_all"] = False

def deselect_all_callback():
    st.session_state["select_all"] = False

if step == "Process Data":
    st.header("🛠️ Process Data")
    uploaded_file = st.file_uploader("Upload CSV File for Processing", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data", df.head())

        st.markdown("### Select Columns to Clean")

        # Checkboxes for selecting/deselecting all with callbacks
        col1, col2 = st.columns(2)
        with col1:
            select_all = st.checkbox("Select All Columns", key="select_all", on_change=select_all_callback)
        with col2:
            deselect_all = st.checkbox("Deselect All Columns", key="deselect_all", on_change=deselect_all_callback)

        # Individual column checkboxes in two columns
        selected_columns = []
        cols_per_row = st.columns(2)

        for i, col in enumerate(df.columns):
            with cols_per_row[i % 2]:
                if select_all:
                    checked = True
                elif deselect_all:
                    checked = False
                else:
                    checked = st.checkbox(f"Include '{col}'", key=f"checkbox_{col}")
                if checked:
                    selected_columns.append(col)

        if st.button("✨ Clean Data"):
            if not selected_columns:
                st.error("Please select at least one column to clean.")
            else:
                df.replace([np.inf, -np.inf], 0, inplace=True)
                df.fillna(0, inplace=True)

                # Convert DataFrame to JSON-safe dictionary
                json_data = df.replace([np.inf, -np.inf], None).where(pd.notnull(df), None).to_dict(orient="records")

                response = requests.post(
                    f"{base_url}/process",
                    json={"data": json_data, "columns": selected_columns}
                )
                if response.status_code == 200:
                    cleaned_data = response.json()["cleaned_data"]
                    cleaned_columns = response.json()["columns"]
                    df_cleaned = pd.DataFrame(cleaned_data, columns=cleaned_columns)

                    st.success("Data cleaned successfully!")
                    st.write("🧹 Cleaned Data", df_cleaned.head())
                    st.download_button("⬇️ Download Cleaned Data", df_cleaned.to_csv(index=False), "cleaned_data.csv")
                else:
                    st.error("Error in data cleaning. Check the server logs for more details.")

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
