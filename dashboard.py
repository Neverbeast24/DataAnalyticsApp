import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.colors as pc
import base64
import re

st.set_page_config(
    page_title="Datafluencers: InsightFlow",
    layout="wide"
)

# Function to convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert your logo to Base64
logo_base64 = get_base64_image("logo2.png")  # Replace with the correct path to your logo file

# Sidebar: Enlarged Logo and Title
st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" alt="InsightFlow Logo" style="width: 150px; margin-bottom: 10px;">
        <h1 style="margin: 0; font-size: 1.8rem;">InsightFlow</h1>
    </div>
    <p style="font-size: 0.9rem; color: #888; text-align: center;">A step-by-step guide to process and visualize your data.</p>
    <hr style="border: none; border-top: 2px solid #eee; margin: 10px 0;">
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation with Progress Bar
st.sidebar.header("Step-by-Step Process")
step = st.sidebar.selectbox("Choose a Step", ["Process Data", "Visualize Data"])

# Progress Indicator
progress = {"Process Data": 50, "Visualize Data": 100}
st.sidebar.markdown(
    f"""
    <div style="margin-top: 20px;">
        <label>Progress:</label>
        <progress value="{progress[step]}" max="100" style="width: 100%;"></progress>
        <span style="font-size: 0.8rem; color: #888;">{progress[step]}% Complete</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Add Background and Styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .css-1d391kg { /* Background for main area */
        background: linear-gradient(135deg, #1e1e2f, #2a2a48);
        color: white;
    }
    .stButton>button { /* Custom button styling */
        background-color: #6c63ff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTextInput input {
        border: 1px solid #6c63ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to categorize columns
def categorize_columns(df):
    """
    Identifies numerical and categorical columns in the dataset.
    """
    numerical_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    categorical_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
    return numerical_cols, categorical_cols

# Default Color Palette
default_palette = px.colors.qualitative.Set1

# Step: Process Data
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
            st.success("Data cleaned successfully!")
            st.write("🧹 Cleaned Data", df.head())
            st.download_button("⬇️ Download Cleaned Data", df.to_csv(index=False), "cleaned_data.csv")
    else:
        st.warning("Please upload a CSV file to proceed.")



# Step: Visualize Data
elif step == "Visualize Data":
    st.header("📊 Visualize Data")
    uploaded_file = st.file_uploader("Upload CSV File for Visualization", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data")
        st.dataframe(df, use_container_width=True)

        # Categorize columns
        numerical_cols, categorical_cols = categorize_columns(df)

        # Allow user to select chart type
        chart_type = st.selectbox("📈 Choose Chart Type", [
            "Polar Area Chart", "Radar Chart", "Area Chart", "Scatter Plot", "Stacked Bar Chart"
        ])

        # Column selection dynamically based on chart type
        st.markdown("### Column Selection")
        if chart_type == "Polar Area Chart":
            theta_col = st.selectbox("Select a column for 'theta' (categorical):", categorical_cols)
            r_col = st.selectbox("Select a column for 'r' (numerical):", numerical_cols)

        elif chart_type == "Radar Chart":
            theta_col = st.selectbox("Select a column for 'theta' (categorical):", categorical_cols)
            r_col = st.selectbox("Select a column for 'r' (numerical):", numerical_cols)

        elif chart_type == "Area Chart":
            x_col = st.selectbox("Select a column for 'x' (numerical or sequential):", numerical_cols)
            y_col = st.selectbox("Select a column for 'y' (numerical):", numerical_cols)

        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-Axis (numerical):", numerical_cols)
            y_col = st.selectbox("Select Y-Axis (numerical):", numerical_cols)

        elif chart_type == "Stacked Bar Chart":
            x_col = st.selectbox("Select a column for 'x' (categorical):", categorical_cols)
            y_col = st.selectbox("Select a column for 'y' (numerical):", numerical_cols)
            color_col = st.selectbox("Select a column for 'color' (categorical):", categorical_cols)

        # Chart customization options
        st.markdown("### Chart Customization and Output")
        chart_title = st.text_input("Chart Title", value=f"My {chart_type}")
        show_legend = st.checkbox("Show Legend", value=True)
        fill_option = st.checkbox("Fill Area (if applicable)", value=False) if chart_type in ["Radar Chart", "Area Chart"] else None

        # Generate the chart
        if st.button("Generate Chart"):
            try:
                if chart_type == "Polar Area Chart":
                    fig = px.bar_polar(df, r=r_col, theta=theta_col, color=theta_col, color_discrete_sequence=px.colors.qualitative.Plotly)
                    fig.update_layout(title=chart_title, showlegend=show_legend)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Radar Chart":
                    st.markdown("### Radar Chart Configuration")

                    # Allow multiple numerical columns for axes
                    selected_numerical_cols = st.multiselect("Select numerical columns for radar axes:", numerical_cols)

                    # Optional grouping column for color encoding
                    grouping_col = st.selectbox("Select a column for grouping (optional):", [None] + categorical_cols)

                    # Check if at least one numerical column is selected
                    if selected_numerical_cols:
                        # Reshape data into long format for radar chart
                        radar_data = pd.melt(
                            df,
                            id_vars=[grouping_col] if grouping_col else None,
                            value_vars=selected_numerical_cols,
                            var_name="Axis",
                            value_name="Value"
                        )

                        # Generate the chart only when the button is clicked
                        generate_chart = st.button("Generate Chart")

                        # Render the radar chart after the button is clicked
                        if generate_chart:
                            try:
                                # Create the radar chart
                                fig = px.line_polar(
                                    radar_data,
                                    r="Value",
                                    theta="Axis",
                                    color=grouping_col if grouping_col else None,
                                    line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.Plotly
                                )

                                # Optional fill and layout customization
                                if fill_option:
                                    fig.update_traces(fill="toself")  # Proper way to fill radar chart
                                fig.update_layout(
                                    title=f"My {chart_type}",
                                    showlegend=show_legend
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.error(f"An error occurred while creating the radar chart: {e}")
                    else:
                        st.warning("Please select at least one numerical column for the radar chart.")

                elif chart_type == "Area Chart":
                    fig = px.area(df, x=x_col, y=y_col)
                    if fill_option:
                        fig.update_traces(fill="tonexty")
                    fig.update_layout(title=chart_title, showlegend=show_legend)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col)
                    fig.update_layout(title=chart_title, showlegend=show_legend)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Stacked Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode="stack", color_discrete_sequence=px.colors.qualitative.Plotly)
                    fig.update_layout(title=chart_title, showlegend=show_legend)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while creating the {chart_type}: {e}")