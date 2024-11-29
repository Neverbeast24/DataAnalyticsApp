import pandas as pd
import numpy as np
import streamlit as st
import requests
import plotly.express as px
import plotly.colors as pc
import base64
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Datafluencers: InsightFlow",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def recommend_chart(df):
    """
    Recommend the best chart type based on the dataset and provide a justification.
    """
    numerical_cols, categorical_cols = categorize_columns(df)

    # Recommendation logic with justification
    if len(numerical_cols) == 1 and len(categorical_cols) >= 1:
        recommendation = "Polar Area Chart"
        justification = (
            "A **Polar Area Chart** is recommended because the dataset has one numerical column "
            f"({numerical_cols[0]}) and one or more categorical columns ({', '.join(categorical_cols)}). "
            "This chart visually represents values grouped by categories in a radial layout."
        )
    elif len(numerical_cols) >= 2 and len(categorical_cols) == 0:
        recommendation = "Scatter Plot"
        justification = (
            "A **Scatter Plot** is recommended because the dataset has multiple numerical columns "
            f"({', '.join(numerical_cols)}). This chart is ideal for visualizing relationships or correlations between two numerical variables."
        )
    elif len(numerical_cols) >= 2 and len(categorical_cols) >= 1:
        recommendation = "Stacked Bar Chart"
        justification = (
            "A **Stacked Bar Chart** is recommended because the dataset contains multiple numerical columns "
            f"({', '.join(numerical_cols)}) and at least one categorical column ({categorical_cols[0]}). "
            "This chart allows comparison of numerical values across categories, with stacking providing additional insights."
        )
    elif len(numerical_cols) > 1:
        recommendation = "Radar Chart"
        justification = (
            "A **Radar Chart** is recommended because the dataset has multiple numerical columns "
            f"({', '.join(numerical_cols)}). This chart displays multivariate data, making it suitable for comparing values across multiple axes."
        )
    elif len(numerical_cols) == 1:
        recommendation = "Area Chart"
        justification = (
            "An **Area Chart** is recommended because the dataset has a single numerical column "
            f"({numerical_cols[0]}). This chart effectively highlights trends over time or sequential data."
        )
    else:
        recommendation = "Scatter Plot"
        justification = (
            "A **Scatter Plot** is recommended as a fallback because it can handle diverse data types and allows flexible customization."
        )

    return recommendation, justification
    
# Convert your logo to Base64
logo_base64 = get_base64_image("logo2.png")  # Replace with the correct path to your logo file

base_url = "http://127.0.0.1:5000"
    
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

# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

if "cleaned_data" not in st.session_state:
    st.session_state["cleaned_data"] = None

if "clustered_data" not in st.session_state:
    st.session_state["clustered_data"] = None
# Ensure the new step is added to the sidebar
st.sidebar.header("Navigation")
step = st.sidebar.radio("Go to:", ["Home Dashboard", "Data Cleaning", "AI Clustering", "Data Visualization", "Export Data"])
# Home Dashboard Page
if step == "Home Dashboard":
    st.title("📊 Welcome to InsightFlow!")
    st.markdown("""
        <div style="text-align: center;">
            <h2 style="color: #6c63ff;">Transform Your Data Into Insights!</h2>
            <p style="font-size: 1.1rem; color: #888;">
                InsightFlow combines AI-powered clustering, data cleaning, and visualization to simplify your data journey.
            </p>
        </div>
        <hr style="border: none; border-top: 2px solid #eee; margin: 10px 0;">
    """, unsafe_allow_html=True)

    # Trivia Section
    st.subheader("📖 Did You Know?")
    trivia = [
        f"90% of the world's data has been created in the last two years.",
        "The average person generates about 1.7 MB of data every second.",
        "Data visualization can improve decision-making efficiency by up to 28%.",
        "The first computer database was developed in 1960 by Charles Bachman."
    ]
    st.markdown(f"💡 **{np.random.choice(trivia)}**")

    # Interactive Chart Section
    st.subheader("🌟 Explore Interactive Data")
    sample_data = pd.DataFrame({
        "Year": [2019, 2020, 2021, 2022],
        "Revenue (in $M)": [50, 60, 70, 85],
        "Profit (in $M)": [10, 15, 20, 25]
    })

    chart_type = st.radio("Choose a Chart Type:", ["Line Chart", "Bar Chart", "Area Chart"])
    if chart_type == "Line Chart":
        fig = px.line(sample_data, x="Year", y=["Revenue (in $M)", "Profit (in $M)"])
    elif chart_type == "Bar Chart":
        fig = px.bar(sample_data, x="Year", y=["Revenue (in $M)", "Profit (in $M)"], barmode="group")
    elif chart_type == "Area Chart":
        fig = px.area(sample_data, x="Year", y=["Revenue (in $M)", "Profit (in $M)"])

    st.plotly_chart(fig, use_container_width=True)

    # Motivational Quote
    st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <h3 style="color: #6c63ff;">"Data is a precious thing and will last longer than the systems themselves." – Tim Berners-Lee</h3>
        </div>
    """, unsafe_allow_html=True)


# Progress Indicator
progress = {
    "Data Cleaning": 20,
    "AI Clustering": 50,
    "Data Visualization": 100,
    "Export Data": 100
}
st.sidebar.markdown(
    f"""
    <div style="margin-top: 20px;">
        <label>Progress:</label>
        <progress value="{progress.get(step, 0)}" max="100" style="width: 100%;"></progress>
        <span style="font-size: 0.8rem; color: #888;">{progress.get(step, 0)}% Complete</span>
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

# Callback functions for mutual exclusivity
def select_all_callback():
    st.session_state["deselect_all"] = False

def deselect_all_callback():
    st.session_state["select_all"] = False

    
if step == "Data Cleaning":
    st.header("🛠️ Process Data")
    uploaded_file = st.file_uploader("Upload CSV File for Processing", type="csv")
    if uploaded_file:
        # Load uploaded data
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data", df.head())
        
        st.markdown("### Select Columns to Include in Cleaning")
        selected_columns = st.multiselect("Choose Columns:", df.columns, default=df.columns)

        # Cleaning Options
        st.markdown("### Choose Cleaning Operations")
        remove_duplicates = st.checkbox("Remove Duplicate Rows")
        handle_missing_values = st.checkbox("Handle Missing Values")
        standardize_column_names = st.checkbox("Standardize Column Names")
        replace_infinite_values = st.checkbox("Replace Infinite Values with NaN")
        fill_missing_values = st.checkbox("Fill Missing Values with a Default Value")

        # Additional options for missing values
        if fill_missing_values:
            fill_value = st.text_input("Enter a default value to replace missing values:", value="0")

        # Clean Data Button Logic
        if st.button("✨ Clean Data"):
            try:
                # Validate column selection
                if not selected_columns:
                    st.error("Please select at least one column to clean.")
                else:
                    # Filter the DataFrame to selected columns
                    cleaned_data = df[selected_columns]

                    # Perform cleaning operations
                    if remove_duplicates:
                        cleaned_data = cleaned_data.drop_duplicates()
                        st.info("Duplicate rows removed.")

                    if handle_missing_values:
                        cleaned_data = cleaned_data.dropna()
                        st.info("Rows with missing values removed.")

                    if standardize_column_names:
                        cleaned_data.columns = [col.strip().lower().replace(" ", "_") for col in cleaned_data.columns]
                        st.info("Column names standardized.")

                    if replace_infinite_values:
                        cleaned_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                        st.info("Infinite values replaced with NaN.")

                    # Handle missing values with validation
                    # Handle missing values with validation
                    if fill_missing_values == 'fill':
                        fill_value = st.text_input("Enter a value to fill missing values")
                        error_detected = False
                        for col in cleaned_data.columns:
                            if cleaned_data[col].isnull().any():
                                # Check if fill_value matches the column data type
                                try:
                                    expected_type = type(cleaned_data[col].dropna().iloc[0])
                                    if not isinstance(fill_value, expected_type):
                                        st.error(f"Column '{col}' expects values of type '{expected_type.__name__}', but the fill value is of type '{type(fill_value).__name__}'.")
                                        error_detected = True
                                except IndexError:
                                    # Handle cases where the column is entirely NaN
                                    st.error(f"Column '{col}' cannot be filled due to no valid type detected.")
                                    error_detected = True
                        if not error_detected:
                            cleaned_data.fillna(fill_value, inplace=True)
                            st.success(f"Missing values filled with '{fill_value}'.")

                    # Download button for cleaned data
                    st.download_button(
                        label="⬇️ Download Cleaned Data",
                        data=cleaned_data.to_csv(index=False),
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error during data cleaning: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")
        
if step == "AI Clustering":
    st.title("🤖 AI Clustering")
    uploaded_file = st.file_uploader("Upload CSV for Clustering", type="csv")

    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data", df.head())

        # Identify numerical columns
        numerical_cols, _ = categorize_columns(df)

        if numerical_cols:
            st.markdown("### Select Columns for Clustering")
            selected_columns = st.multiselect("Choose Numerical Columns:", numerical_cols, default=numerical_cols)

            if selected_columns:
                num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
                if st.button("Perform Clustering"):
                    try:
                        # Data preprocessing
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[selected_columns])

                        # KMeans clustering
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                        clusters = kmeans.fit_predict(scaled_data)
                        df["Cluster"] = clusters

                        st.write("🗂️ Clustered Data", df.head())

                        # Visualize clusters (if 2D or 3D is possible)
                        if len(selected_columns) == 2:
                            fig = px.scatter(
                                df,
                                x=selected_columns[0],
                                y=selected_columns[1],
                                color="Cluster",
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            fig.update_layout(title="Cluster Visualization")
                            st.plotly_chart(fig, use_container_width=True)

                        st.download_button(
                            label="⬇️ Download Clustered Data",
                            data=df.to_csv(index=False),
                            file_name="clustered_data.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error during clustering: {e}")
        else:
            st.warning("The dataset does not have numerical columns for clustering.")
    else:
        st.warning("Please upload a CSV file to proceed.")


# Step: Visualize Data
elif step == "Data Visualization":
    st.header("📊 Visualize Data")
    uploaded_file = st.file_uploader("Upload CSV File for Visualization", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data")
        st.dataframe(df, use_container_width=True)

         # Button to recommend a chart
        if st.button("🔍 Recommend Chart"):
            # Recommend a chart and display justification
            recommended_chart, justification = recommend_chart(df)
            st.success(f"**Recommended Chart Type:** {recommended_chart}")
            st.markdown(f"**Justification:** {justification}")

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
    # Step: Export Data
elif step == "Export Data":
    st.header("📤 Export Data")
    if st.session_state["clustered_data"] is not None:
        st.download_button(
            label="Download Clustered Data",
            data=st.session_state["clustered_data"].to_csv(index=False),
            file_name="clustered_data.csv",
            mime="text/csv"
        )
    elif st.session_state["cleaned_data"] is not None:
        st.download_button(
            label="Download Cleaned Data",
            data=st.session_state["cleaned_data"].to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data available for export. Please process your data first.")