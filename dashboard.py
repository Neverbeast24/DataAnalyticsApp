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
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.plot import plot_plotly
import warnings
import requests

# predictive, data vizualization 
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="InsightFlow Studio",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## InsightFlow Studio\nAI-Driven Analytics Platform"
    }
)

# Function to convert an image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def recommend_chart(df):
    """
    Recommend the best chart type based on the dataset and provide a justification.
    """
    numerical_cols, categorical_cols = categorize_columns(df)

    # Check for empty lists and provide a clear message
    if not numerical_cols:
        return "Error", "No numerical columns found in the dataset. Please upload a dataset with numerical columns."

    if not categorical_cols:
        return "Error", "No categorical columns found in the dataset. Please upload a dataset with categorical columns."

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
    elif len(categorical_cols) > len(numerical_cols):
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
    else:
        recommendation = "Error"
        justification = "The dataset does not meet the required conditions for a specific chart type. Please ensure the dataset has appropriate columns for visualization."

    return recommendation, justification

# Replace with your logo path
logo_base64 = get_base64_image("logo-new.png")

# Initialize Session State for Navigation
if 'active_page' not in st.session_state:
    st.session_state['active_page'] = 'Dashboard'

# Function to Set the Active Page
def set_page(page_name):
    st.session_state['active_page'] = page_name

def sidebar():
    # Sidebar Logo
    st.sidebar.image(f"data:image/png;base64,{logo_base64}", use_container_width=True)
    st.sidebar.markdown("<p style='text-align: center; color: #00bfae; font-size: 1.2rem;'>AI-Driven Analytics Platform</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='border-top: 2px solid #00bfae;'>", unsafe_allow_html=True)

    # Pages Dictionary and Progress Mapping
    pages = {
        "Dashboard": 0,
        "Data Cleaning": 20,
        "Clustering": 50,
        "Data Visualization": 80,
        "Predictive Analytics": 100, 
        "Gemini AI": 100
    }

    st.markdown("""
    <style>
        .stButton>button {
            background-color: #0E1117;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 14px;
            width: 100%;
            text-align: left;
            margin-bottom: 5px;
            font-weight: normal; /* Default font weight */
            border: 2px solid transparent; /* Default border */
        }
        .stButton>button:hover, .stButton>button.selected {
            background-color: #00bfae;
            color: #1b2a41;
            font-weight: bold; /* Bold font on hover */
            border: 2px solid transparent; /* Ensure no border appears on hover */
        }
    </style>
    """, unsafe_allow_html=True)

    # Generate navigation buttons
    for page_name in pages.keys():
        # Add "selected" class if the button is for the current active page
        button_class = "selected" if st.session_state['active_page'] == page_name else ""

        # Create the button
        if st.sidebar.button(page_name, key=page_name):
            st.session_state['active_page'] = page_name  # Update the active page

    progress = {
    "Dashboard": 0,
    "Data Cleaning": 20,
    "Clustering": 50,
    "Data Visualization": 80,
    "Predictive Analytics": 100,
    "Gemini AI": 100
    }

    st.sidebar.markdown(
        f"""
        <div style="margin-top: 20px;">
            <label>Progress:</label>
            <progress value="{progress.get(st.session_state['active_page'], 0)}" max="100" style="width: 100%;"></progress>
            <span style="font-size: 0.8rem; color: #FFFFFF;">{progress.get(st.session_state['active_page'], 0)}% Complete</span>
        </div>
        """,
        unsafe_allow_html=True
    )
     # Add "About" section at the bottom
    st.sidebar.markdown("<hr style='border-top: 2px solid #00bfae;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="text-align: center; font-size: 0.9rem; color: #e4e9f0;">
            <strong>About InsightFlow Studio</strong><br>
            InsightFlow Studio is an AI-driven analytics platform designed to streamline data preprocessing, clustering, visualization, and predictive analytics. Built for ease of use, scalability, and precision.
            <br><br>
            © 2024 InsightFlow Studio. All rights reserved.
        </div>
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
    
def style_page():
    st.markdown("""
    <style>
        /* General body styling */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #121212;
            color: #ffffff;
        }

        /* Sidebar styling */
        .css-1aumxhk {
            background: linear-gradient(135deg, #0066ff, #00ccff);
            color: #ffffff;
        }
        .css-1aumxhk .css-8c4qvf, .css-1aumxhk .css-nqowgj {
            color: #ffffff;
        }
        .css-1aumxhk .css-8c4qvf:hover {
            background-color: #0057e7 !important;
            font-weight: bold;
        }

        /* Title and navigation buttons */
        .title-container {
            text-align: center;
            margin: 20px 0;
        }
        .title-container h1 {
            color: #00ffcc;
            font-size: 2.8rem;
            font-weight: bold;
            animation: fadeIn 2s;
        }
        .title-container p {
            color: #66d9ff;
            font-size: 1.3rem;
            animation: fadeIn 2s ease-in-out;
        }

        /* Button container styling */
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .button-container button {
            background: linear-gradient(to right, #0077ff, #00e6e6);
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            transition: transform 0.3s, background 0.3s;
        }
        .button-container button:hover {
            background: linear-gradient(to right, #00e6e6, #0077ff);
            font-weight: bold;
            transform: scale(1.1);
        }

        /* Cards styling */
        .section-card {
            background: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
            animation: slideIn 1.5s;
        }
        .section-card h2 {
            color: #00bfae;
            text-align: center;
        }
        .section-card p {
            color: #cccccc;
        }

        /* Animations */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }

        @keyframes slideIn {
            0% { transform: translateY(20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }

    </style>
    """, unsafe_allow_html=True)

def dashboard_page():
    style_page()

    # Title Section with Animation
    st.markdown("""
        <div class="title-container">
            <h1>📊 Visualize with InsightFlow!</h1>
            <p>Transform your data into powerful insights with ease.</p>
        </div>
    """, unsafe_allow_html=True)

    # Buttons for Navigation with Smooth Scrolling
    st.markdown("""
        <div class="button-container">
            <button onclick="document.getElementById('trivia').scrollIntoView({ behavior: 'smooth' })"> Data Cleaning </button>
            <button onclick="document.getElementById('interactive-charts').scrollIntoView({ behavior: 'smooth' })">Clustering </button>
            <button onclick="document.getElementById('quote').scrollIntoView({ behavior: 'smooth' })"> Data Visualization</button>
            <button onclick="document.getElementById('interactive-charts').scrollIntoView({ behavior: 'smooth' })">Predictive Analytics </button>
            <button onclick="document.getElementById('quote').scrollIntoView({ behavior: 'smooth' })"> Gemini AI</button>
        </div>
    """, unsafe_allow_html=True)

    # Interactive Chart Section
    with st.container():
        st.markdown('<div id="interactive-charts" class="section-card"><h2>📈 Explore Interactive Data</h2></div>', unsafe_allow_html=True)
        sample_data = pd.DataFrame({
            "Year": [2019, 2020, 2021, 2022],
            "Revenue (in $M)": [50, 60, 70, 85],
            "Profit (in $M)": [10, 15, 20, 25]
        })

        chart_type = st.radio("Choose a Chart Type:", ["Line Chart", "Bar Chart", "Area Chart"], horizontal=True)
        if chart_type == "Line Chart":
            fig = px.line(sample_data, x="Year", y=["Revenue (in $M)", "Profit (in $M)"], title="Line Chart - Revenue and Profit")
        elif chart_type == "Bar Chart":
            fig = px.bar(sample_data, x="Year", y=["Revenue (in $M)", "Profit (in $M)"], barmode="group", title="Bar Chart - Revenue and Profit")
        elif chart_type == "Area Chart":
            fig = px.area(sample_data, x="Year", y=["Revenue (in $M)", "Profit (in $M)"], title="Area Chart - Revenue and Profit")

        st.plotly_chart(fig, use_container_width=True)
        # Trivia Section
    with st.container():
        st.markdown('<div id="trivia" class="section-card"><h2>📖 Did You Know?</h2></div>', unsafe_allow_html=True)
        trivia = [
            "90% of the world's data has been created in the last two years.",
            "The average person generates about 1.7 MB of data every second.",
            "Data visualization can improve decision-making efficiency by up to 28%.",
            "The first computer database was developed in 1960 by Charles Bachman."
        ]
        if st.button("🔄 Shuffle Trivia"):
            selected_trivia = np.random.choice(trivia)
        else:
            selected_trivia = trivia[0]
        st.info(f"💡 **{selected_trivia}**")

# Function to categorize columns as numerical or categorical
def categorize_columns(df):
    numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    return numerical_cols, categorical_cols

# Function to automatically select the most relevant numerical columns
def select_automatic_columns(df, max_columns=3):
    numerical_cols, _ = categorize_columns(df)
    # Select columns with the highest variance (assuming they have the most information)
    variance = df[numerical_cols].var()
    selected_columns = variance.nlargest(max_columns).index.tolist()
    return selected_columns

# Default Color Palette
default_palette = px.colors.qualitative.Set1

# Callback functions for mutual exclusivity
def select_all_callback():
    st.session_state["deselect_all"] = False

def deselect_all_callback():
    st.session_state["select_all"] = False

def data_cleaning_page():
    st.markdown("<h1>🛠️ Data Cleaning</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV File for Processing", type="csv")

    if uploaded_file:
        # Load uploaded data
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        # Categorize columns into numerical and categorical
        numerical_cols, _ = categorize_columns(df)
        
        st.markdown("### Select Columns to Include in Cleaning")
        selected_columns = st.multiselect("Choose Columns:", df.columns, default=df.columns)

        # Cleaning Options
        st.markdown("### Choose Cleaning Operations")
        remove_empty_rows = st.checkbox("Remove Empty Rows")
        remove_duplicates = st.checkbox("Remove Duplicate Rows")
        handle_missing_values = st.checkbox("Remove Rows with Missing Values")
        standardize_column_names = st.checkbox("Standardize Column Names")
        replace_infinite_values = st.checkbox("Replace Infinite Values with NaN")
        fill_missing_values = st.checkbox("Fill Missing Values with a Default Value")
        
        # Additional options for missing values
        if fill_missing_values:
            fill_method = st.selectbox("Choose a method to fill missing values:", ["Mean", "Median", "Mode", "Enter Default Value"])
            fill_value = None
            if fill_method == "Enter Default Value":
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

                    if fill_missing_values:
                        if fill_method == "Mean":
                            cleaned_data = cleaned_data.fillna(cleaned_data.mean(numeric_only=True))
                            st.info("Missing values filled with column mean.")
                        elif fill_method == "Median":
                            cleaned_data = cleaned_data.fillna(cleaned_data.median(numeric_only=True))
                            st.info("Missing values filled with column median.")
                        elif fill_method == "Mode":
                            for column in cleaned_data.columns:
                                mode_value = cleaned_data[column].mode().iloc[0] if not cleaned_data[column].mode().empty else None
                                cleaned_data[column] = cleaned_data[column].fillna(mode_value)
                            st.info("Missing values filled with column mode.")
                        elif fill_method == "Enter Default Value" and fill_value is not None:
                            cleaned_data = cleaned_data.fillna(value=fill_value)
                            st.info(f"Missing values filled with: {fill_value}")
                    
                    if handle_missing_values:
                        cleaned_data = cleaned_data.dropna()
                        st.info("Rows with missing values removed.")

                    if standardize_column_names:
                        cleaned_data.columns = [col.strip().lower().replace(" ", "_") for col in cleaned_data.columns]
                        st.info("Column names standardized.")

                    if replace_infinite_values:
                        cleaned_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                        st.info("Infinite values replaced with NaN.")
                    
                    if remove_empty_rows:  # Handle empty row removal
                        cleaned_data = cleaned_data.dropna(how='all')
                        st.info("Empty rows removed.")
                        
                    # Preview the cleaning operations
                    st.write("### Data Preview After Cleaning")
                    st.dataframe(cleaned_data)

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

def ai_clustering_page():
    st.markdown("<h1>🛠️ Clustering</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV for Clustering", type="csv")

    if uploaded_file:
        # Load the uploaded dataset
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📋 Uploaded Data", df.head())
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

        # Detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.error("The dataset does not contain any numeric columns. Please upload a dataset with numeric data.")
            return

        # Handle missing or infinite values in numeric columns
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            st.error("The dataset is empty after removing missing or invalid values. Please upload a valid dataset.")
            return

        st.write(f"⚙️ Detected numeric columns: {numeric_cols}")

        # User selection of clustering columns
        selected_columns = st.multiselect("Select columns for clustering:", numeric_cols, default=numeric_cols[:3])

        if len(selected_columns) < 2:
            st.error("Please select at least two columns for clustering.")
            return

        # Clustering options
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        clustering_algorithm = st.selectbox("Select Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

        # Perform clustering
        if st.button("Perform Clustering"):
            try:
                # Preprocess the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[selected_columns])

                # Apply clustering algorithm
                if clustering_algorithm == "KMeans":
                    model = KMeans(n_clusters=num_clusters, random_state=42)
                    clusters = model.fit_predict(scaled_data)
                elif clustering_algorithm == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    model = DBSCAN(eps=0.5, min_samples=5)
                    clusters = model.fit_predict(scaled_data)
                elif clustering_algorithm == "Agglomerative":
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering(n_clusters=num_clusters)
                    clusters = model.fit_predict(scaled_data)

                # Add clusters to the DataFrame
                df["Cluster"] = clusters
                st.success(f"Clustering complete using {clustering_algorithm}!")
                st.write("🗂️ Clustered Data", df.head())

                # Visualization
                if len(selected_columns) == 2:
                    fig = px.scatter(
                        df,
                        x=selected_columns[0],
                        y=selected_columns[1],
                        color="Cluster",
                        title="Cluster Visualization (2D)",
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif len(selected_columns) >= 3:
                    fig = px.scatter_3d(
                        df,
                        x=selected_columns[0],
                        y=selected_columns[1],
                        z=selected_columns[2],
                        color="Cluster",
                        title="Cluster Visualization (3D)",
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Download button for clustered data
                st.download_button(
                    label="⬇️ Download Clustered Data",
                    data=df.to_csv(index=False),
                    file_name="clustered_data.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during clustering: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")

def data_visualization_page():
    st.markdown("<h1>📊 Data Visualization</h1>", unsafe_allow_html=True)
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
        # Check for empty column lists
        if not numerical_cols:
            st.error("No numerical columns found in the dataset. Please upload a dataset with numerical data.")
            st.stop()
        if not categorical_cols:
            st.error("No categorical columns found in the dataset. Please upload a dataset with categorical data.")
            st.stop()

            
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
        fill_option = st.checkbox("Fill Area (if applicable)", value=False) 
        if chart_type in ["Polar Area Chart", "Radar Chart"] and (not theta_col or not r_col):     
            st.warning("Please select both 'theta' (categorical) and 'r' (numerical) columns.")
            # return #fix here

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

def predictive_analytics_page():
    st.markdown("<h1> 📈 Predictive Analytics</h1>", unsafe_allow_html=True)

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV File for Forecasting", type="csv")
    if uploaded_file:
        # Load Data
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(df)

        # Prefer 'Date' column as time_col if it exists
        if 'Date' in df.columns:
            time_col = 'Date'
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')  # Ensure proper datetime conversion
            if df[time_col].isnull().any():
                st.error("The 'Date' column contains invalid or missing values. Please clean your data.")
                st.stop()
            df = df.sort_values(by=time_col)
            st.info(f"Time progression detected using column: {time_col}")
        else:
            # Automatically detect date-like columns
            date_columns = [
                col for col in df.columns if pd.to_datetime(df[col], errors='coerce').notna().all()
            ]
            if date_columns:
                time_col = date_columns[0]  # Automatically select the first detected date column
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')  # Ensure proper datetime conversion
                df = df.sort_values(by=time_col)
                st.info(f"Time progression detected using column: {time_col}")
            else:
                st.error("No valid date column detected. Please upload a dataset with a proper date or time column.")
                st.stop()

        # Prefer 'Total' column as metric_col if it exists
        if 'Total' in df.columns:
            metric_col = 'Total'
            st.info(f"Forecasting metric detected: {metric_col}")
        else:
            # Automatically detect numeric columns
            numeric_columns = [
                col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
            ]
            if numeric_columns:
                metric_col = numeric_columns[0]  # Automatically select the first numeric column
                st.info(f"Forecasting metric detected: {metric_col}")
            else:
                st.error("No numeric columns found for forecasting. Please upload a dataset with numerical data.")
                st.stop()

        # Forecast Period
        forecast_period = st.selectbox("Select Forecast Period", ["1 Month", "1 Year", "3 Years", "5 Years"])
        periods = {"1 Month": 1, "1 Year": 12, "3 Years": 36, "5 Years": 60}
        forecast_steps = periods[forecast_period]

        # Forecasting Logic
        if st.button("Run Forecast"):
            try:
                # Prophet Forecasting
                from prophet import Prophet
                from prophet.plot import plot_plotly

                prophet_df = df[[time_col, metric_col]].rename(columns={time_col: 'ds', metric_col: 'y'})

                # Ensure the 'ds' column contains datetime values
                if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
                    st.error("The selected time column does not contain valid datetime values.")
                    st.stop()

                # Create and fit the Prophet model
                prophet_model = Prophet()
                prophet_model.fit(prophet_df)

                # Forecast for the selected period
                future = prophet_model.make_future_dataframe(periods=forecast_steps, freq='M')
                forecast = prophet_model.predict(future)

                # Visualize the forecast
                st.success("Forecasting Complete!")
                fig = plot_plotly(prophet_model, forecast)
                st.plotly_chart(fig)

                # Provide Forecasted Data for Download
                st.download_button(
                    label="Download Forecasted Data",
                    data=forecast[['ds', 'yhat']].to_csv(index=False),
                    file_name="forecasted_data.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error during forecasting: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")

def gemini_ai():
    st.markdown("<h1> 🤖 Gemini AI</h1>", unsafe_allow_html=True)
    st.write("Generate text using Gemini AI.")

    # User input for the text generation prompt
    prompt = st.text_input("Enter your prompt:", placeholder="Type something...")

    if st.button("Generate"):
        if prompt.strip():
            try:
                # Call the Flask API
                response = requests.post(
                    "http://127.0.0.1:5000/api/generate-text",  # Replace with your Flask endpoint URL
                    json={"prompt": prompt}
                )
                if response.status_code == 200:
                    reply = response.json().get("reply", "No response from Gemini AI.")
                    st.text_area("Generated Text:", value=reply, height=200)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to the backend. Error: {e}")
        else:
            st.warning("Please enter a prompt.")
# Footer
def footer():
    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100vw;
            height: 40px;
            background-color: #1f2b38;
            color: #e4e9f0;
            text-align: center;
            line-height: 40px;
            font-size: 0.8rem;
            z-index: 1000;
        ">
            © 2024 InsightFlow Studio. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main Function
def main():
    # Sidebar
    sidebar()

    # Render the selected page
    if st.session_state['active_page'] == "Dashboard":
        dashboard_page()
    elif st.session_state['active_page'] == "Data Cleaning":
        data_cleaning_page()
    elif st.session_state['active_page'] == "Clustering":
        ai_clustering_page()
    elif st.session_state['active_page'] == "Data Visualization":
        data_visualization_page()
    elif st.session_state['active_page'] == "Predictive Analytics":
        predictive_analytics_page()
    elif st.session_state['active_page'] == "Gemini AI":
        gemini_ai()

    # Footer
    footer()

if __name__ == "__main__":
    main()