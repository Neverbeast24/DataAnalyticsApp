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
    st.sidebar.image(f"data:image/png;base64,{logo_base64}", use_column_width=True)
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


# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

if "cleaned_data" not in st.session_state:
    st.session_state["cleaned_data"] = None

if "clustered_data" not in st.session_state:
    st.session_state["clustered_data"] = None
    
    
# Pages Implementation
def dashboard_page():
    st.markdown("""
        <div style="text-align: center;">
            <h2 style="color:  #ffffff;">📊 Visualize with InsightFlow!</h2>
            <p style="font-size: 1.1rem; color: #fffff1;">
                Transform your data into insights!
            </p>
        </div>
        <hr style="border: none; border-top: 2px solid #eee; margin: 10px 0;">
    """, unsafe_allow_html=True)

    # Trivia Section
    st.subheader("📖 Did You Know?")
    trivia = [
        "90% of the world's data has been created in the last two years.",
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
            <h3 style="color: #00bfae;">"Data is a precious thing and will last longer than the systems themselves." – Tim Berners-Lee</h3>
        </div>
    """, unsafe_allow_html=True)

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
    st.markdown("<h1>Data Cleaning</h1>", unsafe_allow_html=True)
    st.header("🛠️ Process Data")
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
    st.markdown("<h1>Clustering</h1>", unsafe_allow_html=True)
    st.title("🤖 Clustering")
    uploaded_file = st.file_uploader("Upload CSV for Clustering", type="csv")

    if uploaded_file:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data", df.head())

        # Check if the 'Cluster' column already exists (indicating clustering has been performed)
        if 'Cluster' not in df.columns:
            st.warning("Please run clustering first!")

            # Automatically select numerical columns
            selected_columns = select_automatic_columns(df, max_columns=3)
            st.write(f"⚙️ Automatically selected columns for clustering: {selected_columns}")

            # If there are at least 2 columns, proceed to clustering
            if len(selected_columns) >= 2:
                # 2D or 3D visualization
                view_type = st.radio("Select View", ["2D", "3D"])

                # Set the number of clusters
                num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

                # Choose clustering algorithm
                clustering_algorithm = st.selectbox("Select Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

                # Button to start clustering
                if st.button("Perform Clustering"):
                    try:
                        # Data Preprocessing: Scaling the data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[selected_columns])

                        # Clustering based on selected algorithm
                        if clustering_algorithm == "KMeans":
                            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                            clusters = kmeans.fit_predict(scaled_data)
                        elif clustering_algorithm == "DBSCAN":
                            from sklearn.cluster import DBSCAN
                            dbscan = DBSCAN(eps=0.5, min_samples=5)
                            clusters = dbscan.fit_predict(scaled_data)
                        elif clustering_algorithm == "Agglomerative":
                            from sklearn.cluster import AgglomerativeClustering
                            agg = AgglomerativeClustering(n_clusters=num_clusters)
                            clusters = agg.fit_predict(scaled_data)

                        # Add the 'Cluster' column to the dataframe
                        df["Cluster"] = clusters

                        # Show success message and preview the clustered data
                        st.success(f"Clustering complete using {clustering_algorithm}!")
                        st.write("🗂️ Clustered Data", df.head())

                        # Calculate silhouette score (optional, if applicable)
                        if clustering_algorithm == "KMeans":
                            silhouette_avg = silhouette_score(scaled_data, clusters)
                            st.write(f"Silhouette Score: {silhouette_avg:.2f}")
                        
                        # 2D Scatter Plot
                        if view_type == "2D" and len(selected_columns) >= 2:
                            fig = px.scatter(
                                df,
                                x=selected_columns[0],
                                y=selected_columns[1],
                                color="Cluster",  # Using 'Cluster' column for coloring
                                color_discrete_sequence=px.colors.qualitative.Plotly,
                                labels={selected_columns[0]: selected_columns[0], selected_columns[1]: selected_columns[1]}
                            )
                            fig.update_layout(title="Cluster Visualization (2D)")
                            st.plotly_chart(fig, use_container_width=True)

                        # 3D Scatter Plot
                        elif view_type == "3D" and len(selected_columns) >= 3:
                            fig = px.scatter_3d(
                                df,
                                x=selected_columns[0],
                                y=selected_columns[1],
                                z=selected_columns[2],
                                color="Cluster",  # Using 'Cluster' column for coloring
                                color_discrete_sequence=px.colors.qualitative.Plotly,
                                labels={selected_columns[0]: selected_columns[0], selected_columns[1]: selected_columns[1], selected_columns[2]: selected_columns[2]}
                            )
                            fig.update_layout(title="Cluster Visualization (3D)")
                            st.plotly_chart(fig, use_container_width=True)

                        # Option to download the clustered data
                        st.download_button(
                            label="⬇️ Download Clustered Data",
                            data=df.to_csv(index=False),
                            file_name="clustered_data.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error during clustering: {e}")

                else:
                    st.info("Select the number of clusters and hit 'Perform Clustering' to begin.")

            else:
                st.error("Please select at least two columns for clustering.")
        else:
            st.warning("Clustering has already been performed. Please upload a new file to reset.")
    else:
        st.warning("Please upload a CSV file to proceed.")

def data_visualization_page():
    st.markdown("<h1>Data Visualization</h1>", unsafe_allow_html=True)
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
    st.markdown("<h1>Predictive Analytics</h1>", unsafe_allow_html=True)
    st.header("📈 Predictive Analytics")

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
    st.markdown("<h1>Gemini AI</h1>", unsafe_allow_html=True)
    st.title("Gemini AI Text Generation")
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