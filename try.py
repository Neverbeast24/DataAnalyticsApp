import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.colors as pc
import base64

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

        # Allow user to select columns for visualization
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        numerical_columns = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]

        st.markdown("### Column Selection")
        col1, col2 = st.columns(2)
        with col1:
            theta_col = st.selectbox("Select a column for 'theta' (categorical):", categorical_columns, key="theta")
        with col2:
            r_col = st.selectbox("Select a column for 'r' (numerical):", numerical_columns, key="r")

        # Allow user to select chart type
        chart_type = st.selectbox("📈 Choose Chart Type", ["Polar Area Chart", "Radar Chart", "Area Chart", "Scatter Plot", "Stacked Bar Chart"])


        # Chart Customization and Chart Rendering
        st.markdown("### Chart Customization and Output")
        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown("#### Custom Chart Options")
            chart_title = st.text_input("Chart Title", value=f"My {chart_type}")
            show_legend = st.checkbox("Show Legend", value=True)
            fill_option = st.checkbox("Fill Area (Radar/Area Chart)", value=False) if chart_type in ["Radar Chart", "Area Chart"] else None
             # Add a color palette selection within chart customization
            color_palette = st.selectbox(
                "Select Color Scheme",
                ["Default"] + dir(pc.qualitative),
                index=0
            )

            # Define the selected palette based on user selection
            if color_palette == "Default":
                selected_palette = pc.qualitative.Plotly
            else:
                try:
                    selected_palette = getattr(pc.qualitative, color_palette)
                except AttributeError:
                    selected_palette = pc.qualitative.Plotly

        with col4:
            if theta_col and r_col:
                try:
                    if chart_type == "Polar Area Chart":
                        fig = px.bar_polar(
                            df,
                            r=r_col,
                            theta=theta_col,
                            color=theta_col,
                            color_discrete_sequence=default_palette
                        )
                        fig.update_layout(title=chart_title, showlegend=show_legend)
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Radar Chart":
                        # Slider for Line Width
                        line_width = st.slider("Line Width (for Radar Chart)", min_value=1, max_value=10, value=2)
                        # Checkbox for Fill Area
                        fill_option = st.checkbox("Fill Area (Radar Chart)", value=False, key="radar_fill")

                        # Create the radar chart
                        fig = px.line_polar(
                            df,
                            r=r_col,
                            theta=theta_col,
                            line_close=True,  # Ensure the radar chart is closed
                            color=theta_col,
                            color_discrete_sequence=selected_palette  # Use the selected palette
                        )

                        # Apply valid properties for line traces
                        fig.update_traces(
                            line=dict(width=line_width)  # Ensure only valid properties are used
                        )

                        # Apply fill only if selected
                        if fill_option:
                            fig.update_traces(fill="toself")  # Correct fill property for radar charts

                        # Update layout for title and legend visibility
                        fig.update_layout(
                            title=chart_title,
                            showlegend=show_legend
                        )

                        # Render the chart
                        st.plotly_chart(fig, use_container_width=True)


                    elif chart_type == "Area Chart":
                        fill_option = st.checkbox("Fill Area (Radar/Area Chart)", value=False, key="area_fill")
                        fig = px.area(
                            df,
                            x=theta_col,
                            y=r_col,
                            color=theta_col,
                            color_discrete_sequence=selected_palette
                        )
                        if fill_option:
                            fig.update_traces(fill="tonexty")
                        fig.update_layout(title=chart_title, showlegend=show_legend)
                        st.plotly_chart(fig, use_container_width=True)


                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(
                            df,
                            x=theta_col,
                            y=r_col,
                            color=theta_col,
                            color_discrete_sequence=default_palette
                        )
                        fig.update_layout(title=chart_title, showlegend=show_legend)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Stacked Bar Chart":
                        color_col = st.selectbox("Select a column for 'color':", categorical_columns, index=0)
                        fig = px.bar(
                            df,
                            x=theta_col,
                            y=r_col,
                            color=color_col,
                            barmode="stack",
                            color_discrete_sequence=selected_palette
                        )
                        fig.update_layout(title=chart_title, showlegend=show_legend)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred while creating the {chart_type}: {e}")
            else:
                st.warning("Please upload a CSV file to proceed.")
