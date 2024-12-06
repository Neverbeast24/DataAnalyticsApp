# testing ui code only, wala pang main logic here
import pandas as pd
import streamlit as st
import base64

# Page Configuration
st.set_page_config(page_title="InsightFlow Studio", layout="wide", initial_sidebar_state="expanded")

# Function to convert an image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert your logo to Base64 (replace with your logo path)
logo_base64 = get_base64_image("logo2.png")

# Custom CSS for the UI design based on your mockups
st.markdown(
    """
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #1b2a41;
            color: #e4e9f0;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        /* Sidebar Styling */
        .sidebar-container {
            background: linear-gradient(135deg, #283d4e, #1f2b38);
            padding: 50px 30px;
            border-radius: 20px;
            position: fixed;
            width: 280px;
            height: 100%;
            box-shadow: 6px 6px 25px rgba(0, 191, 174, 0.2);
            transition: transform 0.3s ease-in-out;
            overflow: auto;
        }

        .sidebar-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #00bfae;
            text-align: center;
            margin-bottom: 30px;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .sidebar-logo img {
            width: 180px;
            border-radius: 50%;
            box-shadow: 0 0 15px rgba(0, 191, 174, 0.3);
            margin-bottom: 20px;
        }

        .sidebar-nav {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 40px;
        }

        .sidebar-nav a {
            color: #ffffff;
            margin: 20px 0;
            display: block;
            text-decoration: none;
            border-radius: 12px;
            padding: 12px 15px;
            transition: all 0.3s ease;
            font-size: 1.1rem;
        }

        .sidebar-nav a:hover, .sidebar-nav a.active {
            background-color: #00bfae;
            color: #1b2a41;
            transform: translateX(10px);
        }

        .sidebar-footer {
            position: absolute;
            bottom: 30px;
            left: 30px;
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 400;
        }

        /* Content Page Styling */
        .content-container {
            padding-left: 320px;
            padding-top: 50px;
            padding-right: 30px;
            padding-bottom: 30px;
        }

        /* Hero Section */
        .hero-section {
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 400px;
            background: url('https://your-image-url.com/hero-background.jpg') no-repeat center center/cover;
            color: white;
            padding: 50px;
            border-radius: 20px;
        }

        .hero-content {
            max-width: 50%;
        }

        .hero-title {
            font-size: 3.8rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-transform: uppercase;
        }

        .hero-subtitle {
            font-size: 1.6rem;
            margin-bottom: 20px;
        }

        .cta-button {
            background-color: #00bfae;
            padding: 15px 30px;
            color: white;
            border-radius: 12px;
            font-size: 1.2rem;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        .cta-button:hover {
            background-color: #00a186;
        }

        /* Footer */
        .footer {
            position: fixed;
            bottom: 10px;
            left: 0;
            right: 0;
            text-align: center;
            padding: 10px 0;
            background-color: #1f2b38;
            color: #e4e9f0;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with navigation menu
def sidebar(active_page):
    st.sidebar.markdown(
        f"""
        <div class="sidebar-container">
            <div class="sidebar-logo" style="text-align: center;">
                <img src="data:image/png;base64,{logo_base64}" alt="InsightFlow Logo">
                <h1 class="sidebar-header">InsightFlow</h1>
                <p style="color: #00bfae;">AI-Driven Analytics Platform</p>
                <hr style="border-top: 2px solid #00bfae;">
            </div>
            <div class="sidebar-nav">
                <a href="?page=home" class="sidebar-link {'active' if active_page == 'home' else ''}">Home</a>
                <a href="?page=data-cleaning" class="sidebar-link {'active' if active_page == 'data-cleaning' else ''}">Data Cleaning</a>
                <a href="?page=ai-clustering" class="sidebar-link {'active' if active_page == 'ai-clustering' else ''}">AI Clustering</a>
                <a href="?page=data-visualization" class="sidebar-link {'active' if active_page == 'data-visualization' else ''}">Data Visualization</a>
                <a href="?page=predictive-analytics" class="sidebar-link {'active' if active_page == 'predictive-analytics' else ''}">Predictive Analytics</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Define the home page with visual appeal and interactivity
def home_page():
    # Page Title and Header Section
    st.markdown(
        """
        <style>
            .hero-section {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background-color: #1b2a41;
                color: #ffffff;
                padding: 60px 50px;
                border-radius: 15px;
                box-shadow: 0px 4px 15px rgba(0, 191, 174, 0.2);
            }
            .hero-content h1 {
                font-size: 3rem;
                font-weight: 700;
            }
            .hero-content p {
                font-size: 1.5rem;
            }
            .cta-button {
                background-color: #00bfae;
                padding: 15px 30px;
                color: white;
                font-size: 1.2rem;
                border-radius: 12px;
                cursor: pointer;
                text-decoration: none;
                transition: background-color 0.3s;
            }
            .cta-button:hover {
                background-color: #00a186;
            }
            .feature-section {
                display: flex;
                justify-content: space-around;
                margin-top: 50px;
            }
            .feature-card {
                background-color: #ffffff;
                color: #1b2a41;
                padding: 20px;
                border-radius: 10px;
                width: 22%;
                box-shadow: 0px 4px 15px rgba(0, 191, 174, 0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-10px);
            }
            .footer {
                position: fixed;
                bottom: 10px;
                width: 100%;
                text-align: center;
                padding: 10px;
                background-color: #1b2a41;
                color: #e4e9f0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero Section (with images and interactive button)
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-content">
                <h1>Welcome to InsightFlow</h1>
                <p>Your go-to platform for AI-driven data analytics, clustering, visualization, and predictive analytics.</p>
                <a class="cta-button" href="#explore">Start Exploring</a>
            </div>
            <img src="https://your-image-url.com/hero-image.jpg" width="300" alt="Hero Image">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scroll to Explore Section
    st.markdown("<div id='explore'></div>", unsafe_allow_html=True)

    # Interactive Features Section
    st.markdown(
        """
        <h2 style="text-align: center; font-size: 2.5rem; color: #00bfae;">Explore Our Key Features</h2>
        <div class="feature-section">
            <div class="feature-card">
                <h3>Data Cleaning</h3>
                <p>Upload your dataset and clean it with advanced AI tools.</p>
                <button class="cta-button" onclick="window.location.href='?page=data-cleaning'">Explore</button>
            </div>
            <div class="feature-card">
                <h3>AI Clustering</h3>
                <p>Segment your data using machine learning for valuable insights.</p>
                <button class="cta-button" onclick="window.location.href='?page=ai-clustering'">Explore</button>
            </div>
            <div class="feature-card">
                <h3>Data Visualization</h3>
                <p>Create stunning visualizations to understand your data better.</p>
                <button class="cta-button" onclick="window.location.href='?page=data-visualization'">Explore</button>
            </div>
            <div class="feature-card">
                <h3>Predictive Analytics</h3>
                <p>Forecast future trends using advanced predictive models.</p>
                <button class="cta-button" onclick="window.location.href='?page=predictive-analytics'">Explore</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Feature Showcase with Button
    st.markdown(
        """
        <h2 style="text-align: center; font-size: 2.5rem; color: #00bfae;">What's New?</h2>
        <p style="text-align: center;">Click the button below to learn about the latest features!</p>
        """,
        unsafe_allow_html=True,
    )

    # Button to show new features
    show_updates = st.button("Show New Features")
    if show_updates:
        st.markdown(
            """
            <div style="padding: 20px; background-color: #f0f5f7; border-radius: 10px; margin-top: 20px;">
                <h4 style="color: #1b2a41;">Latest Updates:</h4>
                <ul style="color: #00bfae;">
                    <li>AI-driven clustering for advanced segmentation of your data.</li>
                    <li>Enhanced interactive charts and data visualizations.</li>
                    <li>New predictive models for time-series forecasting.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

def data_cleaning_page():
    st.markdown(
        """
        <div class="content-container">
            <h1 class="page-title">Data Cleaning</h1>
            <p>Upload your CSV file for cleaning:</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data", df)

def ai_clustering_page():
    st.markdown(
        """
        <div class="content-container">
            <h1 class="page-title">AI Clustering</h1>
            <p>AI Clustering Page Content Here</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def data_visualization_page():
    st.markdown(
        """
        <div class="content-container">
            <h1 class="page-title">Data Visualization</h1>
            <p>Data Visualization Page Content Here</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def predictive_analytics_page():
    st.markdown(
        """
        <div class="content-container">
            <h1 class="page-title">Predictive Analytics</h1>
            <p>Predictive Analytics Page Content Here</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main function to handle navigation
def main():
    # Get the query parameter to determine the active page
    query_params = st.experimental_get_query_params()
    active_page = query_params.get("page", ["home"])[0]

    # Render the sidebar
    sidebar(active_page)

    # Route to the appropriate page
    if active_page == "home":
        home_page()
    elif active_page == "data-cleaning":
        data_cleaning_page()
    elif active_page == "ai-clustering":
        ai_clustering_page()
    elif active_page == "data-visualization":
        data_visualization_page()
    elif active_page == "predictive-analytics":
        predictive_analytics_page()

    # Footer
    st.markdown(
        """
        <div class="footer">
            © 2024 InsightFlow Studio. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Run the app
if __name__ == "__main__":
    main()
