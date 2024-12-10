import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import plotly.express as px

def clean_data(df):
    df.fillna(0, inplace=True)
    return df

def perform_clustering(df):
    numerical_columns = df.select_dtypes(include=["number"]).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_columns])
    kmeans = KMeans(n_clusters=3)
    df["Cluster"] = kmeans.fit_predict(scaled_data)
    return df

def run_forecast(df):
    time_col = df.columns[0]
    metric_col = df.columns[1]
    df.rename(columns={time_col: "ds", metric_col: "y"}, inplace=True)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)
    fig = px.line(forecast, x="ds", y="yhat", title="Forecast")
    return {"figure": fig, "data": forecast}
