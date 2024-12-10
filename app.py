from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import re
from werkzeug.exceptions import BadRequest
from transformers import pipeline
import uuid
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='user_activity.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Configure the Gemini API
genai.configure(api_key="AIzaSyDouj7jzPq-ObX9FsGUb7G8QnEbzo2zF1U")  # Replace with your Gemini API key

@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    """
    Endpoint to generate text using the Gemini API.
    """
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Initialize the GenerativeModel
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')  # Replace with your desired model name

        # Generate text using the Gemini API
        response = model.generate_content(prompt)

        # Return the generated text
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.before_request
def log_request_info():
    logging.info(f"Endpoint: {request.endpoint}, Data: {request.json}")

# Column type specifications
COLUMN_TYPES = {
    0: {'type': 'str', 'format': r'^\d{3}-\d{2}-\d{3}$', 'default': '000-00-000'},
    1: {'type': 'str', 'default': " "},
    2: {'type': 'str', 'default': " "},
    3: {'type': 'str', 'default': " "},
    4: {'type': 'str', 'default': " "},
    5: {'type': 'str', 'default': " "},
    6: {'type': 'float', 'default': 0.0},
    7: {'type': 'int', 'default': 0},
    8: {'type': 'float', 'default': 0.0},
    9: {'type': 'float', 'default': 0.0},
    10: {'type': 'datetime', 'format': '%Y-%m-%d', 'default': '1970-01-01'},
    11: {'type': 'time', 'format': '%H:%M:%S', 'default': '00:00:00'},
    12: {'type': 'str', 'default': " "},
    13: {'type': 'float', 'default': 0.0},
    14: {'type': 'float', 'default': 0.0},
    15: {'type': 'float', 'default': 0.0},
    16: {'type': 'float', 'default': 0.0}
}

def convert_column(data, column_spec):
    """Convert a column to the specified data type and format."""
    def convert(value):
        try:
            if pd.isnull(value):
                return column_spec['default']
            if column_spec['type'] == 'str':
                if isinstance(value, (int, float)) or str(value).strip().isdigit():
                    raise ValueError
                if 'format' in column_spec:
                    if not re.match(column_spec['format'], str(value)):
                        raise ValueError
                return str(value)
            elif column_spec['type'] == 'float':
                return float(value)
            elif column_spec['type'] == 'int':
                return int(float(value))  # Allow float-to-int conversion
            elif column_spec['type'] == 'datetime':
                return datetime.strptime(value, column_spec['format']).strftime(column_spec['format'])
            elif column_spec['type'] == 'time':
                return datetime.strptime(value, column_spec['format']).strftime(column_spec['format'])
        except (ValueError, TypeError):
            return column_spec['default']
    return data.map(convert)

@app.route('/process', methods=['POST'])
def process_data():
    try:
        payload = request.json
        if not payload or 'data' not in payload or 'columns' not in payload:
            raise BadRequest("Payload must contain 'data' and 'columns' keys.")

        data = payload['data']
        selected_columns = payload['columns']

        # Validate columns
        if not isinstance(selected_columns, list) or any(not isinstance(col, int) for col in selected_columns):
            raise BadRequest("The 'columns' key must be a list of integers.")

        df = pd.DataFrame(data)

        # Apply cleaning rules for each column
        for col_idx, spec in COLUMN_TYPES.items():
            if col_idx < len(df.columns):
                df.iloc[:, col_idx] = convert_column(df.iloc[:, col_idx], spec)

        # Drop duplicates and fill NaN with default
        df = df.drop_duplicates().fillna({i: spec['default'] for i, spec in COLUMN_TYPES.items() if i < len(df.columns)})

        return jsonify({'cleaned_data': df.to_dict(orient='records'), 'columns': list(df.columns)})
    except Exception as e:
        logging.error(f"Error in /process: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/ai/clustering', methods=['POST'])
def ai_clustering():
    try:
        # Parse request payload
        payload = request.json
        if not payload or 'data' not in payload:
            raise BadRequest("Payload must contain 'data' key.")

        # Load data into DataFrame
        data = pd.DataFrame(payload['data'])

        # Validate and clean numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise BadRequest("No numeric columns available for clustering.")

        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # Validate the number of clusters
        num_clusters = payload.get('num_clusters', 3)
        if not isinstance(num_clusters, int) or num_clusters <= 0:
            raise BadRequest("'num_clusters' must be a positive integer.")

        if len(data) < num_clusters:
            raise BadRequest("'num_clusters' cannot exceed the number of data points.")

        # Perform clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_cols])

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        data['Cluster'] = clusters

        return jsonify({
            'clustered_data': data.to_dict(orient='records'),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'numeric_columns_used': numeric_cols,
        })
    except Exception as e:
        logging.error(f"Error in /ai/clustering: {e}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("Server is running on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)