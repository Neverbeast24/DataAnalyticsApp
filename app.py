from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

<<<<<<< Updated upstream
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
                # If the value is numeric or empty, use the default value
                if isinstance(value, (int, float)) or str(value).strip().isdigit():
                    raise ValueError
                if 'format' in column_spec:
                    import re
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
    payload = request.json
    data = payload['data']
    selected_columns = payload['columns']
    df = pd.DataFrame(data)

    # Apply cleaning rules for each column
    for col_idx, spec in COLUMN_TYPES.items():
        if col_idx < len(df.columns):
            df.iloc[:, col_idx] = convert_column(df.iloc[:, col_idx], spec)

    # Drop duplicates and fill NaN with default
    df = df.drop_duplicates().fillna({i: spec['default'] for i, spec in COLUMN_TYPES.items() if i < len(df.columns)})

    return jsonify({'cleaned_data': df.to_dict(orient='records'), 'columns': list(df.columns)})
=======
# Set up logging
logging.basicConfig(filename='user_activity.log', level=logging.INFO, format='%(asctime)s - %(message)s')

@app.before_request
def log_request_info():
    logging.info(f"Endpoint: {request.endpoint}, Data: {request.json}")

def is_alphabetic_string(s):
    """
    Check if a string contains only alphabets.
    """
    if isinstance(s, str):
        return s.isalpha()
    return False

def is_numeric(s):
    """
    Check if a string can be converted to a number (int or float).
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def detect_majority_type(series):
    """
    Detect the majority data type for a single column.
    """
    type_counts = {
        'int': series.dropna().apply(lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit())).sum(),
        'float': series.dropna().apply(lambda x: isinstance(x, float) or (isinstance(x, str) and is_numeric(x))).sum(),
        'string': series.dropna().apply(lambda x: isinstance(x, str) and not is_numeric(x) and not x.isdigit()).sum(),
        'datetime': pd.to_datetime(series.dropna(), errors='coerce').notna().sum()
    }

    majority_type = max(type_counts, key=type_counts.get)

    # Default to 'string' if no valid entries are found
    if type_counts[majority_type] == 0:
        return 'string'

    return majority_type

def detect_column_types(df):
    """
    Detect the majority data type for each column.
    """
    column_types = {col: detect_majority_type(df[col]) for col in df.columns}
    return column_types

def clean_data_by_column(df, column_types):
    """
    Validate and clean each column individually based on its detected type.
    """
    cleaned_df = df.copy()

    for col, expected_type in column_types.items():
        if expected_type == 'float':
            # Keep valid floats; replace invalid entries with NaN
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        elif expected_type == 'int':
            # Keep valid integers; replace invalid entries with NaN
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce', downcast='integer')
        elif expected_type == 'string':
            # Keep only valid strings; replace others with NaN
            if df[col].dtype == 'object':  # Check if column is of object type
                cleaned_df[col] = df[col].where(df[col].apply(lambda x: isinstance(x, str) or pd.isna(x))) 
        elif expected_type == 'datetime':
            # Parse valid dates; replace invalid entries with NaN
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')

    return cleaned_df

@app.route('/process', methods=['POST'])
def process_data():
    try:
        payload = request.json
        if not payload or 'data' not in payload:
            return jsonify({'error': "Invalid payload. Must contain 'data'."}), 400

        data = payload['data']
        df = pd.DataFrame(data)

        # Detect column types
        column_types = detect_column_types(df)

        # Clean data by column
        cleaned_df = clean_data_by_column(df, column_types)

        # Replace NaN with None for JSON compatibility
        cleaned_df = cleaned_df.where(pd.notnull(cleaned_df), None)

        return jsonify({
            'cleaned_data': cleaned_df.to_dict(orient='records'),
            'detected_types': column_types
        })
    except Exception as e:
        logging.error(f"Error in /process: {str(e)}")
        return jsonify({'error': f"Error during data cleaning: {str(e)}"}), 500

@app.route('/ai/clustering', methods=['POST'])
def ai_clustering():
    try:
        payload = request.json
        if not payload or 'data' not in payload:
            raise BadRequest("Payload must contain 'data' key.")

        data = pd.DataFrame(payload['data'])
        num_clusters = payload.get('num_clusters', 3)

        # Validate numeric columns exist
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise BadRequest("No numeric columns available for clustering.")

        # Validate num_clusters
        if not isinstance(num_clusters, int) or num_clusters <= 0:
            raise BadRequest("'num_clusters' must be a positive integer.")

        if len(data) < num_clusters:
            raise BadRequest("'num_clusters' cannot exceed the number of data points.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        data['Cluster'] = clusters

        return jsonify({'clustered_data': data.to_dict(orient='records'), 'cluster_centers': kmeans.cluster_centers_.tolist()})
    except Exception as e:
        logging.error(f"Error in /ai/clustering: {str(e)}")
        return jsonify({'error': str(e)}), 400
>>>>>>> Stashed changes

if __name__ == '__main__':
    app.run(debug=True)
