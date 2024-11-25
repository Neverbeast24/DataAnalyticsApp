from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
