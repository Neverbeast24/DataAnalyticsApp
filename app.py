from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    df = pd.read_csv(file)
    return jsonify({'message': 'File uploaded successfully', 'columns': list(df.columns)})

@app.route('/process', methods=['POST'])
def process_data():
    payload = request.json
    data = payload['data']
    selected_columns = payload['columns']
    df = pd.DataFrame(data)

    if selected_columns:
        df = df[selected_columns]

    # Data Cleaning
    df.replace([float('inf'), float('-inf')], None, inplace=True)
    df.fillna(0, inplace=True)
    if 'Weight' in df.columns:
        df['Weight'] = pd.to_numeric(df['Weight'].str.replace('kg', '', regex=True), errors='coerce')
    if 'Price' in df.columns:
        df = df[df['Price'] > 0]
    df.dropna(subset=['Weight'], inplace=True)
    df = df.drop_duplicates()

    return jsonify({'cleaned_data': df.to_dict(orient='records'), 'columns': list(df.columns)})

if __name__ == '__main__':
    app.run(debug=True)
