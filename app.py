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
    data = request.json
    df = pd.DataFrame(data)
    
    # Data cleaning
    df_cleaned = df.drop_duplicates().fillna(0)
    return df_cleaned.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
