from flask_cors import CORS



from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
# ------------------------------
# DATA ANALYSIS FUNCTION
# ------------------------------
def analyze_data(df):
    result = {}

    # Basic info
    result['rows'] = df.shape[0]
    result['columns'] = df.shape[1]

    # Column info
    result['columns_info'] = {col: str(df[col].dtype) for col in df.columns}

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        result['numeric_summary'] = df[numeric_cols].describe().to_dict()

    # Missing values
    result['missing_values'] = df.isnull().sum().to_dict()

    # Trend analysis
    trend_info = {}
    for col in numeric_cols:
        trend_info[col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'trend': 'Increasing' if df[col].mean() > df[col].median() else 'Decreasing'
        }
    result['trend_analysis'] = trend_info

    # Correlation
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        result['correlation_matrix'] = corr_matrix.to_dict()

        # Strong correlations
        strong_corr = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i,j]
                if abs(val) > 0.7:
                    strong_corr[f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}"] = val
        result['strong_correlations'] = strong_corr

    return result

# ------------------------------
# CHART RECOMMENDATION FUNCTION
# ------------------------------
def chart_recommendations(df):
    recommendations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Histogram
    if numeric_cols:
        recommendations.append({
            'chart': 'Histogram',
            'description': 'Shows the distribution of numeric data.',
            'columns': numeric_cols
        })

    # Bar chart
    if categorical_cols:
        recommendations.append({
            'chart': 'Bar Chart',
            'description': 'Compare categories or show composition.',
            'columns': categorical_cols
        })

    # Scatter plot
    if len(numeric_cols) >= 2:
        recommendations.append({
            'chart': 'Scatter Plot',
            'description': 'Shows relationships between two numeric variables.',
            'columns': numeric_cols
        })

    # Line chart
    if len(numeric_cols) >= 1 and df.shape[0] > 10:
        recommendations.append({
            'chart': 'Line Chart',
            'description': 'Shows trends over time or ordered categories.',
            'columns': numeric_cols
        })

    # Correlation heatmap
    if len(numeric_cols) > 2:
        recommendations.append({
            'chart': 'Correlation Heatmap',
            'description': 'Visual representation of correlation matrix.',
            'columns': numeric_cols
        })

    # Box plot
    if numeric_cols and categorical_cols:
        recommendations.append({
            'chart': 'Box Plot',
            'description': 'Shows numeric distribution across categories.',
            'columns': numeric_cols + categorical_cols
        })

    return recommendations

# ------------------------------
# API ENDPOINT
# ------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    filename = file.filename

    # Read file based on extension
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
    except Exception as e:
        return jsonify({'error': f'File read error: {str(e)}'}), 400

    # Analysis
    analysis_result = analyze_data(df)
    chart_result = chart_recommendations(df)

    return jsonify({
        'analysis': analysis_result,
        'chart_recommendations': chart_result
    })

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
      app.run(host="0.0.0.0", port=10000)
