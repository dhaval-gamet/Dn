from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import io
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ------------------------------
# ENHANCED DATA ANALYSIS FUNCTION
# ------------------------------
def analyze_data(df):
    result = {}
    
    # Basic info with more details
    result['rows'] = df.shape[0]
    result['columns'] = df.shape[1]
    result['memory_usage'] = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    
    # Column info with more details
    columns_info = {}
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'unique_values': df[col].nunique() if df[col].dtype == 'object' else None,
            'missing_values': df[col].isnull().sum(),
            'missing_percentage': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%"
        }
        columns_info[col] = col_info
    result['columns_info'] = columns_info

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        result['numeric_summary'] = df[numeric_cols].describe().to_dict()

    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        categorical_summary = {}
        for col in categorical_cols:
            categorical_summary[col] = {
                'count': df[col].count(),
                'unique': df[col].nunique(),
                'top': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'freq': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
            }
        result['categorical_summary'] = categorical_summary

    # Missing values with percentage
    missing_data = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    result['missing_values'] = {
        col: {
            'count': int(missing_data[col]),
            'percentage': f"{missing_percentage[col]:.2f}%"
        } for col in df.columns
    }

    # Advanced trend analysis
    trend_info = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            # Calculate percentage change for trend direction
            if len(col_data) > 1:
                pct_change = ((col_data.iloc[-1] - col_data.iloc[0]) / col_data.iloc[0]) * 100
                trend_direction = 'Increasing' if pct_change > 0 else 'Decreasing' if pct_change < 0 else 'Stable'
            else:
                pct_change = 0
                trend_direction = 'Insufficient data'
            
            trend_info[col] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'range': float(col_data.max() - col_data.min()),
                'trend_direction': trend_direction,
                'percentage_change': f"{pct_change:.2f}%" if len(col_data) > 1 else "N/A"
            }
    result['trend_analysis'] = trend_info

    # Outlier detection using IQR method
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': f"{(len(outliers) / len(df)) * 100:.2f}%",
            'values': outliers.tolist() if len(outliers) > 0 else None
        }
    result['outlier_analysis'] = outlier_info

    # Correlation analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        result['correlation_matrix'] = corr_matrix.to_dict()

        # Strong correlations with interpretation
        strong_corr = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i,j]
                if abs(val) > 0.5:  # Lowered threshold to capture more relationships
                    strength = "Strong" if abs(val) > 0.7 else "Moderate" if abs(val) > 0.5 else "Weak"
                    direction = "Positive" if val > 0 else "Negative"
                    strong_corr[f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}"] = {
                        'correlation': float(val),
                        'strength': strength,
                        'direction': direction
                    }
        result['correlation_analysis'] = strong_corr

    # Data quality score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    data_quality_score = ((total_cells - missing_cells) / total_cells) * 100
    result['data_quality_score'] = f"{data_quality_score:.2f}%"

    return result

# ------------------------------
# ENHANCED CHART RECOMMENDATION FUNCTION
# ------------------------------
def chart_recommendations(df):
    recommendations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Check for datetime columns in object columns
    for col in categorical_cols:
        try:
            pd.to_datetime(df[col])
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass

    # Histogram for numeric distributions
    for col in numeric_cols:
        recommendations.append({
            'chart': 'Histogram',
            'title': f'Distribution of {col}',
            'description': f'Shows the frequency distribution of {col} values.',
            'columns': [col],
            'priority': 'high' if len(numeric_cols) == 1 else 'medium'
        })

    # Bar chart for categorical data
    for col in categorical_cols:
        if df[col].nunique() < 20:  # Avoid charts with too many categories
            recommendations.append({
                'chart': 'Bar Chart',
                'title': f'Count of {col} Categories',
                'description': f'Shows the frequency of each category in {col}.',
                'columns': [col],
                'priority': 'high' if len(categorical_cols) == 1 else 'medium'
            })

    # Scatter plot for relationships
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                recommendations.append({
                    'chart': 'Scatter Plot',
                    'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                    'description': f'Shows relationship between {numeric_cols[i]} and {numeric_cols[j]}.',
                    'columns': [numeric_cols[i], numeric_cols[j]],
                    'priority': 'high' if i == 0 and j == 1 else 'medium'
                })

    # Line chart for time series
    if datetime_cols and numeric_cols:
        for dt_col in datetime_cols:
            for num_col in numeric_cols:
                recommendations.append({
                    'chart': 'Line Chart',
                    'title': f'{num_col} Over Time',
                    'description': f'Shows how {num_col} changes over time.',
                    'columns': [dt_col, num_col],
                    'priority': 'high'
                })

    # Box plot for distribution comparison
    if numeric_cols and categorical_cols:
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if df[cat_col].nunique() < 10:  # Avoid too many categories
                    recommendations.append({
                        'chart': 'Box Plot',
                        'title': f'Distribution of {num_col} by {cat_col}',
                        'description': f'Compares distribution of {num_col} across different {cat_col} categories.',
                        'columns': [cat_col, num_col],
                        'priority': 'medium'
                    })

    # Heatmap for correlation
    if len(numeric_cols) > 2:
        recommendations.append({
            'chart': 'Heatmap',
            'title': 'Correlation Matrix',
            'description': 'Visual representation of correlations between numeric variables.',
            'columns': numeric_cols,
            'priority': 'medium'
        })

    # Pie chart for composition (only if few categories)
    for col in categorical_cols:
        if df[col].nunique() <= 5:
            recommendations.append({
                'chart': 'Pie Chart',
                'title': f'Composition of {col}',
                'description': f'Shows proportional composition of {col} categories.',
                'columns': [col],
                'priority': 'low'
            })

    return recommendations

# ------------------------------
# API ENDPOINTS
# ------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = file.filename.lower()
    
    # Read file into memory
    file_content = file.read()
    file_stream = io.BytesIO(file_content)

    try:
        # Read file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(file_stream)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_stream)
        elif filename.endswith('.json'):
            df = pd.read_json(file_stream)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Basic data cleaning
        df = df.dropna(how='all')  # Remove completely empty rows
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
        
    except Exception as e:
        return jsonify({'error': f'File read error: {str(e)}'}), 400

    # Analysis
    analysis_result = analyze_data(df)
    chart_result = chart_recommendations(df)

    return jsonify({
        'filename': filename,
        'analysis': analysis_result,
        'chart_recommendations': chart_result,
        'processed_at': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)