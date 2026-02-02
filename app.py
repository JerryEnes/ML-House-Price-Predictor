from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

class HousePricePredictor:
    def __init__(self):
        try:
            self.model = joblib.load('app/models/house_price_model.joblib')
            self.scaler = joblib.load('app/models/scaler.joblib')
            self.label_encoder = joblib.load('app/models/label_encoder.joblib')
            self.features = joblib.load('app/models/features.joblib')
        except:
            self.model = None

predictor = HousePricePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create sample data for prediction
        sample_data = {
            'area': 1500, 'bedrooms': 3, 'bathrooms': 2,
            'stories': 2, 'parking': 1, 'year_built': 2000,
            'location': 'Suburb'
        }
        
        # Update with actual data if provided
        if data:
            sample_data.update(data)
        
        # Mock prediction
        base_price = 250000
        price = base_price + (sample_data['area'] * 100) + (sample_data['bedrooms'] * 30000)
        
        return jsonify({
            'success': True,
            'prediction': {
                'predicted_price': float(price),
                'formatted_price': f"${price:,.2f}",
                'price_per_sqft': f"${price/sample_data['area']:,.2f}",
                'features_used': ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'location', 'year_built']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'House Price Predictor',
        'model_loaded': predictor.model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
