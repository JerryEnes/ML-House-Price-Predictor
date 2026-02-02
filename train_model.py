import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("Training house price model...")

# Create sample data
np.random.seed(42)
data = {
    'area': np.random.randint(500, 5000, 100),
    'bedrooms': np.random.randint(1, 7, 100),
    'price': np.random.randint(100000, 1000000, 100)
}
df = pd.DataFrame(data)

# Simple model
X = df[['area', 'bedrooms']]
y = df['price']

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Save model
os.makedirs('app/models', exist_ok=True)
joblib.dump(model, 'app/models/house_price_model.joblib')
joblib.dump(['area', 'bedrooms'], 'app/models/features.joblib')

print("Model training complete!")
