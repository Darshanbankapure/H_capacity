import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('data/tobacco_database.csv')

X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (recommended for Lasso Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Lasso Regression model
lasso = Lasso(alpha=0.01)  # You can adjust the alpha value
lasso.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test)

# Calculate AUE (Absolute Unsigned Error)
aue = abs(y_test - y_pred).mean()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_squared = r2_score(y_test, y_pred)

models = {
    'Ridge and Lasso': (y_test, y_pred),
}

for name, (y_true, y_pred) in models.items():
    plt.figure(figsize=(10, 6))

    # Scatter plot for actual vs predicted values
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    
    # Ideal line for perfect predictions
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.title(f'{name} - Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()
