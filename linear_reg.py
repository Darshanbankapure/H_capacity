import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('data/tobacco_database.csv')

# Define predictors and response variable
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = linear_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_squared = r2_score(y_test, y_pred)

# Calculate AUE (Absolute Unsigned Error)
aue = abs(y_test - y_pred).mean()

# Display the results
print(f"Linear Regression - Mean Squared Error: {mse:.2f}")
print(f"Linear Regression - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Linear Regression - R-squared: {r_squared:.2f}")
print(f"Linear Regression - AUE: {aue:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming y_true and y_pred are the true and predicted values for each model
models = {
    'Linear Regression': (y_test, y_pred),
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

    # Plot for residuals
    residuals = np.array(y_true) - np.array(y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'{name} - Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()
