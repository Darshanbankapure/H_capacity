# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/tobacco_database.csv')

X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but often recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameters for AdaBoostRegressor
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.2]
}

# Create and train the AdaBoostRegressor model with hyperparameter tuning
adaboost = AdaBoostRegressor(base_estimator=None, random_state=42)
grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_adaboost = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_adaboost.predict(X_test)

# Calculate AUE (Absolute Unsigned Error)
aue = abs(y_test - y_pred).mean()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_squared = r2_score(y_test, y_pred)

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Hydrogen Storage Capacity')
plt.ylabel('Predicted Hydrogen Storage Capacity')
plt.title('Regression Line (AdaBoost Regression)')
plt.grid(True)
plt.show()

# Plot R-squared graph
estimators_range = range(1, 101)  # Range of estimator values to plot
r2_scores = []

for n_estimators in estimators_range:
    adaboost = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)
    adaboost.fit(X_train, y_train)
    y_pred = adaboost.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(estimators_range, r2_scores, marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('R-squared Score')
plt.title('R-squared Score vs. Number of Estimators (AdaBoost Regression)')
plt.grid(True)
plt.show()

# Initialize lists to store RMSE and AUE values
rmse_values = []
aue_values = []

for n_estimators in estimators_range:
    # Create and train the AdaBoost Regression model
    adaboost = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)
    adaboost.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = adaboost.predict(X_test)
    
    # Calculate AUE (Absolute Unsigned Error)
    aue = abs(y_test - y_pred).mean()
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    # Append RMSE and AUE values to the respective lists
    rmse_values.append(rmse)
    aue_values.append(aue)

# Plot RMSE and AUE values against the number of estimators
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(estimators_range, rmse_values, marker='o', label='RMSE', color='blue')
plt.xlabel('Number of Estimators')
plt.ylabel('RMSE')
plt.title('RMSE vs. Number of Estimators (AdaBoost Regression)')
plt.grid(True)

plt.subplot(122)
plt.plot(estimators_range, aue_values, marker='o', label='AUE', color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('AUE')
plt.title('AUE vs. Number of Estimators (AdaBoost Regression)')
plt.grid(True)

plt.tight_layout()
plt.show()


print("Best Hyperparameters:", grid_search.best_params_)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")
print(f"AUE: {aue:.2f}")
