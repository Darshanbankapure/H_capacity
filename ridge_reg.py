# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('data/tobacco_database.csv')

X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (recommended for Ridge Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameters for Ridge Regression
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10]
}

# Create and train the Ridge Regression model with hyperparameter tuning
ridge = Ridge()
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_ridge = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_ridge.predict(X_test)

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
plt.title('Regression Line (Ridge Regression)')
plt.grid(True)
plt.show()

# Plot the learning curve
def plot_learning_curve(estimator, X, y, title): 
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='neg_mean_squared_error')
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    test_rmse = np.sqrt(-test_scores.mean(axis=1))
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_rmse, 'o-', color='r', label='Training RMSE')
    plt.plot(train_sizes, test_rmse, 'o-', color='g', label='Validation RMSE')
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curve(ridge, X_train, y_train, 'Learning Curve (Ridge Regression)')


print("Best Hyperparameters:", grid_search.best_params_)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")
print(f"AUE: {aue:.2f}")
