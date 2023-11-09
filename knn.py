import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('data/tobacco_database.csv')

# Define features (X) and target (y)
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It is a good practice to scale features when using KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create KNN regressor object
# You can change n_neighbors to find the best parameter for your model
knn_regressor = KNeighborsRegressor(n_neighbors=5)

# Train the regressor
knn_regressor.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = knn_regressor.predict(X_test_scaled)

# Evaluate the model using various metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
aue = abs(y_test - y_pred).mean()

# Print the performance metrics
print(f"KNN - Mean Squared Error: {mse:.2f}")
print(f"KNN - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"KNN - R-squared: {r2:.2f}")
print(f"KNN - AUE: {aue:.2f}")

def plot_model_results(y_true, y_pred, model_name):
    # Convert to a one-dimensional array if necessary
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Plotting the actual vs predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    # Plotting the perfect prediction line
    max_value = max(y_true.max(), y_pred.max())
    min_value = min(y_true.min(), y_pred.min())
    plt.plot([min_value, max_value], [min_value, max_value], 'k--')

    plt.show()

plot_model_results(y_test, y_pred, 'KNN')
