import pandas as pd
import numpy as np
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

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Surface Area (m²/cm³)')
plt.ylabel('Predicted Surface Area (m²/cm³)')
plt.title('Regression Line (Lasso Regression)')
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

plot_learning_curve(lasso, X_train, y_train, 'Learning Curve (Lasso Regression)')

# Plot R-squared (R2) and AUE values
plt.figure(figsize=(10, 6))
plt.bar(['R-squared (R2)', 'AUE'], [r_squared, aue], color=['blue', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('R-squared (R2) and AUE for Lasso Regression')
plt.ylim(0, 1)  # Adjust the y-axis limit as needed
plt.grid(axis='y')
plt.show()

# Fixed alpha value
alpha = 0.01

# Initialize lists to store RMSE and AUE values
rmse_values = []
aue_values = []

# Create and train the Lasso Regression model with the fixed alpha
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test)

# Calculate AUE (Absolute Unsigned Error)
aue = abs(y_test - y_pred).mean()

# Calculate RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Append RMSE and AUE values to the respective lists
rmse_values.append(rmse)
aue_values.append(aue)

# Plot RMSE and AUE values for the fixed alpha
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot([alpha], rmse_values, marker='o', label='RMSE', color='blue')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs. Alpha (Lasso Regression)')
plt.grid(True)

plt.subplot(122)
plt.plot([alpha], aue_values, marker='o', label='AUE', color='green')
plt.xlabel('Alpha')
plt.ylabel('AUE')
plt.title('AUE vs. Alpha (Lasso Regression)')
plt.grid(True)

plt.tight_layout()
plt.show()
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")
print(f"AUE: {aue:.2f}")
