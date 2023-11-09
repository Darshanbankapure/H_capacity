import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
# Load your dataset
data = pd.read_csv('data/tobacco_database.csv')

# Define features (X) and target (y)
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base models
knn = KNeighborsRegressor(n_neighbors=7)
dt = DecisionTreeRegressor(random_state=42)
ada = AdaBoostRegressor(estimator=dt, n_estimators=200, random_state=42)
tree = DecisionTreeRegressor(random_state=42)
# Stacking model
stacked_model = StackingRegressor(
    estimators=[('knn', knn), ('ada', ada), ('tree', tree)],
    final_estimator=LinearRegression(),
    cv=5
)

# Train the stacking model
stacked_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = stacked_model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE (Mean Squared Error): {mse}')
print(f'RMSE (Root Mean Squared Error): {rmse}')
print(f'MAE (Mean Absolute Error or AUE): {mae}')
print(f'R^2 Score: {r2}')

def plot_model_results(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='black', linewidth=2)
    plt.title(f'{model_name} Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

# Example usage with your AdaBoost stacked model:
# Assuming that `ada_y_true` and `ada_y_pred` are your true and predicted values
plot_model_results(y_test, y_pred, 'AdaBoost + KNN Stacked Model')