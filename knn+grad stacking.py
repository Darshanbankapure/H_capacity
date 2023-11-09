from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
# Load your dataset
data = pd.read_csv('data/tobacco_database.csv')

# Define features (X) and target (y)
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the base models
knn = KNeighborsRegressor(n_neighbors=3)
gbr = GradientBoostingRegressor(n_estimators=200, random_state=42)

# Create the stacking ensemble
stacking_regressor = StackingRegressor(
    estimators=[('knn', knn), ('gbr', gbr)],
    final_estimator=LinearRegression(),
    cv=5
)

# Train the stacking ensemble
stacking_regressor.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = stacking_regressor.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE (Root Mean Squared Error): {rmse}')
print(f'MAE (Mean Absolute Error): {mae}')
print(f'R^2 Score: {r2}')
print(f"Mean Squared Error: {mse}")
