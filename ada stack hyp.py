import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
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
knn = KNeighborsRegressor()
dt = DecisionTreeRegressor(random_state=42)
ada = AdaBoostRegressor(estimator=dt, random_state=42)

# Define parameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'ada__n_estimators': [50, 100, 200],
    'ada__learning_rate': [0.01, 0.1, 1]
}

# Stacking model with GridSearchCV
stacked_model = StackingRegressor(
    estimators=[('knn', knn), ('ada', ada)],
    final_estimator=LinearRegression(),
    cv=5
)

grid_search = GridSearchCV(estimator=stacked_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model after hyperparameter tuning
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE (Root Mean Squared Error): {rmse}')
print(f'MAE (Mean Absolute Error or AUE): {mae}')
print(f'R^2 Score: {r2}')
print(f'Best Parameters: {grid_search.best_params_}')
