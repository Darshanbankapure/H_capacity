import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import pandas as pd
# Load your dataset
data = pd.read_csv('data/tobacco_database.csv')

# Define features (X) and target (y)
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)

# Initialize the AdaBoostRegressor with KNN as the base estimator
ada_boost_knn = AdaBoostRegressor(base_estimator=knn_regressor, n_estimators=50, random_state=42)

# Fit the AdaBoost model to the training data
ada_boost_knn.fit(X_train, y_train)

# Predict on the test data
y_pred = ada_boost_knn.predict(X_test)

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the performance metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared Score (R2): {r2}')
print(f'Mean Absolute Error (MAE): {mae}')
