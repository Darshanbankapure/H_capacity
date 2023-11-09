import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1, l2
from scikeras.wrappers import KerasRegressor
import pandas as pd
# Load your dataset
data = pd.read_csv('data/tobacco_database.csv')

# Define features (X) and target (y)
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

# Normalize/Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction
pca = PCA(n_components=0.95)  # Adjust the number of components
X_pca = pca.fit_transform(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define a neural network model function
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse')
    return model

# Wrap the model with KerasRegressor for compatibility with scikit-learn
keras_model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=1)

# Model Ensembling with Stacking
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('gbr', GradientBoostingRegressor(n_estimators=100)),
    ('nn', keras_model)
]
stacked_regressor = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# Train the model using K-fold cross-validation
kf = KFold(n_splits=5)
cross_val_score(stacked_regressor, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

# Train on the full training set
stacked_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = stacked_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE: {mse}, RMSE: {rmse}, R^2: {r2}, MAE: {mae}')

