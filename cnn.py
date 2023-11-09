import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models


# Assuming your dataset is already loaded and is named 'data'
# Replace 'your_dataset.csv' with your actual dataset file if needed
data = pd.read_csv('data/tobacco_database.csv')

# Splitting features and target
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']].values
y = data['surface_area_m2cm3'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for the CNN input
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# CNN Model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))  # No activation function for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

cnn_model = create_cnn_model((1, X_train_scaled.shape[1]))

# Model summary
cnn_model.summary()

# Fit the model
history = cnn_model.fit(
    X_train_cnn,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_cnn, y_test)
)

# Predict
y_pred_cnn = cnn_model.predict(X_test_cnn).flatten()

# Evaluate
mse_cnn = mean_squared_error(y_test, y_pred_cnn)
rmse_cnn = np.sqrt(mse_cnn)
r_squared_cnn = r2_score(y_test, y_pred_cnn)
aue_cnn = mean_absolute_error(y_test, y_pred_cnn)

print(f"CNN - Mean Squared Error: {mse_cnn:.3f}")
print(f"CNN - Root Mean Squared Error (RMSE): {rmse_cnn:.3f}")
print(f"CNN - R-squared: {r_squared_cnn:.3f}")
print(f"CNN - Absolute Unsigned Error (AUE): {aue_cnn:.2f}")

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



plot_model_results(y_test, y_pred_cnn, 'CNN')
