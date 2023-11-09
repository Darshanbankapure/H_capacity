import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Assuming 'data' is a pandas DataFrame with your dataset
# and 'your_sequence_data.csv' is your actual sequence data file
data = pd.read_csv('data/tobacco_database.csv')

# Splitting features and target
X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']].values
y = data['surface_area_m2cm3'].values

# Data preprocessing
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Reshape data for LSTM [samples, time steps, features]
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM Network architecture
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the LSTM model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predict and evaluate the model
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Transform back to original scale

y_test = scaler_y.inverse_transform(y_test)  # Transform back to original scale

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)
aue_cnn = mean_absolute_error(y_test, y_pred)
# Print metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")
print(f"Absolute Unsigned Error (AUE): {aue_cnn:.2f}")

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

# Now call this function for each model
# Replace lstm_y_true, lstm_y_pred, cnn_y_true, etc., with your actual data
plot_model_results(y_test, y_pred, 'LSTM')





