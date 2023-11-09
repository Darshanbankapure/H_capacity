import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Load the dataset
file_path = 'data/tobacco_database.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Assume the last column is the target variable and the rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Reshape the data to fit the model
# CNN-LSTM expects input in the form of (samples, timesteps, features)
# Since we don't have a natural sequence, we'll treat each feature as a timestep
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Define the CNN-LSTM model architecture
def create_cnn_lstm_model(input_shape):
    model = Sequential()
    
    # CNN layers
    model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
    
    # LSTM layers
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

# Create the model with the correct input shape
input_shape = X_train.shape[1:]
cnn_lstm_model = create_cnn_lstm_model(input_shape)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

# Train the model with early stopping
history = cnn_lstm_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Evaluate the model
mse = cnn_lstm_model.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(mse)

# Predict the values
y_pred = cnn_lstm_model.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Absolute Unsigned Error (AUE)
aue = np.mean(np.abs(y_pred.flatten() - y_test))

print(f"Mean Squared Error on the test set: {mse}")
print(f"Root Mean Squared Error on the test set: {rmse}")
print(f"R-squared on the test set: {r_squared}")
print(f"Mean Absolute Error (MAE) on the test set: {mae}")
print(f"Absolute Unsigned Error (AUE) on the test set: {aue}")

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

plot_model_results(y_test, y_pred, 'CNN+LSTM')