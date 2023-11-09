import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, TimeDistributed

# Define the CNN-LSTM model
def create_cnn_lstm_model(input_shape):
    model = Sequential()
    
    # CNN layers
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    
    # LSTM layer
    model.add(LSTM(50, activation='relu'))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model


# Sample data generation function - replace this with actual data loading
def load_data(num_samples, input_shape, num_classes):
    # Generate random data (replace with actual data loading)
    X = np.random.random((num_samples,) + input_shape).astype(np.float32)
    # Generate random integer labels for each sample
    y_integers = np.random.randint(0, num_classes, size=(num_samples,))
    # Convert integer labels to one-hot encoded labels
    y = tf.keras.utils.to_categorical(y_integers, num_classes=num_classes)
    return X, y

# Set parameters
num_samples = 1000  # Number of samples in the data (replace with actual number)
num_classes = 10    # Number of output classes (replace with actual number of classes)
input_shape = (100, 10, 1)  # Input shape (time steps, features, channels)

# Load data
X_train, y_train = load_data(num_samples, input_shape, num_classes)
X_val, y_val = load_data(num_samples // 10, input_shape, num_classes)  # Validation data

# Create the model
cnn_lstm_model = create_cnn_lstm_model(input_shape)

# Train the model
history = cnn_lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
scores = cnn_lstm_model.evaluate(X_val, y_val, verbose=0)
print(f'Evaluation on the validation set:\nLoss: {scores[0]}\nAccuracy: {scores[1]*100}%')

# Save the model if needed
# cnn_lstm_model.save('cnn_lstm_model.h5')

# Predict on the validation set
y_pred = cnn_lstm_model.predict(X_val)

# Calculate the metrics for the predictions
mse = mean_squared_error(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
rmse = np.sqrt(mse)
r2 = r2_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
aue = np.mean(np.abs(np.argmax(y_val, axis=1) - np.argmax(y_pred, axis=1)))

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared: {r2}')
print(f'Absolute Unsigned Error (AUE): {aue}')
