import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('data/tobacco_database.csv')

X = data[['void_fraction', 'surface_area_m2g', 'pld', 'lcd']]
y = data['surface_area_m2cm3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the DNN model
def build_dnn_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(1)  # Output layer for regression should have 1 neuron and no activation
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model

# Get the input shape from the training data
input_shape = (X_train_scaled.shape[1],)

# Build the model
dnn_model = build_dnn_model(input_shape)

# Define Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with Early Stopping
history = dnn_model.fit(X_train_scaled, y_train, 
                        validation_split=0.2, 
                        epochs=200, 
                        batch_size=32, 
                        callbacks=[early_stopping])
dnn_model.save('dnn.h5')
# Evaluate the model
predicted_y_test = dnn_model.predict(X_test_scaled)

# Calculate additional metrics
mse = mean_squared_error(y_test, predicted_y_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicted_y_test)
aue = mean_absolute_error(y_test, predicted_y_test)


print(f"Mean Squared Error on the test set: {mse}")
print(f"Root Mean Squared Error on the test set: {rmse}")
print(f"R-squared on the test set: {r2}")
print(f"Mean Absolute Error (MAE) on the test set: {aue}")

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

plot_model_results(y_test, predicted_y_test, 'Deep NN')