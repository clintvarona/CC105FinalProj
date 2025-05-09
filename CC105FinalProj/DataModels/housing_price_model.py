import pickle
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and prepare the dataset
dataset = pd.read_csv('DataModels/housing_train.csv')
dataset.head()

# Separate features and target
X = dataset.drop('price', axis=1)
y = dataset['price']

# Handle missing values in numerical columns
for column in X.select_dtypes(include=['float64', 'int64']).columns:
    X[column] = X[column].fillna(X[column].mean())

# Handle categorical features with one-hot encoding
categorical_features = X.select_dtypes(include=['object']).columns
for feature in categorical_features:
    X[feature] = X[feature].fillna(X[feature].mode()[0])
    dummies = pd.get_dummies(X[feature], prefix=feature, drop_first=True)
    X = pd.concat([X, dummies], axis=1)

# Drop original categorical columns
X = X.drop(categorical_features, axis=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize and train the Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# Evaluate on validation set
y_pred = gb_model.predict(X_val_scaled)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Validation Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Evaluate on test set
test_data = pd.read_csv('DataModels/housing_test.csv')
X_test = test_data.drop('price', axis=1)
y_test = test_data['price']

# Preprocess test data similarly
for column in X_test.select_dtypes(include=['float64', 'int64']).columns:
    X_test[column] = X_test[column].fillna(X_test[column].mean())

for feature in X_test.select_dtypes(include=['object']).columns:
    X_test[feature] = X_test[feature].fillna(X_test[feature].mode()[0])
    dummies = pd.get_dummies(X_test[feature], prefix=feature, drop_first=True)
    X_test = pd.concat([X_test, dummies], axis=1)

X_test = X_test.drop(X_test.select_dtypes(include=['object']).columns, axis=1)

# Make sure test data has the same columns as training data
missing_cols = set(X.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

# Ensure the order of columns matches the training data
X_test = X_test[X.columns]

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Make predictions on test set
test_pred = gb_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\nTest Results:")
print(f"Mean Squared Error: {test_mse:.2f}")
print(f"Root Mean Squared Error: {test_rmse:.2f}")
print(f"Mean Absolute Error: {test_mae:.2f}")
print(f"R² Score: {test_r2:.2f}")

# Save model and scaler to current directory
pickle.dump(gb_model, open("DataModels/housing_price_model.pkl", "wb"))
pickle.dump(scaler, open("DataModels/housing_scaler.pkl", "wb"))

# Also save to the parent directory if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pickle.dump(gb_model, open(os.path.join(parent_dir, "housing_price_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(parent_dir, "housing_scaler.pkl"), "wb"))

print(f"\nModel files saved successfully to:")
print(f"- {os.path.abspath('DataModels/housing_price_model.pkl')}")
print(f"- {os.path.join(parent_dir, 'housing_price_model.pkl')}") 