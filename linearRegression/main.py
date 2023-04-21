import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load the data
train_df = pd.read_csv('../dataset/train/train.csv')
test_df = pd.read_csv('../dataset/test/test.csv')

# Define the features and target variable
features = ['year', 'manufacturer', 'fuel', 'odometer', 'title_status', 'transmission', 'type', 'state']
target = 'price'

# Prepare the data
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# One-hot encode categorical features
categorical_features = ['year', 'manufacturer', 'fuel', 'title_status', 'transmission', 'type', 'state']
numerical_features = ['odometer']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Build the linear regression model
model = LinearRegression()

# Train the model on the training data
X_train_transformed = preprocessor.fit_transform(X_train)
model.fit(X_train_transformed, y_train)

joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(model, 'model.joblib')

# Test the model on the test data
X_test_transformed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_transformed)

# Calculate the mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse:.2f}')
