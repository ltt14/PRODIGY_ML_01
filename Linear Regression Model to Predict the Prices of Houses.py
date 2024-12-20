#Importing important Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Loading train and test data
train_data = pd.read_csv('C:\\Users\\cheru\\OneDrive\\Documents\\VSCode\\train.csv')
test_data = pd.read_csv('C:\\Users\\cheru\\OneDrive\\Documents\\VSCode\\test.csv')
##Data Preprocessing
# Display basic information
print(train_data.info())

# Handle missing values (dropping rows with missing target or filling missing feature values)
train_data = train_data.dropna(subset=['SalePrice'])  # Ensure no missing target values
train_data = train_data.fillna(0)  # Simplest approach for missing features

# Select relevant features (square footage, bedrooms, bathrooms)
selected_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']  # Adjust based on Kaggle dataset
features = train_data[selected_features]
target = train_data['SalePrice']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

#Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

#Building and training the model

model = LinearRegression()
model.fit(X_train, y_train)

#Evaluating the model
# Make predictions
y_pred = model.predict(X_val)

# Calculate metrics
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

#Visualizing the results
plt.scatter(y_val, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

#Make predictions on the test dataset
# Process test data using the same selected features and scaling
test_features = test_data[selected_features]
test_features = scaler.transform(test_features)

# Predict SalePrice for the test dataset
test_predictions = model.predict(test_features)

# Create a submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],  # Include the 'Id' column from the test dataset
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
