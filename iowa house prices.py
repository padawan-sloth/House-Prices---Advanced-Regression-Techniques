import pandas as pd

# Load the training dataset
train_data = pd.read_csv('/mnt/data/train.csv')

# Display column names to confirm the presence of 'SalePrice'
print("Columns in training data:", train_data.columns.tolist())

# Specifically check for 'SalePrice' in the columns
if 'SalePrice' not in train_data.columns:
    raise ValueError("The 'SalePrice' column is missing from your training data.")
else:
    print("The 'SalePrice' column is present.")
    print(train_data['SalePrice'].head())  # Display the first few entries of 'SalePrice'

# Assume SalePrice is confirmed to be present from Step 1
# Proceed with data preprocessing while ensuring not to drop SalePrice

# Fill missing values for simplicity
train_data.fillna(method='ffill', inplace=True)

# Encoding categorical variables without affecting SalePrice
train_data = pd.get_dummies(train_data, drop_first=True)

# Confirm that SalePrice is still in the dataframe
if 'SalePrice' not in train_data.columns:
    raise ValueError("The 'SalePrice' column was dropped during preprocessing.")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare feature matrix X and target vector y
X = train_data.drop(['SalePrice'], axis=1)  # Drop 'SalePrice' only here to form feature matrix
y = np.log(train_data['SalePrice'])  # Log-transform SalePrice to form target vector

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')

# Assuming test data needs to be loaded and processed similarly to train_data
test_data = pd.read_csv('/mnt/data/test.csv')
test_data.fillna(method='ffill', inplace=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Align test data with the training features
final_test = test_data.reindex(columns = X_train.columns, fill_value=0)

# Generate predictions
test_predictions = model.predict(final_test)
test_predictions = np.exp(test_predictions)  # Reverse the log transform for final prediction

# Create submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

submission.to_csv('/mnt/data/house_prices_submission.csv', index=False)

print("Submission file created.")
