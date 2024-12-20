# Linear Regression Model for House Price Prediction

This project demonstrates a **Linear Regression** approach to predict house prices using a dataset that includes various features such as square footage, number of bedrooms, and bathrooms. The model is built using Python and popular libraries like `pandas`, `scikit-learn`, and `matplotlib`.

---

## Project Workflow

### 1. **Data Loading**
- Training and test datasets are loaded from CSV files.
- Path to the datasets:
  - `train.csv`: Contains the training data with features and target values (`SalePrice`).
  - `test.csv`: Contains the test data for making predictions.

### 2. **Data Preprocessing**
- **Handling Missing Values**:
  - Rows with missing target values (`SalePrice`) are removed.
  - Missing values in features are filled with 0 as a simple handling strategy.
- **Feature Selection**:
  - Selected features: `GrLivArea`, `BedroomAbvGr`, and `FullBath`.
  - Target: `SalePrice`.

### 3. **Data Splitting**
- The data is split into **training** and **validation** sets using an 80-20 split.

### 4. **Feature Scaling**
- Features are scaled using `StandardScaler` to normalize the data for better model performance.

### 5. **Model Training**
- A **Linear Regression** model is trained on the scaled training data.

### 6. **Model Evaluation**
- Predictions are made on the validation set.
- Performance metrics include:
  - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted prices.
  - **R-squared (R2)**: Indicates the proportion of variance in the dependent variable explained by the model.
- A scatter plot visualizes the comparison of actual vs. predicted prices.

### 7. **Test Predictions**
- The test data undergoes the same preprocessing steps as the training data.
- Predictions are made using the trained model.
- Results are saved to a CSV file named `submission.csv` with the columns:
  - `Id`: The unique identifier for each house in the test dataset.
  - `SalePrice`: The predicted house prices.

---

## How to Run the Code
1. Ensure you have the necessary files:
   - `train.csv`
   - `test.csv`
2. Update the file paths in the code to point to your local files:
   ```python
   train_data = pd.read_csv('C:\\path\\to\\train.csv')
   test_data = pd.read_csv('C:\\path\\to\\test.csv')
   ```
3. Run the script in your Python environment (e.g., VSCode, Jupyter Notebook).

---

## Requirements
Install the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Results
- The model generates a `submission.csv` file with predicted house prices for the test dataset.
- Example file structure:
  ```csv
  Id,SalePrice
  1461,208500.0
  1462,181500.0
  1463,223500.0
  ```

---

## Visualization
The script includes a scatter plot to compare actual vs. predicted prices for the validation set, providing insights into the model's performance.

---

## Future Improvements
- Use advanced feature engineering to include more relevant features.
- Experiment with more sophisticated models like **Ridge Regression** or **Random Forest** for potentially better performance.
- Optimize hyperparameters to further improve prediction accuracy.
