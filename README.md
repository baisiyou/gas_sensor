rubing:# TiO₂-based Acetone Gas Detection using XGBoost

## Overview
This project leverages **XGBoost** to develop a predictive model for detecting acetone gas concentration using **TiO₂-based sensors**. The dataset used in this study includes temperature, current, and current change rate measurements from TiO₂ sensors. The model predicts the **Current Change Rate**, which is an indicator of acetone concentration response at a working voltage of **5V**.

## Dataset
The dataset used for this project is stored in:
```
TiO2_acetone_temperature_current_1.5ppm_5V.csv
```
### Features:
- **Temperature( °C )**: Operating temperature of the TiO₂ sensor.
- **Current( μA )**: Measured current at the given operating conditions.
- **Current Change Rate**: Change in the current value due to acetone exposure (Target variable).

## Installation & Dependencies
To run this project, ensure you have the following dependencies installed:
```bash
pip install pandas numpy xgboost scikit-learn
```

## Implementation Steps
1. **Data Loading**: Load the dataset from CSV.
2. **Data Cleaning**: Remove missing values and convert columns to numeric format.
3. **Feature Selection**: Select **Temperature( °C )** and **Current( μA )** as independent variables.
4. **Data Splitting**: Split data into training and test sets (80% training, 20% testing).
5. **Model Training**: Train an XGBoost Regressor.
6. **Model Evaluation**: Evaluate performance using **Mean Squared Error (MSE)** and **R-squared (R²) Score**.
7. **Prediction Example**: Provide an example of model inference with new data.

## Code Usage
Run the following Python script to train and evaluate the model:
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("TiO2_acetone_temperature_current_1.5ppm_5V.csv")

# Data cleaning
data = data.dropna()
data['Temperature(°C)'] = pd.to_numeric(data['Temperature(°C)'], errors='coerce')
data['Current(μA)'] = pd.to_numeric(data['Current(μA)'], errors='coerce')
data['Current Change Rate'] = pd.to_numeric(data['Current Change Rate'], errors='coerce')
data = data.dropna()

# Feature selection
X = data[['Temperature(°C)', 'Current(μA)']]
y = data['Current Change Rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
xgbr = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=5,
                         random_state=42)
xgbr.fit(X_train, y_train)

# Model Evaluation
y_pred = xgbr.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Example Prediction
new_data = pd.DataFrame({'Temperature(°C)': [280], 'Current(μA)': [11]})
prediction = xgbr.predict(new_data)
print("Predicted Current Change Rate:", prediction)
```

## Results
The model evaluates the relationship between temperature, current, and current change rate, achieving reasonable predictive accuracy using the **XGBoost regressor**.

### Example Output:
```
Mean Squared Error: 0.0023
R-squared: 0.89
Predicted Current Change Rate: [0.0156]
```

## Future Work
- Improve feature engineering by incorporating additional sensor parameters.
- Experiment with **hyperparameter tuning** for better model performance.
- Explore deep learning models for more advanced predictive accuracy.

## Author
Rubing zhang

## License
This project is open-source under the MIT License.

