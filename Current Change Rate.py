import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Loading
data = pd.read_csv("TiO2_acetone_temperature_current_1.5ppm_5V.csv")

# 2. Data Cleaning: Remove rows with missing values or fill with appropriate values
data = data.dropna()

# 3. Data Type Conversion: Ensure numerical columns are of numeric type
data['Temperature(°C)'] = pd.to_numeric(data['Temperature(°C)'], errors='coerce')
data['Current(μA)'] = pd.to_numeric(data['Current(μA)'], errors='coerce')
data['Current Change Rate'] = pd.to_numeric(data['Current Change Rate'], errors='coerce')

# 4. Remove rows containing NaN (due to conversion errors)
data = data.dropna()

# 5. Feature Selection
X = data[['Temperature(°C)', 'Current(μA)']]
y = data['Current Change Rate']

# 6. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training
xgbr = xgb.XGBRegressor(objective='reg:squarederror',
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42)

xgbr.fit(X_train, y_train)

# 8. Model Evaluation
y_pred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# 9. Prediction (Example)
new_data = pd.DataFrame({'Temperature(°C)': [280], 'Current(μA)': [11]})
predictions = xgbr.predict(new_data)
print("Predictions:", predictions)