import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据加载
data = pd.read_csv("TiO2_acetone_temperature_current_1.5ppm_5V.csv")

# 2. 数据清洗: 移除包含缺失值的行或使用合适的值填充
data = data.dropna()

# 3. 数据类型转换: 确保数值列是数值类型
data['Temperature(°C)'] = pd.to_numeric(data['Temperature(°C)'], errors='coerce')
data['Current(μA)'] = pd.to_numeric(data['Current(μA)'], errors='coerce')
data['Relative Response Intensity'] = pd.to_numeric(data['Relative Response Intensity'], errors='coerce')

# 4. 移除包含 NaN 的行 (由于转换错误)
data = data.dropna()

# 5. 特征选择
X = data[['Temperature(°C)', 'Current(μA)']]
y = data['Relative Response Intensity']

# 6. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 模型训练
xgbr = xgb.XGBRegressor(objective='reg:squarederror',
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42)

xgbr.fit(X_train, y_train)

# 8. 模型评估
y_pred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# 9. 预测 (示例)
new_data = pd.DataFrame({'Temperature(°C)': [280], 'Current(μA)': [11]})
predictions = xgbr.predict(new_data)
print("Predictions:", predictions)